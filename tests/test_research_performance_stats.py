from __future__ import annotations

import copy
import hashlib
import inspect
import json
from contextlib import contextmanager
from functools import lru_cache
from types import SimpleNamespace

import pytest
from service_test_client import OperatorClient as TestClient

from baloot_platform.challenge.constraint_deal_search import (
    DealConstraintSpecV1,
    OrToolsCpSatDealGeneratorV1,
)
from baloot_platform.config import build_bundle
from baloot_platform.core.enums import Seat
from baloot_platform.research.contracts import canonical_sha256
from baloot_platform.research.fenced_occurrence import build_fenced_occurrence
from baloot_platform.research.performance_stats import (
    CALLING_THREAD_CPU_BOUNDARY,
    PROCESS_PEAK_RSS_BOUNDARY,
    build_research_environment_declaration,
    build_research_performance_by_lane,
    build_research_performance_rollup,
    build_sequential_search_receipt_item,
    build_sequential_search_receipt_page,
    build_sequential_search_receipt_rollup,
    build_sequential_search_receipt_summary,
    validate_v4_partial_performance_receipt,
    validate_v4_performance_receipt,
)
from baloot_platform.research.postgres_store import PostgresResearchStoreV1
from baloot_platform.research.sequential_candidate_search import (
    SequentialCandidateEvaluationV1,
    SequentialCandidateOutcomeKindV1,
    run_sequential_candidate_search_v1,
)
from baloot_platform.research.sharding import partition_research_shards
from baloot_platform.research.service import ResearchFoundryServiceV1
from baloot_platform.research.store import SQLiteResearchStoreV1


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_SEQUENTIAL_SUMMARY_FIELDS = {
    "admittedSeedCount",
    "exactRejectionCount",
    "noGoodAppliedCount",
    "noGoodUnappliedCount",
    "continuationSolveCount",
    "continuationStatusDistribution",
    "stopReasonDistribution",
}
_EMPTY_SEQUENTIAL_SUMMARY = {
    "admittedSeedCount": 0,
    "exactRejectionCount": 0,
    "noGoodAppliedCount": 0,
    "noGoodUnappliedCount": 0,
    "continuationSolveCount": 0,
    "continuationStatusDistribution": {
        "optimal": 0,
        "infeasible": 0,
    },
    "stopReasonDistribution": {
        "candidates_exhausted": 0,
        "maximum_solutions_reached": 0,
        "qualified": 0,
        "incomplete": 0,
        "cancelled": 0,
        "fatal_quarantine": 0,
    },
}
_SEQUENTIAL_RULES = build_bundle().rules


def _sequential_spec() -> DealConstraintSpecV1:
    return DealConstraintSpecV1.build(
        {
            "schemaVersion": 1,
            "id": "performance_receipt_summary_v1",
            "revision": "1",
            "enabled": True,
            "backend": "ortools_cp_sat_v1",
            "assumedBuyer": None,
            "purchaseSuits": [],
            "initialRequiredCards": {},
            "finalRequiredCards": {},
            "finalForbiddenCards": {},
            "initialSuitCountRanges": {},
            "finalSuitCountRanges": {},
            "projectPattern": None,
            "maximumSolutions": 2,
            "maximumConflicts": 1000,
            "safetyTimeoutSeconds": 1.0,
            "solverWorkers": 1,
            "seedOffset": 0,
            "tags": ["engine_validation_required"],
        }
    )


class _DeterministicSequentialGenerator:
    descriptor = OrToolsCpSatDealGeneratorV1.descriptor

    def __init__(
        self,
        spec: DealConstraintSpecV1,
        *,
        root_seed: int,
        candidate_count: int,
    ) -> None:
        self.model_hash = _hash("performance-sequential-model")
        self.backend_version = "9.15.6755"
        self.no_good_cut_type = "complete_deck_assignment_v1"
        self.search_identity = OrToolsCpSatDealGeneratorV1.search_identity_document(
            spec=spec,
            model_hash=self.model_hash,
            root_seed=root_seed,
            dealer=Seat.SOUTH,
            rules=_SEQUENTIAL_RULES,
            backend_version=self.backend_version,
            no_good_cut_type=self.no_good_cut_type,
        )
        candidates = []
        cuts: list[str] = []
        for ordinal in range(candidate_count + 1):
            deck = tuple(range(ordinal, 32)) + tuple(range(ordinal))
            candidate = OrToolsCpSatDealGeneratorV1._build_candidate(
                ordinal=ordinal,
                deck_order=deck,
                deck_commitment=hashlib.sha256(bytes(deck)).hexdigest(),
                ordering_score=900,
                structural_objective_score=42,
                profile_hash=spec.sha256,
                model_hash=self.model_hash,
                search_identity_hash=self.search_identity["sha256"],
                backend_version=self.backend_version,
                no_good_cut_type=self.no_good_cut_type,
                preceding_no_good_cut_hashes=tuple(cuts),
            )
            candidates.append(candidate)
            cuts.append(
                OrToolsCpSatDealGeneratorV1.exact_no_good_cut_document(
                    candidate,
                    layout=None,
                )["cutSha256"]
            )
        self.candidates = tuple(candidates[:candidate_count])
        self.witness = candidates[-1]
        self.cuts = tuple(cuts[:candidate_count])

    def generate_next(self, _spec, *, rejected_candidates, **_kwargs):
        ordinal = len(rejected_candidates)
        return self.candidates[ordinal] if ordinal < len(self.candidates) else None

    def probe_continuation(
        self,
        spec,
        *,
        rejected_candidates,
        **_kwargs,
    ):
        exhausted = not self.candidates
        return OrToolsCpSatDealGeneratorV1._build_continuation_receipt(
            profile_hash=spec.sha256,
            model_hash=self.model_hash,
            backend_version=self.backend_version,
            solver_status="infeasible" if exhausted else "optimal",
            solution_ordinal=len(rejected_candidates),
            search_identity_hash=self.search_identity["sha256"],
            preceding_no_good_cut_hashes=self.cuts[: len(rejected_candidates)],
            no_good_cut_type=self.no_good_cut_type,
            exhausted=exhausted,
            witness=None if exhausted else self.witness,
            structural_objective_score=42,
        )


@lru_cache(maxsize=None)
def _sequential_receipt(seed: int, stop_reason: str) -> dict:
    spec = _sequential_spec()
    generator = _DeterministicSequentialGenerator(
        spec,
        root_seed=seed,
        candidate_count=(
            0
            if stop_reason == "candidates_exhausted"
            else 2
            if stop_reason == "maximum_solutions_reached"
            else 1
        ),
    )
    if stop_reason == "candidates_exhausted":

        def evaluate(*_args):
            pytest.fail("an empty search cannot evaluate a candidate")
    else:
        outcome = (
            SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION
            if stop_reason == "maximum_solutions_reached"
            else SequentialCandidateOutcomeKindV1(stop_reason)
        )

        def evaluate(candidate, _index):
            return SequentialCandidateEvaluationV1.build(
                outcome=outcome,
                candidate_hash=candidate.candidate_hash,
                evaluation_evidence_hashes=(
                    _hash(f"performance-{seed}-{candidate.ordinal}-{stop_reason}"),
                ),
                failure_codes=(
                    ()
                    if outcome == SequentialCandidateOutcomeKindV1.QUALIFIED
                    else (f"performance_{stop_reason}_{candidate.ordinal}",)
                ),
            )

    result = run_sequential_candidate_search_v1(
        generator,
        spec,
        root_seed=seed,
        dealer=Seat.SOUTH,
        rules=_SEQUENTIAL_RULES,
        evaluate_candidate=evaluate,
    )
    assert result.receipt.to_dict()["stopReason"] == stop_reason
    return result.receipt.verify().to_dict()


def _sequential_receipt_row(
    seed: int,
    stop_reason: str,
    *,
    run_code: str = "RUN-PERF-RECEIPTS",
) -> dict:
    receipt = copy.deepcopy(_sequential_receipt(seed, stop_reason))
    shard_id = f"SHD-{seed}"
    occurrence = build_fenced_occurrence(
        kind="sequential_search_receipt",
        semantic_hash=receipt["sha256"],
        run_code=run_code,
        shard_id=shard_id,
        attempt_count=1,
        fencing_token=1,
        seed=seed,
    )
    return {
        "schemaVersion": 1,
        "occurrenceHash": occurrence["occurrenceHash"],
        "receiptHash": receipt["sha256"],
        "runCode": run_code,
        "shardId": shard_id,
        "attemptCount": 1,
        "fencingToken": 1,
        "seed": seed,
        "admitted": True,
        "createdAt": "2026-08-09T00:00:00+00:00",
        "receiptBundle": receipt,
    }


def _insert_admitted_sqlite_receipt(
    store: SQLiteResearchStoreV1,
    *,
    run_code: str,
    shard_id: str,
    seed: int,
    created_at: str,
    commit: bool = True,
) -> dict:
    receipt = copy.deepcopy(_sequential_receipt(seed, "candidates_exhausted"))
    occurrence = build_fenced_occurrence(
        kind="sequential_search_receipt",
        semantic_hash=receipt["sha256"],
        run_code=run_code,
        shard_id=shard_id,
        attempt_count=1,
        fencing_token=1,
        seed=seed,
    )
    store._db.execute(
        """INSERT INTO sequential_search_receipt_bundles(
           receipt_hash,seed,receipt_json,created_at) VALUES(?,?,?,?)""",
        (
            receipt["sha256"],
            seed,
            json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            created_at,
        ),
    )
    store._db.execute(
        """INSERT INTO sequential_search_receipt_occurrences(
           occurrence_hash,receipt_hash,run_code,shard_id,attempt_count,
           fencing_token,seed,admitted,created_at,updated_at)
           VALUES(?,?,?,?,?,?,?,1,?,?)""",
        (
            occurrence["occurrenceHash"],
            receipt["sha256"],
            run_code,
            shard_id,
            1,
            1,
            seed,
            created_at,
            created_at,
        ),
    )
    if commit:
        store._db.commit()
    return occurrence


def _sqlite_snapshot_store(tmp_path, *, initial_count: int):
    store = SQLiteResearchStoreV1(tmp_path / "receipt-snapshot.sqlite3")
    recipe = {
        "code": "RCP-SNAPSHOT01",
        "sha256": _hash("snapshot-recipe"),
        "status": "published",
    }
    run = {
        "runCode": "RUN-SNAPSHOT0001",
        "runHash": _hash("snapshot-run"),
        "recipeCode": recipe["code"],
        "recipeHash": recipe["sha256"],
        "status": "running",
        "progress": 0.0,
        "events": [],
    }
    store.put_recipe(recipe)
    store.create_run(run)
    shard = partition_research_shards(
        run_code=run["runCode"],
        run_hash=run["runHash"],
        recipe_hash=recipe["sha256"],
        seed_start=0,
        seed_count=max(1, initial_count),
        seeds_per_shard=max(1, initial_count),
    )[0]
    store.create_shards((shard.to_dict(),))
    created_at = "2026-08-09T00:00:00.000000+00:00"
    for seed in range(initial_count):
        _insert_admitted_sqlite_receipt(
            store,
            run_code=run["runCode"],
            shard_id=shard.shard_id,
            seed=seed,
            created_at=created_at,
            commit=False,
        )
    store._db.commit()
    return store, run, shard


def test_sequential_receipt_rollup_uses_only_verified_finalized_evidence():
    rows = [
        _sequential_receipt_row(701, "candidates_exhausted"),
        _sequential_receipt_row(702, "maximum_solutions_reached"),
        _sequential_receipt_row(703, "qualified"),
        _sequential_receipt_row(704, "incomplete"),
        _sequential_receipt_row(705, "cancelled"),
        _sequential_receipt_row(706, "fatal_quarantine"),
    ]

    expected = build_sequential_search_receipt_rollup(rows)

    assert set(expected) == _SEQUENTIAL_SUMMARY_FIELDS
    assert expected == {
        "admittedSeedCount": 6,
        "exactRejectionCount": 2,
        "noGoodAppliedCount": 2,
        "noGoodUnappliedCount": 0,
        "continuationSolveCount": 3,
        "continuationStatusDistribution": {
            "optimal": 2,
            "infeasible": 1,
        },
        "stopReasonDistribution": {
            "candidates_exhausted": 1,
            "maximum_solutions_reached": 1,
            "qualified": 1,
            "incomplete": 1,
            "cancelled": 1,
            "fatal_quarantine": 1,
        },
    }
    assert build_sequential_search_receipt_rollup(reversed(rows)) == expected
    assert build_sequential_search_receipt_rollup(rows[2:] + rows[:2]) == expected


def test_sequential_receipt_item_reuses_the_canonical_summary_boundary():
    row = _sequential_receipt_row(707, "maximum_solutions_reached")
    row["summary"] = {"forged": 1}

    item = build_sequential_search_receipt_item(row)

    assert item is not row
    assert item["summary"] == build_sequential_search_receipt_summary(row)
    assert set(item["summary"]) == _SEQUENTIAL_SUMMARY_FIELDS
    assert item["summary"]["exactRejectionCount"] == 2
    assert item["summary"]["continuationSolveCount"] == 2
    assert item["summary"]["continuationStatusDistribution"] == {
        "optimal": 2,
        "infeasible": 0,
    }
    assert item["receiptBundle"] == row["receiptBundle"]
    assert row["summary"] == {"forged": 1}
    for field in (
        "occurrenceHash",
        "receiptHash",
        "runCode",
        "shardId",
        "attemptCount",
        "fencingToken",
        "seed",
        "admitted",
        "createdAt",
    ):
        assert item[field] == row[field]


def test_sequential_receipt_service_facade_is_admitted_only_and_canonical():
    row = _sequential_receipt_row(716, "qualified")
    row["summary"] = {"forged": 1}

    class Store:
        def __init__(self):
            self.calls = []

        @staticmethod
        def get_run(run_code):
            return {"runCode": run_code} if run_code == "RUN-PERF-RECEIPTS" else None

        def list_sequential_search_receipts(self, **filters):
            self.calls.append(filters)
            return {
                "items": [row],
                "total": 1,
                "offset": filters["offset"],
                "limit": filters["limit"],
                "nextOffset": None,
                "hasMore": False,
            }

    service = object.__new__(ResearchFoundryServiceV1)
    service.store = Store()

    page = service.list_sequential_search_receipts(
        "RUN-PERF-RECEIPTS",
        offset=0,
        limit=9,
    )

    assert service.store.calls == [
        {
            "run_code": "RUN-PERF-RECEIPTS",
            "admitted_only": True,
            "offset": 0,
            "limit": 9,
        }
    ]
    assert {
        key: page[key] for key in ("total", "offset", "limit", "nextOffset", "hasMore")
    } == {
        "total": 1,
        "offset": 0,
        "limit": 9,
        "nextOffset": None,
        "hasMore": False,
    }
    assert page["items"] == [build_sequential_search_receipt_item(row)]
    assert page["items"][0]["summary"] == build_sequential_search_receipt_summary(row)
    assert row["summary"] == {"forged": 1}
    with pytest.raises(KeyError, match="RUN-MISSING"):
        service.list_sequential_search_receipts("RUN-MISSING")
    assert len(service.store.calls) == 1
    for invalid in (True, 0.0, "0"):
        with pytest.raises(ValueError, match="offset"):
            service.list_sequential_search_receipts("RUN-PERF-RECEIPTS", offset=invalid)
    for invalid in (True, 1.0, "1"):
        with pytest.raises(ValueError, match="limit"):
            service.list_sequential_search_receipts("RUN-PERF-RECEIPTS", limit=invalid)
    with pytest.raises(ValueError, match="run code"):
        service.list_sequential_search_receipts(1)
    assert len(service.store.calls) == 1


def test_sequential_receipt_page_metadata_and_types_fail_closed():
    row = _sequential_receipt_row(717, "qualified")
    valid = {
        "items": [row],
        "total": 1,
        "offset": 0,
        "limit": 1,
        "nextOffset": None,
        "hasMore": False,
    }
    assert build_sequential_search_receipt_page(
        valid,
        run_code=row["runCode"],
        offset=0,
        limit=1,
    )["items"] == [build_sequential_search_receipt_item(row)]

    mutations = []
    for field, aliases in {
        "total": (True, 1.0, "1"),
        "offset": (False, 0.0, "0"),
        "limit": (True, 1.0, "1"),
        "hasMore": (0, 0.0, "false"),
    }.items():
        for alias in aliases:
            mutated = copy.deepcopy(valid)
            mutated[field] = alias
            mutations.append(mutated)
    mutations.extend(
        (
            {**copy.deepcopy(valid), "unexpected": True},
            {**copy.deepcopy(valid), "items": []},
            {**copy.deepcopy(valid), "total": 2},
            {**copy.deepcopy(valid), "nextOffset": 1},
        )
    )
    for mutated in mutations:
        with pytest.raises(ValueError):
            build_sequential_search_receipt_page(
                mutated,
                run_code=row["runCode"],
                offset=0,
                limit=1,
            )

    with pytest.raises(ValueError, match="run linkage"):
        build_sequential_search_receipt_page(
            valid,
            run_code="RUN-DIFFERENT",
            offset=0,
            limit=1,
        )

    has_more = {**copy.deepcopy(valid), "total": 2, "hasMore": True, "nextOffset": 1}
    for alias in (True, 1.0, "1"):
        invalid_next = copy.deepcopy(has_more)
        invalid_next["nextOffset"] = alias
        with pytest.raises(ValueError, match="nextOffset"):
            build_sequential_search_receipt_page(
                invalid_next,
                run_code=row["runCode"],
                offset=0,
                limit=1,
            )


@pytest.mark.parametrize(
    "field,alias",
    [
        ("schemaVersion", True),
        ("attemptCount", True),
        ("attemptCount", 1.0),
        ("attemptCount", "1"),
        ("fencingToken", True),
        ("seed", True),
        ("seed", 717.0),
        ("seed", "717"),
        ("admitted", 1),
        ("admitted", "true"),
    ],
)
def test_sequential_receipt_rows_reject_coercive_aliases(field, alias):
    row = _sequential_receipt_row(717, "qualified")
    row[field] = alias
    with pytest.raises(ValueError):
        build_sequential_search_receipt_summary(row)


def test_sequential_receipt_row_rejects_seed_outside_database_range():
    oversized_seed = _sequential_receipt_row(717, "qualified")
    oversized_seed["seed"] = 1 << 63
    with pytest.raises(ValueError, match="database integer range"):
        build_sequential_search_receipt_summary(oversized_seed)


def test_statistics_streams_one_storage_snapshot_and_closes_on_corruption():
    valid = _sequential_receipt_row(718, "qualified")

    class Store:
        def __init__(self, row):
            self.row = row
            self.snapshot_calls = []
            self.closed = False

        @staticmethod
        def get_run(run_code):
            return {"runCode": run_code, "status": "running", "progress": 0.0}

        @staticmethod
        def aggregate_stats(*, run_code):
            return {}

        @staticmethod
        def performance_evidence_snapshot(*, run_code):
            return {
                "runs": [Store.get_run(run_code)],
                "shards": [],
            }

        def list_sequential_search_receipts(self, **_filters):
            pytest.fail("statistics must not consume mutable offset pages")

        @contextmanager
        def iter_admitted_sequential_search_receipts(self, **filters):
            self.snapshot_calls.append(filters)
            try:
                yield iter((self.row,))
            finally:
                self.closed = True

    service = object.__new__(ResearchFoundryServiceV1)
    service.store = Store(valid)
    stats = service.statistics(run_code=valid["runCode"])
    assert stats["sequentialSearch"]["admittedSeedCount"] == 1
    assert service.store.snapshot_calls == [
        {
            "run_code": valid["runCode"],
            "batch_size": 1000,
        }
    ]
    assert service.store.closed is True

    corrupt = copy.deepcopy(valid)
    corrupt["receiptBundle"]["stopReason"] = "cancelled"
    service.store = Store(corrupt)
    with pytest.raises(ValueError):
        service.statistics(run_code=valid["runCode"])
    assert service.store.closed is True


def test_sequential_receipt_rollup_empty_shape_is_closed():
    assert build_sequential_search_receipt_rollup([]) == _EMPTY_SEQUENTIAL_SUMMARY


def test_sequential_receipt_summary_rejects_tampering_and_unknown_schema():
    tampered = _sequential_receipt_row(708, "qualified")
    tampered["receiptBundle"]["stopReason"] = "cancelled"
    with pytest.raises(ValueError):
        build_sequential_search_receipt_summary(tampered)

    unknown = _sequential_receipt_row(709, "qualified")
    unknown_core = copy.deepcopy(unknown["receiptBundle"])
    unknown_core.pop("sha256")
    unknown_core["schemaVersion"] = 2
    unknown["receiptBundle"] = {
        **unknown_core,
        "sha256": canonical_sha256(unknown_core),
    }
    unknown["receiptHash"] = unknown["receiptBundle"]["sha256"]
    with pytest.raises(ValueError, match="unsupported sequential candidate"):
        build_sequential_search_receipt_summary(unknown)


def test_sequential_receipt_summary_rejects_unadmitted_or_unlinked_rows():
    unadmitted = _sequential_receipt_row(710, "incomplete")
    unadmitted["admitted"] = False
    with pytest.raises(ValueError, match="must be admitted"):
        build_sequential_search_receipt_summary(unadmitted)

    wrong_hash = _sequential_receipt_row(711, "fatal_quarantine")
    wrong_hash["receiptHash"] = "0" * 64
    with pytest.raises(ValueError, match="hash linkage"):
        build_sequential_search_receipt_summary(wrong_hash)

    wrong_seed = _sequential_receipt_row(712, "cancelled")
    wrong_seed["seed"] += 1
    with pytest.raises(ValueError, match="seed linkage"):
        build_sequential_search_receipt_summary(wrong_seed)

    wrong_occurrence = _sequential_receipt_row(712, "cancelled")
    wrong_occurrence["occurrenceHash"] = "0" * 64
    with pytest.raises(ValueError, match="occurrence linkage"):
        build_sequential_search_receipt_summary(wrong_occurrence)

    missing_run = _sequential_receipt_row(713, "qualified")
    missing_run["runCode"] = ""
    with pytest.raises(ValueError, match="runCode"):
        build_sequential_search_receipt_summary(missing_run)


def test_sequential_receipt_rollup_rejects_duplicate_admitted_run_seed():
    row = _sequential_receipt_row(714, "candidates_exhausted")
    with pytest.raises(ValueError, match="duplicate admitted.*run seed"):
        build_sequential_search_receipt_rollup([row, copy.deepcopy(row)])

    other_run = _sequential_receipt_row(
        714,
        "candidates_exhausted",
        run_code="RUN-PERF-RECEIPTS-SECOND",
    )
    assert (
        build_sequential_search_receipt_rollup([row, other_run])["admittedSeedCount"]
        == 2
    )


def test_sequential_receipt_summary_ignores_legacy_optimality_cut_counter():
    row = _sequential_receipt_row(715, "maximum_solutions_reached")
    expected = build_sequential_search_receipt_summary(row)
    legacy = copy.deepcopy(row)
    legacy["optimalityCutCount"] = 1_000_000

    assert build_sequential_search_receipt_summary(legacy) == expected
    assert "optimalityCutCount" not in expected


def _completed_shard(
    *, index: int, lane: str, wall: float, qualified: int, codes: list[str]
) -> dict:
    research_store = type("SQLiteResearchStoreV1", (), {})()
    artifact_store = SimpleNamespace(
        descriptor=SimpleNamespace(id="local_content_addressed_v2")
    )
    return {
        "runCode": "RUN-PERF0000001",
        "shardId": f"SHD-{index}",
        "shardIndex": index,
        "status": "completed",
        "attemptCount": 2 if index == 0 else 1,
        "retryCount": 1 if index == 0 else 0,
        "reclaimCount": 0,
        "result": {
            "generationLane": lane,
            "attemptedSeeds": 10,
            "resultCount": len(codes),
            "qualifiedCount": qualified,
            "resultCodes": codes,
            "constructionSeconds": wall,
            "simulationSeconds": wall + 1,
            "semanticParitySeconds": wall + 2,
            "analysisSeconds": wall + 3,
            "tacticalVerificationSeconds": wall + 4,
            "shardWallWorkSeconds": wall + 5,
            "cpuSeconds": wall / 2,
            "cpuMeasurementBoundary": CALLING_THREAD_CPU_BOUNDARY,
            "firstFeasibleCandidateSeconds": wall + 0.1,
            "firstOptimalCandidateSeconds": wall + 0.2,
            "firstQualifiedResultSeconds": wall + 0.3,
            "peakProcessRssBytes": 100_000_000 + index,
            "peakRssMeasurementBoundary": PROCESS_PEAK_RSS_BOUNDARY,
            "environment": build_research_environment_declaration(
                generation_lane=lane,
                configured_shard_workers=4,
                effective_shard_workers=2,
                cp_sat_workers_per_solve=1 if lane == "ortools" else None,
                research_store=research_store,
                artifact_store=artifact_store,
            ),
            "cacheScopes": {
                "compiled_domain": {
                    "hits": index,
                    "misses": 1,
                    "evictions": 0,
                    "entries": 1,
                    "capacity": 16,
                }
            },
            "solverOptimal": 2,
            "solverIncomplete": 1 if index == 1 else 0,
            "playCompilerOptimal": 1,
            "playCompilerFeasible": 2,
            "playCompilerIncomplete": 3,
        },
    }


def _v4_completed_run_receipt(
    run_code: str,
    *,
    result_hash: str = "a" * 64,
    scenario_hash: str = "b" * 64,
    wall_seconds: float | None = 0.5,
    cpu_seconds: float | None = 0.3,
    qualified: bool = True,
) -> dict:
    research_store = type("SQLiteResearchStoreV1", (), {})()
    artifact_store = SimpleNamespace(
        descriptor=SimpleNamespace(id="local_content_addressed_v2")
    )
    core = {
        "schemaVersion": 1,
        "measurementScope": "research_v4_completed_run_v1",
        "runCode": run_code,
        "executionKind": "scenario_search",
        "generationLane": "ortools",
        "attemptedSeeds": 1,
        "attemptedCandidates": 1,
        "structuralCandidates": 1,
        "auctionFeasible": 1,
        "solverOptimal": 1,
        "solverIncomplete": 0,
        "runtimeParityPassed": 1,
        "semanticReplayPassed": 1,
        "proofCompleted": 1,
        "resultCount": 1,
        "qualifiedCount": int(qualified),
        "rejectedCount": int(not qualified),
        "quarantineCount": 0,
        "resultCodes": [f"LAF-{run_code}"],
        "distinctQualifiedResultHashes": [result_hash] if qualified else [],
        "qualifiedScenarioHashes": [scenario_hash] if qualified else [],
        "funnel": {
            "attempted": 1,
            "structurallyFeasible": 1,
            "legalReplay": 1,
            "semanticSatisfied": 1,
            "proofCompleted": 1,
            "qualified": int(qualified),
        },
        "constructionSeconds": 0.1,
        "simulationSeconds": None,
        "semanticParitySeconds": None,
        "analysisSeconds": 0.2,
        "tacticalVerificationSeconds": None,
        "candidatePipelineSeconds": 0.4,
        "stageTimingBoundary": {"candidatePipeline": "executor_call_v1"},
        "firstFeasibleCandidateSeconds": 0.1,
        "firstOptimalCandidateSeconds": 0.2,
        "firstQualifiedResultSeconds": 0.4 if qualified else None,
        "shardWallWorkSeconds": wall_seconds,
        "cpuSeconds": cpu_seconds,
        "cpuMeasurementBoundary": CALLING_THREAD_CPU_BOUNDARY,
        "peakProcessRssBytes": 100_000_000,
        "peakRssMeasurementBoundary": PROCESS_PEAK_RSS_BOUNDARY,
        "environment": build_research_environment_declaration(
            generation_lane="ortools",
            configured_shard_workers=1,
            effective_shard_workers=1,
            cp_sat_workers_per_solve=1,
            research_store=research_store,
            artifact_store=artifact_store,
        ),
        "cacheScopes": {},
        "proofResources": {
            "nodes": 10,
            "cacheEntries": 2,
            "cacheBytes": 32,
            "hiddenWorlds": 1,
            "aggregation": "sum_per_proof_invocation_v1",
        },
        "cancellationObserved": False,
        "terminalStatus": "satisfied" if qualified else "not_found_within_budget",
    }
    return {**core, "sha256": canonical_sha256(core)}


def _v4_partial_run_receipt(run_code: str) -> dict:
    research_store = type("SQLiteResearchStoreV1", (), {})()
    artifact_store = SimpleNamespace(
        descriptor=SimpleNamespace(id="local_content_addressed_v2")
    )
    core = {
        "schemaVersion": 1,
        "measurementScope": "research_v4_cancelled_or_incomplete_run_v1",
        "runCode": run_code,
        "executionKind": "scenario_search",
        "generationLane": "ortools",
        "terminalStatus": "cancelled",
        "terminalReason": "cancelled_after_candidate",
        "evidenceCompleteness": {
            "counters": "complete_through_terminal_observation_v1",
            "timings": "complete_through_terminal_observation_v1",
            "results": "retained_results_only_v1",
            "proofResources": "not_started",
        },
        "attemptedSeeds": 1,
        "attemptedCandidates": 1,
        "structuralCandidates": 1,
        "auctionFeasible": 1,
        "solverOptimal": 1,
        "solverIncomplete": 0,
        "proofCompleted": 0,
        "resultCount": 0,
        "qualifiedCount": 0,
        "rejectedCount": 0,
        "quarantineCount": 0,
        "resultCodes": [],
        "funnel": {
            "attempted": 1,
            "structurallyFeasible": 1,
            "legalReplay": 0,
            "semanticSatisfied": 0,
            "proofCompleted": 0,
            "qualified": 0,
        },
        "constructionSeconds": 0.1,
        "simulationSeconds": None,
        "semanticParitySeconds": None,
        "analysisSeconds": 0.0,
        "tacticalVerificationSeconds": None,
        "candidatePipelineSeconds": 0.2,
        "stageTimingBoundary": {
            "construction": "sum_structural_receipt_elapsed_seconds_v1",
            "simulation": "unavailable_combined_candidate_pipeline_v1",
            "semanticParity": "unavailable_combined_candidate_pipeline_v1",
            "analysis": "sum_proof_resource_elapsed_wall_seconds_v1",
            "tacticalVerification": "not_applicable_no_dedicated_stage_v1",
            "candidatePipeline": "sum_scenario_search_executor_calls_until_cancellation_v1",
        },
        "firstFeasibleCandidateSeconds": 0.1,
        "firstOptimalCandidateSeconds": 0.1,
        "firstQualifiedResultSeconds": None,
        "shardWallWorkSeconds": 0.3,
        "cpuSeconds": 0.2,
        "cpuMeasurementBoundary": CALLING_THREAD_CPU_BOUNDARY,
        "peakProcessRssBytes": 100_000_000,
        "peakRssMeasurementBoundary": PROCESS_PEAK_RSS_BOUNDARY,
        "environment": build_research_environment_declaration(
            generation_lane="ortools",
            configured_shard_workers=1,
            effective_shard_workers=1,
            cp_sat_workers_per_solve=1,
            research_store=research_store,
            artifact_store=artifact_store,
        ),
        "cacheScopes": {},
        "proofResources": {
            "nodes": None,
            "cacheEntries": None,
            "cacheBytes": None,
            "hiddenWorlds": None,
            "aggregation": "not_started_v1",
            "availability": "not_started",
        },
        "cancellationObserved": True,
    }
    return {**core, "sha256": canonical_sha256(core)}


def test_v4_completed_run_receipt_is_validated_and_included_in_statistics():
    run_code = "RUN-V4-PERF-STATS"
    expected_run_code = run_code
    receipt = _v4_completed_run_receipt(run_code)
    assert validate_v4_performance_receipt(receipt, run_code=run_code) == receipt

    class Store:
        @staticmethod
        def aggregate_stats(*, run_code):
            assert run_code is None
            return {}

        @staticmethod
        def performance_evidence_snapshot(*, run_code):
            assert run_code is None
            return {
                "runs": [
                    {
                        "runCode": expected_run_code,
                        "status": "completed",
                        "performanceReceipt": receipt,
                    }
                ],
                "shards": [],
            }

        @staticmethod
        @contextmanager
        def iter_admitted_sequential_search_receipts(**filters):
            assert filters == {"run_code": None, "batch_size": 1000}
            yield iter(())

    service = object.__new__(ResearchFoundryServiceV1)
    service.store = Store()
    stats = service.statistics()
    assert stats["measuredRunCount"] == 1
    assert stats["funnel"]["attemptedCandidates"] == 1
    assert stats["funnel"]["proofCompleted"] == 1
    assert stats["performance"]["shardCount"] == 1
    assert stats["performance"]["qualifiedCount"] == 1
    assert stats["performance"]["qualifiedDistinctScenarioCount"] == 1
    assert stats["performance"]["qualifiedDistinctScenariosPerWallHour"] == 7200.0
    assert stats["performance"]["qualifiedDistinctScenariosPerCpuHour"] == 12000.0
    assert stats["performance"]["measurementAvailability"][
        "qualifiedDistinctScenarios"
    ] == {
        "status": "measured",
        "reason": "complete_hash_bound_v4_scenario_identity_receipts_v1",
    }
    assert stats["performanceByGenerationLane"]["ortools"]["shardCount"] == 1

    tampered = copy.deepcopy(receipt)
    tampered["qualifiedCount"] = 0
    with pytest.raises(ValueError, match="hash is invalid|counters disagree"):
        validate_v4_performance_receipt(tampered, run_code=run_code)

    omitted_identity = copy.deepcopy(receipt)
    omitted_identity["qualifiedScenarioHashes"] = []
    omitted_identity["sha256"] = canonical_sha256(
        {key: value for key, value in omitted_identity.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="qualified-scenario evidence"):
        validate_v4_performance_receipt(omitted_identity, run_code=run_code)

    aliased_auction_stage = copy.deepcopy(receipt)
    aliased_auction_stage["auctionFeasible"] = 0
    aliased_auction_stage["sha256"] = canonical_sha256(
        {key: value for key, value in aliased_auction_stage.items() if key != "sha256"}
    )
    with pytest.raises(
        ValueError, match="search performance receipt counters disagree"
    ):
        validate_v4_performance_receipt(aliased_auction_stage, run_code=run_code)

    exact = copy.deepcopy(receipt)
    exact.update(
        {
            "executionKind": "exact_trace_prefix",
            "generationLane": "exact_input",
            "attemptedSeeds": 0,
            "structuralCandidates": 0,
            "auctionFeasible": 0,
            "solverOptimal": 0,
        }
    )
    exact["funnel"]["structurallyFeasible"] = 0
    exact["sha256"] = canonical_sha256(
        {key: value for key, value in exact.items() if key != "sha256"}
    )
    assert validate_v4_performance_receipt(exact, run_code=run_code) == exact


def test_global_v4_statistics_do_not_truncate_or_double_count_scenarios():
    runs = []
    for index in range(501):
        run_code = f"RUN-V4-GLOBAL-{index:04d}"
        receipt = _v4_completed_run_receipt(
            run_code,
            result_hash=_hash(f"v4-result-{index % 500}"),
            scenario_hash="c" * 64,
        )
        runs.append(
            {
                "runCode": run_code,
                "status": "completed",
                "performanceReceipt": receipt,
            }
        )

    class Store:
        @staticmethod
        def aggregate_stats(*, run_code):
            assert run_code is None
            return {}

        @staticmethod
        def performance_evidence_snapshot(*, run_code):
            assert run_code is None
            return {"runs": runs, "shards": []}

        @staticmethod
        @contextmanager
        def iter_admitted_sequential_search_receipts(**filters):
            assert filters == {"run_code": None, "batch_size": 1000}
            yield iter(())

    service = object.__new__(ResearchFoundryServiceV1)
    service.store = Store()
    stats = service.statistics()

    assert stats["measuredRunCount"] == 501
    assert stats["performance"]["qualifiedCount"] == 501
    assert stats["performance"]["qualifiedDistinctScenarioCount"] == 1
    assert stats["performance"]["qualifiedDistinctScenariosPerWallHour"] == (
        pytest.approx(3600.0 / (501 * 0.5))
    )
    assert stats["performance"]["qualifiedDistinctScenariosPerCpuHour"] == (
        pytest.approx(3600.0 / (501 * 0.3))
    )


def test_global_v4_statistics_fail_closed_when_completed_legacy_run_has_no_receipt():
    v4_run_code = "RUN-V4-WITH-LEGACY"
    receipt = _v4_completed_run_receipt(v4_run_code)

    class Store:
        @staticmethod
        def aggregate_stats(*, run_code):
            assert run_code is None
            return {}

        @staticmethod
        def performance_evidence_snapshot(*, run_code):
            assert run_code is None
            return {
                "runs": [
                    {
                        "runCode": v4_run_code,
                        "status": "completed",
                        "performanceReceipt": receipt,
                    },
                    {
                        "runCode": "RUN-LEGACY-WITHOUT-RECEIPT",
                        "status": "completed",
                    },
                ],
                "shards": [],
            }

        @staticmethod
        @contextmanager
        def iter_admitted_sequential_search_receipts(**filters):
            assert filters == {"run_code": None, "batch_size": 1000}
            yield iter(())

    service = object.__new__(ResearchFoundryServiceV1)
    service.store = Store()
    performance = service.statistics()["performance"]

    assert performance["qualifiedDistinctScenarioCount"] is None
    assert performance["qualifiedDistinctScenariosPerWallHour"] is None
    assert performance["qualifiedDistinctScenariosPerCpuHour"] is None
    assert performance["measurementAvailability"]["qualifiedDistinctScenarios"] == {
        "status": "mixed",
        "reason": "legacy_or_partial_completed_result_coverage",
    }


def test_global_v4_statistics_fail_closed_when_partial_run_receipt_is_present():
    completed_run_code = "RUN-V4-WITH-PARTIAL"
    completed_receipt = _v4_completed_run_receipt(completed_run_code)
    partial_run_code = "RUN-V4-CANCELLED-PARTIAL"
    partial_receipt = _v4_partial_run_receipt(partial_run_code)
    assert (
        validate_v4_partial_performance_receipt(
            partial_receipt,
            run_code=partial_run_code,
        )
        == partial_receipt
    )

    class Store:
        @staticmethod
        def aggregate_stats(*, run_code):
            assert run_code is None
            return {}

        @staticmethod
        def performance_evidence_snapshot(*, run_code):
            assert run_code is None
            return {
                "runs": [
                    {
                        "runCode": completed_run_code,
                        "status": "completed",
                        "performanceReceipt": completed_receipt,
                    },
                    {
                        "runCode": partial_run_code,
                        "status": "cancelled",
                        "partialPerformanceReceipt": partial_receipt,
                    },
                ],
                "shards": [],
            }

        @staticmethod
        @contextmanager
        def iter_admitted_sequential_search_receipts(**filters):
            assert filters == {"run_code": None, "batch_size": 1000}
            yield iter(())

    service = object.__new__(ResearchFoundryServiceV1)
    service.store = Store()
    performance = service.statistics()["performance"]

    assert performance["shardCount"] == 2
    assert performance["completedShardCount"] == 1
    assert performance["operations"]["currentCancelledShardCount"] == 1
    assert performance["qualifiedDistinctScenarioCount"] is None
    assert performance["qualifiedDistinctScenariosPerWallHour"] is None
    assert performance["qualifiedDistinctScenariosPerCpuHour"] is None
    assert performance["measurementAvailability"]["qualifiedDistinctScenarios"] == {
        "status": "mixed",
        "reason": "cancelled_or_incomplete_v4_execution_receipts_present",
    }


def test_performance_rollup_is_deterministic_and_evidence_bounded():
    shards = [
        _completed_shard(
            index=0, lane="ortools", wall=1.0, qualified=1, codes=["A", "B"]
        ),
        _completed_shard(
            index=1, lane="organic_runtime", wall=3.0, qualified=1, codes=["B"]
        ),
        {
            "runCode": "RUN-PERF0000001",
            "shardId": "SHD-2",
            "shardIndex": 2,
            "status": "queued",
            "attemptCount": 0,
            "retryCount": 0,
            "reclaimCount": 0,
            "result": None,
        },
    ]
    expected = build_research_performance_rollup(shards)
    assert build_research_performance_rollup(reversed(shards)) == expected
    assert expected["attemptedSeeds"] == 20
    assert expected["resultCount"] == 3
    assert expected["qualifiedCount"] == 2
    assert expected["qualifiedDistinctScenarioCount"] is None
    assert expected["qualifiedDistinctScenariosPerWallHour"] is None
    assert expected["qualifiedDistinctScenariosPerCpuHour"] is None
    assert expected["measurementAvailability"]["qualifiedDistinctScenarios"] == {
        "status": "unavailable",
        "reason": "no_completed_hash_bound_v4_scenario_identity_receipts",
    }
    assert expected["qualifiedPer100Attempts"] == 10.0
    assert expected["duplicateCount"] == 1
    assert expected["duplicateRate"] == 1 / 3
    assert expected["timing"]["construction"] == {
        "sampleCount": 2,
        "p50Seconds": 2.0,
        "p95Seconds": 2.9,
        "totalSeconds": 4.0,
        "percentileMethod": "linear_interpolation_v1",
        "sampleUnit": "completed_shard_stage_total",
    }
    assert expected["cpu"]["nativeWorkOnCallingThreadIncluded"] is True
    assert expected["cpu"]["auxiliaryThreadsExcluded"] is True
    assert expected["cpu"]["measurementBoundary"] == CALLING_THREAD_CPU_BOUNDARY
    assert expected["cpu"]["totalSeconds"] == 2.0
    assert expected["milestoneTiming"]["firstQualified"]["p50Seconds"] == 2.3
    assert expected["environment"]["status"] == "measured"
    assert expected["peakMemory"]["peakProcessRssBytes"] == 100_000_001
    assert expected["cacheScopes"]["compiled_domain"]["hits"] == 1
    assert expected["operations"]["currentQueuedShardCount"] == 1
    assert expected["operations"]["repeatedExecutionCount"] == 1
    assert expected["operations"]["retryCount"] == 1
    assert expected["operations"]["reclaimCount"] == 0
    assert expected["measurementAvailability"]["hardware"]["status"] == "measured"
    assert expected["measurementAvailability"]["peakMemory"]["status"] == "measured"
    assert expected["measurementAvailability"]["cache"]["status"] == "measured"
    assert expected["solverStatusDistribution"]["construction"] == {
        "INCOMPLETE": 1,
        "OPTIMAL": 4,
    }
    assert expected["solverStatusDistribution"]["playCompiler"] == {
        "FEASIBLE": 4,
        "INCOMPLETE": 6,
        "OPTIMAL": 2,
    }

    lanes = build_research_performance_by_lane(shards)
    assert sorted(lanes) == ["organic_runtime", "ortools", "unavailable"]
    assert lanes["ortools"]["timing"]["construction"]["p95Seconds"] == 1.0


def test_v4_distinct_scenario_rollup_deduplicates_hashes_and_requires_cpu_coverage():
    scenario_hash = "c" * 64
    duplicate_result_hash = "d" * 64
    first_receipt = _v4_completed_run_receipt(
        "RUN-V4-DISTINCT-ONE",
        result_hash=duplicate_result_hash,
        scenario_hash=scenario_hash,
        wall_seconds=1.0,
        cpu_seconds=0.5,
    )
    second_receipt = _v4_completed_run_receipt(
        "RUN-V4-DISTINCT-TWO",
        result_hash="e" * 64,
        scenario_hash=scenario_hash,
        wall_seconds=2.0,
        cpu_seconds=None,
    )
    rollup = build_research_performance_rollup(
        [
            {
                "runCode": first_receipt["runCode"],
                "shardId": "SHD-V4-DISTINCT-ONE",
                "shardIndex": 0,
                "status": "completed",
                "attemptCount": 1,
                "retryCount": 0,
                "reclaimCount": 0,
                "result": first_receipt,
            },
            {
                "runCode": second_receipt["runCode"],
                "shardId": "SHD-V4-DISTINCT-TWO",
                "shardIndex": 0,
                "status": "completed",
                "attemptCount": 1,
                "retryCount": 0,
                "reclaimCount": 0,
                "result": second_receipt,
            },
        ]
    )

    assert rollup["qualifiedCount"] == 2
    assert rollup["qualifiedDistinctScenarioCount"] == 1
    assert rollup["qualifiedDistinctScenariosPerWallHour"] == 1200.0
    assert rollup["qualifiedDistinctScenariosPerCpuHour"] is None
    assert rollup["measurementAvailability"]["qualifiedDistinctScenarioCpuRate"] == {
        "status": "mixed",
        "reason": "cpu_measurement_coverage_is_incomplete",
    }

    mixed = build_research_performance_rollup(
        [
            {
                "runCode": first_receipt["runCode"],
                "shardId": "SHD-V4-DISTINCT-ONE",
                "shardIndex": 0,
                "status": "completed",
                "attemptCount": 1,
                "retryCount": 0,
                "reclaimCount": 0,
                "result": first_receipt,
            },
            _completed_shard(
                index=9,
                lane="legacy",
                wall=1.0,
                qualified=1,
                codes=["LEGACY"],
            ),
        ]
    )
    assert mixed["qualifiedDistinctScenarioCount"] is None
    assert mixed["qualifiedDistinctScenariosPerWallHour"] is None
    assert mixed["qualifiedDistinctScenariosPerCpuHour"] is None
    assert mixed["measurementAvailability"]["qualifiedDistinctScenarios"] == {
        "status": "mixed",
        "reason": "legacy_or_partial_completed_result_coverage",
    }

    missing = build_research_performance_rollup(
        [
            {
                "runCode": first_receipt["runCode"],
                "shardId": "SHD-V4-DISTINCT-ONE",
                "shardIndex": 0,
                "status": "completed",
                "attemptCount": 1,
                "retryCount": 0,
                "reclaimCount": 0,
                "result": first_receipt,
            },
            {
                "runCode": "RUN-V4-MISSING-RECEIPT",
                "shardId": "SHD-V4-MISSING-RECEIPT",
                "shardIndex": 0,
                "status": "completed",
                "attemptCount": 1,
                "retryCount": 0,
                "reclaimCount": 0,
                "result": None,
            },
        ]
    )
    assert missing["qualifiedDistinctScenarioCount"] is None
    assert (
        missing["measurementAvailability"]["qualifiedDistinctScenarios"]["reason"]
        == "legacy_or_partial_completed_result_coverage"
    )


def test_v4_distinct_scenario_rates_preserve_zero_and_missing_boundaries():
    def shard(receipt: dict) -> dict:
        return {
            "runCode": receipt["runCode"],
            "shardId": f"{receipt['runCode']}:performance",
            "shardIndex": 0,
            "status": "completed",
            "attemptCount": 1,
            "retryCount": 0,
            "reclaimCount": 0,
            "result": receipt,
        }

    zero_qualified = build_research_performance_rollup(
        [shard(_v4_completed_run_receipt("RUN-V4-ZERO", qualified=False))]
    )
    assert zero_qualified["qualifiedDistinctScenarioCount"] == 0
    assert zero_qualified["qualifiedDistinctScenariosPerWallHour"] == 0.0
    assert zero_qualified["qualifiedDistinctScenariosPerCpuHour"] == 0.0

    zero_duration = build_research_performance_rollup(
        [
            shard(
                _v4_completed_run_receipt(
                    "RUN-V4-ZERO-DURATION",
                    wall_seconds=0.0,
                    cpu_seconds=0.0,
                )
            )
        ]
    )
    assert zero_duration["qualifiedDistinctScenarioCount"] == 1
    assert zero_duration["qualifiedDistinctScenariosPerWallHour"] is None
    assert zero_duration["qualifiedDistinctScenariosPerCpuHour"] is None
    assert (
        zero_duration["measurementAvailability"]["qualifiedDistinctScenarioWallRate"][
            "reason"
        ]
        == "zero_wall_duration_denominator"
    )
    assert (
        zero_duration["measurementAvailability"]["qualifiedDistinctScenarioCpuRate"][
            "reason"
        ]
        == "zero_cpu_duration_denominator"
    )

    missing_wall = build_research_performance_rollup(
        [
            shard(
                _v4_completed_run_receipt(
                    "RUN-V4-MISSING-WALL",
                    wall_seconds=None,
                    cpu_seconds=0.5,
                )
            )
        ]
    )
    assert missing_wall["qualifiedDistinctScenarioCount"] == 1
    assert missing_wall["qualifiedDistinctScenariosPerWallHour"] is None
    assert missing_wall["qualifiedDistinctScenariosPerCpuHour"] == 7200.0
    assert missing_wall["measurementAvailability"][
        "qualifiedDistinctScenarioWallRate"
    ] == {
        "status": "mixed",
        "reason": "wall_duration_coverage_is_incomplete",
    }


def test_cpu_rates_are_unavailable_when_measurement_coverage_is_partial():
    first = _completed_shard(
        index=0, lane="ortools", wall=1.0, qualified=1, codes=["A"]
    )
    second = _completed_shard(
        index=1, lane="ortools", wall=2.0, qualified=1, codes=["B"]
    )
    second["result"].pop("cpuSeconds")
    rollup = build_research_performance_rollup([first, second])
    assert rollup["cpu"]["availability"] == "mixed"
    assert rollup["costProxy"]["callingOsThreadCpuSecondsPerQualified"] is None

    class Store:
        @staticmethod
        def aggregate_stats(*, run_code):
            assert run_code is None
            return {}

        @staticmethod
        def performance_evidence_snapshot(*, run_code):
            assert run_code is None
            return {
                "runs": [
                    {
                        "generationLane": "ortools",
                        "attemptedSeeds": 1,
                        "structuralCandidates": 1,
                        "qualifiedCount": 1,
                        "elapsedSeconds": 1.0,
                        "cpuSeconds": 0.5,
                        "cpuMeasurementBoundary": CALLING_THREAD_CPU_BOUNDARY,
                    },
                    {
                        "generationLane": "ortools",
                        "attemptedSeeds": 1,
                        "structuralCandidates": 1,
                        "qualifiedCount": 1,
                        "elapsedSeconds": 1.0,
                        "cpuSeconds": 0.25,
                        "cpuMeasurementBoundary": "incompatible_boundary_v1",
                    },
                    {
                        "generationLane": "organic",
                        "attemptedSeeds": 1,
                        "structuralCandidates": 1,
                        "qualifiedCount": 1,
                        "elapsedSeconds": 1.0,
                    },
                ],
                "shards": [],
            }

        @staticmethod
        @contextmanager
        def iter_admitted_sequential_search_receipts(**filters):
            assert filters == {"run_code": None, "batch_size": 1000}
            yield iter(())

    service = object.__new__(ResearchFoundryServiceV1)
    service.store = Store()
    stats = service.statistics()
    assert stats["cpuMeasuredRunCount"] == 2
    assert stats["cpuSeconds"] is None
    assert stats["qualifiedPerCpuHour"] is None
    assert stats["byGenerationLane"]["ortools"]["cpuSeconds"] is None
    assert stats["byGenerationLane"]["ortools"]["cpuMeasurementAvailability"] == "mixed"
    assert stats["byGenerationLane"]["ortools"]["qualifiedPerCpuHour"] is None
    assert (
        stats["byGenerationLane"]["organic"]["cpuMeasurementAvailability"]
        == "unavailable"
    )


def test_environment_declaration_is_sanitized_and_comparable():
    research_store = type("SQLiteResearchStoreV1", (), {})()
    artifact_store = SimpleNamespace(
        descriptor=SimpleNamespace(id="local_content_addressed_v2")
    )
    declaration = build_research_environment_declaration(
        generation_lane="ortools",
        configured_shard_workers=4,
        effective_shard_workers=2,
        cp_sat_workers_per_solve=1,
        research_store=research_store,
        artifact_store=artifact_store,
    )
    encoded = json.dumps(declaration, sort_keys=True).lower()
    assert declaration["storage"] == {
        "researchControlPlane": "sqlite_wal",
        "artifactStorePlugin": "local_content_addressed_v2",
    }
    assert declaration["cpuMeasurementBoundary"] == CALLING_THREAD_CPU_BOUNDARY
    assert "hostname" not in encoded
    assert "dsn" not in encoded
    assert "cwd" not in encoded


def test_sqlite_performance_projection_and_postgres_contract_are_aligned(tmp_path):
    assert inspect.signature(
        SQLiteResearchStoreV1.performance_evidence_snapshot
    ) == inspect.signature(PostgresResearchStoreV1.performance_evidence_snapshot)
    sqlite_snapshot_source = inspect.getsource(
        SQLiteResearchStoreV1.performance_evidence_snapshot
    )
    postgres_snapshot_source = inspect.getsource(
        PostgresResearchStoreV1.performance_evidence_snapshot
    )
    assert 'self._db.execute("BEGIN")' in sqlite_snapshot_source
    assert "REPEATABLE READ, READ ONLY" in postgres_snapshot_source
    assert ".transaction()" in postgres_snapshot_source
    assert " LIMIT " not in sqlite_snapshot_source
    assert " LIMIT " not in postgres_snapshot_source
    assert inspect.signature(SQLiteResearchStoreV1.list_runs) == inspect.signature(
        PostgresResearchStoreV1.list_runs
    )
    assert inspect.signature(
        SQLiteResearchStoreV1.list_performance_shards
    ) == inspect.signature(PostgresResearchStoreV1.list_performance_shards)
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    try:
        recipe = {
            "code": "RCP-PERF000001",
            "sha256": _hash("recipe"),
            "status": "published",
        }
        run = {
            "runCode": "RUN-PERF0000001",
            "runHash": _hash("run"),
            "recipeCode": recipe["code"],
            "recipeHash": recipe["sha256"],
            "status": "running",
            "progress": 0.0,
            "events": [],
        }
        store.put_recipe(recipe)
        store.create_run(run)
        assert [item["runCode"] for item in store.list_runs(limit=None)] == [
            run["runCode"]
        ]
        spec = partition_research_shards(
            run_code=run["runCode"],
            run_hash=run["runHash"],
            recipe_hash=recipe["sha256"],
            seed_start=1,
            seed_count=1,
            seeds_per_shard=1,
        )[0]
        store.create_shards((spec.to_dict(),))
        claimed = store.claim_shard(
            run["runCode"], worker_id="worker", lease_seconds=30
        )
        assert claimed is not None
        store.complete_shard(
            claimed["shardId"],
            worker_id="worker",
            fencing_token=claimed["fencingToken"],
            result=_completed_shard(
                index=0, lane="ortools", wall=1.0, qualified=1, codes=["A"]
            )["result"],
        )
        scoped = store.list_performance_shards(run_code=run["runCode"])
        assert len(scoped) == 1
        assert scoped[0]["result"]["generationLane"] == "ortools"
        assert build_research_performance_rollup(scoped)["qualifiedCount"] == 1
        assert store.list_performance_shards(run_code="RUN-MISSING") == []
        snapshot = store.performance_evidence_snapshot(run_code=run["runCode"])
        assert [item["runCode"] for item in snapshot["runs"]] == [run["runCode"]]
        assert [item["shardId"] for item in snapshot["shards"]] == [
            scoped[0]["shardId"]
        ]

        service = object.__new__(ResearchFoundryServiceV1)
        service.store = store
        stats = service.statistics(run_code=run["runCode"])
        assert stats["performance"]["qualifiedCount"] == 1
        assert stats["performanceByGenerationLane"]["ortools"]["attemptedSeeds"] == 10
        assert stats["sequentialSearch"] == _EMPTY_SEQUENTIAL_SUMMARY
    finally:
        store.close()


def test_sqlite_receipt_snapshot_is_fixed_across_a_second_connection(tmp_path):
    store, run, shard = _sqlite_snapshot_store(tmp_path, initial_count=2)
    writer = SQLiteResearchStoreV1(store.path)
    try:
        with store.iter_admitted_sequential_search_receipts(
            run_code=run["runCode"],
            batch_size=1,
        ) as rows:
            first = next(rows)
            assert first["seed"] == 0
            _insert_admitted_sqlite_receipt(
                writer,
                run_code=run["runCode"],
                shard_id=shard.shard_id,
                seed=2,
                created_at="2026-08-09T00:00:01.000000+00:00",
            )
            assert [row["seed"] for row in rows] == [1]

        with store.iter_admitted_sequential_search_receipts(
            run_code=run["runCode"], batch_size=1
        ) as rows:
            assert [row["seed"] for row in rows] == [0, 1, 2]
        assert store._db.in_transaction is False
    finally:
        writer.close()
        store.close()


def test_sqlite_receipt_snapshot_streams_1001_rows_with_rollup_parity(tmp_path):
    store, run, _shard = _sqlite_snapshot_store(tmp_path, initial_count=1001)
    expected = {
        **copy.deepcopy(_EMPTY_SEQUENTIAL_SUMMARY),
        "admittedSeedCount": 1001,
        "continuationSolveCount": 1001,
        "continuationStatusDistribution": {
            "optimal": 0,
            "infeasible": 1001,
        },
        "stopReasonDistribution": {
            **_EMPTY_SEQUENTIAL_SUMMARY["stopReasonDistribution"],
            "candidates_exhausted": 1001,
        },
    }
    try:
        with store.iter_admitted_sequential_search_receipts(
            run_code=run["runCode"],
            batch_size=73,
        ) as rows:
            assert build_sequential_search_receipt_rollup(rows) == expected

        service = object.__new__(ResearchFoundryServiceV1)
        service.store = store
        assert (
            service.statistics(run_code=run["runCode"])["sequentialSearch"] == expected
        )
        service_source = inspect.getsource(ResearchFoundryServiceV1.statistics)
        assert "iter_admitted_sequential_search_receipts" in service_source
        assert "list_sequential_search_receipts" not in service_source
    finally:
        store.close()


def test_sqlite_receipt_snapshot_releases_cursor_transaction_and_lock_on_corruption(
    tmp_path,
):
    store, run, _shard = _sqlite_snapshot_store(tmp_path, initial_count=1)
    try:
        with pytest.raises(KeyboardInterrupt):
            with store.iter_admitted_sequential_search_receipts(
                run_code=run["runCode"], batch_size=1
            ) as rows:
                next(rows)
                raise KeyboardInterrupt
        assert store._db.in_transaction is False

        store._db.execute(
            "UPDATE sequential_search_receipt_bundles SET receipt_json='{}'"
        )
        store._db.commit()
        with pytest.raises((KeyError, ValueError)):
            with store.iter_admitted_sequential_search_receipts(
                run_code=run["runCode"], batch_size=1
            ) as rows:
                build_sequential_search_receipt_rollup(rows)

        assert store._db.in_transaction is False
        assert store._lock.acquire(blocking=False) is True
        store._lock.release()
        store._db.execute("BEGIN IMMEDIATE")
        store._db.rollback()
    finally:
        store.close()


def test_comparison_promotion_uses_distinct_finalists_and_release_evidence():
    profile = {
        "schemaVersion": 2,
        "id": "matched_research_comparison_v2",
        "revision": "2",
        "minimumQualifiedPerLane": 1,
        "minimumDistinctSelectedFinalistsPerLane": 1,
        "minimumAttemptedSeedsPerLane": 1000,
        "minimumYieldMultiplier": 2.0,
        "minimumCostReduction": 0.4,
        "requiredPromotionCpuBoundary": "isolated_process_cpu_v1",
        "requireZeroSemanticDefects": True,
    }
    runs = {
        "RUN-ORTOOLS": {
            "runCode": "RUN-ORTOOLS",
            "status": "completed",
            "progress": 1.0,
            "attemptedSeeds": 1000,
            "structuralCandidates": 100,
            "qualifiedCount": 8,
            "selectedCount": 4,
            "elapsedSeconds": 3600.0,
            "cpuSeconds": 3600.0,
        },
        "RUN-ORGANIC": {
            "runCode": "RUN-ORGANIC",
            "status": "completed",
            "progress": 1.0,
            "attemptedSeeds": 1000,
            "structuralCandidates": 100,
            "qualifiedCount": 7,
            "selectedCount": 1,
            "elapsedSeconds": 1800.0,
            "cpuSeconds": 1800.0,
        },
    }

    class Store:
        @staticmethod
        def get_recipe(_code):
            return {"code": "RCP-COMPARISON"}

        @staticmethod
        def get_run(code):
            return runs.get(code)

    service = object.__new__(ResearchFoundryServiceV1)
    service.store = Store()
    context = SimpleNamespace(
        comparison_profile=profile,
        comparison_profile_hash=_hash("comparison-profile"),
    )
    service._build_recipe_and_context = lambda _value: (None, context)

    def statistics(*, run_code):
        return {
            "semanticDefects": {},
            "qualifiedPerWallHour": 0.0,
            "qualifiedPerCpuHour": 0.0,
            "stageSeconds": {},
            "performance": {
                "environment": {
                    "status": "measured",
                    "declaration": {"schemaVersion": 1},
                    "comparisonHashes": [_hash("matched-environment")],
                },
                "cpu": {
                    "availability": "measured",
                    "measurementBoundary": "isolated_process_cpu_v1",
                },
            },
        }

    service.statistics = statistics
    campaign = {
        "schemaVersion": 1,
        "campaignCode": "CMP-PERFORMANCE1",
        "baseRecipeCode": "RCP-COMPARISON",
        "runCodes": {
            "ortools": "RUN-ORTOOLS",
            "organic_runtime": "RUN-ORGANIC",
        },
        "createdAt": "2026-08-09T00:00:00+00:00",
    }
    hydrated = service._hydrate_comparison_campaign(campaign)
    comparison = hydrated["comparison"]
    assert hydrated["runs"]["ortools"]["qualifiedPer100Candidates"] == 8.0
    assert (
        hydrated["runs"]["ortools"]["distinctSelectedFinalistsPer100Candidates"] == 4.0
    )
    assert hydrated["runs"]["ortools"]["distinctSelectedFinalistsPer100Seeds"] == 0.4
    assert (
        hydrated["runs"]["organic_runtime"]["distinctSelectedFinalistsPer100Seeds"]
        == 0.1
    )
    assert comparison["smokeEvidence"]["scope"] == ("smoke_only_not_promotion_evidence")
    assert comparison["yieldMultiplier"] == 2.0
    assert comparison["passed"] is True
    assert comparison["recommendation"] == "promote_ortools"

    runs["RUN-ORTOOLS"]["attemptedSeeds"] = 999
    insufficient = service._hydrate_comparison_campaign(campaign)["comparison"]
    assert insufficient["passed"] is False
    assert "insufficient_release_attempts:ortools" in insufficient["reasons"]
    assert insufficient["smokeEvidence"]["passed"] is True

    runs["RUN-ORTOOLS"]["attemptedSeeds"] = 0
    zero_seed_lane = service._hydrate_comparison_campaign(campaign)
    assert (
        zero_seed_lane["runs"]["ortools"]["distinctSelectedFinalistsPer100Seeds"]
        == 0.0
    )


def test_historical_comparison_checksum_failure_requires_readjudication(tmp_path):
    store = SQLiteResearchStoreV1(tmp_path / "historical.sqlite3")
    try:
        store.put_recipe(
            {
                "code": "RCP-HISTORICAL",
                "sha256": _hash("not-the-recipe-checksum"),
                "status": "published",
            }
        )
        for code in ("RUN-HIST-ORT", "RUN-HIST-ORG"):
            store.create_run(
                {
                    "runCode": code,
                    "runHash": _hash(code),
                    "recipeCode": "RCP-HISTORICAL",
                    "recipeHash": _hash("not-the-recipe-checksum"),
                    "status": "completed",
                    "progress": 1.0,
                    "events": [],
                }
            )
        store.put_comparison_campaign(
            {
                "schemaVersion": 1,
                "campaignCode": "CMP-HISTORICAL1",
                "campaignHash": _hash("historical-campaign"),
                "baseRecipeCode": "RCP-HISTORICAL",
                "runCodes": {
                    "ortools": "RUN-HIST-ORT",
                    "organic_runtime": "RUN-HIST-ORG",
                },
                "createdAt": "2026-08-09T00:00:00+00:00",
            }
        )
        service = object.__new__(ResearchFoundryServiceV1)
        service.store = store
        item = service.list_comparison_campaigns(limit=10)[0]
        assert item["comparison"]["evidenceStatus"] == (
            "historical_record_requires_readjudication"
        )
        assert item["comparison"]["eligibleForPromotion"] is False
        assert item["comparison"]["recommendation"] == "insufficient_evidence"
    finally:
        store.close()


def test_research_stats_openapi_and_empty_runtime_are_typed(tmp_path, monkeypatch):
    from baloot_platform.service import api

    store = SQLiteResearchStoreV1(tmp_path / "empty-api.sqlite3")
    service = object.__new__(ResearchFoundryServiceV1)
    service.store = store
    monkeypatch.setattr(api, "RESEARCH", service)
    try:
        with TestClient(api.app) as client:
            response = client.get("/api/v2/research-stats")
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["performance"]["schemaVersion"] == 1
            assert (
                payload["performance"]["measurementAvailability"]["hardware"]["status"]
                == "unavailable"
            )
            assert payload["performance"]["operations"]["reclaimCount"] is None

            schema = client.get("/openapi.json").json()
            stats_schema = schema["components"]["schemas"]["ResearchStatsResponse"]
            assert "performance" in stats_schema["required"]
            assert "performanceByGenerationLane" in stats_schema["properties"]
            for route in ("/api/v2/research-stats", "/api/v2/research-runs/{run_code}/stats"):
                assert schema["paths"][route]["get"]["responses"]["409"]["content"]["application/json"]["schema"] == {
                    "$ref": "#/components/schemas/ResearchMutationErrorResponse"}
            performance_schema = schema["components"]["schemas"][
                "ResearchPerformanceRollupResponse"
            ]
            for field in (
                "qualifiedDistinctScenarioCount",
                "qualifiedDistinctScenariosPerWallHour",
                "qualifiedDistinctScenariosPerCpuHour",
                "milestoneTiming",
                "environment",
                "peakMemory",
                "cacheScopes",
                "operations",
            ):
                assert field in performance_schema["required"]
            availability_property = performance_schema["properties"][
                "measurementAvailability"
            ]
            assert availability_property["$ref"].endswith(
                "/ResearchMeasurementAvailabilityResponse"
            )
            availability_schema = schema["components"]["schemas"][
                "ResearchMeasurementAvailabilityResponse"
            ]
            for field in (
                "qualifiedDistinctScenarios",
                "qualifiedDistinctScenarioWallRate",
                "qualifiedDistinctScenarioCpuRate",
            ):
                assert field in availability_schema["required"]
        comparison_schema = schema["components"]["schemas"][
            "ComparisonCampaignResponse"
        ]
        assert "comparison" in comparison_schema["required"]
        assert comparison_schema["properties"]["comparison"]["$ref"].endswith(
            "/ComparisonPromotionEvidenceResponse"
        )
        assert "/api/v2/research-runs/{run_code}/stats" in schema["paths"]
    finally:
        store.close()


def _ortools_zero_result_run(run_code: str) -> dict:
    semantic_core = {
        "schemaVersion": 1,
        "status": "incomplete",
        "requestedResultCount": 5,
        "maximumCandidates": 50,
        "symmetryPolicyId": "rules_preserving_v1",
        "distancePolicyId": "decision_state_card_hamming_v1",
        "minimumDistance": 4,
        "executionSemanticHashes": [_hash(f"semantic-{index}") for index in range(5)],
        "acceptedSemanticHashes": [],
        "noGoodHashes": [],
        "variantReceipts": [],
    }
    semantic_hash = canonical_sha256(semantic_core)
    receipt_core = {
        **semantic_core,
        "semanticHash": semantic_hash,
        "wallTimeSeconds": 12.5,
        "cpuSeconds": 11.0,
    }
    receipt = {
        "schemaVersion": 1,
        "status": "incomplete",
        "requestedResultCount": 5,
        "maximumCandidates": 50,
        "symmetryPolicyId": "rules_preserving_v1",
        "distancePolicyId": "decision_state_card_hamming_v1",
        "minimumDistance": 4,
        "completedCandidates": 5,
        "executionHashes": [_hash(f"execution-{index}") for index in range(5)],
        "executionSemanticHashes": semantic_core["executionSemanticHashes"],
        "acceptedExecutionHashes": [],
        "acceptedSemanticHashes": [],
        "noGoodVectors": [],
        "variantReceipts": [],
        "wallTimeSeconds": 12.5,
        "cpuSeconds": 11.0,
        "semanticHash": semantic_hash,
        "sha256": canonical_sha256(receipt_core),
    }
    return {
        "runCode": run_code,
        "runHash": _hash(run_code),
        "recipeCode": "RCP-ORTOOLS-ZERO",
        "recipeHash": _hash("recipe-ortools-zero"),
        "status": "completed",
        "progress": 1.0,
        "generationLane": "ortools",
        "requestedResultCount": 5,
        "attemptedCandidates": 5,
        "resultCount": 0,
        "qualifiedCount": 0,
        "rejectedCount": 5,
        "quarantineCount": 0,
        "scenarioResearchStatus": "proof_incomplete",
        "scenarioYieldCounts": {
            "attempted": 5,
            "structurallyFeasible": 5,
            "legalReplay": 0,
            "semanticSatisfied": 0,
            "proofCompleted": 0,
            "qualified": 0,
        },
        "scenarioExecutorIdentity": {
            "backendIdentity": {"backendId": "ortools_cp_sat_v2"}
        },
        "performanceReceipt": receipt,
        "events": [],
    }


def _coordinated_ortools_campaign_run(
    run_code: str,
    *,
    campaign_status: str,
    coordinator_status: str | None,
    scenario_status: str,
    accepted_count: int = 0,
    attempted_count: int = 1,
    audit_result_count: int = 0,
) -> dict:
    run = _ortools_zero_result_run(run_code)
    accepted_hashes = [
        _hash(f"{run_code}-accepted-{index}") for index in range(accepted_count)
    ]
    variant_hashes = [
        _hash(f"{run_code}-variant-{index}") for index in range(accepted_count)
    ]
    coordinator_core = {
        "schemaVersion": 1,
        "status": coordinator_status,
        "shards": [
            {
                "sha256": _hash(f"{run_code}-shard"),
                "recipeHash": _hash("coordinated-recipe"),
                "shardIndex": 0,
                "seedStart": 0,
                "seedCount": 1,
                "maximumAttempts": 1,
            }
        ],
        "budget": {
            "resultCount": int(run["requestedResultCount"]),
            "shardWorkerCount": 1,
            "maximumWallTimeSeconds": 60.0,
            "cancellationPollMilliseconds": 25,
            "cpSatWorkersPerSolve": 1,
        },
        "workerDeclaration": {
            "shardWorkerCount": 1,
            "cpSatWorkersPerSolve": 1,
        },
        "duplicatePolicyId": "semantic_hash_v1",
        "symmetryPolicyId": "rules_preserving_v1",
        "distancePolicyId": "decision_state_card_hamming_v1",
        "minimumDistance": 4,
        "targetSelectionPolicyId": "anchor_descending_partition_prefix_v1",
        "attempts": [],
        "variants": [],
        "acceptedVariants": [
            {
                "semanticHash": variant_hash,
                "semanticDocument": {"semanticResultHash": semantic_result_hash},
            }
            for variant_hash, semantic_result_hash in zip(
                variant_hashes, accepted_hashes, strict=True
            )
        ],
        "semanticSha256": None,
        "wallTimeNs": 1_000_000,
        "cpuTimeNs": 500_000,
    }
    if coordinator_status in {"completed", "not_found_within_budget"}:
        coordinator_core["semanticSha256"] = canonical_sha256(
            {
                "schemaVersion": 1,
                "status": coordinator_status,
                "recipeHash": coordinator_core["shards"][0]["recipeHash"],
                "shardPartitions": [
                    {
                        "shardIndex": 0,
                        "seedStart": 0,
                        "seedCount": 1,
                        "maximumAttempts": 1,
                    }
                ],
                "requestedResultCount": run["requestedResultCount"],
                "duplicatePolicyId": coordinator_core["duplicatePolicyId"],
                "symmetryPolicyId": coordinator_core["symmetryPolicyId"],
                "distancePolicyId": coordinator_core["distancePolicyId"],
                "minimumDistance": coordinator_core["minimumDistance"],
                "targetSelectionPolicyId": coordinator_core[
                    "targetSelectionPolicyId"
                ],
                "acceptedSemanticHashes": variant_hashes,
            }
        )
    coordinator = {
        **coordinator_core,
        "receiptSha256": canonical_sha256(coordinator_core),
    }
    if coordinator_status is None:
        coordinator = None
    receipt = copy.deepcopy(run["performanceReceipt"])
    receipt.update(
        {
            "status": campaign_status,
            "completedCandidates": attempted_count,
            "executionHashes": [
                _hash(f"{run_code}-execution-{index}")
                for index in range(attempted_count)
            ],
            "executionSemanticHashes": [
                _hash(f"{run_code}-execution-semantic-{index}")
                for index in range(attempted_count)
            ],
            "acceptedExecutionHashes": [
                _hash(f"{run_code}-accepted-execution-{index}")
                for index in range(accepted_count)
            ],
            "acceptedSemanticHashes": accepted_hashes,
            "coordinatorReceipt": coordinator,
        }
    )
    semantic_core = {
        key: copy.deepcopy(receipt[key])
        for key in (
            "schemaVersion",
            "status",
            "requestedResultCount",
            "maximumCandidates",
            "symmetryPolicyId",
            "distancePolicyId",
            "minimumDistance",
            "executionSemanticHashes",
            "acceptedSemanticHashes",
            "variantReceipts",
        )
    }
    semantic_core["noGoodHashes"] = [
        canonical_sha256(item) for item in receipt["noGoodVectors"]
    ]
    semantic_core["coordinatorSemanticHash"] = (
        coordinator["semanticSha256"] if coordinator is not None else None
    )
    receipt["semanticHash"] = canonical_sha256(semantic_core)
    receipt["sha256"] = canonical_sha256(
        {
            **semantic_core,
            "semanticHash": receipt["semanticHash"],
            "wallTimeSeconds": receipt["wallTimeSeconds"],
            "cpuSeconds": receipt["cpuSeconds"],
            "coordinatorReceipt": coordinator,
        }
    )
    run.update(
        {
            "attemptedCandidates": attempted_count,
            "resultCount": accepted_count,
            "qualifiedCount": accepted_count,
            "auditResultCount": audit_result_count,
            "scenarioResearchStatus": scenario_status,
            "performanceReceipt": receipt,
        }
    )
    run["scenarioYieldCounts"].update(
        {
            "attempted": attempted_count,
            "structurallyFeasible": attempted_count,
            "qualified": accepted_count,
        }
    )
    return run


@pytest.mark.parametrize(
    (
        "campaign_status",
        "coordinator_status",
        "scenario_status",
        "accepted_count",
        "attempted_count",
        "audit_result_count",
    ),
    (
        ("completed", "completed", "satisfied", 1, 1, 0),
        ("incomplete", "not_found_within_budget", "proof_incomplete", 0, 1, 1),
        (
            "structurally_infeasible",
            "not_found_within_budget",
            "structurally_infeasible",
            0,
            0,
            0,
        ),
        ("proof_failed", "not_found_within_budget", "proof_failed", 0, 1, 1),
    ),
)
def test_coordinated_ortools_campaign_statistics_cover_terminal_outcomes(
    tmp_path,
    campaign_status,
    coordinator_status,
    scenario_status,
    accepted_count,
    attempted_count,
    audit_result_count,
):
    store = SQLiteResearchStoreV1(tmp_path / "ortools-coordinated.sqlite3")
    service = object.__new__(ResearchFoundryServiceV1)
    service.store = store
    run = _coordinated_ortools_campaign_run(
        f"RUN-ORTOOLS-COORDINATED-{campaign_status.upper()}",
        campaign_status=campaign_status,
        coordinator_status=coordinator_status,
        scenario_status=scenario_status,
        accepted_count=accepted_count,
        attempted_count=attempted_count,
        audit_result_count=audit_result_count,
    )
    try:
        store.put_recipe(
            {
                "code": run["recipeCode"],
                "sha256": run["recipeHash"],
                "status": "published",
            }
        )
        store.create_run(run)
        stats = service.statistics(run_code=run["runCode"])
        assert stats["measuredRunCount"] == 1
        assert stats["funnel"]["attemptedCandidates"] == attempted_count
        assert stats["performance"]["resultCount"] == accepted_count
    finally:
        store.close()


@pytest.mark.parametrize("mutation", ["valid", "historical_counter", "historical_empty_shards", "historical_failed_shard", "inflated_run", "wrong_owner", "boolean_count", "sampled_cpu", "invalid_cpu_coverage", "coercive_cpu_coverage"])
def test_cancelled_campaign_statistics_preserve_owned_candidate_progress(tmp_path, mutation):
    run = _coordinated_ortools_campaign_run(
        "RUN-CANCELLED-CHECKPOINT", campaign_status="incomplete",
        coordinator_status="incomplete", scenario_status="proof_incomplete",
        attempted_count=0, accepted_count=0, audit_result_count=0,
    )
    receipt = run["performanceReceipt"]
    coordinator = receipt["coordinatorReceipt"]
    if mutation in {"sampled_cpu", "invalid_cpu_coverage", "coercive_cpu_coverage"}:
        coordinator["cpuTimeCoverage"] = (
            "sampled_lower_bound" if mutation == "sampled_cpu"
            else [] if mutation == "coercive_cpu_coverage" else "invented_exact")
    coordinator["attempts"] = [{"shardId": "RSH-OWNED", "attempt": 1,
        "status": "cancelled", "partialEvidence": {"shardId": "RSH-OWNED",
            "attempt": 1, "completedCandidates": 1}}]
    if mutation == "wrong_owner":
        coordinator["attempts"][0]["partialEvidence"]["shardId"] = "RSH-OTHER"
    if mutation == "boolean_count":
        coordinator["attempts"][0]["partialEvidence"]["completedCandidates"] = True
    if mutation == "historical_empty_shards":
        for shard in ("RSH-EMPTY-1", "RSH-EMPTY-2"):
            coordinator["attempts"].append({"shardId": shard, "attempt": 1,
                "status": "completed", "partialEvidence": {"shardId": shard,
                    "attempt": 1, "completedCandidates": 0,
                    "currentStage": "shard_budget_exhausted", "shardBudgetExhausted": True,
                    "campaignStatus": "incomplete"}})
    if mutation == "historical_failed_shard":
        coordinator["attempts"] = [{"shardId": "RSH-FAILED", "attempt": 1,
            "status": "failed", "errorCode": "builtins.RuntimeError",
            "partialEvidence": {"shardId": "RSH-FAILED", "attempt": 1,
                "currentStage": "worker_failed", "errorCode": "builtins.RuntimeError"}}]
    coordinator["receiptSha256"] = canonical_sha256(
        {k: v for k, v in coordinator.items() if k != "receiptSha256"})
    semantic = {k: receipt[k] for k in (
        "schemaVersion", "status", "requestedResultCount", "maximumCandidates",
        "symmetryPolicyId", "distancePolicyId", "minimumDistance",
        "executionSemanticHashes", "acceptedSemanticHashes", "variantReceipts")}
    semantic.update(noGoodHashes=[canonical_sha256(v) for v in receipt["noGoodVectors"]],
        coordinatorSemanticHash=coordinator["semanticSha256"])
    receipt["sha256"] = canonical_sha256({**semantic, "semanticHash": receipt["semanticHash"],
        "wallTimeSeconds": receipt["wallTimeSeconds"], "cpuSeconds": receipt["cpuSeconds"],
        "coordinatorReceipt": coordinator})
    run["attemptedCandidates"] = (0 if mutation == "historical_counter"
        else 3 if mutation == "historical_empty_shards"
        else 2 if mutation == "inflated_run" else 1)
    run["scenarioYieldCounts"].update(attempted=1, structurallyFeasible=1)
    store = SQLiteResearchStoreV1(tmp_path / "cancelled-checkpoint.sqlite3")
    service = object.__new__(ResearchFoundryServiceV1)
    service.store = store
    try:
        store.put_recipe({"code": run["recipeCode"], "sha256": run["recipeHash"], "status": "published"})
        store.create_run(run)
        if mutation not in {"valid", "historical_counter", "historical_empty_shards", "historical_failed_shard", "sampled_cpu"}:
            with pytest.raises(ValueError, match="campaign performance receipt is invalid"):
                service.statistics(run_code=run["runCode"])
        else:
            stats = service.statistics(run_code=run["runCode"])
            assert stats["funnel"]["attemptedCandidates"] == (0 if mutation == "historical_failed_shard" else 1)
            assert stats["performance"]["resultCount"] == 0
            assert receipt["completedCandidates"] == 0
            assert store.get_run(run["runCode"])["attemptedCandidates"] == run["attemptedCandidates"]
    finally:
        store.close()


def test_current_ortools_campaign_statistics_accept_hash_bound_null_coordinator(
    tmp_path,
):
    run = _coordinated_ortools_campaign_run(
        "RUN-ORTOOLS-NULL-COORDINATOR",
        campaign_status="incomplete",
        coordinator_status=None,
        scenario_status="proof_incomplete",
    )
    store = SQLiteResearchStoreV1(tmp_path / "ortools-null-coordinator.sqlite3")
    service = object.__new__(ResearchFoundryServiceV1)
    service.store = store
    try:
        store.put_recipe(
            {
                "code": run["recipeCode"],
                "sha256": run["recipeHash"],
                "status": "published",
            }
        )
        store.create_run(run)
        assert service.statistics(run_code=run["runCode"])["measuredRunCount"] == 1
    finally:
        store.close()


@pytest.mark.parametrize(
    "mutation", ("nested_substitution", "nested_unknown", "outer_unknown")
)
def test_coordinated_ortools_campaign_statistics_reject_corrupt_receipts(
    tmp_path, mutation
):
    store = SQLiteResearchStoreV1(tmp_path / f"ortools-coordinated-{mutation}.sqlite3")
    service = object.__new__(ResearchFoundryServiceV1)
    service.store = store
    run = _coordinated_ortools_campaign_run(
        f"RUN-ORTOOLS-COORDINATED-{mutation.upper()}",
        campaign_status="incomplete",
        coordinator_status="not_found_within_budget",
        scenario_status="proof_incomplete",
        audit_result_count=1,
    )
    receipt = run["performanceReceipt"]
    if mutation == "nested_substitution":
        receipt["coordinatorReceipt"]["wallTimeNs"] += 1
    elif mutation == "nested_unknown":
        receipt["coordinatorReceipt"]["unknown"] = True
        nested = receipt["coordinatorReceipt"]
        nested["receiptSha256"] = canonical_sha256(
            {key: value for key, value in nested.items() if key != "receiptSha256"}
        )
    else:
        receipt["unknown"] = True
    try:
        store.put_recipe(
            {
                "code": run["recipeCode"],
                "sha256": run["recipeHash"],
                "status": "published",
            }
        )
        store.create_run(run)
        with pytest.raises(
            ValueError,
            match="OR-Tools Research campaign performance receipt is invalid",
        ):
            service.statistics(run_code=run["runCode"])
    finally:
        store.close()


@pytest.mark.parametrize("mutation", ["valid", "ordinary", "partition", "coverage", "non_null_hash", "shard_hash", "coercive_partition"])
def test_historical_proposal_stream_statistics_preserve_hash_bound_receipts(mutation):
    from baloot_platform.research.service import _project_ortools_campaign_statistics_run
    from baloot_platform.research.sharding import ResearchShardSpecV1

    run = _coordinated_ortools_campaign_run("RUN-HISTORICAL-STREAM",
        campaign_status="not_found_within_budget", coordinator_status="not_found_within_budget",
        scenario_status="not_found_within_budget", attempted_count=1)
    receipt = run["performanceReceipt"]
    coordinator = receipt["coordinatorReceipt"]
    coordinator["semanticSha256"] = "a" * 64 if mutation == "non_null_hash" else None
    coordinator["cpuTimeCoverage"] = "terminal_measurements" if mutation == "coverage" else "sampled_lower_bound"
    coordinator["shards"] = [ResearchShardSpecV1.build(
        run_code="ordinary" if mutation == "ordinary" else "proposal_stream",
        run_hash=_hash("historical-stream"), recipe_hash=run["recipeHash"],
        shard_index=0, seed_start=0, seed_count=2 if mutation == "partition" else 1,
        maximum_attempts=1).to_dict()]
    if mutation == "shard_hash":
        coordinator["shards"][0]["sha256"] = "0" * 64
    if mutation == "coercive_partition":
        coordinator["shards"][0]["seedCount"] = True
    coordinator["receiptSha256"] = canonical_sha256({
        key: value for key, value in coordinator.items() if key != "receiptSha256"})
    semantic = {key: receipt[key] for key in (
        "schemaVersion", "status", "requestedResultCount", "maximumCandidates",
        "symmetryPolicyId", "distancePolicyId", "minimumDistance", "executionSemanticHashes",
        "acceptedSemanticHashes", "variantReceipts")}
    semantic.update(noGoodHashes=[canonical_sha256(item) for item in receipt["noGoodVectors"]],
        coordinatorSemanticHash=coordinator["semanticSha256"])
    receipt["semanticHash"] = canonical_sha256(semantic)
    receipt["sha256"] = canonical_sha256({**semantic, "semanticHash": receipt["semanticHash"],
        "wallTimeSeconds": receipt["wallTimeSeconds"], "cpuSeconds": receipt["cpuSeconds"],
        "coordinatorReceipt": coordinator})
    saved = copy.deepcopy(run)
    if mutation == "valid":
        assert _project_ortools_campaign_statistics_run(run, receipt)["attemptedCandidates"] == 1
        assert run == saved
        assert coordinator["semanticSha256"] is None
    else:
        with pytest.raises(ValueError):
            _project_ortools_campaign_statistics_run(run, receipt)


def test_coordinated_ortools_one_audit_campaign_stats_api_returns_typed_200(
    tmp_path, monkeypatch
):
    from baloot_platform.service import api

    store = SQLiteResearchStoreV1(tmp_path / "ortools-coordinated-api.sqlite3")
    service = object.__new__(ResearchFoundryServiceV1)
    service.store = store
    run = _coordinated_ortools_campaign_run(
        "RUN-ORTOOLS-COORDINATED-API",
        campaign_status="incomplete",
        coordinator_status="not_found_within_budget",
        scenario_status="proof_incomplete",
        accepted_count=0,
        attempted_count=1,
        audit_result_count=1,
    )
    store.put_recipe(
        {
            "code": run["recipeCode"],
            "sha256": run["recipeHash"],
            "status": "published",
        }
    )
    store.create_run(run)
    monkeypatch.setattr(api, "RESEARCH", service)
    try:
        with TestClient(api.app) as client:
            response = client.get("/api/v2/research-stats")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["measuredRunCount"] == 1
        assert payload["resultCount"] == 0
        assert payload["funnel"]["attemptedCandidates"] == 1
    finally:
        store.close()


def test_ortools_zero_result_campaign_statistics_are_truthful_and_fail_closed(tmp_path):
    store = SQLiteResearchStoreV1(tmp_path / "ortools-zero.sqlite3")
    service = object.__new__(ResearchFoundryServiceV1)
    service.store = store
    run = _ortools_zero_result_run("RUN-ORTOOLS-ZERO")
    try:
        store.put_recipe(
            {
                "code": run["recipeCode"],
                "sha256": run["recipeHash"],
                "status": "published",
            }
        )
        store.create_run(run)
        stats = service.statistics(run_code=run["runCode"])
        assert stats["resultCount"] == 0
        assert stats["measuredRunCount"] == 1
        assert stats["elapsedWallSeconds"] == 12.5
        assert stats["cpuSeconds"] is None
        assert stats["funnel"]["requested"] == 5
        assert stats["funnel"]["attemptedCandidates"] == 5
        assert stats["funnel"]["structuralCandidates"] == 5
        assert stats["funnel"]["proofCompleted"] == 0
        assert stats["performance"]["resultCount"] == 0
        assert stats["performance"]["qualifiedDistinctScenarioCount"] is None
        assert (
            stats["performance"]["measurementAvailability"][
                "qualifiedDistinctScenarios"
            ]["status"]
            == "unavailable"
        )

        corrupt = copy.deepcopy(run)
        corrupt["runCode"] = "RUN-ORTOOLS-ZERO-CORRUPT"
        corrupt["runHash"] = _hash(corrupt["runCode"])
        corrupt["performanceReceipt"]["wallTimeSeconds"] = 99.0
        store.create_run(corrupt)
        with pytest.raises(
            ValueError,
            match="OR-Tools Research campaign performance receipt is invalid",
        ):
            service.statistics(run_code=corrupt["runCode"])
    finally:
        store.close()


def test_ortools_zero_result_campaign_stats_api_returns_typed_200(
    tmp_path, monkeypatch
):
    from baloot_platform.service import api

    store = SQLiteResearchStoreV1(tmp_path / "ortools-zero-api.sqlite3")
    service = object.__new__(ResearchFoundryServiceV1)
    service.store = store
    run = _ortools_zero_result_run("RUN-ORTOOLS-ZERO-API")
    store.put_recipe(
        {
            "code": run["recipeCode"],
            "sha256": run["recipeHash"],
            "status": "published",
        }
    )
    store.create_run(run)
    monkeypatch.setattr(api, "RESEARCH", service)
    try:
        with TestClient(api.app) as client:
            response = client.get(f"/api/v2/research-runs/{run['runCode']}/stats")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["scope"] == "run"
        assert payload["runCode"] == run["runCode"]
        assert payload["resultCount"] == 0
        assert payload["funnel"]["attemptedCandidates"] == 5
        assert payload["performance"]["resultCount"] == 0
        assert payload["cpuSeconds"] is None
        corrupt = copy.deepcopy(run)
        corrupt["runCode"] = "RUN-ORTOOLS-INVALID-API"
        corrupt["runHash"] = _hash(corrupt["runCode"])
        corrupt["performanceReceipt"]["wallTimeSeconds"] = 99.0
        store.create_run(corrupt)
        with TestClient(api.app) as client:
            for url in ("/api/v2/research-stats", f"/api/v2/research-runs/{corrupt['runCode']}/stats"):
                invalid = client.get(url)
                assert invalid.status_code == 409, invalid.text
                detail = invalid.json()["detail"]
                assert "campaign performance receipt is invalid" in detail
                assert corrupt["runCode"] in detail
            assert client.get("/api/v2/research-runs/RUN-MISSING/stats").status_code == 404
        assert store.get_run(corrupt["runCode"])["performanceReceipt"] == corrupt["performanceReceipt"]
        assert store.get_run(run["runCode"])["performanceReceipt"] == run["performanceReceipt"]
    finally:
        store.close()
