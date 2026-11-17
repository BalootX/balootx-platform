from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from baloot_platform.analysis.search import SearchTimeBudgetExceeded
from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.research import ResearchRecipeV2
from baloot_platform.research.contracts import canonical_sha256
from baloot_platform.research.service import (
    ResearchFoundryServiceV1,
    _shard_failure_evidence,
)
from baloot_platform.research.sharding import partition_research_shards
from baloot_platform.research.store import SQLiteResearchStoreV1


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _store_with_run(tmp_path):
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    recipe = {
        "code": "RCP-SHARD01",
        "sha256": _hash("recipe"),
        "status": "published",
    }
    store.put_recipe(recipe)
    run = {
        "runCode": "RUN-SHARD000001",
        "runHash": _hash("run"),
        "recipeCode": recipe["code"],
        "recipeHash": recipe["sha256"],
        "status": "queued",
        "progress": 0.0,
        "events": [],
    }
    store.create_run(run)
    return store, run, recipe


def _store_validated_legacy_recipe(research, payload):
    validated = research.validate_recipe(payload)
    assert validated["valid"] is True, validated["issues"]
    return research.store.put_recipe(validated["recipe"])


def test_create_run_normalizes_required_timestamp_authority(tmp_path):
    store, run, _recipe = _store_with_run(tmp_path)
    try:
        stored = store.get_run(run["runCode"])
        assert stored is not None
        assert stored["createdAt"].endswith("+00:00")
        assert stored["updatedAt"].endswith("+00:00")
    finally:
        store.close()


def test_seed_partition_is_immutable_complete_and_deterministic():
    values = partition_research_shards(
        run_code="RUN-SHARD000001",
        run_hash=_hash("run"),
        recipe_hash=_hash("recipe"),
        seed_start=100,
        seed_count=12,
        seeds_per_shard=5,
    )
    repeated = partition_research_shards(
        run_code="RUN-SHARD000001",
        run_hash=_hash("run"),
        recipe_hash=_hash("recipe"),
        seed_start=100,
        seed_count=12,
        seeds_per_shard=5,
    )
    assert values == repeated
    assert [(item.seed_start, item.seed_count) for item in values] == [
        (100, 5), (105, 5), (110, 2)
    ]
    assert [seed for item in values for seed in range(item.seed_start, item.seed_end_exclusive)] == list(range(100, 112))
    assert len({item.shard_id for item in values}) == 3


@pytest.mark.real_clock
def test_deterministic_solver_exhaustion_is_not_retried():
    error, retryable = _shard_failure_evidence(RuntimeError(
        "CP-SAT deal construction did not prove an optimal deterministic result"
    ))
    assert error["category"] == "solver_incomplete"
    assert retryable is False

    other, other_retryable = _shard_failure_evidence(RuntimeError("temporary worker failure"))
    assert other["category"] == "system_failure"
    assert other_retryable is True

    timeout, timeout_retryable = _shard_failure_evidence(
        SearchTimeBudgetExceeded("wall-clock budget exhausted")
    )
    assert timeout == {
        "type": "SearchTimeBudgetExceeded",
        "category": "analysis_incomplete",
        "message": "wall-clock budget exhausted",
    }
    assert timeout_retryable is False


@pytest.mark.real_clock
def test_sqlite_shard_lease_heartbeat_fencing_and_reclaim(tmp_path):
    store, run, recipe = _store_with_run(tmp_path)
    try:
        specs = partition_research_shards(
            run_code=run["runCode"], run_hash=run["runHash"],
            recipe_hash=recipe["sha256"], seed_start=10, seed_count=4,
            seeds_per_shard=2,
        )
        created = store.create_shards(tuple(item.to_dict() for item in specs))
        assert [item["shardIndex"] for item in created] == [0, 1]
        assert [(item["retryCount"], item["reclaimCount"]) for item in created] == [
            (0, 0),
            (0, 0),
        ]
        assert store.create_shards(tuple(item.to_dict() for item in specs)) == created

        reclaim_at = datetime.now(timezone.utc)
        started = reclaim_at - timedelta(seconds=16)
        first = store.claim_shard(
            run["runCode"], worker_id="worker-a", lease_seconds=10, now=started
        )
        assert first is not None
        assert first["fencingToken"] == 1
        assert first["retryCount"] == 0
        assert first["reclaimCount"] == 0
        heartbeat = store.heartbeat_shard(
            first["shardId"], worker_id="worker-a",
            fencing_token=first["fencingToken"], lease_seconds=10,
            now=started + timedelta(seconds=5),
        )
        assert datetime.fromisoformat(heartbeat["heartbeatAt"]) == (
            started + timedelta(seconds=5)
        )

        second = store.claim_shard(
            run["runCode"], worker_id="worker-b", lease_seconds=10,
            now=started + timedelta(seconds=6),
        )
        assert second is not None and second["shardIndex"] == 1
        reclaimed = store.claim_shard(
            run["runCode"], worker_id="worker-c", lease_seconds=10,
            now=reclaim_at,
        )
        assert reclaimed is not None and reclaimed["shardIndex"] == 0
        assert reclaimed["fencingToken"] == 2
        assert reclaimed["retryCount"] == 0
        assert reclaimed["reclaimCount"] == 1
        with pytest.raises(RuntimeError, match="stale"):
            store.complete_shard(
                first["shardId"], worker_id="worker-a",
                fencing_token=first["fencingToken"], result={"attemptedSeeds": 2},
            )
        with pytest.raises(RuntimeError, match="stale"):
            store.fail_shard(
                first["shardId"],
                worker_id="worker-a",
                fencing_token=first["fencingToken"],
                error={"type": "stale"},
                retryable=True,
            )
        after_stale = store.get_shard(first["shardId"])
        assert after_stale is not None
        assert after_stale["retryCount"] == 0
        assert after_stale["reclaimCount"] == 1
        complete = store.complete_shard(
            reclaimed["shardId"], worker_id="worker-c",
            fencing_token=reclaimed["fencingToken"], result={"attemptedSeeds": 2},
        )
        assert complete["status"] == "completed"
    finally:
        store.close()


def test_partial_shard_results_and_disagreements_are_not_admitted(tmp_path):
    store, run, recipe = _store_with_run(tmp_path)
    try:
        spec = partition_research_shards(
            run_code=run["runCode"], run_hash=run["runHash"],
            recipe_hash=recipe["sha256"], seed_start=1, seed_count=1,
            seeds_per_shard=1,
        )[0]
        store.create_shards((spec.to_dict(),))
        lease = store.claim_shard(
            run["runCode"], worker_id="worker", lease_seconds=30
        )
        assert lease is not None
        result = {
            "resultCode": "LAF-STAGED",
            "resultHash": _hash("staged-result"),
            "runCode": run["runCode"],
            "truthLabel": "provisional",
            "confidence": 0.5,
            "score": 50,
            "disposition": "rejected",
            "provenance": {"researchShardId": lease["shardId"]},
        }
        disagreement = {
            "disagreementId": "DSG-STAGED",
            "sha256": _hash("staged-disagreement"),
            "runCode": run["runCode"],
            "category": "compiler_parity_defect",
            "status": "open",
        }
        store.put_result(
            result,
            attempt_count=lease["attemptCount"],
            fencing_token=lease["fencingToken"],
            worker_id="worker",
        )
        store.put_disagreement(
            disagreement,
            shard_id=lease["shardId"],
            attempt_count=lease["attemptCount"],
            fencing_token=lease["fencingToken"],
            worker_id="worker",
        )
        assert store.get_result(result["resultCode"]) is None
        assert store.list_results(run_code=run["runCode"])["total"] == 0
        assert store.list_disagreements(run_code=run["runCode"]) == []
        assert store.aggregate_stats(run_code=run["runCode"])["resultCount"] == 0

        store.complete_shard(
            lease["shardId"], worker_id="worker",
            fencing_token=lease["fencingToken"], result={"attemptedSeeds": 1},
        )
        assert store.get_result(result["resultCode"]) == result
        assert store.list_results(run_code=run["runCode"])["total"] == 1
        assert store.list_disagreements(run_code=run["runCode"]) == [disagreement]
        stats = store.aggregate_stats(run_code=run["runCode"])
        assert stats["resultCount"] == 1
        assert stats["semanticDefects"] == {"compiler_parity_defect": 1}
    finally:
        store.close()


def test_completed_v2_run_applies_frozen_diversity_selection(tmp_path):
    recipe = ResearchRecipeV2.build({
        "schemaVersion": 2,
        "titleAr": "اختبار تنوع نتائج البحث",
        "output": {"kind": "laffa", "resultCount": 1},
        "profiles": {
            "domainHash": _hash("domain"),
            "bundleHash": _hash("bundle"),
            "rulesHash": _hash("rules"),
            "conventionHash": _hash("convention"),
            "agentRelease": "agent@frozen",
            "analysisProfileHash": _hash("analysis"),
            "scoreProfileHash": _hash("score"),
        },
        "situation": {
            "dealer": "east",
            "buyer": None,
            "contracts": ["sun"],
            "matchScore": [0, 0],
            "requiredCards": {},
            "forbiddenCards": {},
            "projects": [],
            "escalations": [],
        },
        "tactics": [],
        "qualification": {},
        "resources": {
            "seedStart": 1,
            "seedCount": 1,
            "seedsPerShard": 1,
        },
        "diversity": {
            "schemaVersion": 1,
            "cellDimensions": ["contract"],
            "topKPerCell": 2,
            "rejectedAuditModulus": 100,
        },
        "renderPolicy": {"mode": "on_demand"},
        "status": "published",
    })
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    try:
        store.put_recipe(recipe.to_dict())
        run = {
            "runCode": "RUN-DIVERSE0001",
            "runHash": _hash("diverse-run"),
            "recipeCode": recipe.code,
            "recipeHash": recipe.sha256,
            "status": "running",
            "progress": 0.0,
            "events": [],
        }
        store.create_run(run)
        spec = partition_research_shards(
            run_code=run["runCode"],
            run_hash=run["runHash"],
            recipe_hash=recipe.sha256,
            seed_start=1,
            seed_count=1,
            seeds_per_shard=1,
        )[0]
        store.create_shards((spec.to_dict(),))
        lease = store.claim_shard(
            run["runCode"], worker_id="worker", lease_seconds=30
        )
        assert lease is not None
        for code, contract, score in (
            ("LAF-DIVERSE-A", "sun", 90),
            ("LAF-DIVERSE-B", "hokum", 80),
        ):
            store.put_result(
                {
                    "resultCode": code,
                    "resultHash": _hash(f"result:{code}"),
                    "traceHash": _hash(f"trace:{code}"),
                    "gameHash": _hash(f"game:{code}"),
                    "runCode": run["runCode"],
                    "truthLabel": "robust_best_known",
                    "confidence": 0.9,
                    "score": score,
                    "disposition": "qualified",
                    "qualification": {
                        "qualified": True,
                        "minimumSurroundingQuality": 80,
                    },
                    "libraryProjection": {
                        "contract": contract,
                        "actingSeats": ["south"],
                        "observerSeats": ["south"],
                        "generationLane": "ortools",
                    },
                    "provenance": {"researchShardId": lease["shardId"]},
                },
                attempt_count=lease["attemptCount"],
                fencing_token=lease["fencingToken"],
                worker_id="worker",
            )
        store.complete_shard(
            lease["shardId"],
            worker_id="worker",
            fencing_token=lease["fencingToken"],
            result={
                "attemptedSeeds": 1,
                "resultCount": 2,
                "qualifiedCount": 2,
            },
        )
        service = object.__new__(ResearchFoundryServiceV1)
        service.store = store
        assert service._reduce_sharded_run(run["runCode"]) == "completed"
        completed = store.get_run(run["runCode"])
        assert completed is not None
        assert completed["selectedCount"] == 1
        assert completed["diversitySelection"]["selectedResultCodes"] == [
            "LAF-DIVERSE-B"
        ]
        assert completed["diversitySelection"]["qualifiedCount"] == 2
    finally:
        store.close()


def test_shard_retry_limit_cancellation_and_process_recovery(tmp_path):
    store, run, recipe = _store_with_run(tmp_path)
    try:
        spec = partition_research_shards(
            run_code=run["runCode"], run_hash=run["runHash"],
            recipe_hash=recipe["sha256"], seed_start=1, seed_count=1,
            seeds_per_shard=1, maximum_attempts=2,
        )[0]
        store.create_shards((spec.to_dict(),))
        first = store.claim_shard(run["runCode"], worker_id="a", lease_seconds=30)
        assert first is not None
        retried = store.fail_shard(
            first["shardId"], worker_id="a", fencing_token=first["fencingToken"],
            error={"type": "temporary"}, retryable=True,
        )
        assert retried["status"] == "queued"
        assert retried["retryCount"] == 1
        assert retried["reclaimCount"] == 0
        second = store.claim_shard(run["runCode"], worker_id="b", lease_seconds=30)
        assert second is not None and second["attemptCount"] == 2
        assert second["retryCount"] == 1
        assert second["reclaimCount"] == 0
        failed = store.fail_shard(
            second["shardId"], worker_id="b", fencing_token=second["fencingToken"],
            error={"type": "repeat"}, retryable=True,
        )
        assert failed["status"] == "failed"
        assert failed["retryCount"] == 1
        assert failed["reclaimCount"] == 0

        # A separate run demonstrates safe recovery and cancellation.
        run2 = {**run, "runCode": "RUN-SHARD000002", "runHash": _hash("run2")}
        store.create_run(run2)
        spec2 = partition_research_shards(
            run_code=run2["runCode"], run_hash=run2["runHash"],
            recipe_hash=recipe["sha256"], seed_start=2, seed_count=2,
            seeds_per_shard=1,
        )
        store.create_shards(tuple(item.to_dict() for item in spec2))
        owned = store.claim_shard(run2["runCode"], worker_id="old-process", lease_seconds=900)
        assert owned is not None
        assert store.recover_orphaned_shards() == [run2["runCode"]]
        recovered = store.get_shard(owned["shardId"])
        assert recovered is not None and recovered["status"] == "queued"
        assert recovered["retryCount"] == 0
        assert recovered["reclaimCount"] == 1
        assert store.recover_orphaned_shards() == []
        recovered_again = store.get_shard(owned["shardId"])
        assert recovered_again is not None
        assert recovered_again["reclaimCount"] == 1
        claimed_after_recovery = store.claim_shard(
            run2["runCode"], worker_id="new-process", lease_seconds=900
        )
        assert claimed_after_recovery is not None
        assert claimed_after_recovery["reclaimCount"] == 1
        assert store.cancel_shards(run2["runCode"]) == 2
        assert {item["status"] for item in store.list_shards(run2["runCode"])} == {"cancelled"}
    finally:
        store.close()


def test_sqlite_shard_counter_migration_preserves_historical_unavailability(tmp_path):
    path = tmp_path / "legacy-research.sqlite3"
    historical_spec = partition_research_shards(
        run_code="RUN-HISTORIC0001",
        run_hash=_hash("historical-run"),
        recipe_hash=_hash("historical-recipe"),
        seed_start=1,
        seed_count=1,
        seeds_per_shard=1,
    )[0].to_dict()
    legacy = sqlite3.connect(path)
    try:
        legacy.execute(
            """CREATE TABLE research_shards(
               shard_id TEXT PRIMARY KEY, shard_hash TEXT NOT NULL UNIQUE,
               run_code TEXT NOT NULL, shard_index INTEGER NOT NULL,
               status TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0,
               fencing_token INTEGER NOT NULL DEFAULT 0,
               lease_owner TEXT, lease_expires_at TEXT, heartbeat_at TEXT,
               spec_json TEXT NOT NULL, result_json TEXT, error_json TEXT,
               created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
               UNIQUE(run_code,shard_index))"""
        )
        legacy.execute(
            """INSERT INTO research_shards(
               shard_id,shard_hash,run_code,shard_index,status,attempt_count,
               fencing_token,lease_owner,lease_expires_at,heartbeat_at,
               spec_json,result_json,error_json,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                historical_spec["shardId"],
                historical_spec["sha256"],
                historical_spec["runCode"],
                historical_spec["shardIndex"],
                "completed",
                1,
                1,
                None,
                None,
                "2026-08-08T00:00:00+00:00",
                json.dumps(historical_spec, sort_keys=True, separators=(",", ":")),
                json.dumps({"attemptedSeeds": 1}),
                None,
                "2026-08-08T00:00:00+00:00",
                "2026-08-08T00:00:01+00:00",
            ),
        )
        legacy.commit()
    finally:
        legacy.close()

    store = SQLiteResearchStoreV1(path)
    try:
        historical = store.get_shard(historical_spec["shardId"])
        assert historical is not None
        assert historical["retryCount"] is None
        assert historical["reclaimCount"] is None

        recipe = {
            "code": "RCP-MIGRATE01",
            "sha256": _hash("migration-recipe"),
            "status": "published",
        }
        run = {
            "runCode": "RUN-MIGRATE0001",
            "runHash": _hash("migration-run"),
            "recipeCode": recipe["code"],
            "recipeHash": recipe["sha256"],
            "status": "queued",
            "progress": 0.0,
            "events": [],
        }
        store.put_recipe(recipe)
        store.create_run(run)
        spec = partition_research_shards(
            run_code=run["runCode"],
            run_hash=run["runHash"],
            recipe_hash=recipe["sha256"],
            seed_start=1,
            seed_count=1,
            seeds_per_shard=1,
        )[0]
        shard = store.create_shards((spec.to_dict(),))[0]
        assert shard["retryCount"] == 0
        assert shard["reclaimCount"] == 0
    finally:
        store.close()


def test_service_creates_bounded_shards_and_reduces_out_of_order(tmp_path):
    from baloot_platform.service.container import build_container

    class DeferredExecutor:
        def __init__(self):
            self.calls = []

        def submit(self, function, *args):
            self.calls.append((function, args))

    container = build_container(writable_root=tmp_path / "runtime")
    try:
        research = container.research
        store = research.store
        recipe = _store_validated_legacy_recipe(research, {
            "schemaVersion": 3,
            "titleAr": "اختبار تقسيم البحث القابل للاسترداد",
            "output": {"kind": "laffa_collection", "resultCount": 1},
            "configuration": {
                "schemaVersion": 1,
                "bundleRevisionId": container.configuration.resolve()["bundle"][
                    "revisionId"
                ],
            },
            "situation": {
                "dealer": "east", "buyer": None, "contracts": [],
                "matchScore": [0, 0], "requiredCards": {},
                "forbiddenCards": {}, "projects": [], "escalations": [],
            },
            "tactics": [],
            "qualification": {
                "minimumTruth": "provisional", "minimumConfidence": 0,
                "minimumSurroundingQuality": 0, "requireObserverSafety": True,
                "requireAllDecisionAudit": True, "maximumRegret": 999,
            },
            "resources": {
                "generationLane": "organic_runtime", "seedStart": 10,
                "seedCount": 5, "seedsPerShard": 2,
                "analysisMode": "screening",
            },
            "diversity": {
                "schemaVersion": 1,
                "cellDimensions": ["contract"],
                "topKPerCell": 1,
                "rejectedAuditModulus": 10,
            },
            "renderPolicy": {"mode": "on_demand"},
            "status": "published",
        })
        run = research._submit_recipe_run(
            recipe["code"],
            idempotency_key="sharded-service-run",
            schedule=False,
        )

        deferred = DeferredExecutor()
        research.executor = deferred
        research.maximum_local_shard_workers = 2
        research._scheduled_runs = {run["runCode"]}
        worker_calls = []
        research._shard_worker_loop = (
            lambda code, index, _context=None: worker_calls.append((code, index))
        )

        shards = research.store.list_shards(run["runCode"])
        assert [(item["seedStart"], item["seedCount"]) for item in shards] == [
            (10, 2), (12, 2), (14, 1)
        ]
        research._execute(run["runCode"])
        assert worker_calls == [(run["runCode"], 0)]
        assert len(deferred.calls) == 1
        deferred.calls[0][0](*deferred.calls[0][1])
        assert worker_calls == [(run["runCode"], 0), (run["runCode"], 1)]

        claimed = []
        for index in range(3):
            lease = research.store.claim_shard(
                run["runCode"], worker_id=f"worker-{index}", lease_seconds=30
            )
            assert lease is not None
            claimed.append(lease)
        for lease in reversed(claimed):
            count = int(lease["seedCount"])
            research.store.complete_shard(
                lease["shardId"], worker_id=f"worker-{lease['shardIndex']}",
                fencing_token=lease["fencingToken"],
                result={
                    "attemptedSeeds": count,
                    "structuralCandidates": count,
                    "semanticReplayPassed": count,
                    "resultCount": count,
                    "qualifiedCount": 0,
                    "rejectedCount": count,
                    "shardWallWorkSeconds": float(count),
                    "cpuSeconds": float(count) / 10,
                },
            )
        assert research._reduce_sharded_run(run["runCode"]) == "completed"
        finished = research.get_run(run["runCode"])
        assert finished is not None
        assert finished["attemptedSeeds"] == 5
        assert finished["resultCount"] == 5
        assert finished["completedShardCount"] == 3
        assert finished["shardWallWorkSeconds"] == 5
        assert finished["cpuSeconds"] == pytest.approx(0.5)
        assert finished["cpuMeasurementBoundary"] == (
            "calling_os_worker_thread_cpu_v1"
        )
        assert [
            item["shardIndex"] for item in finished["events"]
            if item["type"] == "shard_completed"
        ] == [0, 1, 2]

        failed_run = store.create_run({
            **run,
            "runCode": "RUN-SHARDFAIL001",
            "runHash": _hash("sharded-service-failure"),
            "status": "running",
            "events": [],
        })
        failed_spec = partition_research_shards(
            run_code=failed_run["runCode"],
            run_hash=failed_run["runHash"],
            recipe_hash=recipe["sha256"],
            seed_start=99,
            seed_count=1,
            seeds_per_shard=1,
            maximum_attempts=1,
        )[0]
        store.create_shards((failed_spec.to_dict(),))
        failed_lease = store.claim_shard(
            failed_run["runCode"], worker_id="failure-worker", lease_seconds=30
        )
        assert failed_lease is not None
        store.fail_shard(
            failed_lease["shardId"],
            worker_id="failure-worker",
            fencing_token=failed_lease["fencingToken"],
            error={
                "type": "RuntimeError",
                "category": "solver_incomplete",
                "message": "CP-SAT deal construction did not prove an optimal deterministic result",
            },
            retryable=True,
        )
        assert research._reduce_sharded_run(failed_run["runCode"]) == "failed"
        failed_result = research.get_run(failed_run["runCode"])
        assert failed_result is not None
        assert failed_result["error"]["category"] == "solver_incomplete"
        assert failed_result["error"]["attemptCount"] == 1
        assert failed_result["shardErrors"] == [failed_result["error"]]
        assert any(
            item["type"] == "shard_failed"
            and item["message"].startswith("CP-SAT deal construction")
            for item in failed_result["events"]
        )
    finally:
        container.shutdown()


@pytest.mark.real_clock
def test_real_research_is_identical_with_one_two_four_workers_and_reverse_completion(
    tmp_path, monkeypatch
):
    """Bounded end-to-end proof over real simulation and shard commits."""
    from baloot_platform.service.container import build_container

    # The global test harness sets these paths for API tests. Explicit
    # writable_root values must own this proof so its measured runs cannot
    # affect another test's aggregate statistics.
    monkeypatch.delenv("BALOOT_RUNTIME", raising=False)
    monkeypatch.delenv("BALOOT_OUTPUTS", raising=False)

    container = build_container(writable_root=tmp_path / "shared-runtime")
    research = container.research
    active = container.configuration.resolve()
    challenge = copy.deepcopy(active["profiles"]["challenge"])
    # This proof compares shard scheduling, not wall-clock search exhaustion.
    # Use the governed heuristic path so host load cannot change admission.
    challenge["screening"]["shallow_exact_search_remaining"] = 0
    challenge_revision = container.configuration.create_profile(
        family="challenge",
        payload=challenge,
        created_by="TEST",
        scope="tenant",
        tenant_id="RESEARCH-SHARDING-PROOF",
        publish=True,
    )
    component_revisions = {
        family: str(value["revisionId"])
        for family, value in active["profileRevisions"].items()
    }
    component_revisions["challenge"] = str(challenge_revision["revisionId"])
    proof_bundle = container.configuration.create_bundle(
        bundle_id="research-sharding-proof",
        version="99.0.0",
        mode=str(active["bundle"]["mode"]),
        component_revisions=component_revisions,
        created_by="TEST",
        scope="tenant",
        tenant_id="RESEARCH-SHARDING-PROOF",
        publish=True,
        force_new_revision=True,
    )
    recipe = _store_validated_legacy_recipe(research, {
        "schemaVersion": 3,
        "titleAr": "إثبات ثبات تقسيم البحث",
        "output": {"kind": "laffa_collection", "resultCount": 1},
        "configuration": {
            "schemaVersion": 1,
            "bundleRevisionId": proof_bundle["revisionId"],
        },
        "situation": {
            "dealer": "east", "buyer": None, "contracts": [],
            "matchScore": [0, 0], "requiredCards": {},
            "forbiddenCards": {}, "projects": [], "escalations": [],
        },
        "tactics": [],
        "qualification": {
            "minimumTruth": "provisional", "minimumConfidence": 0,
            "minimumSurroundingQuality": 0,
            "requireObserverSafety": True,
            "requireAllDecisionAudit": True, "maximumRegret": 999,
        },
        "resources": {
            "generationLane": "organic_runtime", "seedStart": 31,
            "seedCount": 4, "seedsPerShard": 1,
            "analysisMode": "screening",
        },
        "diversity": {
            "schemaVersion": 1,
            "cellDimensions": ["contract"],
            "topKPerCell": 1,
            "rejectedAuditModulus": 10,
        },
        "renderPolicy": {"mode": "on_demand"},
        "status": "published",
    })
    build_recipe_and_context = research._build_recipe_and_context
    frozen_recipe, context = build_recipe_and_context(
        research.store.get_recipe(recipe["code"]) or {}
    )
    assert context is not None
    throughput_simulator = NaturalLaffaSimulatorV1(
        context.bundle.rules,
        context.bundle.convention,
        bundle_hash=context.bundle.sha256,
        configuration_snapshot=context.bundle.configuration_snapshot(),
        structural_domain=context.bundle.rules.structural_domain,
        play_domain=context.bundle.rules.play_domain,
        agent_profile=AgentProfileRegistry.from_mapping(
            context.bundle.component("agents")
        ).get("foundry_throughput"),
        agent_release="foundry-throughput-sharding-proof",
        generation_tier="throughput_reference",
        maximum_deal_attempts=context.simulator.maximum_deal_attempts,
        all_pass_dealer_policy=context.simulator.all_pass_dealer_policy,
        exact_search_backend=context.exact_search_backend,
        analysis_policy=context.analysis_policy,
        belief_policy=context.belief_policy,
        match_value_profile=context.match_value_profile,
    )
    throughput_context = replace(context, simulator=throughput_simulator)

    def use_throughput_context(_payload):
        return frozen_recipe, throughput_context

    def execute(worker_count: int, *, reverse_completion: bool = False):
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix=f"research-proof-{worker_count}",
        )
        completion_order: list[int] = []
        try:
            research.executor = executor
            research.maximum_local_shard_workers = worker_count
            if reverse_completion:
                barrier = threading.Barrier(4, timeout=180)
                condition = threading.Condition()
                next_index = {"value": 3}
                complete = research.store.complete_shard

                def complete_in_reverse(shard_id, **kwargs):
                    shard = research.store.get_shard(shard_id)
                    assert shard is not None
                    index = int(shard["shardIndex"])
                    barrier.wait()
                    with condition:
                        assert condition.wait_for(
                            lambda: next_index["value"] == index, timeout=180
                        )
                        value = complete(shard_id, **kwargs)
                        completion_order.append(index)
                        next_index["value"] -= 1
                        condition.notify_all()
                        return value

                research.store.complete_shard = complete_in_reverse

            research._build_recipe_and_context = build_recipe_and_context
            run = research._submit_recipe_run(
                recipe["code"],
                idempotency_key=f"deterministic-shard-proof-{worker_count}",
                schedule=False,
            )
            research._build_recipe_and_context = use_throughput_context
            research._execute(run["runCode"])
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                current = research.get_run(run["runCode"])
                assert current is not None
                if current["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.02)
            else:
                pytest.fail("real sharded Research proof timed out")
            assert current["status"] == "completed", current

            results = research.list_results(
                run_code=run["runCode"], sort="score_desc", limit=100
            )["items"]
            normalized_results = []
            for result in results:
                provenance = dict(result["provenance"])
                projection = dict(result["libraryProjection"])
                normalized_results.append({
                    "researchSeed": provenance["researchSeed"],
                    "candidateHash": provenance["candidateHash"],
                    "modelHash": provenance["modelHash"],
                    "solverStatus": provenance["solverStatus"],
                    "traceHash": result["traceHash"],
                    "gameHash": result["gameHash"],
                    "completeTraceSha256": result["artifacts"]["completeTrace"][
                        "sha256"
                    ],
                    "contract": projection["contract"],
                    "purchaseRound": projection["purchaseRound"],
                    "trumpSuit": projection["trumpSuit"],
                    "escalation": projection["escalation"],
                    "projectKinds": projection["projectKinds"],
                    "generationLane": projection["generationLane"],
                    "recipeHash": provenance["recipeHash"],
                    "domainHash": provenance["domainHash"],
                })
            normalized_results.sort(key=lambda item: item["researchSeed"])
            shard_evidence = []
            count_keys = (
                "attemptedSeeds", "structuralCandidates", "auctionFeasible",
                "solverOptimal", "runtimeParityPassed", "semanticReplayPassed",
                "playCompilerAttempted", "playCompilerOptimal",
                "playCompilerFeasible", "playCompilerIncomplete", "resultCount",
                "qualifiedCount", "rejectedCount", "quarantineCount",
                "tacticalVerificationCandidates", "tacticalVerificationDecisions",
                "optimalityCutCount",
            )
            for shard in research.store.list_shards(run["runCode"]):
                metrics = dict(shard.get("result") or {})
                shard_evidence.append({
                    "shardIndex": shard["shardIndex"],
                    "seedStart": shard["seedStart"],
                    "seedCount": shard["seedCount"],
                    "status": shard["status"],
                    "counts": {key: metrics.get(key, 0) for key in count_keys},
                    "resultCountByShard": len(metrics.get("resultCodes") or ()),
                })
            disagreements = research.list_disagreements(run_code=run["runCode"])
            normalized_disagreements = []
            for disagreement in disagreements:
                normalized = dict(disagreement)
                for key in (
                    "disagreementId", "sha256", "runCode", "candidateHash",
                ):
                    normalized.pop(key, None)
                normalized_disagreements.append(normalized)
            evidence = {
                "results": normalized_results,
                "shards": shard_evidence,
                "disagreements": normalized_disagreements,
                "runCounts": {
                    key: current.get(key, 0)
                    for key in (
                        "attemptedSeeds", "structuralCandidates",
                        "semanticReplayPassed", "resultCount", "qualifiedCount",
                        "rejectedCount", "quarantineCount", "completedShardCount",
                    )
                },
            }
            return canonical_sha256(evidence), evidence, completion_order
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    try:
        one = execute(1)
        two = execute(2)
        four = execute(4, reverse_completion=True)
        assert four[2] == [3, 2, 1, 0]
        assert one[1] == two[1]
        assert two[1] == four[1]
        assert one[0] == two[0] == four[0]
    finally:
        container.shutdown()
