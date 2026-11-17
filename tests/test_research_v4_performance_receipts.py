from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from baloot_platform.core.enums import Seat
from baloot_platform.research import service as service_module
from baloot_platform.research.performance_stats import (
    CALLING_THREAD_CPU_BOUNDARY,
    PROCESS_PEAK_RSS_BOUNDARY,
    qualified_scenario_identity_hash_v1,
    validate_v4_partial_performance_receipt,
    validate_v4_performance_receipt,
)
from baloot_platform.research.scenario_contracts import canonical_sha256
from baloot_platform.research.scenario_search import ScenarioSearchExecutorV1
from baloot_platform.research.service import ResearchFoundryServiceV1


_RECEIPT_FIELDS = {
    "schemaVersion",
    "measurementScope",
    "runCode",
    "executionKind",
    "generationLane",
    "attemptedSeeds",
    "attemptedCandidates",
    "structuralCandidates",
    "auctionFeasible",
    "solverOptimal",
    "solverIncomplete",
    "runtimeParityPassed",
    "semanticReplayPassed",
    "proofCompleted",
    "resultCount",
    "qualifiedCount",
    "rejectedCount",
    "quarantineCount",
    "resultCodes",
    "distinctQualifiedResultHashes",
    "qualifiedScenarioHashes",
    "funnel",
    "constructionSeconds",
    "simulationSeconds",
    "semanticParitySeconds",
    "analysisSeconds",
    "tacticalVerificationSeconds",
    "candidatePipelineSeconds",
    "stageTimingBoundary",
    "firstFeasibleCandidateSeconds",
    "firstOptimalCandidateSeconds",
    "firstQualifiedResultSeconds",
    "shardWallWorkSeconds",
    "cpuSeconds",
    "cpuMeasurementBoundary",
    "peakProcessRssBytes",
    "peakRssMeasurementBoundary",
    "environment",
    "cacheScopes",
    "proofResources",
    "cancellationObserved",
    "terminalStatus",
    "sha256",
}
_PARTIAL_RECEIPT_FIELDS = {
    "schemaVersion",
    "measurementScope",
    "runCode",
    "executionKind",
    "generationLane",
    "terminalStatus",
    "terminalReason",
    "evidenceCompleteness",
    "attemptedSeeds",
    "attemptedCandidates",
    "structuralCandidates",
    "auctionFeasible",
    "solverOptimal",
    "solverIncomplete",
    "proofCompleted",
    "resultCount",
    "qualifiedCount",
    "rejectedCount",
    "quarantineCount",
    "resultCodes",
    "funnel",
    "constructionSeconds",
    "simulationSeconds",
    "semanticParitySeconds",
    "analysisSeconds",
    "tacticalVerificationSeconds",
    "candidatePipelineSeconds",
    "stageTimingBoundary",
    "firstFeasibleCandidateSeconds",
    "firstOptimalCandidateSeconds",
    "firstQualifiedResultSeconds",
    "shardWallWorkSeconds",
    "cpuSeconds",
    "cpuMeasurementBoundary",
    "peakProcessRssBytes",
    "peakRssMeasurementBoundary",
    "environment",
    "cacheScopes",
    "proofResources",
    "cancellationObserved",
    "sha256",
}


def test_candidate_receipt_preserves_feasible_auction_before_decision_miss(rules):
    deck_order = tuple(range(32))

    class _Structural:
        @staticmethod
        def solve(*_args, **_kwargs):
            return SimpleNamespace(
                status="satisfied",
                deck_order=deck_order,
                to_dict=lambda: {
                    "status": "satisfied",
                    "solverStatus": "optimal",
                    "deckOrder": list(deck_order),
                },
            )

    class _Auction:
        @staticmethod
        def compile(**_kwargs):
            return SimpleNamespace(
                actions=(
                    SimpleNamespace(actor=Seat.SOUTH, action="pass", trump=None),
                )
            )

    recipe = SimpleNamespace(
        scenario=SimpleNamespace(
            origin=SimpleNamespace(kind="search"),
            decision_point={"kind": "bid", "selector": {"actionIndex": 1}},
            predicates=lambda: (),
        ),
        resources={
            "solverTimeoutSeconds": 1.0,
            "maximumConflicts": 100,
            "generationPolicy": "natural_policy",
        },
    )

    candidate = ScenarioSearchExecutorV1(
        structural=_Structural(),
        auction=_Auction(),
    ).build_candidate(
        recipe,
        rules=rules,
        bundle_hash="b" * 64,
        seed=12001,
    )

    assert candidate.status == "construction_target_miss"
    assert candidate.normalized is None
    assert candidate.construction_receipt["auctionFeasible"] is True
    assert candidate.construction_receipt["failure"] == (
        "bid_decision_action_index_unavailable"
    )


class _RunStore:
    def __init__(self, run):
        self.run = dict(run)

    def update_run(self, run_code, **changes):
        assert run_code == self.run["runCode"]
        self.run.update(changes)
        return dict(self.run)

    def get_run(self, run_code):
        assert run_code == self.run["runCode"]
        return dict(self.run)


class _ArtifactStore:
    def __init__(self):
        self.values = {}

    def put(self, data, *, media_type, name=None):
        digest = hashlib.sha256(data).hexdigest()
        uri = f"cas://sha256/{digest}"
        self.values[uri] = bytes(data)
        return {
            "uri": uri,
            "sha256": digest,
            "bytes": len(data),
            "mediaType": media_type,
            "name": name,
        }


def _run(run_code: str) -> dict:
    return {
        "runCode": run_code,
        "runHash": "a" * 64,
        "recipeCode": "RCP-V4-PERFORMANCE",
        "recipeHash": "c" * 64,
        "status": "queued",
        "progress": 0.0,
        "events": [],
        "createdAt": "2026-08-23T00:00:00+00:00",
    }


def _recipe(kind: str, *, seed_count: int = 1, result_count: int = 1):
    origin = SimpleNamespace(
        kind=kind,
        exact_trace_prefix={"schemaVersion": 1}
        if kind == "exact_trace_prefix"
        else None,
    )
    scenario = SimpleNamespace(
        origin=origin,
        decision_point={"kind": "play", "selector": {}},
        sha256="d" * 64,
    )
    return SimpleNamespace(
        code="RCP-V4-PERFORMANCE",
        sha256="c" * 64,
        scenario=scenario,
        proof=SimpleNamespace(),
        output={
            "kind": "laffa" if result_count == 1 else "laffa_collection",
            "resultCount": result_count,
        },
        diversity={
            "schemaVersion": 1,
            "cellDimensions": ["decision_kind", "proof_mode"],
            "topKPerCell": 3,
            "rejectedAuditModulus": 100,
        },
        render_policy={"mode": "on_demand", "videoCount": 1},
        resources={
            "generationLane": "ortools",
            "generationPolicy": "scenario_guided_actor_safe",
            "seedStart": 5,
            "seedCount": seed_count,
            "candidatesPerSeed": 1,
        },
    )


def _harness(run: dict):
    return SimpleNamespace(
        store=_RunStore(run),
        artifact_store=_ArtifactStore(),
        execution_context_factory=None,
        maximum_local_shard_workers=3,
        _cancel_check=lambda _run_code: False,
    )


def _assert_hash_bound_receipt(receipt: dict, run_code: str) -> None:
    assert set(receipt) == _RECEIPT_FIELDS
    assert receipt["runCode"] == run_code
    core = {key: value for key, value in receipt.items() if key != "sha256"}
    assert receipt["sha256"] == canonical_sha256(core)
    assert validate_v4_performance_receipt(receipt, run_code=run_code) == receipt
    assert receipt["cpuSeconds"] >= 0
    assert receipt["cpuMeasurementBoundary"] == CALLING_THREAD_CPU_BOUNDARY
    assert receipt["shardWallWorkSeconds"] >= receipt["candidatePipelineSeconds"]
    if receipt["peakProcessRssBytes"] is None:
        assert receipt["peakRssMeasurementBoundary"] is None
    else:
        assert receipt["peakProcessRssBytes"] > 0
        assert receipt["peakRssMeasurementBoundary"] == PROCESS_PEAK_RSS_BOUNDARY


def _assert_hash_bound_partial_receipt(receipt: dict, run_code: str) -> None:
    assert set(receipt) == _PARTIAL_RECEIPT_FIELDS
    assert receipt["runCode"] == run_code
    core = {key: value for key, value in receipt.items() if key != "sha256"}
    assert receipt["sha256"] == canonical_sha256(core)
    assert (
        validate_v4_partial_performance_receipt(receipt, run_code=run_code)
        == receipt
    )


@pytest.mark.real_clock
def test_exact_v4_run_persists_hash_bound_honest_performance_receipt(monkeypatch):
    normalized = SimpleNamespace()
    proof = SimpleNamespace(
        payload={
            "completionStatus": "exact_complete",
            "resources": {
                "nodes": 17,
                "cacheEntries": 4,
                "cacheBytes": 512,
                "cancelled": False,
            },
            "hiddenWorlds": {"count": 2},
        }
    )
    satisfaction = SimpleNamespace(status="satisfied")

    class _Normalizer:
        def __init__(self, _rules):
            pass

        def normalize(self, _prefix, *, decision_point):
            assert decision_point["kind"] == "play"
            return normalized

    class _ProofService:
        def prove(self, received, _request, **_kwargs):
            assert received is normalized
            return proof

    class _ProofFactory:
        @staticmethod
        def from_default_authority():
            return _ProofService()

    class _RuntimeVerifier:
        def verify(self, _scenario, received, *, proof_outcome):
            assert received is normalized
            assert proof_outcome is proof
            return satisfaction

    monkeypatch.setattr(service_module, "ExactStateNormalizerV1", _Normalizer)
    monkeypatch.setattr(service_module, "UnifiedScenarioProofServiceV1", _ProofFactory)
    monkeypatch.setattr(service_module, "ScenarioRuntimeVerifierV1", _RuntimeVerifier)

    def _put_result(_self, **kwargs):
        assert kwargs["normalized"] is normalized
        return {
            "resultCode": "LAF-EXACT",
            "resultHash": "1" * 64,
            "disposition": "qualified",
            "qualification": {"qualified": True},
            "scenarioProjection": {
                "scenarioTemplateId": "SCN-EXACTPERF001",
                "exactStateId": "STATE-EXACTPERFORMANCE001",
            },
        }

    monkeypatch.setattr(
        ResearchFoundryServiceV1, "_put_v4_scenario_result", _put_result
    )
    run = _run("RUN-V4-PERF-EXACT")
    harness = _harness(run)
    context = SimpleNamespace(bundle=SimpleNamespace(rules=object(), sha256="b" * 64))

    ResearchFoundryServiceV1._execute_v4_exact_run(
        harness,
        run,
        _recipe("exact_trace_prefix"),
        context,
    )

    receipt = harness.store.run["performanceReceipt"]
    _assert_hash_bound_receipt(receipt, run["runCode"])
    assert receipt["executionKind"] == "exact_trace_prefix"
    assert receipt["generationLane"] == "exact_input"
    assert receipt["attemptedSeeds"] == 0
    assert receipt["attemptedCandidates"] == 1
    assert receipt["funnel"] == {
        "attempted": 1,
        "structurallyFeasible": 0,
        "legalReplay": 1,
        "semanticSatisfied": 1,
        "proofCompleted": 1,
        "qualified": 1,
    }
    assert receipt["constructionSeconds"] is None
    assert receipt["simulationSeconds"] >= 0
    assert receipt["semanticParitySeconds"] >= 0
    assert receipt["analysisSeconds"] >= 0
    assert receipt["resultCodes"] == ["LAF-EXACT"]
    assert harness.store.run["resultCodes"] == ["LAF-EXACT"]
    assert receipt["distinctQualifiedResultHashes"] == ["1" * 64]
    assert receipt["qualifiedScenarioHashes"] == [
        qualified_scenario_identity_hash_v1(
            {
                "scenarioProjection": {
                    "scenarioTemplateId": "SCN-EXACTPERF001",
                    "exactStateId": "STATE-EXACTPERFORMANCE001",
                }
            }
        )
    ]
    assert receipt["proofResources"] == {
        "nodes": 17,
        "cacheEntries": 4,
        "cacheBytes": 512,
        "hiddenWorlds": 2,
        "aggregation": "single_proof_invocation_v1",
    }
    assert (
        harness.store.run["events"][-1]["performanceReceiptHash"] == receipt["sha256"]
    )


@pytest.mark.real_clock
def test_search_v4_run_persists_funnel_stage_and_duplicate_result_evidence(
    monkeypatch,
):
    proofs = [
        SimpleNamespace(
            payload={
                "completionStatus": "exact_complete",
                "resources": {
                    "nodes": 10,
                    "cacheEntries": 2,
                    "cacheBytes": 100,
                    "elapsedWallSeconds": 0.10,
                    "cancelled": False,
                },
                "hiddenWorlds": {"count": 1},
            }
        ),
        SimpleNamespace(
            payload={
                "completionStatus": "exact_complete",
                "resources": {
                    "nodes": 20,
                    "cacheEntries": 3,
                    "cacheBytes": 200,
                    "elapsedWallSeconds": 0.20,
                    "cancelled": False,
                },
                "hiddenWorlds": {"count": 2},
            }
        ),
    ]
    satisfactions = [
        SimpleNamespace(status="not_satisfied"),
        SimpleNamespace(status="satisfied"),
    ]

    def _candidate(index: int, *, solver_status: str, elapsed: float):
        structural = {
            "status": "satisfied",
            "solverStatus": solver_status,
            "elapsedSeconds": elapsed,
            "deckOrder": list(range(index, index + 4)),
        }
        receipt = {
            "auctionFeasible": True,
            "structural": structural,
            "sha256": str(index + 1) * 64,
        }
        return SimpleNamespace(
            status="satisfied" if index else "semantic_postcondition_miss",
            normalized=SimpleNamespace(index=index),
            proof=proofs[index],
            satisfaction=satisfactions[index],
            construction_receipt=receipt,
            sha256=str(index + 3) * 64,
        )

    candidates = iter(
        [
            _candidate(0, solver_status="feasible", elapsed=0.25),
            _candidate(1, solver_status="optimal", elapsed=0.50),
        ]
    )

    class _Executor:
        def build_candidate(self, *_args, **_kwargs):
            return next(candidates)

    monkeypatch.setattr(service_module, "ScenarioSearchExecutorV1", _Executor)
    emitted = 0

    def _put_result(_self, **kwargs):
        nonlocal emitted
        index = emitted
        emitted += 1
        return {
            "resultCode": "LAF-DUPLICATE",
            "resultHash": str(index + 7) * 64,
            "disposition": (
                "qualified"
                if kwargs["satisfaction"].status == "satisfied"
                else "rejected"
            ),
            "qualification": {
                "qualified": kwargs["satisfaction"].status == "satisfied"
            },
            "scenarioProjection": {
                "scenarioTemplateId": "SCN-SEARCHPERF01",
                "exactStateId": f"STATE-SEARCHPERFORMANCE{index:03d}",
            },
        }

    monkeypatch.setattr(
        ResearchFoundryServiceV1, "_put_v4_scenario_result", _put_result
    )
    run = _run("RUN-V4-PERF-SEARCH")
    harness = _harness(run)
    context = SimpleNamespace(bundle=SimpleNamespace(rules=object(), sha256="b" * 64))

    ResearchFoundryServiceV1._execute_v4_search_run(
        harness,
        run,
        _recipe("search", seed_count=2),
        context,
    )

    receipt = harness.store.run["performanceReceipt"]
    _assert_hash_bound_receipt(receipt, run["runCode"])
    assert receipt["executionKind"] == "scenario_search"
    assert receipt["attemptedSeeds"] == 2
    assert receipt["attemptedCandidates"] == 2
    assert receipt["structuralCandidates"] == 2
    assert receipt["auctionFeasible"] == 2
    assert receipt["funnel"] == {
        "attempted": 2,
        "structurallyFeasible": 2,
        "legalReplay": 2,
        "semanticSatisfied": 1,
        "proofCompleted": 2,
        "qualified": 1,
    }
    assert receipt["solverOptimal"] == 1
    assert receipt["solverIncomplete"] == 1
    assert receipt["constructionSeconds"] == 0.75
    assert receipt["analysisSeconds"] == pytest.approx(0.30)
    assert receipt["simulationSeconds"] is None
    assert receipt["semanticParitySeconds"] is None
    assert receipt["resultCodes"] == ["LAF-DUPLICATE", "LAF-DUPLICATE"]
    assert harness.store.run["resultCodes"] == ["LAF-DUPLICATE", "LAF-DUPLICATE"]
    assert receipt["distinctQualifiedResultHashes"] == ["8" * 64]
    assert receipt["qualifiedScenarioHashes"] == [
        qualified_scenario_identity_hash_v1(
            {
                "scenarioProjection": {
                    "scenarioTemplateId": "SCN-SEARCHPERF01",
                    "exactStateId": "STATE-SEARCHPERFORMANCE001",
                }
            }
        )
    ]
    assert receipt["proofResources"] == {
        "nodes": 30,
        "cacheEntries": 5,
        "cacheBytes": 300,
        "hiddenWorlds": 3,
        "aggregation": "sum_per_proof_invocation_v1",
    }
    evidence_reference = harness.store.run["scenarioSearchEvidence"]
    evidence = json.loads(harness.artifact_store.values[evidence_reference["uri"]])
    assert evidence["performanceReceiptHash"] == receipt["sha256"]
    assert (
        harness.store.run["events"][-1]["performanceReceiptHash"] == receipt["sha256"]
    )


def test_search_v4_receipt_does_not_alias_auction_feasible_to_legal_replay(
    monkeypatch,
):
    candidate = SimpleNamespace(
        status="construction_target_miss",
        normalized=None,
        proof=None,
        satisfaction=None,
        construction_receipt={
            "auctionFeasible": True,
            "structural": {
                "status": "satisfied",
                "solverStatus": "optimal",
                "elapsedSeconds": 0.0,
                "deckOrder": [0, 1, 2, 3],
            },
            "sha256": "1" * 64,
        },
        sha256="2" * 64,
    )

    class _Executor:
        @staticmethod
        def build_candidate(*_args, **_kwargs):
            return candidate

    monkeypatch.setattr(service_module, "ScenarioSearchExecutorV1", _Executor)
    run = _run("RUN-V4-AUCTION-FEASIBLE-MISS")
    harness = _harness(run)
    context = SimpleNamespace(bundle=SimpleNamespace(rules=object(), sha256="b" * 64))

    ResearchFoundryServiceV1._execute_v4_search_run(
        harness,
        run,
        _recipe("search"),
        context,
    )

    receipt = harness.store.run["performanceReceipt"]
    _assert_hash_bound_receipt(receipt, run["runCode"])
    assert receipt["attemptedCandidates"] == 1
    assert receipt["structuralCandidates"] == 1
    assert receipt["auctionFeasible"] == 1
    assert receipt["funnel"]["structurallyFeasible"] == 1
    assert receipt["funnel"]["legalReplay"] == 0
    assert receipt["resultCount"] == 0
    assert receipt["terminalStatus"] == "not_found_within_budget"


def test_search_v4_cancellation_before_candidate_persists_typed_partial_receipt():
    run = {**_run("RUN-V4-CANCEL-BEFORE"), "status": "cancelled"}
    harness = _harness(run)
    harness._cancel_check = lambda _run_code: True
    context = SimpleNamespace(bundle=SimpleNamespace(rules=object(), sha256="b" * 64))

    ResearchFoundryServiceV1._execute_v4_search_run(
        harness,
        run,
        _recipe("search", seed_count=2),
        context,
    )

    frozen = harness.store.run
    assert frozen["status"] == "cancelled"
    assert "performanceReceipt" not in frozen
    receipt = frozen["partialPerformanceReceipt"]
    _assert_hash_bound_partial_receipt(receipt, run["runCode"])
    assert receipt["terminalReason"] == "cancelled_before_candidate"
    assert receipt["attemptedSeeds"] == 0
    assert receipt["attemptedCandidates"] == 0
    assert receipt["resultCount"] == 0
    assert receipt["proofResources"] == {
        "nodes": None,
        "cacheEntries": None,
        "cacheBytes": None,
        "hiddenWorlds": None,
        "aggregation": "not_started_v1",
        "availability": "not_started",
    }
    assert frozen["events"][-1]["performanceReceiptHash"] == receipt["sha256"]


def test_search_v4_mid_search_cancellation_preserves_actual_partial_counters(
    monkeypatch,
):
    candidate = SimpleNamespace(
        status="construction_target_miss",
        normalized=None,
        proof=None,
        satisfaction=None,
        construction_receipt={
            "auctionFeasible": True,
            "structural": {
                "status": "satisfied",
                "solverStatus": "optimal",
                "elapsedSeconds": 0.25,
                "deckOrder": [0, 1, 2, 3],
            },
            "sha256": "3" * 64,
        },
        sha256="4" * 64,
    )

    class _Executor:
        @staticmethod
        def build_candidate(*_args, **_kwargs):
            return candidate

    monkeypatch.setattr(service_module, "ScenarioSearchExecutorV1", _Executor)
    run = _run("RUN-V4-CANCEL-MID-SEARCH")
    harness = _harness(run)
    cancellation_checks = 0

    def cancel_after_first_candidate(_run_code):
        nonlocal cancellation_checks
        cancellation_checks += 1
        if cancellation_checks >= 3:
            harness.store.run["status"] = "cancelled"
            return True
        return False

    harness._cancel_check = cancel_after_first_candidate
    context = SimpleNamespace(bundle=SimpleNamespace(rules=object(), sha256="b" * 64))

    ResearchFoundryServiceV1._execute_v4_search_run(
        harness,
        run,
        _recipe("search", seed_count=2),
        context,
    )

    receipt = harness.store.run["partialPerformanceReceipt"]
    _assert_hash_bound_partial_receipt(receipt, run["runCode"])
    assert harness.store.run["status"] == "cancelled"
    assert receipt["terminalReason"] == "cancelled_between_candidates"
    assert receipt["attemptedSeeds"] == 1
    assert receipt["attemptedCandidates"] == 1
    assert receipt["structuralCandidates"] == 1
    assert receipt["auctionFeasible"] == 1
    assert receipt["solverOptimal"] == 1
    assert receipt["funnel"] == {
        "attempted": 1,
        "structurallyFeasible": 1,
        "legalReplay": 0,
        "semanticSatisfied": 0,
        "proofCompleted": 0,
        "qualified": 0,
    }
    assert receipt["constructionSeconds"] == 0.25
    assert receipt["proofResources"]["availability"] == "not_started"
    assert "performanceReceipt" not in harness.store.run


@pytest.mark.parametrize(
    ("fingerprints", "expected_attempted", "expected_status", "target_satisfied"),
    [
        (["a", "a", "b", "c"], 3, "satisfied", True),
        (["a", "a"], 2, "not_found_within_budget", False),
    ],
)
def test_search_v4_result_target_counts_distinct_diversity_selection(
    monkeypatch,
    fingerprints,
    expected_attempted,
    expected_status,
    target_satisfied,
):
    candidates = []
    for index in range(len(fingerprints)):
        proof = SimpleNamespace(
            payload={
                "completionStatus": "exact_complete",
                "resources": {
                    "nodes": 1,
                    "cacheEntries": 0,
                    "cacheBytes": 0,
                    "elapsedWallSeconds": 0.0,
                    "cancelled": False,
                },
                "hiddenWorlds": {"count": 1},
            }
        )
        candidates.append(
            SimpleNamespace(
                status="satisfied",
                normalized=SimpleNamespace(index=index),
                proof=proof,
                satisfaction=SimpleNamespace(status="satisfied"),
                construction_receipt={
                    "auctionFeasible": True,
                    "structural": {
                        "status": "satisfied",
                        "solverStatus": "optimal",
                        "elapsedSeconds": 0.0,
                        "deckOrder": [index, index + 1, index + 2, index + 3],
                    },
                    "sha256": f"{index + 1:x}" * 64,
                },
                sha256=f"{index + 5:x}" * 64,
            )
        )
    candidate_iter = iter(candidates)

    class _Executor:
        def build_candidate(self, *_args, **_kwargs):
            return next(candidate_iter)

    monkeypatch.setattr(service_module, "ScenarioSearchExecutorV1", _Executor)
    emitted = 0

    def _put_result(_self, **_kwargs):
        nonlocal emitted
        index = emitted
        emitted += 1
        return {
            "resultCode": f"LAF-TARGET-{index + 1}",
            "resultHash": fingerprints[index] * 64,
            "disposition": "qualified",
            "qualification": {"qualified": True},
            "scenarioProjection": {
                "scenarioTemplateId": "SCN-TARGETPERF01",
                "exactStateId": "STATE-TARGETPERFORMANCE-SAME",
                "decisionKind": "play",
                "achievedProofMode": "exact",
            },
        }

    monkeypatch.setattr(
        ResearchFoundryServiceV1, "_put_v4_scenario_result", _put_result
    )
    run = _run("RUN-V4-TARGET-TEST")
    harness = _harness(run)
    context = SimpleNamespace(bundle=SimpleNamespace(rules=object(), sha256="b" * 64))

    ResearchFoundryServiceV1._execute_v4_search_run(
        harness,
        run,
        _recipe("search", seed_count=len(fingerprints), result_count=2),
        context,
    )

    frozen = harness.store.run
    _assert_hash_bound_receipt(frozen["performanceReceipt"], run["runCode"])
    assert frozen["attemptedCandidates"] == expected_attempted
    assert frozen["scenarioSearchStatus"] == expected_status
    assert frozen["selectedCount"] == (2 if target_satisfied else 1)
    assert frozen["duplicateQualifiedCount"] == 1
    assert frozen["outputReceipt"]["resultTargetSatisfied"] is target_satisfied
    assert frozen["outputReceipt"]["sha256"] == canonical_sha256(
        {
            key: value
            for key, value in frozen["outputReceipt"].items()
            if key != "sha256"
        }
    )
    assert frozen["diversitySelection"]["selectedCount"] == frozen["selectedCount"]
    assert (
        len(frozen["performanceReceipt"]["qualifiedScenarioHashes"])
        == frozen["performanceReceipt"]["qualifiedCount"]
    )
    assert len(set(frozen["performanceReceipt"]["qualifiedScenarioHashes"])) == 1
