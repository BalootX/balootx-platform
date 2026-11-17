from __future__ import annotations

import pytest

from baloot_platform.research.service import (
    _operator_incomplete_compilation_v1,
    _operator_candidate_progress_receipt_v1,
    _operator_candidate_progress_snapshot_v1,
    _operator_stage_counters_v1,
    _operator_quality_solver_progress_v1,
    _operator_research_run_v1,
)
from baloot_platform.research.ortools_research_executor import _quality_failure_cause_v1


def test_family_rejection_projection_is_visible_but_never_reveals_an_opening():
    import copy
    source = {'type': 'progress', 'currentStage': 'family_proposal',
        'constructionStatus': 'seed_miss', 'constructionReason': 'opening_mismatch',
        'constructionElapsedSeconds': 3.5, 'screenedSeed': 42, 'screenedDealer': 'north',
        'auctionQualification': {'hands': ['hidden'], 'buyer': 'north'}, 'deckOrder': list(range(32))}
    original = copy.deepcopy(source)
    for status in ('running', 'completed'):
        for reason in ('opening_mismatch', 'fixed_deal_projects_infeasible', 'fixed_deal_projects_incomplete'):
            event = {**source, 'constructionReason': reason}
            projected = _operator_research_run_v1({'status': status, 'events': [event]})
            assert projected['events'] == [{key: event[key] for key in (
                'type', 'currentStage', 'constructionStatus', 'constructionReason', 'constructionElapsedSeconds')}]
    for mutation in ({'constructionStatus': []}, {'type': 'candidate'}, {'currentStage': 'playingQuality'}):
        event, = _operator_research_run_v1({'events': [{**source, **mutation}]})['events']
        assert not any(key.startswith('construction') for key in event)
    for seconds in (True, -1, float('nan'), float('inf'), '3.5', 10**400):
        event, = _operator_research_run_v1({'events': [{**source, 'constructionElapsedSeconds': seconds,
            'constructionReason': 'S7 is hidden'}]})['events']
        assert 'constructionReason' not in event and 'constructionElapsedSeconds' not in event
    assert source == original


def test_live_quality_solver_progress_excludes_private_assignments():
    from baloot_platform.research.scenario_contracts import SCENARIO_DECISION_KINDS
    source = {"kind": "quality_solver_progress_v1", "phase": "counterexample",
        "masterCount": 2, "queryCount": 3, "bindingCount": 1,
        "decisionKind": "bid", "proofPhase": "best",
        "deckOrder": list(range(32)), "hands": ["private"], "candidate": {"utility": 42}}
    assert _operator_quality_solver_progress_v1(source) == {
        "phase": "counterexample", "masterCount": 2, "queryCount": 3,
        "bindingCount": 1, "decisionKind": "bid", "proofPhase": "best"}
    for kind in SCENARIO_DECISION_KINDS:
        assert _operator_quality_solver_progress_v1({**source, "decisionKind": kind})["decisionKind"] == kind
    assert "decisionKind" not in _operator_quality_solver_progress_v1({**source, "decisionKind": "unknown"})
    assert _operator_quality_solver_progress_v1({**source, "proofPhase": "fallback"})["proofPhase"] == "fallback"
    assert _operator_quality_solver_progress_v1({**source, "queryCount": -1}) is None
    assert _operator_quality_solver_progress_v1({**source, "phase": "qualified"}) is None


def test_quality_failure_cause_reports_stage_without_private_evidence():
    failure = {"stage": "native_observed_play", "evidence": {"causeReceipt": {
        "stage": "native_observed_root_actions", "evidence": {"causeReceipt": {
            "stage": "hidden_world_enumeration", "hands": ["hidden"],
        }},
    }}}
    assert _quality_failure_cause_v1(failure) == ("hidden_world_enumeration", None, None, None)
    assert _quality_failure_cause_v1({"evidence": {"cause": {"stage": "native_observed_counterexample",
        "evidence": {"queries": [{"queryModelHash": "a" * 64}], "queryPhase": "path_query",
            "pathPhase": "solver"}}}}) == ("native_observed_counterexample", 1, "path_query", "solver")
    assert _quality_failure_cause_v1({"stage": "native_observed_play"}) == (None, None, None, None)
    failure = {"stage": "quality_readback", "evidence": {
        "generationReceipt": {"stage": "hidden_world_enumeration", "exhaustionReason": "maximum_seconds",
                              "evidence": {"hands": ["private"], "acceptedWorlds": 109}},
        "readbackReceipt": {"stage": "quality_readback", "exhaustionReason": "maximum_seconds"},
    }}
    assert _quality_failure_cause_v1(failure) == ("hidden_world_enumeration", None, None, None)
    for wrapper in ("nativeBidDominance", "nativeObservedBidDominance"):
        failure = {"stage": "complete_state_quality", "evidence": {wrapper: {
            "queriesUsed": 18, "hands": ["private"], "cause": {
                "stage": "native_bid_preparation", "evidence": {
                    "queries": [{}], "queryPhase": "preparation", "pathPhase": "solver",
                },
            },
        }}}
        assert _quality_failure_cause_v1(failure) == ("native_bid_preparation", 18, "preparation", "solver")
        failure["evidence"][wrapper]["queriesUsed"] = True
        assert _quality_failure_cause_v1(failure) == ("native_bid_preparation", 1, "preparation", "solver")


def test_live_incomplete_receipt_excludes_private_evidence_without_mutating_source():
    private = {
        "stage": "playingQuality", "exhaustionReason": "maximum_seconds",
        "sha256": "a" * 64, "candidateHash": "b" * 64,
        "targetPlayIndex": 23, "targetActor": "west",
        "generatedActorAtTarget": "north",
        "firstConstructedPlayDivergenceIndex": 0,
        "qualityStatus": "quality_failed",
        "qualityFailureStage": "action_analysis",
        "qualityFailureCauseStage": "native_play_preparation",
        "qualityFailureReason": "recorded_action_not_optimal",
        "qualityEvidence": {"finalPrefix": {"deckOrder": list(range(32))}},
        "qualityCheckpoint": {"sourcePrefix": {"deckOrder": list(range(32))},
                              "artifact": {"legalActions": ["CA"], "value": 42}},
        "qualityElapsedSeconds": 3.5,
        "recordedTrace": {"hands": [[0, 1]]},
        "recordedEscalationOpportunities": [{"actor": "west", "actionCode": "decline"}],
        "graph": {"legalActions": ["play:0"], "value": 42},
        "evidence": {"graphReceipt": {"exactStateId": "private-state"},
                     "decisionStateStatusVector": list(range(32))},
        "reason": "unstructured diagnostic containing private state",
        "futureEvidenceField": {"principalVariation": ["play:0"]},
    }
    expected = {key: private[key] for key in (
        "stage", "exhaustionReason", "sha256", "candidateHash",
        "targetPlayIndex", "targetActor", "generatedActorAtTarget",
        "firstConstructedPlayDivergenceIndex", "qualityStatus",
        "qualityFailureStage", "qualityFailureCauseStage", "qualityFailureReason")}
    assert _operator_incomplete_compilation_v1(private) == expected
    receipt = _operator_candidate_progress_receipt_v1({
        "currentStage": "candidate_terminal", "candidateStatus": "incomplete",
        "incompleteCompilation": private,
    })
    assert receipt["incompleteCompilation"] == expected
    assert private["qualityEvidence"]["finalPrefix"]["deckOrder"] == list(range(32))
    assert _operator_incomplete_compilation_v1(None) is None
    assert _operator_incomplete_compilation_v1({"stage": {"hands": [0]}}) == {}
    assert _operator_incomplete_compilation_v1({
        "stage": "playingQuality", "reason": "hard_project_award_miss",
        "publicAwardedProjects": [{"owner": "north", "team": "north_south"}],
    }) == {"stage": "playingQuality", "reason": "hard_project_award_miss"}


def test_live_solver_counters_expose_only_safe_nonnegative_counts() -> None:
    assert _operator_stage_counters_v1({
        "tailCandidates": 24,
        "proofBoundaryAttempts": 24,
        "prefixLiftAttempts": 0,
        "ownerOfCard": ["north"] * 32,
        "proofRejectedZeroWinningRoots": True,
    }) == {
        "tailCandidates": 24,
        "proofBoundaryAttempts": 24,
        "prefixLiftAttempts": 0,
    }
    assert _operator_stage_counters_v1({"tailCandidates": -1}) is None
    assert _operator_stage_counters_v1(None) is None


def test_candidate_progress_receipts_cover_every_operator_status_deterministically() -> None:
    evidence = (
        {
            "currentStage": "candidate_terminal",
            "candidateStatus": "qualified",
            "shardIndex": 0,
            "shardId": "SHARD-0",
            "attempt": 1,
            "executionOrdinal": 1,
            "candidateHash": "qualified-hash",
            "candidateSummary": {"candidateHash": "qualified-hash"},
        },
        {
            "currentStage": "candidate_terminal",
            "candidateStatus": "semantic_postcondition_miss",
            "shardIndex": 1,
            "shardId": "SHARD-1",
            "attempt": 1,
            "executionOrdinal": 1,
            "candidateHash": "rejected-hash",
        },
        {
            "currentStage": "variant_selection_terminal",
            "candidateStatus": "duplicate",
            "shardIndex": 2,
            "shardId": "SHARD-2",
            "attempt": 1,
            "candidateHash": "variant-semantic-hash",
            "nearestAcceptedDistance": 0,
        },
        {
            "currentStage": "candidate_terminal",
            "candidateStatus": "incomplete",
            "shardIndex": 3,
            "shardId": "SHARD-3",
            "attempt": 1,
            "executionOrdinal": 1,
            "incompleteCompilation": {
                "stage": "proof",
                "exhaustionReason": "node_limit",
            },
            "candidateAttemptReason": "tail_proof_budget",
        },
        {
            "currentStage": "proof_retry_scheduled",
            "candidateStatus": "proof_incomplete",
            "shardIndex": 3,
            "shardId": "SHARD-3",
            "attempt": 1,
            "retryAttempt": 2,
        },
    )
    receipts = tuple(
        receipt
        for item in evidence
        if (receipt := _operator_candidate_progress_receipt_v1(item)) is not None
    )

    snapshot = _operator_candidate_progress_snapshot_v1(receipts, history_limit=3)
    reversed_snapshot = _operator_candidate_progress_snapshot_v1(
        reversed(receipts), history_limit=3
    )

    assert snapshot == reversed_snapshot
    assert snapshot["candidateReceiptCount"] == 5
    assert snapshot["candidateReceiptCounts"] == {
        "qualified": 1,
        "rejected": 1,
        "duplicate": 1,
        "incomplete": 2,
    }
    assert snapshot["candidateStatusCounts"] == {
        "duplicate": 1,
        "incomplete": 1,
        "proof_incomplete": 1,
        "qualified": 1,
        "semantic_postcondition_miss": 1,
    }
    assert snapshot["candidateReceiptHistoryTruncated"] is True
    assert len(snapshot["candidateReceipts"]) == 3
    duplicate = next(item for item in receipts if item["classification"] == "duplicate")
    assert duplicate["candidateHash"] is None
    assert duplicate["variantSemanticHash"] == "variant-semantic-hash"
    assert duplicate["nearestAcceptedDistance"] == 0
    incomplete = next(
        item
        for item in receipts
        if item.get("candidateAttemptReason") == "tail_proof_budget"
    )
    assert incomplete["classification"] == "incomplete"
    assert _operator_candidate_progress_receipt_v1(
        {"currentStage": "proof", "nodes": 10}
    ) is None


def test_completed_execution_replaces_its_intermediate_candidate_receipt() -> None:
    shared = {
        "currentStage": "candidate_terminal",
        "candidateStatus": "semantic_postcondition_miss",
        "shardIndex": 0,
        "shardId": "SHARD-0",
        "attempt": 1,
    }
    intermediate = _operator_candidate_progress_receipt_v1(
        {**shared, "candidateHash": "construction-hash"}
    )
    completed = _operator_candidate_progress_receipt_v1(
        {
            **shared,
            "executionOrdinal": 1,
            "candidateHash": "execution-hash",
        }
    )

    snapshot = _operator_candidate_progress_snapshot_v1(
        (intermediate, completed)
    )

    assert snapshot["candidateReceiptCount"] == 1
    assert snapshot["candidateReceiptCounts"]["rejected"] == 1
    assert snapshot["candidateReceipts"][0]["candidateHash"] == "execution-hash"


@pytest.mark.real_clock
def test_worker_and_coordinator_deadline_count_the_same_candidate_once() -> None:
    shared = {
        "candidateStatus": "incomplete", "shardId": "SHARD-0",
        "attempt": 1, "executionOrdinal": 1, "candidateHash": "candidate-hash",
    }
    terminal = _operator_candidate_progress_receipt_v1(
        {**shared, "currentStage": "candidate_terminal"})
    deadline = _operator_candidate_progress_receipt_v1(
        {**shared, "currentStage": "shard_budget_exhausted"})
    snapshot = _operator_candidate_progress_snapshot_v1((terminal, deadline))
    assert snapshot == _operator_candidate_progress_snapshot_v1((deadline, terminal))
    assert snapshot["candidateReceiptCounts"]["incomplete"] == 1
    assert snapshot["candidateReceipts"] == [terminal]
    # A lone worker receipt, different candidate, or later retry must survive.
    assert _operator_candidate_progress_snapshot_v1((deadline,))["candidateReceipts"] == [deadline]
    for change in ({"candidateHash": "other"}, {"attempt": 2}, {"executionOrdinal": 2}):
        distinct = _operator_candidate_progress_receipt_v1(
            {**shared, **change, "currentStage": "shard_budget_exhausted"})
        assert _operator_candidate_progress_snapshot_v1(
            (terminal, distinct))["candidateReceiptCounts"]["incomplete"] == 2
