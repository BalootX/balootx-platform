from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from baloot_platform.core.project_visibility import project_rank_visibility_policy
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.serialization import replay_trusted_exact_state_context
from baloot_platform.research.exact_state import ExactStateNormalizerV1
from baloot_platform.research.result_authority import RuntimeCompatibilityV1
from baloot_platform.research.runtime_compatibility import (
    RuntimeCompatibilityAdmissionError,
    check_runtime_compatibility_v1,
    quarantine_runtime_compatibility_failure_v1,
    require_runtime_compatibility_passed_v1,
    runtime_compatibility_admission_v1,
)
from baloot_platform.research.scenario_contracts import canonical_sha256


def _qualified_document(root: Path, rules) -> dict[str, object]:
    fixture = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    exact_state = (
        ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"]).to_dict()
    )
    semantic_hash = canonical_sha256(
        {
            "scenario": fixture["scenarioId"],
            "qualificationAuthority": "cp_sat",
            "proof": fixture["proof"],
        }
    )
    return {
        "semanticResultHash": semantic_hash,
        "qualification": {"authority": "cp_sat", "status": "qualified"},
        "proof": copy.deepcopy(fixture["proof"]),
        "exactState": exact_state,
    }


def _qualified_escalation_document(root: Path, rules) -> dict[str, object]:
    fixture = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    prefix = copy.deepcopy(fixture["exactTracePrefix"])
    prefix.update(
        {
            "playActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "stateAssertions": {},
        }
    )
    exact_state = (
        ExactStateNormalizerV1(rules)
        .normalize(
            prefix,
            decision_point={
                "kind": "escalation",
                "selector": {"actingSeat": "west"},
            },
        )
        .to_dict()
    )
    return {
        "semanticResultHash": canonical_sha256(
            {
                "scenario": "exact-escalation-decision",
                "qualificationAuthority": "cp_sat",
            }
        ),
        "qualification": {"authority": "cp_sat", "status": "qualified"},
        "exactState": exact_state,
    }


def _check(document: dict[str, object], rules):
    return check_runtime_compatibility_v1(
        exact_state=document["exactState"],
        semantic_result_hash=str(document["semanticResultHash"]),
        bundle_hash="b" * 64,
        trace_hash="c" * 64,
        rules=rules,
        engine_version="21.1.0",
    )


def test_authoritative_replay_passes_without_mutating_cp_qualification(
    root: Path, rules
) -> None:
    document = _qualified_document(root, rules)
    before = copy.deepcopy(document)
    semantic_hash_before = canonical_sha256(document)

    check = _check(document, rules)

    assert document == before
    assert canonical_sha256(document) == semantic_hash_before
    receipt = check.receipt.to_dict()
    evidence = check.evidence.to_dict()
    assert receipt["status"] == "passed"
    assert receipt["failureCodes"] == []
    assert receipt["evidenceHash"] == evidence["sha256"]
    assert evidence["semanticResultHash"] == document["semanticResultHash"]
    assert evidence["replayStatus"] == "completed"
    assert (
        evidence["expectedStateProjectionHash"]
        == evidence["derivedStateProjectionHash"]
    )
    assert (
        evidence["expectedTraceProjectionHash"]
        == evidence["derivedTraceProjectionHash"]
    )
    assert require_runtime_compatibility_passed_v1(check.receipt) == check.receipt
    admission = runtime_compatibility_admission_v1(check.receipt)
    assert admission["playableEligible"] is True
    assert admission["publishableVideoEligible"] is True
    assert admission["auditMediaEligible"] is True
    assert admission["auditLabelRequired"] is False
    assert admission["blockers"] == []


def test_exact_escalation_replay_preserves_decision_stage_and_is_admitted(
    root: Path, rules
) -> None:
    document = _qualified_escalation_document(root, rules)
    exact_state = document["exactState"]
    assert isinstance(exact_state, dict)
    state = exact_state["state"]
    assert isinstance(state, dict)
    assert state["phase"] == "final_dealt"
    assert state["currentTrick"] is None
    assert state["projects"] == []
    assert any(action["code"] == "decline" for action in exact_state["legalActions"])

    check = _check(document, rules)

    receipt = check.receipt.to_dict()
    evidence = check.evidence.to_dict()
    assert receipt["status"] == "passed"
    assert receipt["failureCodes"] == []
    assert (
        evidence["expectedStateProjectionHash"]
        == evidence["derivedStateProjectionHash"]
    )
    assert (
        evidence["expectedTraceProjectionHash"]
        == evidence["derivedTraceProjectionHash"]
    )
    admission = runtime_compatibility_admission_v1(check.receipt)
    assert admission["playableEligible"] is True
    assert admission["publishableVideoEligible"] is True
    assert admission["blockers"] == []


@pytest.mark.parametrize(
    "document_builder", (_qualified_document, _qualified_escalation_document)
)
def test_replay_preserves_rules_bound_project_visibility(root, document_builder):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")
    document = document_builder(root, rules)
    before = copy.deepcopy(document)

    check = _check(document, rules)

    assert check.receipt.to_dict()["status"] == "passed"
    assert check.receipt.to_dict()["failureCodes"] == []
    replayed = replay_trusted_exact_state_context(document["exactState"]["state"], rules)
    assert replayed.to_dict()["projectRankVisibility"] == project_rank_visibility_policy(rules)
    assert document == before


@pytest.mark.parametrize("forged_policy", (None, {"mode": "declared"}))
def test_replay_rejects_client_authored_project_visibility(root, forged_policy):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")
    document = _qualified_document(root, rules)
    exact = document["exactState"]
    exact["state"]["projectRankVisibility"] = forged_policy
    exact["stateHash"] = canonical_sha256(exact["state"])

    replayed = replay_trusted_exact_state_context(exact["state"], rules)

    assert replayed.to_dict()["projectRankVisibility"] == project_rank_visibility_policy(rules)
    assert _check(document, rules).receipt.to_dict()["failureCodes"] == ["derived_state_mismatch"]


def test_exact_escalation_replay_rejects_forged_domain_state(root: Path, rules) -> None:
    document = _qualified_escalation_document(root, rules)
    exact_state = document["exactState"]
    assert isinstance(exact_state, dict)
    state = exact_state["state"]
    assert isinstance(state, dict)
    state["escalation"] = "double"
    exact_state["stateHash"] = canonical_sha256(state)

    check = _check(document, rules)

    assert check.receipt.to_dict()["status"] == "failed"
    assert check.receipt.to_dict()["failureCodes"] == ["engine_replay_failed"]
    assert (
        runtime_compatibility_admission_v1(check.receipt)["playableEligible"] is False
    )


def test_replay_state_disagreement_is_quarantined_and_blocks_admission(
    root: Path, rules
) -> None:
    document = _qualified_document(root, rules)
    exact_state = document["exactState"]
    assert isinstance(exact_state, dict)
    state = exact_state["state"]
    assert isinstance(state, dict)
    state["hands"] = copy.deepcopy(state["initialHands"])
    exact_state["stateHash"] = canonical_sha256(state)

    check = _check(document, rules)

    assert check.receipt.to_dict()["status"] == "failed"
    assert check.receipt.to_dict()["failureCodes"] == ["derived_state_mismatch"]
    assert check.evidence.to_dict()["replayStatus"] == "completed"
    with pytest.raises(
        RuntimeCompatibilityAdmissionError,
        match="playable or video admission",
    ):
        require_runtime_compatibility_passed_v1(check.receipt)


def test_illegal_replay_is_quarantined_and_unchecked_is_not_admitted(
    root: Path, rules
) -> None:
    document = _qualified_document(root, rules)
    exact_state = document["exactState"]
    assert isinstance(exact_state, dict)
    state = exact_state["state"]
    assert isinstance(state, dict)
    state["currentPlayer"] = "north"
    exact_state["stateHash"] = canonical_sha256(state)

    failed = _check(document, rules)

    assert failed.receipt.to_dict()["status"] == "failed"
    assert failed.receipt.to_dict()["failureCodes"] == ["engine_replay_failed"]
    assert failed.evidence.to_dict()["replayFailureHash"] is not None

    unchecked = RuntimeCompatibilityV1.build(
        status="not_checked",
        bundle_hash="b" * 64,
        trace_hash="c" * 64,
    )
    for receipt in (failed.receipt, unchecked):
        admission = runtime_compatibility_admission_v1(receipt)
        assert admission["playableEligible"] is False
        assert admission["publishableVideoEligible"] is False
        assert admission["auditMediaEligible"] is False
        assert admission["auditLabelRequired"] is True
        with pytest.raises(RuntimeCompatibilityAdmissionError):
            require_runtime_compatibility_passed_v1(receipt)


def test_malformed_exact_state_is_quarantined_instead_of_raising(rules) -> None:
    check = check_runtime_compatibility_v1(
        exact_state={},
        semantic_result_hash="a" * 64,
        bundle_hash="b" * 64,
        trace_hash="c" * 64,
        rules=rules,
        engine_version="21.1.0",
    )

    receipt = check.receipt.to_dict()
    assert receipt["status"] == "failed"
    assert receipt["failureCodes"] == ["exact_state_document_malformed"]
    assert check.evidence.to_dict()["replayStatus"] == "quarantined"
    admission = runtime_compatibility_admission_v1(receipt)
    assert admission["playableEligible"] is False
    assert admission["publishableVideoEligible"] is False
    assert admission["auditMediaEligible"] is False
    assert admission["auditLabelRequired"] is True


def test_quarantine_receipt_is_deterministic_and_does_not_capture_host_errors() -> None:
    common = {
        "semantic_result_hash": "a" * 64,
        "bundle_hash": "b" * 64,
        "trace_hash": "c" * 64,
        "failure_code": "runtime_compatibility_callback_failed",
    }
    first = quarantine_runtime_compatibility_failure_v1(
        **common,
        failure=RuntimeError(r"C:\\machine-a\\private\\state.json"),
    )
    second = quarantine_runtime_compatibility_failure_v1(
        **common,
        failure=RuntimeError("/machine-b/private/state.json"),
    )

    assert first.to_dict() == second.to_dict()
    assert "machine-a" not in json.dumps(first.to_dict())
    assert "machine-b" not in json.dumps(second.to_dict())


def test_replay_compares_canonical_trick_trace_not_only_remaining_state(
    root: Path, rules
) -> None:
    document = _qualified_document(root, rules)
    exact_state = document["exactState"]
    assert isinstance(exact_state, dict)
    state = exact_state["state"]
    assert isinstance(state, dict)
    completed = state["completedTricks"]
    assert isinstance(completed, list) and completed
    completed[0]["winner"] = "east"
    exact_state["stateHash"] = canonical_sha256(state)

    check = _check(document, rules)

    assert check.receipt.to_dict()["failureCodes"] == [
        "derived_state_mismatch",
        "derived_trace_mismatch",
    ]
    evidence = check.evidence.to_dict()
    assert evidence["replayStatus"] == "completed"
    assert (
        evidence["expectedTraceProjectionHash"]
        != evidence["derivedTraceProjectionHash"]
    )


def test_failed_compatibility_cannot_enter_labelled_audit_media_path() -> None:
    receipt = RuntimeCompatibilityV1.build(
        status="failed",
        bundle_hash="b" * 64,
        trace_hash="c" * 64,
        engine_version="test-engine",
        evidence_hash="d" * 64,
        failure_codes=("engine_replay_failed",),
    )

    admission = runtime_compatibility_admission_v1(receipt)

    assert admission["auditMediaEligible"] is False
    assert admission["blockers"] == ["runtime_compatibility_failed"]
