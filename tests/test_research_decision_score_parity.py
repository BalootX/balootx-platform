from __future__ import annotations

import copy
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

import pytest

from baloot_platform.analysis.bidding import BidChoice, BidChoiceEvaluation
from baloot_platform.analysis.decision_scores import (
    BID_DECISION_OBJECTIVE_V1,
    BID_DECISION_TIE_BREAK_V1,
    PLAY_DECISION_TIE_BREAK_V1,
    PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
    bid_score_rows_v1,
    canonical_sha256_v1,
    legal_action_set_hash_v1,
    play_score_rows_v1,
    primary_decision_score_evidence_v1,
)
from baloot_platform.analysis.models import ActionEvaluation
from baloot_platform.challenge.decision_cohort import (
    PRIMARY_DECISION_COHORT_SALT_V1,
    derive_actor_observer_cohort_seed_v1,
)
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat
from baloot_platform.research.service import ResearchFoundryServiceV1


_RULES_HASH = "1" * 64
_PROFILE_HASH = "2" * 64
_IDENTITY_HASH = "3" * 64
_TRACE_HASH = "4" * 64
_ANALYSIS_HASH = "5" * 64
_CONTEXT_HASH = "6" * 64
_BACKEND_ID = "python_reference_exact_search_v1"
_ANALYSIS_BUDGET = {"requested_samples": 4, "exact_allocation_limit": 4}


def _bid_evidence(public_prefix_hash: str, *, margin_delta: float = 0.0):
    evaluations = (
        BidChoiceEvaluation(
            choice=BidChoice("sun"),
            expected_match_margin=12.0 + margin_delta,
            team_win_probability=0.7,
            contract_success_probability=0.65,
            downside_cvar10=-4.0,
            volatility=3.0,
            best_share=0.75,
            samples=4,
        ),
        BidChoiceEvaluation(
            choice=BidChoice("pass"),
            expected_match_margin=2.0,
            team_win_probability=0.4,
            contract_success_probability=0.0,
            downside_cvar10=-7.0,
            volatility=5.0,
            best_share=0.25,
            samples=4,
        ),
    )
    seed = derive_actor_observer_cohort_seed_v1(
        public_prefix_hash=public_prefix_hash,
        decision_kind="bid",
        acting_seat=Seat.NORTH,
        cohort_authority_hash=_IDENTITY_HASH,
        cohort_index=0,
        governed_salt=PRIMARY_DECISION_COHORT_SALT_V1,
    )
    return primary_decision_score_evidence_v1(
        decision_kind="bid",
        actor=Seat.NORTH,
        public_prefix_hash=public_prefix_hash,
        rules_hash=_RULES_HASH,
        analysis_identity_hash=_IDENTITY_HASH,
        trace_analysis_profile_hash=_PROFILE_HASH,
        cohort_seed_policy=PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
        cohort_index=0,
        cohort_seed=seed,
        objective=BID_DECISION_OBJECTIVE_V1,
        tie_break=BID_DECISION_TIE_BREAK_V1,
        backend=_BACKEND_ID,
        method="paired_hidden_deal_rollout_v1",
        hidden_deal_support=4,
        effective_sample_size=4.0,
        recommended_action=evaluations[0].choice.to_dict(),
        rows=bid_score_rows_v1(evaluations),
    )


def _play_evaluations(*, margin_delta: float = 0.0):
    return (
        ActionEvaluation(
            card=Card.parse("D10"),
            expected_round_margin=8.0 + margin_delta,
            expected_team_awarded_points=20.0,
            expected_opponent_awarded_points=12.0,
            round_lead_probability=0.7,
            draw_probability=0.05,
            confidence_low=0.5,
            confidence_high=0.85,
            best_share=0.75,
            samples=4,
            effective_samples=4.0,
            exact_share=0.5,
            contract_success_probability=0.7,
            match_win_probability=0.6,
            match_confidence_low=0.45,
            match_confidence_high=0.75,
            match_value_model="policy_conditioned_exact_dynamic_program_v2",
            deal_margins=(10.0, 8.0, 4.0, -2.0),
            downside_cvar=-2.0,
            volatility=4.0,
            principal_variation=("D10", "C7"),
            search_nodes=20,
            search_completion_status="heuristic_complete",
        ),
        ActionEvaluation(
            card=Card.parse("C7"),
            expected_round_margin=3.0,
            expected_team_awarded_points=16.0,
            expected_opponent_awarded_points=13.0,
            round_lead_probability=0.55,
            draw_probability=0.1,
            confidence_low=0.35,
            confidence_high=0.7,
            best_share=0.25,
            samples=4,
            effective_samples=4.0,
            exact_share=0.25,
            contract_success_probability=0.6,
            match_win_probability=0.5,
            match_confidence_low=0.35,
            match_confidence_high=0.65,
            match_value_model="policy_conditioned_exact_dynamic_program_v2",
            deal_margins=(5.0, 3.0, 1.0, -4.0),
            downside_cvar=-4.0,
            volatility=5.0,
            principal_variation=("C7", "D10"),
            search_nodes=18,
            search_completion_status="heuristic_complete",
        ),
    )


def _play_evidence(public_prefix_hash: str, *, margin_delta: float = 0.0):
    evaluations = _play_evaluations(margin_delta=margin_delta)
    seed = derive_actor_observer_cohort_seed_v1(
        public_prefix_hash=public_prefix_hash,
        decision_kind="play",
        acting_seat=Seat.WEST,
        cohort_authority_hash=_IDENTITY_HASH,
        cohort_index=0,
        governed_salt=PRIMARY_DECISION_COHORT_SALT_V1,
    )
    return primary_decision_score_evidence_v1(
        decision_kind="play",
        actor=Seat.WEST,
        public_prefix_hash=public_prefix_hash,
        rules_hash=_RULES_HASH,
        analysis_identity_hash=_IDENTITY_HASH,
        trace_analysis_profile_hash=_PROFILE_HASH,
        cohort_seed_policy=PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
        cohort_index=0,
        cohort_seed=seed,
        objective="match_win_probability",
        tie_break=PLAY_DECISION_TIE_BREAK_V1,
        backend=_BACKEND_ID,
        method="paired_hidden_deal_rollout_v1",
        hidden_deal_support=4,
        effective_sample_size=4.0,
        recommended_action=evaluations[0].card.to_dict(),
        rows=play_score_rows_v1(evaluations),
    )


def _forced_proof(
    *,
    decision_kind: str,
    seat: Seat,
    public_prefix_hash: str,
    selected_action: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schemaVersion": 1,
        "proofType": "authoritative_forced_legal_action_v1",
        "decisionKind": decision_kind,
        "actor": seat.name.lower(),
        "publicPrefixHash": public_prefix_hash,
        "rulesHash": _RULES_HASH,
        "legalActionCount": 1,
        "legalActionSetHash": legal_action_set_hash_v1((selected_action,)),
        "selectedAction": dict(selected_action),
        "selectedActionHash": canonical_sha256_v1(dict(selected_action)),
    }
    return {**core, "sha256": canonical_sha256_v1(core)}


class _AuditItem:
    def __init__(
        self,
        record: Any,
        *,
        evidence: Any = None,
        proof: Mapping[str, Any] | None = None,
    ) -> None:
        self.decision_id = record.decision_id
        self.ordinal = record.ordinal
        self.decision_kind = record.decision_kind
        self.forced = record.forced
        self.primary_decision_score_evidence = evidence
        self.forced_legality_proof = proof

    def validate_ledger_binding(
        self,
        record: Any,
        *,
        trace_analysis_profile_hash: str,
        rules_hash: str,
    ) -> None:
        if (
            record.decision_id != self.decision_id
            or record.ordinal != self.ordinal
            or trace_analysis_profile_hash != _PROFILE_HASH
            or rules_hash != _RULES_HASH
        ):
            raise ValueError("synthetic direct-deep ledger binding mismatch")


def _combined_policy(evidence: Any) -> dict[str, Any]:
    primary = {
        "selectedAction": dict(evidence.recommended_action),
        "primaryDecisionScoreEvidence": evidence.to_dict(),
        "primaryDecisionScoreEvidenceHash": evidence.sha256,
    }
    return {
        "source": "finalist_multi_cohort_agent",
        "decisionKind": evidence.decision_kind,
        "selectionMode": "primary_policy_only",
        "primaryActionPreserved": True,
        "observerSafe": True,
        "traceAnalysisProfileHash": _PROFILE_HASH,
        "analysisIdentityHash": _IDENTITY_HASH,
        "analysisBudget": dict(_ANALYSIS_BUDGET),
        "selectedAction": dict(evidence.recommended_action),
        "primaryTrace": primary,
        "primaryDecisionScoreEvidence": evidence.to_dict(),
        "primaryDecisionScoreEvidenceHash": evidence.sha256,
        "decisionScoreRowsHash": evidence.rows_hash,
        "legalActionSetHash": evidence.legal_action_set_hash,
        "forcedLegalityProof": None,
    }


def _fixture():
    bid_prefix = "7" * 64
    play_prefix = "8" * 64
    forced_prefix = "9" * 64
    bid_evidence = _bid_evidence(bid_prefix)
    play_evidence = _play_evidence(play_prefix)
    forced_action = Card.parse("H7").to_dict()
    forced_proof = _forced_proof(
        decision_kind="play",
        seat=Seat.SOUTH,
        public_prefix_hash=forced_prefix,
        selected_action=forced_action,
    )
    records = (
        SimpleNamespace(
            decision_id="bid-1",
            ordinal=1,
            acting_seat=Seat.NORTH,
            decision_kind="bid",
            forced=False,
            chosen_action=dict(bid_evidence.recommended_action),
            public_prefix_hash=bid_prefix,
            legal_actions=tuple(row["action"] for row in bid_evidence.rows),
        ),
        SimpleNamespace(
            decision_id="play-2",
            ordinal=2,
            acting_seat=Seat.WEST,
            decision_kind="play",
            forced=False,
            chosen_action=dict(play_evidence.recommended_action),
            public_prefix_hash=play_prefix,
            legal_actions=tuple(row["action"] for row in play_evidence.rows),
        ),
        SimpleNamespace(
            decision_id="play-3",
            ordinal=3,
            acting_seat=Seat.SOUTH,
            decision_kind="play",
            forced=True,
            chosen_action=forced_action,
            public_prefix_hash=forced_prefix,
            legal_actions=(forced_action,),
        ),
    )
    policies = [
        {"ordinal": 1, "trace": _combined_policy(bid_evidence)},
        {"ordinal": 2, "trace": _combined_policy(play_evidence)},
        {
            "ordinal": 3,
            "trace": {
                "selectedAction": forced_action,
                "primaryDecisionScoreEvidence": None,
                "primaryDecisionScoreEvidenceHash": None,
                "forcedLegalityProof": forced_proof,
            },
        },
    ]
    trace = SimpleNamespace(
        sha256=_TRACE_HASH,
        rules_hash=_RULES_HASH,
        decision_ledger=SimpleNamespace(records=records),
        trusted_final_state={"decisionPolicyTraces": policies},
    )
    analysis = SimpleNamespace(
        trace_hash=_TRACE_HASH,
        profile_hash=_PROFILE_HASH,
        sha256=_ANALYSIS_HASH,
        decisions=(
            _AuditItem(records[0], evidence=bid_evidence),
            _AuditItem(records[1], evidence=play_evidence),
            _AuditItem(records[2], proof=forced_proof),
        ),
    )
    profile = SimpleNamespace(
        sha256=_PROFILE_HASH,
        objective="match_win_probability",
    )
    context = SimpleNamespace(
        analyzers={"deep": SimpleNamespace(profile=profile)},
        analysis_identity_hashes={"deep": _IDENTITY_HASH},
        exact_search_backend=SimpleNamespace(
            descriptor=SimpleNamespace(id=_BACKEND_ID)
        ),
        simulator=SimpleNamespace(
            agent_profile=SimpleNamespace(id="finalist_candidate"),
            decision_agreement=SimpleNamespace(
                exact_primary_policy=True,
                primary_analysis_budget=dict(_ANALYSIS_BUDGET),
            ),
            analysis_identity_hash=_IDENTITY_HASH,
            trace_analysis_profile_hash=_PROFILE_HASH,
        ),
        context_hash=_CONTEXT_HASH,
    )
    return trace, analysis, context


def test_exact_finalist_bid_play_and_forced_evidence_match_direct_deep() -> None:
    trace, analysis, context = _fixture()

    parity = ResearchFoundryServiceV1._decision_score_parity_evidence(
        trace,
        analysis,
        context=context,
    )

    assert parity["matched"] is True
    assert parity["decisionCount"] == 3
    assert parity["matchedDecisionCount"] == 3
    assert parity["nonForcedDecisionCount"] == 2
    assert parity["forcedDecisionCount"] == 1
    assert parity["mismatches"] == []
    assert parity["crossLaneComparison"] is False
    assert parity["structuralObjectiveCompared"] is False
    assert parity["officialBalootScoreCompared"] is False


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    (
        ("missing_policy", "generation_policy_trace_missing"),
        ("duplicate_policy", "decision_policy_trace_ordinal_duplicate"),
        ("missing_evidence", "generation_primary_score_evidence_missing"),
        (
            "self_valid_score_drift",
            "primary_decision_score_evidence_bytes_mismatch",
        ),
        ("primary_trace_drift", "generation_primary_trace_binding_mismatch"),
        ("forced_proof_missing", "forced_generation_legality_proof_missing"),
        ("forced_proof_drift", "forced_legality_proof_bytes_mismatch"),
    ),
)
def test_finalist_decision_score_drift_fails_closed(
    mutation: str,
    expected_reason: str,
) -> None:
    trace, analysis, context = _fixture()
    rows = trace.trusted_final_state["decisionPolicyTraces"]
    if mutation == "missing_policy":
        rows.pop(1)
    elif mutation == "duplicate_policy":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "missing_evidence":
        rows[1]["trace"].pop("primaryDecisionScoreEvidence")
    elif mutation == "self_valid_score_drift":
        replacement = _play_evidence("8" * 64, margin_delta=1.0)
        policy = rows[1]["trace"]
        policy["primaryDecisionScoreEvidence"] = replacement.to_dict()
        policy["primaryDecisionScoreEvidenceHash"] = replacement.sha256
        policy["decisionScoreRowsHash"] = replacement.rows_hash
        policy["legalActionSetHash"] = replacement.legal_action_set_hash
        policy["primaryTrace"]["primaryDecisionScoreEvidence"] = (
            replacement.to_dict()
        )
        policy["primaryTrace"]["primaryDecisionScoreEvidenceHash"] = (
            replacement.sha256
        )
    elif mutation == "primary_trace_drift":
        rows[1]["trace"]["primaryTrace"][
            "primaryDecisionScoreEvidence"
        ] = None
    elif mutation == "forced_proof_missing":
        rows[2]["trace"].pop("forcedLegalityProof")
    elif mutation == "forced_proof_drift":
        proof = rows[2]["trace"]["forcedLegalityProof"]
        changed = {key: value for key, value in proof.items() if key != "sha256"}
        changed["rulesHash"] = "a" * 64
        rows[2]["trace"]["forcedLegalityProof"] = {
            **changed,
            "sha256": canonical_sha256_v1(changed),
        }
    else:  # pragma: no cover - parameter list is closed above
        raise AssertionError(mutation)

    parity = ResearchFoundryServiceV1._decision_score_parity_evidence(
        trace,
        analysis,
        context=context,
    )

    assert parity["matched"] is False
    assert any(
        expected_reason in str(item["reason"])
        for item in parity["mismatches"]
    )


def test_distinct_games_are_only_compared_with_their_own_deep_evidence() -> None:
    trace, analysis, context = _fixture()
    replacement = _bid_evidence("7" * 64, margin_delta=2.0)
    record = trace.decision_ledger.records[0]
    policy = trace.trusted_final_state["decisionPolicyTraces"][0]["trace"]
    policy["primaryDecisionScoreEvidence"] = replacement.to_dict()
    policy["primaryDecisionScoreEvidenceHash"] = replacement.sha256
    policy["decisionScoreRowsHash"] = replacement.rows_hash
    policy["legalActionSetHash"] = replacement.legal_action_set_hash
    policy["primaryTrace"]["primaryDecisionScoreEvidence"] = replacement.to_dict()
    policy["primaryTrace"]["primaryDecisionScoreEvidenceHash"] = replacement.sha256
    analysis.decisions = (
        _AuditItem(record, evidence=replacement),
        *analysis.decisions[1:],
    )

    parity = ResearchFoundryServiceV1._decision_score_parity_evidence(
        trace,
        analysis,
        context=context,
    )

    assert parity["matched"] is True
    assert parity["crossLaneComparison"] is False


def test_copied_bypass_evidence_cannot_impersonate_a_finalist_policy() -> None:
    trace, analysis, context = _fixture()
    policy = trace.trusted_final_state["decisionPolicyTraces"][0]["trace"]
    policy["source"] = "scripted_compiler"
    policy.pop("selectionMode")
    policy.pop("primaryActionPreserved")
    policy.pop("primaryTrace")

    parity = ResearchFoundryServiceV1._decision_score_parity_evidence(
        trace,
        analysis,
        context=context,
    )

    reasons = {str(item["reason"]) for item in parity["mismatches"]}
    assert {
        "generation_finalist_policy_source_mismatch",
        "generation_finalist_selection_mode_mismatch",
        "generation_primary_action_preservation_missing",
        "generation_primary_trace_missing",
    }.issubset(reasons)


def test_current_finalist_context_frozen_budget_matches_and_drift_fails() -> None:
    from baloot_platform.service import api

    factory = api.RESEARCH.execution_context_factory
    assert factory is not None
    revision_id = str(
        api.CONTAINER.configuration.resolve()["bundle"]["revisionId"]
    )
    current_context = factory.resolve_bundle_revision(revision_id)[1]
    agreement = current_context.simulator.decision_agreement
    assert agreement is not None
    assert isinstance(agreement.primary_analysis_budget, MappingProxyType)
    current_budget = dict(agreement.primary_analysis_budget)

    trace, analysis, context = _fixture()
    context.simulator.decision_agreement = agreement
    for row in trace.trusted_final_state["decisionPolicyTraces"][:2]:
        row["trace"]["analysisBudget"] = copy.deepcopy(current_budget)

    matched = ResearchFoundryServiceV1._decision_score_parity_evidence(
        trace,
        analysis,
        context=context,
    )

    assert matched["matched"] is True
    trace.trusted_final_state["decisionPolicyTraces"][0]["trace"][
        "analysisBudget"
    ]["requested_samples"] += 1

    drifted = ResearchFoundryServiceV1._decision_score_parity_evidence(
        trace,
        analysis,
        context=context,
    )

    assert drifted["matched"] is False
    assert any(
        item["reason"] == "generation_finalist_analysis_budget_mismatch"
        and item["decisionId"] == "bid-1"
        for item in drifted["mismatches"]
    )
