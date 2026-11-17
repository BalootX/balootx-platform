from __future__ import annotations

import copy
import hashlib
import json

import pytest

from baloot_platform.analysis.bidding import BidChoice, BidChoiceEvaluation
from baloot_platform.analysis.decision_scores import (
    PrimaryDecisionScoreEvidenceV1,
    bid_score_rows_v1,
    canonical_json_bytes_v1,
    canonical_sha256_v1,
    legal_action_set_hash_v1,
    play_score_rows_v1,
    primary_decision_score_evidence_from_mapping_v1,
    primary_decision_score_evidence_v1,
    validate_primary_decision_score_evidence_v1,
)
from baloot_platform.analysis.models import ActionEvaluation
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat, Suit


def _bid_evaluations() -> tuple[BidChoiceEvaluation, ...]:
    return (
        BidChoiceEvaluation(
            choice=BidChoice("sun"),
            expected_match_margin=12.5,
            team_win_probability=0.75,
            contract_success_probability=0.625,
            downside_cvar10=-8.0,
            volatility=4.5,
            best_share=0.7,
            samples=16,
        ),
        BidChoiceEvaluation(
            choice=BidChoice("hokum", Suit.HEARTS),
            expected_match_margin=9.0,
            team_win_probability=0.6,
            contract_success_probability=0.55,
            downside_cvar10=-11.0,
            volatility=6.0,
            best_share=0.3,
            samples=16,
        ),
    )


def _play_evaluations() -> tuple[ActionEvaluation, ...]:
    return (
        ActionEvaluation(
            card=Card(31),
            expected_round_margin=18.25,
            expected_team_awarded_points=31.5,
            expected_opponent_awarded_points=13.25,
            round_lead_probability=0.8,
            draw_probability=0.05,
            confidence_low=0.7,
            confidence_high=0.9,
            best_share=0.75,
            samples=4,
            effective_samples=3.5,
            exact_share=0.5,
            contract_success_probability=0.875,
            match_win_probability=0.625,
            match_confidence_low=0.5,
            match_confidence_high=0.75,
            match_value_model="policy_conditioned_exact_dynamic_program_v2",
            deal_margins=(22.0, 18.0, -3.0, 36.0),
            downside_cvar=-3.0,
            volatility=13.5,
            principal_variation=("CA", "S7"),
            search_nodes=412,
            search_completion_status="budget_exhausted",
        ),
        ActionEvaluation(
            card=Card(30),
            expected_round_margin=11.0,
            expected_team_awarded_points=25.0,
            expected_opponent_awarded_points=14.0,
            round_lead_probability=0.6,
            draw_probability=0.1,
            confidence_low=0.45,
            confidence_high=0.75,
            best_share=0.25,
            samples=4,
            effective_samples=3.5,
            exact_share=0.25,
            contract_success_probability=0.75,
            match_win_probability=0.5,
            match_confidence_low=0.35,
            match_confidence_high=0.65,
            match_value_model="policy_conditioned_exact_dynamic_program_v2",
            deal_margins=(15.0, 10.0, -8.0, 27.0),
            downside_cvar=-8.0,
            volatility=12.0,
            principal_variation=("CK", "S8"),
            search_nodes=390,
            search_completion_status="heuristic_complete",
        ),
    )


def _bid_evidence() -> PrimaryDecisionScoreEvidenceV1:
    rows = bid_score_rows_v1(_bid_evaluations())
    return primary_decision_score_evidence_v1(
        decision_kind="bid",
        actor=Seat.NORTH,
        public_prefix_hash="1" * 64,
        rules_hash="2" * 64,
        analysis_identity_hash="3" * 64,
        trace_analysis_profile_hash="4" * 64,
        cohort_seed_policy="actor_observer_cohort_seed_v1",
        cohort_index=0,
        cohort_seed=0x0123456789ABCDEF,
        objective="expected_round_margin",
        tie_break="win_probability_then_best_share_then_action_id",
        backend="python_reference_exact_search_v1",
        method="paired_hidden_deal_rollout_v1",
        hidden_deal_support=16,
        effective_sample_size=16.0,
        recommended_action=_bid_evaluations()[0].choice.to_dict(),
        rows=rows,
    )


def _assert_forbidden_scores_absent(value: object) -> None:
    forbidden = {
        "finalGameScore",
        "balootGameScore",
        "structuralObjective",
        "structuralObjectiveScore",
        "contentScore",
    }
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value)
        for child in value.values():
            _assert_forbidden_scores_absent(child)
    elif isinstance(value, list):
        for child in value:
            _assert_forbidden_scores_absent(child)


def test_bid_score_rows_preserve_analyzer_order_and_every_metric() -> None:
    evaluations = _bid_evaluations()
    rows = bid_score_rows_v1(evaluations)

    assert [row["actionId"] for row in rows] == ["sun", "hokum:hearts"]
    assert [row["rank"] for row in rows] == [1, 2]
    assert rows[0] == {
        "rank": 1,
        "actionId": "sun",
        "action": evaluations[0].choice.to_dict(),
        "expectedRoundMargin": 12.5,
        "roundLeadProbability": 0.75,
        "contractSuccessProbability": 0.625,
        "downsideCvar10": -8.0,
        "volatility": 4.5,
        "bestShare": 0.7,
        "samples": 16,
    }
    _assert_forbidden_scores_absent(list(rows))


def test_play_score_rows_preserve_full_semantic_evaluation() -> None:
    evaluations = _play_evaluations()
    rows = play_score_rows_v1(evaluations)

    assert [row["actionId"] for row in rows] == ["card:31", "card:30"]
    assert rows[0]["action"] == evaluations[0].card.to_dict()
    assert rows[0]["supportMarginsHash"] == canonical_sha256_v1(
        [22.0, 18.0, -3.0, 36.0]
    )
    assert rows[0]["supportMarginCount"] == 4
    assert rows[0]["drawProbability"] == 0.05
    assert rows[0]["roundLeadConfidence95"] == [0.7, 0.9]
    assert rows[0]["matchWinConfidence95"] == [0.5, 0.75]
    assert rows[0]["matchValueModel"] == (
        "policy_conditioned_exact_dynamic_program_v2"
    )
    assert rows[0]["principalVariationHash"] == canonical_sha256_v1(["CA", "S7"])
    assert rows[0]["principalVariationLength"] == 2
    assert rows[0]["searchCompletionStatus"] == "budget_exhausted"
    assert set(rows[0]) == {
        "rank",
        "actionId",
        "action",
        "expectedRoundMargin",
        "expectedTeamAwardedPoints",
        "expectedOpponentAwardedPoints",
        "roundLeadProbability",
        "drawProbability",
        "roundLeadConfidence95",
        "bestShare",
        "samples",
        "effectiveSamples",
        "exactShare",
        "contractSuccessProbability",
        "matchWinProbability",
        "matchWinConfidence95",
        "matchValueModel",
        "supportMarginsHash",
        "supportMarginCount",
        "downsideCvar10",
        "volatility",
        "principalVariationHash",
        "principalVariationLength",
        "searchNodes",
        "searchCompletionStatus",
    }
    _assert_forbidden_scores_absent(list(rows))


def test_evidence_binds_exact_bytes_hashes_and_round_trips() -> None:
    evidence = _bid_evidence()
    value = evidence.to_dict()
    core = {key: item for key, item in value.items() if key != "sha256"}

    assert canonical_json_bytes_v1({"z": 1, "a": "صن"}) == (
        '{"a":"صن","z":1}'.encode()
    )
    assert evidence.sha256 == hashlib.sha256(
        json.dumps(
            core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert value["rowsHash"] == canonical_sha256_v1({
        "schemaVersion": 1,
        "decisionKind": "bid",
        "rows": value["rows"],
    })
    assert value["legalActionSetHash"] == legal_action_set_hash_v1(
        row["action"] for row in value["rows"]
    )
    assert evidence.canonical_bytes() == canonical_json_bytes_v1(value)
    assert primary_decision_score_evidence_from_mapping_v1(value) == evidence
    assert validate_primary_decision_score_evidence_v1(
        value,
        decision_kind="bid",
        actor=Seat.NORTH,
        public_prefix_hash="1" * 64,
        legal_action_set_hash=value["legalActionSetHash"],
        rules_hash="2" * 64,
        analysis_identity_hash="3" * 64,
        trace_analysis_profile_hash="4" * 64,
        cohort_seed_policy="actor_observer_cohort_seed_v1",
        cohort_index=0,
        cohort_seed=0x0123456789ABCDEF,
    ) == evidence
    _assert_forbidden_scores_absent(value)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["rows"].reverse(),
        lambda value: value["rows"][0].__setitem__("expectedRoundMargin", 12.75),
        lambda value: value["rows"][0]["action"].__setitem__("action", "pass"),
        lambda value: value.__setitem__("recommendedAction", {"action": "pass", "trumpSuit": None}),
        lambda value: value.__setitem__("publicPrefixHash", "a" * 64),
        lambda value: value.__setitem__("rulesHash", "b" * 64),
        lambda value: value.__setitem__("analysisIdentityHash", "c" * 64),
        lambda value: value.__setitem__("traceAnalysisProfileHash", "d" * 64),
        lambda value: value.__setitem__("cohortSeedPolicy", "different_seed_policy_v1"),
        lambda value: value.__setitem__("cohortIndex", 1),
        lambda value: value.__setitem__("cohortSeed", value["cohortSeed"] + 1),
    ),
    ids=(
        "order",
        "value",
        "row-action",
        "recommended-action",
        "public-state",
        "rules-authority",
        "analysis-authority",
        "trace-profile-authority",
        "seed-policy",
        "cohort-index",
        "seed",
    ),
)
def test_evidence_rejects_mutated_order_value_action_state_authority_or_seed(
    mutate,
) -> None:
    value = copy.deepcopy(_bid_evidence().to_dict())
    mutate(value)

    with pytest.raises((TypeError, ValueError)):
        PrimaryDecisionScoreEvidenceV1.from_mapping(value)


def test_evidence_is_closed_and_requires_independent_binding_match() -> None:
    evidence = _bid_evidence()
    with_unknown_envelope = evidence.to_dict()
    with_unknown_envelope["contentScore"] = 100
    with pytest.raises(ValueError, match="unknown keys"):
        PrimaryDecisionScoreEvidenceV1.from_mapping(with_unknown_envelope)

    with_unknown_row = evidence.to_dict()
    with_unknown_row["rows"][0]["structuralObjectiveScore"] = 999
    with pytest.raises(ValueError, match="unknown keys"):
        PrimaryDecisionScoreEvidenceV1.from_mapping(with_unknown_row)

    with pytest.raises(ValueError, match="binding mismatch"):
        validate_primary_decision_score_evidence_v1(
            evidence,
            public_prefix_hash="f" * 64,
        )

    rows = bid_score_rows_v1(_bid_evaluations())
    with pytest.raises(ValueError, match="legal action set hash"):
        primary_decision_score_evidence_v1(
            decision_kind="bid",
            actor="north",
            public_prefix_hash="1" * 64,
            rules_hash="2" * 64,
            analysis_identity_hash="3" * 64,
            trace_analysis_profile_hash="4" * 64,
            cohort_seed_policy="actor_observer_cohort_seed_v1",
            cohort_index=0,
            cohort_seed=1,
            objective="expected_round_margin",
            tie_break="action_id",
            backend="python_reference_exact_search_v1",
            method="paired_hidden_deal_rollout_v1",
            hidden_deal_support=16,
            effective_sample_size=16.0,
            recommended_action=rows[0]["action"],
            rows=rows,
            legal_action_set_hash="f" * 64,
        )


def test_non_finite_metrics_fail_closed() -> None:
    item = _play_evaluations()[0]
    invalid = ActionEvaluation(
        card=item.card,
        expected_round_margin=float("nan"),
        expected_team_awarded_points=item.expected_team_awarded_points,
        expected_opponent_awarded_points=item.expected_opponent_awarded_points,
        round_lead_probability=item.round_lead_probability,
        draw_probability=item.draw_probability,
        confidence_low=item.confidence_low,
        confidence_high=item.confidence_high,
        best_share=item.best_share,
        samples=item.samples,
        effective_samples=item.effective_samples,
        exact_share=item.exact_share,
    )

    with pytest.raises(ValueError, match="finite"):
        play_score_rows_v1((invalid,))
