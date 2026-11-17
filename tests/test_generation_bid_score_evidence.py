from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from baloot_platform.agents.profiles import (
    AdaptiveBiddingStrategy,
    AgentProfile,
    actor_observer_bid_public_state_v1,
    primary_bid_decision_score_authority_v1,
    validate_primary_bid_decision_score_trace_v1,
)
from baloot_platform.analysis.bidding import (
    BidAnalysisResult,
    BidChoice,
    BidChoiceEvaluation,
    legal_bid_choices,
)
from baloot_platform.analysis.decision_scores import (
    bid_score_rows_v1,
    canonical_json_bytes_v1,
    canonical_sha256_v1,
)
from baloot_platform.analysis.search_backend import PythonReferenceSearchBackend
from baloot_platform.challenge.analysis_pipeline import (
    TraceAnalysisProfileV1,
    build_finalist_reanalysis_trace_profile_v1,
)
from baloot_platform.challenge.finalist_agent_agreement import FinalistAgentAgreementV1
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat


def _exact_agent():
    return replace(
        AgentProfile.from_mapping("finalist_candidate", {}),
        skill=1.0,
        play_search_mode="paired_information_set",
        play_search_objective="match_win_probability",
        play_search_max_remaining=32,
        play_search_budget=80_000,
        play_search_samples=192,
        play_exact_allocation_limit=128,
        play_exact_search_remaining=9,
        play_maximum_sampling_attempts=4_000,
        bid_samples=192,
        mistake_probability=0.0,
        deception_probability=0.0,
        use_production_model=False,
    )


def _auction_state(rules):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=73_001)
    engine.shuffle_and_initial_deal(state, cut_position=7)
    assert state.purchase_card is not None
    machine = BiddingMachine(
        state.dealer,
        state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(rules.data),
    )
    return state, machine


def _analysis(
    actor,
    choices,
    *,
    recommended_action: str = "sun",
    hidden_deals: int = 192,
):
    evaluations = []
    for index, choice in enumerate(choices):
        preferred = choice.action == recommended_action
        evaluations.append(BidChoiceEvaluation(
            choice=choice,
            expected_match_margin=10.0 - index if preferred else -5.0 - index,
            team_win_probability=0.7 if preferred else 0.3,
            contract_success_probability=0.7 if preferred else 0.3,
            downside_cvar10=-1.0 if preferred else -8.0,
            volatility=1.0,
            best_share=1.0 if preferred else 0.0,
            samples=hidden_deals,
        ))
    evaluations.sort(
        key=lambda item: (
            item.expected_match_margin,
            item.team_win_probability,
            item.best_share,
        ),
        reverse=True,
    )
    return BidAnalysisResult(
        actor=actor,
        recommended=evaluations[0].choice,
        choices=tuple(evaluations),
        method="fixture_shared_bid_cohort_v1",
        hidden_deals=hidden_deals,
        difficulty=50,
        debate_score=50,
        explanation=(),
    )


def _authority(rules, state, machine, *, seed=771, cohort_index=0):
    assert machine.actor is not None
    choices = tuple(legal_bid_choices(machine))
    return primary_bid_decision_score_authority_v1(
        actor=machine.actor,
        public_prefix_hash=canonical_sha256_v1(
            actor_observer_bid_public_state_v1(state, machine, machine.actor)
        ),
        legal_choices=choices,
        rules_hash=rules.sha256,
        analysis_identity_hash="3" * 64,
        trace_analysis_profile_hash="2" * 64,
        cohort_index=cohort_index,
        cohort_seed=seed,
        backend=PythonReferenceSearchBackend.descriptor.id,
    )


def test_exact_generation_bid_rows_hash_and_repeat_bytes(rules, monkeypatch):
    state, machine = _auction_state(rules)
    choices = tuple(legal_bid_choices(machine))
    result = _analysis(machine.actor, choices)
    monkeypatch.setattr("baloot_platform.agents.profiles.analyze_bid", lambda *_a, **_k: result)
    strategy = AdaptiveBiddingStrategy(_exact_agent())
    authority = _authority(rules, state, machine)

    first = strategy.choose_with_trace(
        state,
        machine,
        rules,
        seed=authority["cohortSeed"],
        primary_decision_score_authority=authority,
    )
    repeated = strategy.choose_with_trace(
        state,
        machine,
        rules,
        seed=authority["cohortSeed"],
        primary_decision_score_authority=authority,
    )

    evidence = first[2]["primaryDecisionScoreEvidence"]
    assert first[:2] == repeated[:2] == (
        result.recommended.action,
        result.recommended.trump_suit,
    )
    assert evidence["rows"] == list(bid_score_rows_v1(result.choices))
    assert first[2]["decisionScoreRowsHash"] == evidence["rowsHash"]
    assert first[2]["primaryDecisionScoreEvidenceHash"] == evidence["sha256"]
    assert canonical_json_bytes_v1(evidence) == canonical_json_bytes_v1(
        repeated[2]["primaryDecisionScoreEvidence"]
    )


def test_generation_bid_evidence_mutation_and_missing_analysis_fail_closed(
    rules,
    monkeypatch,
):
    state, machine = _auction_state(rules)
    choices = tuple(legal_bid_choices(machine))
    result = _analysis(machine.actor, choices)
    monkeypatch.setattr("baloot_platform.agents.profiles.analyze_bid", lambda *_a, **_k: result)
    strategy = AdaptiveBiddingStrategy(_exact_agent())
    authority = _authority(rules, state, machine)
    action, trump, trace = strategy.choose_with_trace(
        state,
        machine,
        rules,
        seed=authority["cohortSeed"],
        primary_decision_score_authority=authority,
    )
    selected = {
        "action": action,
        "trumpSuit": trump.name.lower() if trump is not None else None,
    }
    mutated = copy.deepcopy(trace)
    mutated["primaryDecisionScoreEvidence"]["rows"][0][
        "expectedRoundMargin"
    ] += 1.0
    with pytest.raises(ValueError, match="checksum"):
        validate_primary_bid_decision_score_trace_v1(
            mutated,
            authority,
            selected_action=selected,
            forced=False,
        )

    monkeypatch.setattr(
        "baloot_platform.agents.profiles.analyze_bid",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    with pytest.raises(ValueError, match="failed closed"):
        strategy.choose_with_trace(
            state,
            machine,
            rules,
            seed=authority["cohortSeed"],
            primary_decision_score_authority=authority,
        )


class _ForcedBidMachine:
    actor = Seat.NORTH
    round_index = 1
    in_recall = False
    records = ()
    purchase_suit = None

    @staticmethod
    def legal_actions():
        return ("pass",)


def test_forced_exact_generation_bid_is_analyzer_free_with_legality_proof(
    rules,
    monkeypatch,
):
    state, _machine = _auction_state(rules)
    machine = _ForcedBidMachine()
    choices = tuple(legal_bid_choices(machine))
    assert choices == (BidChoice("pass"),)
    authority = primary_bid_decision_score_authority_v1(
        actor=machine.actor,
        public_prefix_hash=canonical_sha256_v1(
            actor_observer_bid_public_state_v1(state, machine, machine.actor)
        ),
        legal_choices=choices,
        rules_hash=rules.sha256,
        analysis_identity_hash="3" * 64,
        trace_analysis_profile_hash="2" * 64,
        cohort_index=0,
        cohort_seed=772,
        backend=PythonReferenceSearchBackend.descriptor.id,
    )
    monkeypatch.setattr(
        "baloot_platform.agents.profiles.analyze_bid",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("forced bid invoked analyzer")
        ),
    )

    action, trump, trace = AdaptiveBiddingStrategy(_exact_agent()).choose_with_trace(
        state,
        machine,
        rules,
        seed=authority["cohortSeed"],
        primary_decision_score_authority=authority,
    )

    assert (action, trump) == ("pass", None)
    assert trace["primaryDecisionScoreEvidence"] is None
    assert trace["forcedLegalityProof"]["decisionKind"] == "bid"
    validate_primary_bid_decision_score_trace_v1(
        trace,
        authority,
        selected_action=BidChoice("pass").to_dict(),
        forced=True,
    )


def _deep_profile():
    return TraceAnalysisProfileV1.build(
        bid_samples=192,
        play_samples=192,
        exact_allocation_limit=128,
        exact_search_max_remaining=9,
        maximum_sampling_attempts=4_000,
        exact_search_max_nodes=80_000,
        exact_search_max_seconds=0.4,
        exact_search_max_depth=32,
        exact_search_max_cache_entries=100_000,
        objective="match_win_probability",
        use_conventions=False,
    )


def _stability():
    return {
        "reanalysis_samples": 256,
        "maximum_sampling_attempts": 6_000,
        "exact_allocation_limit": 256,
        "exact_search_remaining": 9,
        "exact_search_max_nodes": 120_000,
        "exact_search_max_seconds": 0.8,
        "seed_offset": 104729,
        "minimum_hidden_deals": 8,
        "minimum_best_share": 0.5,
        "maximum_competence_expected_margin_regret": 1.0,
        "maximum_competence_match_win_drop": 0.01,
        "maximum_expected_margin_drift": 5.0,
        "maximum_cohorts": 3,
        "minimum_agreement": 2,
        "third_cohort_on_disagreement": True,
        "require_same_recommended_action": True,
    }


def _reanalysis_profile(
    primary: TraceAnalysisProfileV1,
) -> TraceAnalysisProfileV1:
    return build_finalist_reanalysis_trace_profile_v1(
        _stability(),
        {
            "objective": primary.objective,
            "exact_search_max_depth": primary.exact_search_max_depth,
            "exact_search_max_cache_entries": (
                primary.exact_search_max_cache_entries
            ),
        },
        search_backend_id=PythonReferenceSearchBackend.descriptor.id,
    )


def _analysis_identity(role: str, profile: TraceAnalysisProfileV1) -> str:
    return canonical_sha256_v1({
        "schemaVersion": 1,
        "role": role,
        "traceAnalysisProfileHash": profile.sha256,
    })


def _primary_budget():
    return {
        "requested_samples": 192,
        "exact_allocation_limit": 128,
        "exact_search_max_remaining": 9,
        "maximum_sampling_attempts": 4_000,
        "exact_search_max_nodes": 80_000,
        "exact_search_max_seconds": 0.4,
        "exact_search_max_depth": 32,
        "exact_search_max_cache_entries": 100_000,
    }


def _reanalysis_budget():
    return {
        "requested_samples": 256,
        "exact_allocation_limit": 256,
        "exact_search_max_remaining": 9,
        "maximum_sampling_attempts": 6_000,
        "exact_search_max_nodes": 120_000,
        "exact_search_max_seconds": 0.8,
        "exact_search_max_depth": 32,
        "exact_search_max_cache_entries": 100_000,
    }


def _exact_agreement():
    primary = _deep_profile()
    reanalysis = _reanalysis_profile(primary)
    return FinalistAgentAgreementV1(
        _exact_agent(),
        _stability(),
        maximum_decisions_per_trace=64,
        minimum_confidence=0.75,
        convention={},
        primary_trace_analysis_profile=primary,
        primary_analysis_identity_hash=_analysis_identity(
            "deep_confirmation", primary
        ),
        reanalysis_trace_analysis_profile=reanalysis,
        reanalysis_analysis_identity_hash=_analysis_identity(
            "finalist_reanalysis", reanalysis
        ),
    )


def test_exact_bid_primary_minority_is_preserved_and_blocked(rules, monkeypatch):
    state, machine = _auction_state(rules)
    agreement = _exact_agreement()
    assert machine.actor is not None
    public_hash = canonical_sha256_v1(
        actor_observer_bid_public_state_v1(state, machine, machine.actor)
    )
    primary_seed = agreement.actor_observer_seed(
        public_prefix_hash=public_hash,
        decision_kind="bid",
        acting_seat=machine.actor,
    )

    observed_samples = []

    def analyze_for_seed(_state, current, actor, _rules, **kwargs):
        observed_samples.append(kwargs["requested_samples"])
        recommendation = "sun" if kwargs["seed"] == primary_seed else "pass"
        return _analysis(
            actor,
            tuple(legal_bid_choices(current)),
            recommended_action=recommendation,
            hidden_deals=kwargs["requested_samples"],
        )

    monkeypatch.setattr("baloot_platform.agents.profiles.analyze_bid", analyze_for_seed)
    action, trump, trace = agreement.choose_bid(
        AdaptiveBiddingStrategy(_exact_agent()),
        state,
        machine,
        rules,
        tuple(legal_bid_choices(machine)),
        seed=primary_seed,
        agreement_ordinal=1,
        public_prefix_hash=public_hash,
    )

    assert (action, trump) == ("sun", None)
    assert trace["qualificationBlocked"] is True
    assert trace["primaryActionPreserved"] is True
    evidence = trace["primaryDecisionScoreEvidence"]
    assert evidence == trace["primaryTrace"]["primaryDecisionScoreEvidence"]
    assert evidence["recommendedAction"] == BidChoice("sun").to_dict()
    cohorts = trace["cohorts"]
    assert observed_samples == [192, 256, 256]
    assert [item["primaryDecisionScoreEvidence"]["cohortIndex"] for item in cohorts] == [
        0,
        1,
        2,
    ]
    assert [item["primaryDecisionScoreEvidence"]["hiddenDealSupport"] for item in cohorts] == [
        192,
        256,
        256,
    ]

    primary_profile_hash = agreement.primary_trace_analysis_profile_hash
    primary_identity_hash = agreement.primary_analysis_identity_hash
    reanalysis_profile_hash = agreement.reanalysis_trace_analysis_profile_hash
    reanalysis_identity_hash = agreement.reanalysis_analysis_identity_hash
    assert primary_profile_hash != reanalysis_profile_hash
    assert primary_identity_hash != reanalysis_identity_hash
    assert [item["traceAnalysisProfileHash"] for item in cohorts] == [
        primary_profile_hash,
        reanalysis_profile_hash,
        reanalysis_profile_hash,
    ]
    assert [item["analysisIdentityHash"] for item in cohorts] == [
        primary_identity_hash,
        reanalysis_identity_hash,
        reanalysis_identity_hash,
    ]
    assert [item["analysisBudget"] for item in cohorts] == [
        _primary_budget(),
        _reanalysis_budget(),
        _reanalysis_budget(),
    ]
    for item in cohorts:
        cohort_evidence = item["primaryDecisionScoreEvidence"]
        assert item["traceAnalysisProfileHash"] == cohort_evidence[
            "traceAnalysisProfileHash"
        ]
        assert item["analysisIdentityHash"] == cohort_evidence[
            "analysisIdentityHash"
        ]
        assert item["primaryDecisionScoreEvidenceHash"] == cohort_evidence["sha256"]

    assert canonical_json_bytes_v1(evidence) == canonical_json_bytes_v1(
        cohorts[0]["primaryDecisionScoreEvidence"]
    )
    assert canonical_json_bytes_v1(evidence) != canonical_json_bytes_v1(
        cohorts[1]["primaryDecisionScoreEvidence"]
    )
    assert evidence["rows"] == cohorts[0]["primaryDecisionScoreEvidence"]["rows"]
    assert evidence["rows"] != cohorts[1]["primaryDecisionScoreEvidence"]["rows"]
    assert trace["analysisAuthorities"] == {
        "primary": {
            "role": "deep_confirmation",
            "traceAnalysisProfileHash": primary_profile_hash,
            "analysisIdentityHash": primary_identity_hash,
            "analysisBudget": _primary_budget(),
        },
        "reanalysis": {
            "role": "finalist_reanalysis",
            "traceAnalysisProfileHash": reanalysis_profile_hash,
            "analysisIdentityHash": reanalysis_identity_hash,
            "analysisBudget": _reanalysis_budget(),
        },
    }


@pytest.mark.parametrize(
    "bypass",
    (
        {"schemaVersion": 1, "method": "ortools_cp_sat_auction_v1"},
        {
            "schemaVersion": 1,
            "source": "research_optimizer",
            "decisionSource": "paired_actor_safe_cp_sat_bid_objective_v1",
        },
    ),
)
def test_scripted_and_strategic_bid_bypass_never_claim_primary_score_evidence(
    bypass,
):
    simulator = object.__new__(NaturalLaffaSimulatorV1)
    simulator.trace_analysis_profile_hash = "2" * 64
    simulator.analysis_identity_hash = "3" * 64
    enriched = simulator._policy_with_analysis_identity(bypass)
    assert "primaryDecisionScoreEvidence" not in enriched
    assert "primaryDecisionScoreEvidenceHash" not in enriched
