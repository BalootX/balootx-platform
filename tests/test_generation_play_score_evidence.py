from __future__ import annotations

import copy
from dataclasses import replace
import pytest

from baloot_platform.agents.profiles import (
    AdaptivePlayStrategy,
    AgentProfile,
    primary_play_decision_score_authority_v1,
    validate_primary_play_decision_score_trace_v1,
)
from baloot_platform.analysis.decision_scores import (
    canonical_json_bytes_v1,
    canonical_sha256_v1,
    play_score_rows_v1,
)
from baloot_platform.analysis.models import (
    ActionEvaluation,
    AnalysisResult,
    InformationState,
)
from baloot_platform.analysis.search_backend import PythonReferenceSearchBackend
from baloot_platform.challenge.analysis_pipeline import (
    TraceAnalysisProfileV1,
    build_finalist_reanalysis_trace_profile_v1,
)
from baloot_platform.challenge.finalist_agent_agreement import FinalistAgentAgreementV1
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat


def _exact_agent():
    return replace(
        AgentProfile.from_mapping("finalist_candidate", {}),
        skill=1.0,
        play_search_mode="paired_information_set",
        play_search_objective="match_win_probability",
        play_search_max_remaining=32,
        play_search_samples=192,
        play_exact_allocation_limit=128,
        play_exact_search_remaining=9,
        play_maximum_sampling_attempts=4_000,
        play_search_budget=80_000,
        bid_samples=192,
        mistake_probability=0.0,
        deception_probability=0.0,
        use_production_model=False,
    )


def _playing_state(rules, *, seed: int = 904):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=seed, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.SUN,
        forced_buyer=Seat.SOUTH,
    )
    engine.prepare_projects(state)
    return engine, state


def _evaluation(card: Card, rank: int, samples: int) -> ActionEvaluation:
    margin = float(12 - rank)
    probability = max(0.05, 0.8 - rank * 0.05)
    return ActionEvaluation(
        card=card,
        expected_round_margin=margin,
        expected_team_awarded_points=100.0 + margin,
        expected_opponent_awarded_points=100.0,
        round_lead_probability=probability,
        draw_probability=0.0,
        confidence_low=probability,
        confidence_high=probability,
        best_share=1.0 if rank == 1 else 0.0,
        samples=samples,
        effective_samples=float(samples),
        exact_share=1.0,
        contract_success_probability=probability,
        match_win_probability=probability,
        match_confidence_low=probability,
        match_confidence_high=probability,
        match_value_model="fixture_match_value_v1",
        deal_margins=(margin,),
        downside_cvar=margin,
        volatility=0.0,
        principal_variation=(card.code,),
        search_nodes=rank,
        search_completion_status="exact",
    )


def _analysis(
    state,
    legal: tuple[Card, ...],
    *,
    recommended_card: Card | None = None,
    samples: int | None = None,
) -> AnalysisResult:
    assert state.current_player is not None
    recommended = recommended_card or legal[0]
    ordered = (recommended,) + tuple(card for card in legal if card != recommended)
    support = int(samples if samples is not None else len(legal))
    actions = tuple(
        _evaluation(card, rank, support)
        for rank, card in enumerate(ordered, start=1)
    )
    return AnalysisResult(
        observer=state.current_player,
        recommended_card=actions[0].card,
        actions=actions,
        deductions=(),
        signals=(),
        method="fixture_exact_hidden_information_v1",
        hidden_deals=support,
        effective_sample_size=float(support),
        difficulty=50,
        debate_score=50,
        conventions_applied=False,
        objective="match_win_probability",
        analysis_profile_id="deep_fixture",
        analysis_profile_version="1",
    )


def _authority(engine, state, legal: tuple[Card, ...], *, seed: int = 991):
    assert state.current_player is not None
    return primary_play_decision_score_authority_v1(
        actor=state.current_player,
        public_prefix_hash=canonical_sha256_v1(
            InformationState.from_complete(state, state.current_player).to_dict()
        ),
        legal_cards=legal,
        rules_hash=engine.rules.sha256,
        analysis_identity_hash="3" * 64,
        trace_analysis_profile_hash="2" * 64,
        cohort_index=0,
        cohort_seed=seed,
        objective="match_win_probability",
        backend=PythonReferenceSearchBackend.descriptor.id,
    )


def _deep_profile() -> TraceAnalysisProfileV1:
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


def _exact_agreement() -> FinalistAgentAgreementV1:
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


def test_exact_generation_play_rows_hash_and_repeat_bytes(rules, monkeypatch):
    engine, state = _playing_state(rules)
    legal = tuple(engine.legal_moves(state))
    assert len(legal) > 1
    result = _analysis(state, legal)
    monkeypatch.setattr("baloot_platform.agents.profiles.analyze", lambda *_a, **_k: result)
    strategy = AdaptivePlayStrategy(_exact_agent(), {})
    authority = _authority(engine, state, legal)

    first_card, first = strategy.choose_with_trace(
        engine,
        state,
        legal,
        seed=authority["cohortSeed"],
        primary_decision_score_authority=authority,
    )
    repeated_card, repeated = strategy.choose_with_trace(
        engine,
        state,
        legal,
        seed=authority["cohortSeed"],
        primary_decision_score_authority=authority,
    )

    evidence = first["primaryDecisionScoreEvidence"]
    assert first_card == repeated_card == result.recommended_card
    assert evidence["rows"] == list(play_score_rows_v1(result.actions))
    assert first["decisionScoreRowsHash"] == evidence["rowsHash"]
    assert first["primaryDecisionScoreEvidenceHash"] == evidence["sha256"]
    assert canonical_json_bytes_v1(evidence) == canonical_json_bytes_v1(
        repeated["primaryDecisionScoreEvidence"]
    )
    assert first["primaryDecisionScoreEvidenceHash"] == repeated[
        "primaryDecisionScoreEvidenceHash"
    ]


def test_generation_play_evidence_mutation_and_missing_analysis_fail_closed(
    rules,
    monkeypatch,
):
    engine, state = _playing_state(rules, seed=905)
    legal = tuple(engine.legal_moves(state))
    result = _analysis(state, legal)
    monkeypatch.setattr("baloot_platform.agents.profiles.analyze", lambda *_a, **_k: result)
    strategy = AdaptivePlayStrategy(_exact_agent(), {})
    authority = _authority(engine, state, legal)
    selected, trace = strategy.choose_with_trace(
        engine,
        state,
        legal,
        seed=authority["cohortSeed"],
        primary_decision_score_authority=authority,
    )

    mutated = copy.deepcopy(trace)
    mutated["primaryDecisionScoreEvidence"]["rows"][0][
        "expectedRoundMargin"
    ] += 1.0
    with pytest.raises(ValueError, match="checksum"):
        validate_primary_play_decision_score_trace_v1(
            mutated,
            authority,
            selected_action=selected.to_dict(),
            forced=False,
        )

    drifted_authority = {**authority, "publicPrefixHash": "4" * 64}
    with pytest.raises(ValueError, match="public prefix drifted"):
        strategy.choose_with_trace(
            engine,
            state,
            legal,
            seed=authority["cohortSeed"],
            primary_decision_score_authority=drifted_authority,
        )

    def fail_analysis(*_args, **_kwargs):
        raise RuntimeError("fixture analyzer unavailable")

    monkeypatch.setattr("baloot_platform.agents.profiles.analyze", fail_analysis)
    with pytest.raises(ValueError, match="failed closed"):
        strategy.choose_with_trace(
            engine,
            state,
            legal,
            seed=authority["cohortSeed"],
            primary_decision_score_authority=authority,
        )


def test_forced_exact_generation_play_is_analyzer_free_with_legality_proof(
    rules,
    monkeypatch,
):
    engine, state = _playing_state(rules, seed=906)
    while sum(hand.bit_count() for hand in state.hands) > 4:
        engine.play_card(state, engine.legal_moves(state)[0])
    legal = tuple(engine.legal_moves(state))
    assert len(legal) == 1
    authority = _authority(engine, state, legal, seed=992)

    def forbidden_analysis(*_args, **_kwargs):
        raise AssertionError("forced play invoked analyzer")

    monkeypatch.setattr("baloot_platform.agents.profiles.analyze", forbidden_analysis)
    selected, trace = AdaptivePlayStrategy(_exact_agent(), {}).choose_with_trace(
        engine,
        state,
        legal,
        seed=authority["cohortSeed"],
        primary_decision_score_authority=authority,
    )

    assert trace["primaryDecisionScoreEvidence"] is None
    assert trace["forcedLegalityProof"]["selectedAction"] == selected.to_dict()
    assert trace["forcedLegalityProof"]["legalActionCount"] == 1
    validate_primary_play_decision_score_trace_v1(
        trace,
        authority,
        selected_action=selected.to_dict(),
        forced=True,
    )


def test_exact_play_primary_score_bytes_and_dual_cohort_authority(
    rules,
    monkeypatch,
):
    engine, state = _playing_state(rules, seed=907)
    legal = tuple(engine.legal_moves(state))
    assert len(legal) > 1
    agreement = _exact_agreement()
    assert state.current_player is not None
    public_hash = canonical_sha256_v1(
        InformationState.from_complete(state, state.current_player).to_dict()
    )
    primary_seed = agreement.actor_observer_seed(
        public_prefix_hash=public_hash,
        decision_kind="play",
        acting_seat=state.current_player,
    )
    observed_budgets = []

    def analyze_for_seed(_information, _rules, _convention, **kwargs):
        observed_budgets.append({
            key: kwargs[key]
            for key in _primary_budget()
        })
        recommended = legal[0] if kwargs["seed"] == primary_seed else legal[1]
        return _analysis(
            state,
            legal,
            recommended_card=recommended,
            samples=kwargs["requested_samples"],
        )

    monkeypatch.setattr("baloot_platform.agents.profiles.analyze", analyze_for_seed)
    selected, trace = agreement.choose_play(
        AdaptivePlayStrategy(_exact_agent(), {}),
        engine,
        state,
        legal,
        seed=primary_seed,
        agreement_ordinal=1,
        public_prefix_hash=public_hash,
    )

    assert selected == legal[0]
    assert trace["primaryActionPreserved"] is True
    cohorts = trace["cohorts"]
    assert [item["cohortIndex"] for item in cohorts] == [0, 1, 2]
    assert observed_budgets == [
        _primary_budget(),
        _reanalysis_budget(),
        _reanalysis_budget(),
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

    primary_evidence = trace["primaryDecisionScoreEvidence"]
    assert canonical_json_bytes_v1(primary_evidence) == canonical_json_bytes_v1(
        trace["primaryTrace"]["primaryDecisionScoreEvidence"]
    )
    assert canonical_json_bytes_v1(primary_evidence) == canonical_json_bytes_v1(
        cohorts[0]["primaryDecisionScoreEvidence"]
    )
    assert canonical_json_bytes_v1(primary_evidence) != canonical_json_bytes_v1(
        cohorts[1]["primaryDecisionScoreEvidence"]
    )
    assert primary_evidence["rows"] == cohorts[0][
        "primaryDecisionScoreEvidence"
    ]["rows"]
    assert primary_evidence["rows"] != cohorts[1][
        "primaryDecisionScoreEvidence"
    ]["rows"]
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
        {
            "schemaVersion": 1,
            "method": "ortools_cp_sat_complete_play_v1",
            "selectionReason": "compiled_legal_play",
        },
        {
            "schemaVersion": 1,
            "source": "research_optimizer",
            "decisionSource": "policy_conditioned_actor_safe_late_game_v1",
        },
    ),
)
def test_scripted_and_strategic_bypass_never_claim_primary_score_evidence(bypass):
    simulator = object.__new__(NaturalLaffaSimulatorV1)
    simulator.trace_analysis_profile_hash = "2" * 64
    simulator.analysis_identity_hash = "3" * 64

    enriched = simulator._policy_with_analysis_identity(bypass)

    assert "primaryDecisionScoreEvidence" not in enriched
    assert "primaryDecisionScoreEvidenceHash" not in enriched
    assert enriched["traceAnalysisProfileHash"] == "2" * 64
    assert enriched["analysisIdentityHash"] == "3" * 64
