from __future__ import annotations

import copy
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import pytest
import yaml

import baloot_platform.analysis.belief_v7 as belief_module
import baloot_platform.analysis.evaluator as evaluator_module
import baloot_platform.agents.profiles as profiles_module
from baloot_platform.agents.profiles import (
    AdaptivePlayStrategy,
    AgentProfileRegistry,
)
from baloot_platform.analysis.belief_v7 import BeliefPolicy
from baloot_platform.analysis.evaluator import AnalysisPolicy
from baloot_platform.analysis.models import (
    ActionEvaluation,
    AnalysisResult,
    InformationState,
)
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.challenge.finalist_agent_agreement import (
    FinalistAgentAgreementV1,
)
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.core.cards import Card, iter_mask
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit


def _explicit_profiles(
    root,
    *,
    suffix: str,
    objective: str,
    requested: int,
) -> tuple[AnalysisPolicy, BeliefPolicy, dict[str, Any]]:
    belief_profile = yaml.safe_load(
        (root / "configs/belief/particle_v1.yaml").read_text(encoding="utf-8")
    )
    belief_profile["profile"] = {
        **dict(belief_profile["profile"]),
        "id": f"belief_{suffix}",
        "version": "1.0.0",
    }
    belief_profile["particles"]["requested"] = requested
    belief_policy = BeliefPolicy.from_profile(belief_profile)

    analysis_profile = yaml.safe_load(
        (root / "configs/analysis/deep.yaml").read_text(encoding="utf-8")
    )
    analysis_profile["profile"] = {
        **dict(analysis_profile["profile"]),
        "id": f"analysis_{suffix}",
        "version": "1.0.0",
    }
    analysis_profile["requested_samples"] = requested
    analysis_profile["exact_allocation_limit"] = 100
    analysis_profile["exact_search_max_remaining"] = 32
    analysis_profile["maximum_sampling_attempts"] = 100
    analysis_profile["objective"] = objective
    analysis_policy = AnalysisPolicy.from_profile(
        analysis_profile,
        belief_policy=belief_policy,
    )

    match_value_profile = yaml.safe_load(
        (
            root / "configs/match_value/exact_dynamic_v2.yaml"
        ).read_text(encoding="utf-8")
    )
    match_value_profile["profile"] = {
        **dict(match_value_profile["profile"]),
        "id": f"match_value_{suffix}",
        "version": "2.0.0",
    }
    match_value_profile["future_rounds"][
        "future_team_round_win_probability"
    ] = 0.4 if suffix == "alpha" else 0.6
    return analysis_policy, belief_policy, match_value_profile


def _search_profile(root):
    base = AgentProfileRegistry.from_yaml(
        root / "configs/agents/default.yaml"
    ).get("strong")
    return replace(
        base,
        play_search_mode="paired_information_set",
        play_search_max_remaining=32,
        play_search_budget=1_000,
        mistake_probability=0.0,
        deception_probability=0.0,
        use_production_model=False,
    )


def _policy_snapshot_payloads(explicit) -> dict[str, dict[str, Any]]:
    analysis, belief, match_value = explicit
    return {
        "analysis": {
            "profile": {
                "id": analysis.profile_id,
                "version": analysis.profile_version,
            },
            "requested_samples": analysis.requested_samples,
            "exact_allocation_limit": analysis.exact_allocation_limit,
            "exact_search_max_remaining": analysis.exact_search_max_remaining,
            "maximum_sampling_attempts": analysis.maximum_sampling_attempts,
            "objective": analysis.objective,
        },
        "belief": {
            "profile": {
                "id": belief.profile_id,
                "version": belief.profile_version,
            },
            "particles": {
                "requested": belief.requested_particles,
                "maximum": belief.maximum_particles,
                "minimum_effective_fraction": (
                    belief.minimum_effective_fraction
                ),
                "resampling": "systematic",
            },
            "construction": {
                "exact_when_allocations_below": (
                    belief.exact_allocation_limit
                )
            },
        },
        "match_value": copy.deepcopy(match_value),
    }


def _simulation_snapshot(rules, explicit) -> tuple[str, dict[str, Any]]:
    bundle_hash = "a" * 64
    policy_payloads = _policy_snapshot_payloads(explicit)
    snapshot = {
        "schemaVersion": 1,
        "bundleHash": bundle_hash,
        "components": {
            "rules": {
                "sha256": rules.sha256,
                "payload": copy.deepcopy(rules.data),
            },
            **{
                family: {
                    "sha256": canonical_sha256(payload),
                    "payload": copy.deepcopy(payload),
                }
                for family, payload in policy_payloads.items()
            },
        },
    }
    snapshot["sha256"] = canonical_sha256(snapshot)
    return bundle_hash, snapshot


def _state(rules):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=8401, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=13)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.SOUTH,
        forced_trump=Suit.CLUBS,
    )
    engine.prepare_projects(state)
    assert state.current_player is not None
    assert len(engine.legal_moves(state)) > 1
    return engine, state


def _unexpected_active_lookup(*_args, **_kwargs):
    raise AssertionError("packaged active policy must not be read")


def _fake_explicit_analysis(captured: list[dict[str, Any]]):
    def fake(
        info: InformationState,
        _rules,
        _convention=None,
        **kwargs: Any,
    ) -> AnalysisResult:
        analysis_policy = kwargs.get("analysis_policy")
        belief_policy = kwargs.get("belief_policy")
        match_value_profile = kwargs.get("match_value_profile")
        assert isinstance(info, InformationState)
        assert isinstance(analysis_policy, AnalysisPolicy)
        assert isinstance(belief_policy, BeliefPolicy)
        assert isinstance(match_value_profile, dict)
        captured.append(
            {
                "information": info,
                "analysisPolicy": analysis_policy,
                "beliefPolicy": belief_policy,
                "matchValueProfile": copy.deepcopy(match_value_profile),
            }
        )
        cards = sorted(
            iter_mask(info.observer_hand),
            key=lambda card: card.id,
            reverse=analysis_policy.profile_id.endswith("omega"),
        )
        actions = tuple(
            ActionEvaluation(
                card=card,
                expected_round_margin=float(len(cards) - index),
                expected_team_awarded_points=20.0,
                expected_opponent_awarded_points=10.0,
                round_lead_probability=0.6,
                draw_probability=0.0,
                confidence_low=0.4,
                confidence_high=0.8,
                best_share=1.0 if index == 0 else 0.0,
                samples=belief_policy.requested_particles,
                effective_samples=float(belief_policy.requested_particles),
                exact_share=0.0,
                match_win_probability=0.6,
                downside_cvar=5.0,
                volatility=1.0,
                search_completion_status="heuristic_complete",
            )
            for index, card in enumerate(cards)
        )
        return AnalysisResult(
            observer=info.observer,
            recommended_card=actions[0].card,
            actions=actions,
            deductions=(),
            signals=(),
            method="explicit_policy_probe_v1",
            hidden_deals=belief_policy.requested_particles,
            effective_sample_size=float(belief_policy.requested_particles),
            difficulty=50,
            debate_score=50,
            conventions_applied=False,
            objective=analysis_policy.objective,
            analysis_profile_id=analysis_policy.profile_id,
            analysis_profile_version=analysis_policy.profile_version,
        )

    return fake


def _block_active_policy_reads(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluator_module,
        "_active_analysis_policy",
        _unexpected_active_lookup,
    )
    monkeypatch.setattr(
        evaluator_module,
        "_active_analysis_profile",
        _unexpected_active_lookup,
    )
    monkeypatch.setattr(
        belief_module,
        "_active_belief_policy",
        _unexpected_active_lookup,
    )


def test_adaptive_play_requires_complete_explicit_policy_triplet(
    root,
):
    analysis, belief, _match_value = _explicit_profiles(
        root,
        suffix="alpha",
        objective="round_margin",
        requested=2,
    )
    with pytest.raises(ValueError, match="requires analysis, belief, and"):
        AdaptivePlayStrategy(
            _search_profile(root),
            analysis_policy=analysis,
            belief_policy=belief,
        )


def test_finalist_paired_cohorts_retain_the_same_explicit_authority(
    root,
    convention,
):
    analysis, belief, match_value = _explicit_profiles(
        root,
        suffix="alpha",
        objective="round_margin",
        requested=2,
    )
    challenge = yaml.safe_load(
        (
            root / "configs/challenge/progressive_mining_v2.yaml"
        ).read_text(encoding="utf-8")
    )
    agreement = FinalistAgentAgreementV1(
        _search_profile(root),
        challenge["stability"],
        maximum_decisions_per_trace=4,
        minimum_confidence=0.75,
        convention=convention,
        analysis_policy=analysis,
        belief_policy=belief,
        match_value_profile=match_value,
    )

    assert agreement.independent_play.analysis_policy is analysis
    assert agreement.independent_play.belief_policy is belief
    assert agreement.third_play.analysis_policy is analysis
    assert agreement.third_play.belief_policy is belief
    assert (
        agreement.independent_play.policy_authority
        == agreement.third_play.policy_authority
    )


def test_simulator_rejects_policy_from_a_different_frozen_bundle(
    root,
    rules,
    convention,
):
    alpha = _explicit_profiles(
        root,
        suffix="alpha",
        objective="round_margin",
        requested=2,
    )
    omega = _explicit_profiles(
        root,
        suffix="omega",
        objective="low_volatility",
        requested=4,
    )
    bundle_hash, snapshot = _simulation_snapshot(rules, alpha)

    with pytest.raises(ValueError, match="does not match the frozen bundle"):
        NaturalLaffaSimulatorV1(
            rules,
            convention,
            bundle_hash=bundle_hash,
            configuration_snapshot=snapshot,
            agent_profile=_search_profile(root),
            analysis_policy=omega[0],
            belief_policy=omega[1],
            match_value_profile=omega[2],
        )


def test_root_simulator_factory_compiles_one_explicit_bundle_authority(root):
    simulator = NaturalLaffaSimulatorV1.from_root(
        root,
        maximum_deal_attempts=1,
    )

    assert simulator.analysis_policy is not None
    assert simulator.belief_policy is not None
    assert simulator.match_value_profile is not None
    assert simulator.analysis_policy_authority["mode"] == "explicit_governed"
    assert (
        simulator.analysis_policy_authority["analysis"]["profileId"]
        == simulator.analysis_policy.profile_id
    )


def test_explicit_profiles_change_play_and_provenance_without_active_reads(
    root,
    rules,
    convention,
    monkeypatch,
):
    alpha = _explicit_profiles(
        root,
        suffix="alpha",
        objective="round_margin",
        requested=2,
    )
    omega = _explicit_profiles(
        root,
        suffix="omega",
        objective="low_volatility",
        requested=4,
    )
    captured: list[dict[str, Any]] = []
    _block_active_policy_reads(monkeypatch)
    monkeypatch.setattr(
        profiles_module,
        "analyze",
        _fake_explicit_analysis(captured),
    )
    engine, state = _state(rules)
    profile = _search_profile(root)
    first_strategy = AdaptivePlayStrategy(
        profile,
        convention,
        analysis_policy=alpha[0],
        belief_policy=alpha[1],
        match_value_profile=alpha[2],
    )
    second_strategy = AdaptivePlayStrategy(
        profile,
        convention,
        analysis_policy=omega[0],
        belief_policy=omega[1],
        match_value_profile=omega[2],
    )
    retained_match_hash = first_strategy.policy_authority["matchValue"][
        "sha256"
    ]
    alpha[2]["target"] = 999
    exposed_copy = first_strategy.match_value_profile
    assert exposed_copy is not None
    exposed_copy["target"] = 998
    assert (
        first_strategy.policy_authority["matchValue"]["sha256"]
        == retained_match_hash
    )
    assert first_strategy.match_value_profile["target"] == 152

    first_card, first_trace = first_strategy.choose_with_trace(
        engine,
        state,
        seed=99,
    )
    repeated_card, repeated_trace = first_strategy.choose_with_trace(
        engine,
        state,
        seed=99,
    )
    second_card, second_trace = second_strategy.choose_with_trace(
        engine,
        state,
        seed=99,
    )

    assert first_card == repeated_card
    assert first_trace == repeated_trace
    assert first_card != second_card
    assert first_trace["analysisHiddenDeals"] == 2
    assert second_trace["analysisHiddenDeals"] == 4
    assert first_trace["analysisPolicyAuthority"]["mode"] == "explicit_governed"
    assert (
        first_trace["analysisPolicyAuthority"]["analysis"]["profileId"]
        == "analysis_alpha"
    )
    assert (
        second_trace["analysisPolicyAuthority"]["analysis"]["profileId"]
        == "analysis_omega"
    )
    assert (
        first_trace["analysisPolicyAuthority"]["matchValue"]["sha256"]
        != second_trace["analysisPolicyAuthority"]["matchValue"]["sha256"]
    )
    assert set(first_trace["analysisPolicyAuthority"]) == {
        "mode",
        "analysis",
        "belief",
        "matchValue",
    }
    assert set(first_trace["analysisPolicyAuthority"]["analysis"]) == {
        "profileId",
        "profileVersion",
        "sha256",
    }
    assert all(item["observerSafe"] for item in (first_trace, second_trace))
    legal = set(engine.legal_moves(state))
    assert {
        Card.parse(item["action"])
        for item in first_trace["rankedActions"]
    } <= legal
    first_information = captured[0]["information"]
    assert isinstance(first_information, InformationState)
    assert first_information.observer == state.current_player
    assert first_information.observer_hand == state.hands[int(state.current_player)]
    assert all(
        first_information.known_unplayed[int(seat)].bit_count() <= 1
        for seat in Seat
        if seat != first_information.observer
    )


def test_natural_simulator_reuses_explicit_policy_and_records_game_provenance(
    root,
    rules,
    convention,
    monkeypatch,
):
    profile = _search_profile(root)
    alpha = _explicit_profiles(
        root,
        suffix="alpha",
        objective="round_margin",
        requested=2,
    )
    omega = _explicit_profiles(
        root,
        suffix="omega",
        objective="low_volatility",
        requested=4,
    )
    captured: list[dict[str, Any]] = []
    _block_active_policy_reads(monkeypatch)
    monkeypatch.setattr(
        profiles_module,
        "analyze",
        _fake_explicit_analysis(captured),
    )

    def simulator(explicit) -> NaturalLaffaSimulatorV1:
        bundle_hash, snapshot = _simulation_snapshot(rules, explicit)
        return NaturalLaffaSimulatorV1(
            rules,
            convention,
            bundle_hash=bundle_hash,
            configuration_snapshot=snapshot,
            agent_profile=profile,
            maximum_deal_attempts=4,
            analysis_policy=explicit[0],
            belief_policy=explicit[1],
            match_value_profile=explicit[2],
        )

    first_simulator = simulator(alpha)
    second_simulator = simulator(omega)
    first = first_simulator.simulate(1, dealer=Seat.EAST)
    repeated = first_simulator.simulate(1, dealer=Seat.EAST)
    second = second_simulator.simulate(1, dealer=Seat.EAST)

    assert first.sha256 == repeated.sha256
    assert first.to_dict(trusted=True) == repeated.to_dict(trusted=True)
    assert first.final_trace.game_hash != second.final_trace.game_hash
    first_generation = first.final_trace.trusted_final_state["generation"]
    second_generation = second.final_trace.trusted_final_state["generation"]
    assert (
        first_generation["analysisPolicyAuthority"]
        == first_simulator.analysis_policy_authority
    )
    assert (
        first_generation["analysisPolicyAuthority"]["analysis"]["profileId"]
        == "analysis_alpha"
    )
    assert (
        second_generation["analysisPolicyAuthority"]["analysis"]["profileId"]
        == "analysis_omega"
    )
    first_play_traces = [
        item["trace"]
        for item in first.final_trace.trusted_final_state[
            "decisionPolicyTraces"
        ]
        if item["trace"].get("strategyId") == "adaptive_observer_safe_v2"
    ]
    assert first_play_traces
    assert all(
        item["analysisPolicyAuthority"]
        == first_simulator.analysis_policy_authority
        for item in first_play_traces
    )
    first_plays = tuple(
        item.chosen_action
        for item in first.final_trace.decision_ledger.records
        if item.decision_kind == "play"
    )
    second_plays = tuple(
        item.chosen_action
        for item in second.final_trace.decision_ledger.records
        if item.decision_kind == "play"
    )
    assert first_plays != second_plays
    assert captured
    assert all(
        item["information"].observer == item["information"].current_player
        for item in captured
    )


def test_frozen_research_context_profile_round_trips_to_finalist_simulator(
    root,
    tmp_path,
    monkeypatch,
):
    from baloot_platform.service.container import build_container

    monkeypatch.setenv("BALOOT_RUNTIME", str(tmp_path / "runtime"))
    monkeypatch.setenv("BALOOT_OUTPUTS", str(tmp_path / "outputs"))
    container = build_container(
        root=root,
        writable_root=tmp_path / "data",
        database_url="",
    )
    try:
        factory = container.research.execution_context_factory
        assert factory is not None
        revision = str(container.configuration.resolve()["bundle"]["revisionId"])
        _effective, context = factory.resolve_bundle_revision(revision)
        raw_profile = copy.deepcopy(
            dict(context.bundle.component("match_value"))
        )
        expected_hash = canonical_sha256(raw_profile)
        original_target = raw_profile["target"]
        original_maximum_rounds = raw_profile["future_rounds"][
            "maximum_rounds"
        ]

        assert isinstance(context.match_value_profile, MappingProxyType)
        assert isinstance(
            context.match_value_profile["profile"], MappingProxyType
        )
        assert isinstance(
            context.match_value_profile["future_rounds"][
                "increment_distribution"
            ],
            tuple,
        )
        with pytest.raises(TypeError):
            context.match_value_profile["target"] = 999
        with pytest.raises(TypeError):
            context.match_value_profile["profile"]["id"] = "tampered"
        with pytest.raises(TypeError):
            context.match_value_profile["future_rounds"][
                "increment_distribution"
            ][0]["weight"] = 1.0

        _block_active_policy_reads(monkeypatch)
        challenge = dict(context.bundle.component("challenge"))
        agreement = FinalistAgentAgreementV1(
            context.simulator.agent_profile,
            challenge["stability"],
            maximum_decisions_per_trace=4,
            minimum_confidence=float(
                context.decision_score_profile.minimum_surrounding_confidence
            ),
            convention=context.bundle.convention,
            exact_search_backend=context.exact_search_backend,
            analysis_policy=context.analysis_policy,
            belief_policy=context.belief_policy,
            match_value_profile=context.match_value_profile,
        )

        def build_bound_simulator() -> NaturalLaffaSimulatorV1:
            return NaturalLaffaSimulatorV1(
                context.bundle.rules,
                context.bundle.convention,
                bundle_hash=context.bundle.sha256,
                configuration_snapshot=(
                    context.simulator.configuration_snapshot
                ),
                structural_domain=context.bundle.rules.structural_domain,
                play_domain=context.bundle.rules.play_domain,
                agent_profile=context.simulator.agent_profile,
                agent_release=context.simulator.agent_release,
                generation_tier=context.simulator.generation_tier,
                maximum_deal_attempts=(
                    context.simulator.maximum_deal_attempts
                ),
                all_pass_dealer_policy=(
                    context.simulator.all_pass_dealer_policy
                ),
                exact_search_backend=context.exact_search_backend,
                decision_agreement=agreement,
                analysis_policy=context.analysis_policy,
                belief_policy=context.belief_policy,
                match_value_profile=context.match_value_profile,
            )

        first = build_bound_simulator()
        repeated = build_bound_simulator()
        assert first.match_value_profile == raw_profile
        assert repeated.match_value_profile == raw_profile
        assert first.analysis_policy_authority == (
            repeated.analysis_policy_authority
        )
        assert (
            first.analysis_policy_authority["matchValue"]["sha256"]
            == expected_hash
            == context.bundle.components["match_value"].sha256
        )
        assert (
            agreement.independent_play.policy_authority
            == first.analysis_policy_authority
        )

        deep_profile = context.analyzers["deep"].profile
        deep_identity_hash = context.analysis_identity_hashes["deep"]
        frozen_context_agreement = context.simulator.decision_agreement
        assert frozen_context_agreement is not None
        reanalysis_profile = (
            frozen_context_agreement.reanalysis_trace_analysis_profile
        )
        assert reanalysis_profile is not None
        reanalysis_identity_hash = context.analysis_identity_hashes[
            "finalist_reanalysis"
        ]
        finalist_profile = AgentProfileRegistry.from_mapping(
            context.bundle.component("agents")
        ).get("finalist_candidate")
        exact_agreement = FinalistAgentAgreementV1(
            finalist_profile,
            challenge["stability"],
            maximum_decisions_per_trace=4,
            minimum_confidence=float(
                context.decision_score_profile.minimum_surrounding_confidence
            ),
            convention=context.bundle.convention,
            exact_search_backend=context.exact_search_backend,
            analysis_policy=context.analysis_policy,
            belief_policy=context.belief_policy,
            match_value_profile=context.match_value_profile,
            primary_trace_analysis_profile=deep_profile,
            primary_analysis_identity_hash=deep_identity_hash,
            reanalysis_trace_analysis_profile=reanalysis_profile,
            reanalysis_analysis_identity_hash=reanalysis_identity_hash,
        )
        exact_simulator = NaturalLaffaSimulatorV1(
            context.bundle.rules,
            context.bundle.convention,
            bundle_hash=context.bundle.sha256,
            configuration_snapshot=context.simulator.configuration_snapshot,
            structural_domain=context.bundle.rules.structural_domain,
            play_domain=context.bundle.rules.play_domain,
            agent_profile=finalist_profile,
            maximum_deal_attempts=context.simulator.maximum_deal_attempts,
            all_pass_dealer_policy=context.simulator.all_pass_dealer_policy,
            exact_search_backend=context.exact_search_backend,
            decision_agreement=exact_agreement,
            analysis_policy=context.analysis_policy,
            belief_policy=context.belief_policy,
            match_value_profile=context.match_value_profile,
            trace_analysis_profile=deep_profile,
            analysis_identity_hash=deep_identity_hash,
        )
        assert exact_simulator.trace_analysis_profile is deep_profile
        assert exact_agreement.trace_analysis_profile is deep_profile
        assert exact_simulator.trace_analysis_profile_hash == deep_profile.sha256
        assert exact_simulator.analysis_identity_hash == deep_identity_hash
        assert exact_simulator.primary_analysis_budget == dict(
            exact_agreement.primary_analysis_budget
        )
        assert exact_simulator.play_strategy.convention == {}
        assert exact_agreement.independent_play.convention == {}

        raw_profile["target"] = 999
        exposed = first.match_value_profile
        assert exposed is not None
        exposed["future_rounds"]["maximum_rounds"] = 1
        assert context.match_value_profile["target"] == original_target
        assert first.match_value_profile["target"] == original_target
        assert (
            first.match_value_profile["future_rounds"]["maximum_rounds"]
            == original_maximum_rounds
        )
        assert (
            first.analysis_policy_authority["matchValue"]["sha256"]
            == expected_hash
        )
    finally:
        container.shutdown()
