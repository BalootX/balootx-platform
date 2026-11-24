from __future__ import annotations

import pytest
from ortools.sat.python import cp_model  # noqa: F401
import yaml
from pathlib import Path

from baloot_platform.challenge.replay import CompleteTraceReplayerV1
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.agents.profiles import AgentProfile
from baloot_platform.analysis.hidden_deals import generate_hidden_deals
from baloot_platform.analysis.models import InformationState
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract
from baloot_platform.research.best_response_proof import (
    EngineReducedBestResponseProblemBuilderV1,
    OrToolsBoundedBestResponseProofV1,
    ReducedBestResponseProfileV1,
    exhaustive_backward_induction,
)
from baloot_platform.research.decision_proof_adapter import (
    DecisionProofResourcesV1,
    TraceDecisionPolicyProofAdapterV1,
)
from baloot_platform.research.policy_proof_benchmark import (
    authoritative_late_game_fixture,
    benchmark_reduced_policy_proof,
)


def _simulator(rules, convention) -> NaturalLaffaSimulatorV1:
    bundle_hash = canonical_sha256({"fixture": "decision-proof-adapter"})
    snapshot = {
        "schemaVersion": 1,
        "bundleHash": bundle_hash,
        "components": {
            "rules": {"sha256": rules.sha256, "payload": rules.data},
        },
    }
    snapshot["sha256"] = canonical_sha256(snapshot)
    profile_document = yaml.safe_load(
        Path("configs/agents/default.yaml").read_text(encoding="utf-8")
    )
    profile = AgentProfile.from_mapping(
        "foundry_throughput",
        profile_document["profiles"]["foundry_throughput"],
    )
    return NaturalLaffaSimulatorV1(
        rules,
        convention,
        bundle_hash=bundle_hash,
        configuration_snapshot=snapshot,
        agent_profile=profile,
    )


@pytest.mark.parametrize("contract", (Contract.HOKUM, Contract.SUN))
def test_real_late_game_fixture_matches_reference_backward_induction(rules, contract):
    engine, state = authoritative_late_game_fixture(rules, contract)
    assert state.current_player is not None
    problem = EngineReducedBestResponseProblemBuilderV1().build(
        states=(state,),
        engine=engine,
        actor=state.current_player,
        coverage_method="sampled_actual_deal_fixture",
        maximum_plies=8,
    )
    proof = OrToolsBoundedBestResponseProofV1().prove(problem)

    assert state.contract == contract
    assert len(state.played_cards) == 24
    assert len(engine.legal_actions(state)) == 2
    assert proof.policy_proof.strategic_truth == "provisional"
    assert proof.small_card_big_gain is None
    assert all(item.parity for item in proof.continuations)
    for world in problem.worlds:
        for root in world.root_actions:
            expected, _line = exhaustive_backward_induction(root.continuation)
            actual = next(
                item
                for item in proof.continuations
                if item.world_fingerprint == world.world_fingerprint
                and item.root_action == root.action
            )
            assert actual.cp_sat_margin == expected


def test_real_state_builder_cancellation_is_cooperative(rules):
    engine, state = authoritative_late_game_fixture(rules, Contract.HOKUM)
    with pytest.raises(InterruptedError, match="cancelled"):
        EngineReducedBestResponseProblemBuilderV1().build(
            states=(state,),
            engine=engine,
            actor=state.current_player,
            coverage_method="sampled_actual_deal_fixture",
            cancel_check=lambda: True,
        )
    with pytest.raises(ValueError, match="node budget"):
        EngineReducedBestResponseProblemBuilderV1().build(
            states=(state,),
            engine=engine,
            actor=state.current_player,
            coverage_method="sampled_actual_deal_fixture",
            profile=ReducedBestResponseProfileV1(maximum_nodes=2),
        )


def test_trace_adapter_returns_deterministic_audit_v2_solver_evidence(rules, convention):
    simulator = _simulator(rules, convention)
    trace = simulator.simulate(1).final_trace
    replay = CompleteTraceReplayerV1(simulator.rules).replay(trace)
    decision = next(
        item
        for item in replay.decisions
        if item.decision.decision_kind == "play"
        and len(item.state.played_cards) >= 24
        and not item.decision.forced
    )
    adapter = TraceDecisionPolicyProofAdapterV1(simulator.rules)
    information = InformationState.from_complete(
        decision.state,
        decision.decision.acting_seat,
    )
    cohort = generate_hidden_deals(
        information,
        BalootEngine(simulator.rules),
        requested_samples=1,
        exact_allocation_limit=1,
        maximum_attempts=100,
        seed=91,
    )
    actor_safe_worlds = tuple(state.clone() for state in cohort.states)
    assert all(
        InformationState.from_complete(state, decision.decision.acting_seat).to_dict()
        == information.to_dict()
        for state in actor_safe_worlds
    )
    arguments = {
        "trace": trace,
        "decision_id": decision.decision.decision_id,
        "hidden_worlds": actor_safe_worlds,
        "coverage_method": cohort.method + "+hard_facts_only",
        "resources": DecisionProofResourcesV1(
            maximum_nodes=20_000,
            maximum_plies=8,
            maximum_wall_seconds=20.0,
        ),
    }
    first = adapter.adapt(**arguments)
    second = adapter.adapt(**arguments)

    assert first.sha256 == second.sha256
    assert first.truth_label == "provisional"
    assert first.information_set_evidence.observer_safe
    assert first.information_set_evidence.action_coverage_complete
    assert first.sequential_response_evidence.all_alternatives_evaluated
    assert first.to_dict()["hiddenStateIdentifiersExposed"] is False
    assert "worldFingerprint" not in str(first.to_dict())


def test_trace_adapter_resource_and_cancellation_bounds(rules, convention):
    simulator = _simulator(rules, convention)
    trace = simulator.simulate(1).final_trace
    replay = CompleteTraceReplayerV1(simulator.rules).replay(trace)
    decision = next(
        item
        for item in replay.decisions
        if item.decision.decision_kind == "play"
        and len(item.state.played_cards) >= 24
        and not item.decision.forced
    )
    adapter = TraceDecisionPolicyProofAdapterV1(simulator.rules)
    with pytest.raises(ValueError, match="cohort"):
        adapter.adapt(
            trace=trace,
            decision_id=decision.decision.decision_id,
            hidden_worlds=(decision.state, decision.state.clone()),
            coverage_method="sampled_actual_deal_fixture",
            resources=DecisionProofResourcesV1(maximum_worlds=1),
        )
    with pytest.raises(InterruptedError, match="cancelled"):
        adapter.adapt(
            trace=trace,
            decision_id=decision.decision.decision_id,
            hidden_worlds=(decision.state,),
            coverage_method="sampled_actual_deal_fixture",
            cancel_check=lambda: True,
        )
    with pytest.raises(ValueError, match="actor-safe cohort method"):
        adapter.adapt(
            trace=trace,
            decision_id=decision.decision.decision_id,
            hidden_worlds=(decision.state,),
            coverage_method="sampled_actual_deal_fixture",
        )


@pytest.mark.real_clock
def test_benchmark_covers_real_hokum_and_sun_without_speed_claim(rules, convention):
    evidence = benchmark_reduced_policy_proof(rules, convention, iterations=1)
    by_contract = {case["contract"]: case for case in evidence["cases"]}
    assert set(by_contract) == {"hokum", "sun"}
    assert all(case["kernelMedianMs"] > 0 for case in evidence["cases"])
    assert all(case["currentAnalysisMedianMs"] > 0 for case in evidence["cases"])
    assert all(case["recommendationAgreement"] for case in evidence["cases"])
    assert by_contract["hokum"]["kernelTruth"] == "exact_reduced_actor_safe"
    assert by_contract["sun"]["kernelTruth"] == "provisional"
    assert "matched observations" in evidence["scopeNote"]
