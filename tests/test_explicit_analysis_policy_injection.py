from __future__ import annotations

from types import SimpleNamespace
from copy import deepcopy

import pytest

import baloot_platform.analysis.belief_v7 as belief_module
import baloot_platform.analysis.evaluator as evaluator_module
import baloot_platform.challenge.analysis_pipeline as pipeline_module
from baloot_platform.analysis.belief_v7 import BeliefPolicy, build_particle_belief
from baloot_platform.analysis.evaluator import AnalysisPolicy, analyze
from baloot_platform.analysis.models import InformationState
from baloot_platform.challenge.analysis_pipeline import (
    StagedTraceAnalyzerV1,
    TraceAnalysisProfileV1,
    _initialize_decision_analysis_worker,
)
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit


def _state_and_information(rules):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=27101, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=13)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.SOUTH,
        forced_trump=Suit.CLUBS,
    )
    engine.prepare_projects(state)
    return engine, state, InformationState.from_complete(state, state.current_player)


def _belief_policy(*, profile_id: str, requested: int) -> BeliefPolicy:
    return BeliefPolicy.from_profile(
        {
            "profile": {"id": profile_id, "version": "1.0.0"},
            "particles": {
                "requested": requested,
                "maximum": 16,
                "minimum_effective_fraction": 0.35,
                "resampling": "systematic",
            },
            "construction": {"exact_when_allocations_below": 100},
        }
    )


def _analysis_policy(
    belief_policy: BeliefPolicy,
    *,
    profile_id: str,
    requested: int,
    objective: str,
) -> AnalysisPolicy:
    return AnalysisPolicy.from_profile(
        {
            "profile": {"id": profile_id, "version": "1.0.0"},
            "requested_samples": requested,
            "exact_allocation_limit": 100,
            "exact_search_max_remaining": 0,
            "maximum_sampling_attempts": 100,
            "objective": objective,
        },
        belief_policy=belief_policy,
    )


def _unexpected_active_lookup(*_args, **_kwargs):
    raise AssertionError("packaged active policy must not be read")


def test_explicit_belief_policy_controls_defaults_without_active_lookup(
    rules,
    monkeypatch,
):
    engine, _state, info = _state_and_information(rules)
    first_policy = _belief_policy(profile_id="bundle_belief_two", requested=2)
    second_policy = _belief_policy(profile_id="bundle_belief_four", requested=4)
    monkeypatch.setattr(
        belief_module,
        "_active_belief_policy",
        _unexpected_active_lookup,
    )

    first = build_particle_belief(
        info,
        engine,
        belief_policy=first_policy,
        maximum_attempts=100,
        seed=17,
    )
    second = build_particle_belief(
        info,
        engine,
        belief_policy=second_policy,
        maximum_attempts=100,
        seed=17,
    )

    assert len(first.particles) == 2
    assert first.profile_id == "bundle_belief_two"
    assert len(second.particles) == 4
    assert second.profile_id == "bundle_belief_four"


def test_explicit_analysis_and_belief_policies_bypass_packaged_defaults(
    rules,
    monkeypatch,
):
    _engine, _state, info = _state_and_information(rules)
    belief_policy = _belief_policy(profile_id="bundle_belief", requested=3)
    analysis_policy = _analysis_policy(
        belief_policy,
        profile_id="bundle_analysis",
        requested=3,
        objective="low_volatility",
    )
    monkeypatch.setattr(
        belief_module,
        "_active_belief_policy",
        _unexpected_active_lookup,
    )
    monkeypatch.setattr(
        evaluator_module,
        "_active_analysis_profile",
        _unexpected_active_lookup,
    )

    result = analyze(
        info,
        rules,
        analysis_policy=analysis_policy,
        belief_policy=belief_policy,
        seed=23,
    )

    assert result.analysis_profile_id == "bundle_analysis"
    assert result.analysis_profile_version == "1.0.0"
    assert result.objective == "low_volatility"
    assert result.hidden_deals == 3


def test_staged_and_process_analyzers_retain_explicit_bundle_policies(
    rules,
    convention,
    monkeypatch,
):
    _engine, state, _info = _state_and_information(rules)
    belief_policy = _belief_policy(profile_id="worker_belief", requested=2)
    analysis_policy = _analysis_policy(
        belief_policy,
        profile_id="worker_analysis",
        requested=2,
        objective="round_margin",
    )
    trace_profile = TraceAnalysisProfileV1.build(
        bid_samples=2,
        play_samples=2,
        exact_allocation_limit=50,
        exact_search_max_remaining=0,
        maximum_sampling_attempts=100,
        exact_search_max_nodes=1_000,
        exact_search_max_seconds=0.05,
        exact_search_max_cache_entries=100,
        objective="round_margin",
    )
    analyzer = StagedTraceAnalyzerV1(
        rules,
        convention,
        trace_profile,
        analysis_policy=analysis_policy,
        belief_policy=belief_policy,
    )
    captured: dict[str, object] = {}

    def capture_analyze(*_args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setattr(pipeline_module, "analyze", capture_analyze)
    snapshot = SimpleNamespace(
        state=state,
        decision=SimpleNamespace(acting_seat=state.current_player),
    )
    with pytest.raises(RuntimeError, match="captured"):
        analyzer._play(snapshot, 29, None)
    assert captured["analysis_policy"] is analysis_policy
    assert captured["belief_policy"] is belief_policy

    cancel_event = SimpleNamespace(is_set=lambda: False)
    _initialize_decision_analysis_worker(
        rules.data,
        convention,
        trace_profile,
        {},
        70.0,
        0.75,
        analysis_policy,
        belief_policy,
        "python_reference_exact_search_v1",
        cancel_event,
    )
    worker_analyzer = pipeline_module._PROCESS_ANALYZER
    assert worker_analyzer is not None
    assert worker_analyzer.analysis_policy == analysis_policy
    assert worker_analyzer.belief_policy == belief_policy


@pytest.mark.parametrize("failure", ["unsupported_rules", "unavailable_library"])
def test_process_analyzer_uses_declared_reference_fallback_on_native_refusal(rules, monkeypatch, failure):
    from baloot_platform.analysis.search_backend import PythonReferenceSearchBackend
    from baloot_platform.native import accelerator

    configured = deepcopy(rules.data)
    configured["scoring"]["contract"]["buyer_tie_outcome"] = "buyer_makes"
    captured = {}

    def refuse(_rules):
        raise (accelerator.NativeUnavailable("unsupported Sun tie")
               if failure == "unsupported_rules" else OSError("missing native library"))

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(accelerator, "NativeAccelerator", refuse)
    monkeypatch.setattr(pipeline_module, "StagedTraceAnalyzerV1", capture)
    monkeypatch.setattr(pipeline_module, "_PROCESS_ANALYZER", None)
    monkeypatch.setattr(pipeline_module, "_PROCESS_CANCEL_EVENT", None)
    _initialize_decision_analysis_worker(
        configured, {}, None, {}, 70.0, 0.75, None, None,
        "native_exact_v3_python_fallback_v1", SimpleNamespace(is_set=lambda: False),
    )
    assert isinstance(captured["exact_search_backend"], PythonReferenceSearchBackend)
