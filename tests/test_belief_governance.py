from __future__ import annotations

import pytest

from baloot_platform.analysis.belief_v7 import (
    MAXIMUM_HIDDEN_DEAL_ATTEMPTS,
    MAXIMUM_EXACT_ALLOCATION_LIMIT,
    BeliefPolicy,
    build_particle_belief,
    hidden_deal_attempt_budget,
)
from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.evaluator import (
    ANALYSIS_OBJECTIVES,
    AnalysisPolicy,
    analyze,
    analysis_objective_key,
)
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit


def _information_state(rules):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=26001, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=11)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.SOUTH,
        forced_trump=Suit.CLUBS,
    )
    engine.prepare_projects(state)
    return engine, InformationState.from_complete(
        state,
        state.current_player,
    )


def test_v21_belief_policy_resolves_active_governed_limits(root):
    policy = BeliefPolicy.from_profile(root=root)
    assert policy.profile_id == "constrained_particle_belief_v1"
    assert policy.requested_particles == 256
    assert policy.maximum_particles == 4096
    assert policy.exact_allocation_limit == 20_000
    assert (
        policy.maximum_exact_allocation_limit
        == MAXIMUM_EXACT_ALLOCATION_LIMIT
    )
    assert policy.minimum_effective_fraction == 0.35
    assert policy.resolve(
        particles=None,
        exact_allocation_limit=None,
        resample_threshold=None,
        maximum_attempts=None,
    ) == (
        256,
        20_000,
        0.35,
        MAXIMUM_HIDDEN_DEAL_ATTEMPTS,
    )
    assert hidden_deal_attempt_budget(
        4096,
        multiplier=1000,
        minimum=10_000,
    ) == MAXIMUM_HIDDEN_DEAL_ATTEMPTS


@pytest.mark.parametrize(
    ("override", "value"),
    (
        ("particles", 4097),
        (
            "exact_allocation_limit",
            MAXIMUM_EXACT_ALLOCATION_LIMIT + 1,
        ),
        ("resample_threshold", 1.01),
        ("maximum_attempts", MAXIMUM_HIDDEN_DEAL_ATTEMPTS + 1),
    ),
)
def test_v21_belief_policy_rejects_unsafe_overrides(root, override, value):
    policy = BeliefPolicy.from_profile(root=root)
    inputs = {
        "particles": 4,
        "exact_allocation_limit": 100,
        "resample_threshold": 0.35,
        "maximum_attempts": 100,
    }
    inputs[override] = value
    with pytest.raises(ValueError):
        policy.resolve(**inputs)


def test_v21_particle_belief_carries_profile_provenance(rules):
    engine, info = _information_state(rules)
    belief = build_particle_belief(
        info,
        engine,
        particles=4,
        exact_allocation_limit=100,
        maximum_attempts=100,
        seed=27,
    )
    payload = belief.to_dict(include_card_locations=False)
    assert payload["profileId"] == "constrained_particle_belief_v1"
    assert payload["profileVersion"] == "1.0.0"
    assert payload["particleCount"] == 4


def test_v21_analysis_uses_active_configured_objective(rules):
    engine, info = _information_state(rules)
    del engine
    result = analyze(
        info,
        rules,
        requested_samples=4,
        exact_allocation_limit=100,
        maximum_sampling_attempts=100,
        exact_search_max_remaining=0,
        seed=31,
    )
    assert result.objective == "match_point_margin"
    assert result.analysis_profile_id == "deep_analysis_v1"
    assert result.analysis_profile_version == "1.0.0"
    assert list(result.actions) == sorted(
        result.actions,
        key=lambda action: analysis_objective_key(
            action,
            result.objective,
        ),
        reverse=True,
    )
    assert result.to_dict()["objective"] == result.objective
    assert result.to_dict()["analysisProfile"] == {
        "id": "deep_analysis_v1",
        "version": "1.0.0",
    }


def test_v21_analysis_rejects_unknown_objective_before_sampling(rules):
    _engine, info = _information_state(rules)
    assert "cvar" in ANALYSIS_OBJECTIVES
    with pytest.raises(ValueError, match="unsupported analysis objective"):
        analyze(info, rules, objective="unbounded_magic")


def test_v21_analysis_profile_defaults_are_validated(root):
    import yaml

    profile = yaml.safe_load(
        (root / "configs/analysis/deep.yaml").read_text(
            encoding="utf-8"
        )
    )
    policy = AnalysisPolicy.from_profile(
        profile,
        belief_policy=BeliefPolicy.from_profile(root=root),
    )
    assert policy.requested_samples == 160
    assert policy.exact_allocation_limit == 25_000
    assert policy.maximum_sampling_attempts == 80_000
    assert policy.objective == "match_point_margin"
