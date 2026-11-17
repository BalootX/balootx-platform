from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from baloot_platform.agents.profiles import (
    AdaptivePlayStrategy,
    AgentProfile,
    AgentProfileRegistry,
)
from baloot_platform.agents.qualification import (
    validate_finalist_analysis_compatibility,
)
from baloot_platform.config import build_bundle
from baloot_platform.config import _validate_research_agent_bindings
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.native.accelerator import NativeAccelerator


def test_active_agent_profile_is_v2_and_snapshot_round_trips(root: Path):
    registry = AgentProfileRegistry.from_bundle(root)
    solver = registry.get("solver_fast")

    assert solver.play_search_mode == "paired_information_set"
    assert solver.play_search_samples == 16
    assert solver.play_exact_allocation_limit == 64
    assert solver.mistake_probability == 0.0
    finalist = registry.get("finalist_candidate")
    assert finalist.play_search_mode == "paired_information_set"
    assert finalist.play_search_objective == "match_win_probability"
    assert finalist.play_search_samples == 192
    assert finalist.play_exact_allocation_limit == 128
    assert finalist.play_exact_search_remaining == 9
    assert finalist.bid_samples == 192
    assert finalist.mistake_probability == 0.0
    challenge = build_bundle(root).component("challenge")
    assert challenge["finalist_validation"]["agentProfileId"] == finalist.id
    validate_finalist_analysis_compatibility(finalist, challenge["confirmation"])
    assert challenge["confirmation"]["bid_samples"] >= finalist.bid_samples
    assert challenge["confirmation"]["deep_samples"] >= finalist.play_search_samples
    assert (
        challenge["confirmation"]["exact_search_remaining"]
        >= finalist.play_exact_search_remaining
    )
    assert (
        challenge["stability"]["reanalysis_samples"]
        >= challenge["confirmation"]["deep_samples"]
    )
    assert (
        challenge["stability"]["exact_search_remaining"]
        >= challenge["confirmation"]["exact_search_remaining"]
    )
    assert (
        challenge["stability"]["exact_search_max_nodes"]
        >= challenge["confirmation"]["exact_search_max_nodes"]
    )
    assert (
        challenge["stability"]["exact_search_max_seconds"]
        >= challenge["confirmation"]["exact_search_max_seconds"]
    )
    with pytest.raises(ValueError, match="incompatible with confirmation"):
        validate_finalist_analysis_compatibility(
            replace(finalist, play_search_samples=4),
            challenge["confirmation"],
        )
    snapshot = registry.snapshot_lineup(
        {
            "north": "solver_fast",
            "west": "solver_fast",
            "south": "solver_fast",
            "east": "solver_fast",
        }
    )
    restored = AgentProfileRegistry.from_snapshot(snapshot)
    assert restored.get("solver_fast") == solver


def test_default_lineup_is_fair_and_exact_analysis_is_deterministic(
    root: Path,
) -> None:
    registry = AgentProfileRegistry.from_bundle(root)
    assert set(registry.default_lineup.values()) == {
        registry.default_profile
    }

    analysis = registry.get("expert_deterministic")
    assert analysis.play_search_mode == "paired_information_set"
    assert analysis.play_exact_allocation_limit == 20_000
    assert analysis.mistake_probability == 0.0
    assert analysis.deception_probability == 0.0


def test_research_lineages_are_equal_seat_zero_variation_and_budgeted(
    root: Path,
) -> None:
    bundle = build_bundle(root)
    agents = bundle.component("agents")
    challenge = bundle.component("challenge")
    profiles = agents["profiles"]
    discovery_ids = {
        challenge["campaign_screening"]["agentProfileId"],
        challenge["guided_search"]["agentProfileId"],
        challenge["candidate_resimulation"]["agentProfileId"],
    }
    assert discovery_ids == {"foundry_balanced"}

    stage_ids = discovery_ids | {
        challenge["finalist_validation"]["agentProfileId"]
    }
    for profile_id in stage_ids:
        profile = profiles[profile_id]
        assert profile["skill"] == 1.0
        assert profile["mistake_probability"] == 0.0
        assert profile["deception_probability"] == 0.0
        assert profile["signal_discipline"] == 1.0
        assert profile["play_search_mode"] == "paired_information_set"
        assert profile["play_search_budget"] > 0
        assert profile["play_search_samples"] > 0
        assert profile["bid_samples"] > 0
        assert profile["use_production_model"] is False
        assert profile["model_weight"] == 0.0

    # Online skill modes remain intentionally separate from Research policy.
    assert profiles["beginner"]["mistake_probability"] > 0.0
    assert profiles["regular"]["mistake_probability"] > 0.0
    assert agents["default_profile"] == "strong"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("mistake_probability", 0.01, "stochastic mistakes"),
        ("deception_probability", 0.01, "stochastic deception"),
        ("play_search_mode", "disabled", "paired information-set search"),
        ("play_search_budget", 0, "positive search and bidding budgets"),
        ("bid_samples", 0, "positive search and bidding budgets"),
        ("use_production_model", True, "unfrozen production model"),
    ),
)
def test_research_agent_binding_fails_closed(
    root: Path,
    field: str,
    value,
    message: str,
) -> None:
    bundle = build_bundle(root)
    agents = copy.deepcopy(bundle.component("agents"))
    challenge = copy.deepcopy(bundle.component("challenge"))
    agents["profiles"]["foundry_balanced"][field] = value
    with pytest.raises(ValueError, match=message):
        _validate_research_agent_bindings(agents, challenge)


def test_zero_configured_mistakes_are_not_reintroduced_by_skill(
    root: Path,
):
    bundle = build_bundle(root)
    engine = BalootEngine(bundle.rules)
    state = engine.create_round(seed=904, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.SUN,
        forced_buyer=Seat.SOUTH,
    )
    engine.prepare_projects(state)
    profile = replace(
        AgentProfileRegistry.from_bundle(root).get("beginner"),
        mistake_probability=0.0,
        deception_probability=0.0,
    )
    strategy = AdaptivePlayStrategy(profile, bundle.convention)

    for seed in range(32):
        _, trace = strategy.choose_with_trace(
            engine,
            state,
            seed=seed,
        )
        assert trace["mistakeApplied"] is False
        assert trace["deceptionApplied"] is False


def test_unknown_search_mode_fails_closed():
    with pytest.raises(ValueError, match="search mode"):
        AgentProfile.from_mapping(
            "invalid",
            {
                "play_search_mode": "magic",
            },
        )


def test_unknown_search_objective_fails_closed():
    with pytest.raises(ValueError, match="search objective"):
        AgentProfile.from_mapping(
            "invalid",
            {
                "play_search_objective": "win_somehow",
            },
        )


def test_agent_search_node_budget_is_enforced_at_analysis_boundary(
    root: Path, monkeypatch,
):
    bundle = build_bundle(root)
    engine = BalootEngine(bundle.rules)
    state = engine.create_round(seed=905, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.SUN,
        forced_buyer=Seat.SOUTH,
    )
    engine.prepare_projects(state)
    legal = engine.legal_moves(state)
    captured = {}

    def fake_analyze(_info, _rules, _convention, **kwargs):
        captured.update(kwargs)
        action = SimpleNamespace(
            card=legal[0],
            expected_match_margin=1.0,
            win_probability=0.5,
            match_win_probability=0.5,
            downside_cvar=0.0,
            best_share=1.0,
            exact_share=0.0,
        )
        return SimpleNamespace(
            actions=(action,),
            method="test_search",
            effective_sample_size=2.0,
            hidden_deals=2,
        )

    monkeypatch.setattr("baloot_platform.agents.profiles.analyze", fake_analyze)
    profile = replace(
        AgentProfileRegistry.from_bundle(root).get("finalist_candidate"),
        play_search_budget=777,
        play_search_samples=2,
        play_search_max_remaining=32,
    )
    selected, trace = AdaptivePlayStrategy(profile, bundle.convention).choose_with_trace(
        engine, state, legal, seed=1
    )
    assert selected == legal[0]
    assert trace["decisionSource"] == "hidden_information_search"
    assert captured["exact_search_max_nodes"] == 777
    assert captured["objective"] == "match_win_probability"


def test_native_library_discovery_is_platform_specific() -> None:
    assert all(
        path.endswith(".dll")
        for path in NativeAccelerator._library_candidates("win32")
    )
    assert all(
        path.endswith(".so")
        for path in NativeAccelerator._library_candidates("linux")
    )
    assert all(
        path.endswith(".dylib")
        for path in NativeAccelerator._library_candidates("darwin")
    )
