from __future__ import annotations

import yaml

from baloot_platform.analysis.match_value_v7 import (
    MatchValueConfig,
    estimate_match_win_probability,
)
from baloot_platform.core.enums import Team


def test_active_match_value_profile_uses_exact_dynamic_program(root):
    profile = yaml.safe_load(
        (root / "configs/match_value/exact_dynamic_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = MatchValueConfig.from_profile(profile)
    first = estimate_match_win_probability(
        (10, 6),
        Team.NORTH_SOUTH,
        config=config,
        seed_parts=("first",),
    )
    second = estimate_match_win_probability(
        (10, 6),
        Team.NORTH_SOUTH,
        config=config,
        seed_parts=("different",),
    )
    mirrored = estimate_match_win_probability(
        (6, 10),
        Team.WEST_EAST,
        config=config,
    )
    assert first == second == mirrored
    assert first.exact is True
    assert first.confidence_low == first.probability == first.confidence_high
    assert first.model == "policy_conditioned_exact_dynamic_program_v2"


def test_exact_match_value_respects_declared_future_round_policy():
    config = MatchValueConfig(
        target=10,
        maximum_rounds=2,
        future_team_round_win_probability=1.0,
        increment_distribution=((10, 0, 1.0),),
        solver="exact_dynamic_program_v2",
        maximum_cache_states=100,
    )
    result = estimate_match_win_probability(
        (0, 0),
        Team.NORTH_SOUTH,
        config=config,
    )
    assert result.probability == 1.0
    assert result.exact is True


def test_exact_match_value_terminal_state_is_not_reestimated():
    config = MatchValueConfig(
        target=10,
        maximum_rounds=2,
        future_team_round_win_probability=0.5,
        increment_distribution=((10, 0, 1.0),),
        solver="exact_dynamic_program_v2",
        maximum_cache_states=100,
    )
    result = estimate_match_win_probability(
        (12, 9),
        Team.WEST_EAST,
        config=config,
    )
    assert result.probability == 0.0
    assert result.exact is True
    assert result.model == "terminal_exact_v1"
