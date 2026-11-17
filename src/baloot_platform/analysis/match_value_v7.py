from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import yaml

from baloot_platform.config import platform_root
from baloot_platform.core.enums import Team


@dataclass(frozen=True, slots=True)
class MatchValueConfig:
    target: int
    maximum_rounds: int
    future_team_round_win_probability: float
    increment_distribution: tuple[tuple[int, int, float], ...]
    solver: str
    maximum_cache_states: int

    @classmethod
    def from_profile(
        cls, profile: dict[str, Any] | None
    ) -> "MatchValueConfig":
        # Bounded V1 read adapter. New runtime paths pass the frozen profile
        # explicitly; legacy callers read the same packaged, schema-validated
        # profile instead of applying private code defaults.
        if profile is None:
            profile = yaml.safe_load(
                (
                    platform_root()
                    / "configs/match_value/exact_dynamic_v2.yaml"
                ).read_text(encoding="utf-8")
            )
        data = dict(profile)
        allowed_top = {
            "schema_version", "profile", "target", "future_rounds",
            "outputs", "compatibility",
        }
        unknown_top = set(data) - allowed_top
        if unknown_top:
            raise ValueError(
                f"unknown match-value profile keys: {sorted(unknown_top)}"
            )
        future = dict(data["future_rounds"])
        required_future = {
            "maximum_rounds", "dealer_rotation", "policy",
            "future_team_round_win_probability", "increment_distribution",
            "solver", "maximum_cache_states",
        }
        if set(future) != required_future:
            raise ValueError(
                "match-value future_rounds keys mismatch: "
                f"missing={sorted(required_future - set(future))} "
                f"unknown={sorted(set(future) - required_future)}"
            )
        distribution = future["increment_distribution"]
        values: list[tuple[int, int, float]] = []
        for item in distribution:
            values.append(
                (
                    int(item["winner"]),
                    int(item["loser"]),
                    float(item["weight"]),
                )
            )
        if not values or sum(max(0.0, item[2]) for item in values) <= 0:
            raise ValueError("match-value increment distribution requires positive weight")
        probability = float(future["future_team_round_win_probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError("future team round win probability must be in [0, 1]")
        solver = str(future["solver"])
        if solver != "exact_dynamic_program_v2":
            raise ValueError(f"unsupported match-value solver: {solver}")
        maximum_cache_states = int(future["maximum_cache_states"])
        if maximum_cache_states < 1:
            raise ValueError("match-value cache limit must be positive")
        return cls(
            target=int(data["target"]),
            maximum_rounds=max(1, int(future["maximum_rounds"])),
            future_team_round_win_probability=probability,
            increment_distribution=tuple(values),
            solver=solver,
            maximum_cache_states=maximum_cache_states,
        )


@dataclass(frozen=True, slots=True)
class MatchValueEstimate:
    probability: float
    confidence_low: float
    confidence_high: float
    samples: int
    exact: bool
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "confidence95": [self.confidence_low, self.confidence_high],
            "samples": self.samples,
            "exact": self.exact,
            "model": self.model,
        }


class _ExactMatchValueSolver:
    """Bounded exact solver for the versioned future-round policy."""

    def __init__(self, config: MatchValueConfig):
        self.config = config
        weighted = tuple(
            (winner, loser, max(0.0, weight))
            for winner, loser, weight in config.increment_distribution
            if weight > 0.0
        )
        total = sum(weight for _winner, _loser, weight in weighted)
        if total <= 0.0:
            raise ValueError("match-value increment distribution has no positive mass")
        round_win = config.future_team_round_win_probability
        transitions: dict[tuple[int, int], float] = {}
        for winner, loser, weight in weighted:
            normalized = weight / total
            transitions[(winner, loser)] = (
                transitions.get((winner, loser), 0.0)
                + normalized * round_win
            )
            transitions[(loser, winner)] = (
                transitions.get((loser, winner), 0.0)
                + normalized * (1.0 - round_win)
            )
        self.transitions = tuple(
            (own, opponent, probability)
            for (own, opponent), probability in sorted(transitions.items())
            if probability > 0.0
        )
        self._probability = lru_cache(maxsize=config.maximum_cache_states)(
            self._probability_uncached
        )

    def _probability_uncached(
        self,
        own_score: int,
        opponent_score: int,
        rounds_remaining: int,
    ) -> float:
        target = self.config.target
        if max(own_score, opponent_score) >= target and own_score != opponent_score:
            return float(own_score > opponent_score)
        if rounds_remaining <= 0:
            return float(own_score > opponent_score)
        return math.fsum(
            probability
            * self._probability(
                own_score + own_increment,
                opponent_score + opponent_increment,
                rounds_remaining - 1,
            )
            for own_increment, opponent_increment, probability in self.transitions
        )

    def probability(self, own_score: int, opponent_score: int) -> float:
        return self._probability(
            int(own_score),
            int(opponent_score),
            self.config.maximum_rounds,
        )


@lru_cache(maxsize=16)
def _exact_solver(config: MatchValueConfig) -> _ExactMatchValueSolver:
    return _ExactMatchValueSolver(config)


def estimate_match_win_probability(
    score_after_round: tuple[int, int],
    team: Team,
    *,
    config: MatchValueConfig,
    seed_parts: Iterable[Any] = (),
) -> MatchValueEstimate:
    """Estimate policy-conditioned probability of winning the match to target.

    The current round score is exact. If it ends the match, the result is exact.
    Otherwise future rounds are simulated using an explicit, versioned policy
    model. This is intentionally distinct from current-round lead probability.
    """

    score = [int(score_after_round[0]), int(score_after_round[1])]
    target = int(config.target)
    if max(score) >= target and score[0] != score[1]:
        winner = Team.NORTH_SOUTH if score[0] > score[1] else Team.WEST_EAST
        probability = 1.0 if winner == team else 0.0
        return MatchValueEstimate(probability, probability, probability, 1, True, "terminal_exact_v1")

    # V2 is an exact finite-horizon evaluation of the declared future-round
    # policy. ``seed_parts`` remains a bounded read-compatibility argument;
    # exact evaluation intentionally has no sampling seed.
    del seed_parts
    probability = _exact_solver(config).probability(
        score[int(team)],
        score[int(team.other)],
    )
    return MatchValueEstimate(
        probability,
        probability,
        probability,
        0,
        True,
        "policy_conditioned_exact_dynamic_program_v2",
    )
