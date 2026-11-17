from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.rules import RuleBook

from .environment import BalootEnvironment
from .models import ModelPlugin, UniformLegalModel, rank_legal_actions


@dataclass(frozen=True, slots=True)
class SelfPlayResult:
    seed: int
    steps: int
    final_match_score: tuple[int, int]
    examples: tuple[dict[str, Any], ...]


def _choose(model: ModelPlugin, observation: np.ndarray, legal_mask: tuple[int, ...]) -> int:
    output = model.infer(observation, legal_mask)
    legal = tuple(index for index, allowed in enumerate(legal_mask) if allowed)
    return rank_legal_actions(output, legal)[0][0]


def play_round(
    rules: RuleBook,
    *,
    seed: int,
    contract: Contract = Contract.HOKUM,
    buyer: Seat = Seat.SOUTH,
    trump_suit: Suit | None = Suit.HEARTS,
    model: ModelPlugin | None = None,
) -> SelfPlayResult:
    """Generate deterministic observer-safe policy/value training examples."""
    model = model or UniformLegalModel()
    environment = BalootEnvironment(rules)
    result = environment.reset(seed=seed, contract=contract, buyer=buyer, trump_suit=trump_suit)
    rows: list[dict[str, Any]] = []
    while not result.terminated:
        assert result.observation is not None and environment.state is not None
        actor = environment.state.current_player
        action = _choose(model, result.observation, result.legal_mask)
        rows.append({
            "seed": seed,
            "step": len(rows),
            "actor": actor.name.lower(),
            "observation": result.observation.tolist(),
            "legalMask": list(result.legal_mask),
            "action": action,
        })
        result = environment.step(action)
    assert environment.state is not None
    reward_ns = environment.state.round_score.awarded_points[0] - environment.state.round_score.awarded_points[1]
    for row in rows:
        row["targetValueNorthSouth"] = reward_ns
    return SelfPlayResult(seed, len(rows), tuple(environment.state.match_score), tuple(rows))
