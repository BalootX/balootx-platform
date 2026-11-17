from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine

from .belief_v7 import build_particle_belief
from .models import InformationState, SignalObservation
from .search import exact_team_value


@dataclass(frozen=True, slots=True)
class CoordinatorAction:
    card: Card
    probability: float
    expected_margin: float
    worst_case_margin: float
    downside_cvar: float
    positive_probability: float
    regret: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "card": self.card.to_dict(), "probability": self.probability,
            "expectedMargin": self.expected_margin, "worstCaseMargin": self.worst_case_margin,
            "downsideCvar": self.downside_cvar, "positiveProbability": self.positive_probability,
            "regret": self.regret,
        }


@dataclass(frozen=True, slots=True)
class CoordinatorSearchResult:
    recommended_card: Card
    actions: tuple[CoordinatorAction, ...]
    iterations: int
    method: str
    guarantee: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendedCard": self.recommended_card.to_dict(),
            "actions": [item.to_dict() for item in self.actions],
            "iterations": self.iterations, "method": self.method,
            "guarantee": self.guarantee, "diagnostics": dict(self.diagnostics),
        }


def _cvar(values: list[tuple[float, float]], alpha: float) -> float:
    ordered = sorted(values, key=lambda item: item[0])
    target = max(1e-9, min(1.0, float(alpha)))
    used = 0.0
    total = 0.0
    for value, weight in ordered:
        take = min(weight, target - used)
        if take > 0:
            total += value * take; used += take
        if used >= target - 1e-12:
            break
    return total / used if used > 0 else ordered[0][0]


def analyze_team_public_coordinator(
    info: InformationState,
    engine: BalootEngine,
    observations: tuple[SignalObservation, ...] = (),
    convention_profile: dict[str, Any] | None = None,
    *,
    particles: int = 128,
    exact_allocation_limit: int = 10_000,
    iterations: int = 2_000,
    cvar_alpha: float = 0.20,
    risk_weight: float = 0.15,
    max_nodes: int = 120_000,
    max_seconds: float = 1.0,
    seed: int = 1,
) -> CoordinatorSearchResult:
    """Root-consistent no-regret coordinator for observer-safe Baloot.

    Unlike independent determinization, this procedure learns one public root
    policy shared across all compatible hidden particles. Continuations are
    still solved per particle, so this is not a full-game equilibrium
    certificate. The guarantee is stated explicitly in the result.
    """
    belief = build_particle_belief(
        info, engine, observations, convention_profile,
        particles=particles, exact_allocation_limit=exact_allocation_limit, seed=seed,
    )
    if not belief.particles:
        raise ValueError("belief contains no particles")
    legal = tuple(engine.legal_moves(belief.particles[0].state))
    if not legal:
        raise ValueError("no legal root action")
    weights = belief.normalized_weights
    matrix: list[list[float]] = [[0.0 for _ in belief.particles] for _ in legal]
    exact_count = 0
    for action_index, card in enumerate(legal):
        for particle_index, particle in enumerate(belief.particles):
            branch = particle.state.clone()
            engine.play_card(branch, card)
            result = exact_team_value(branch, engine, info.observer.team, max_nodes=max_nodes, max_seconds=max_seconds)
            matrix[action_index][particle_index] = float(result.value)
            exact_count += int(result.exact)

    # Regret matching across a shared public action policy. Each iteration draws
    # one compatible hidden state from the observer-safe belief and updates the
    # same action regrets, preventing root strategy fusion.
    rng = random.Random(seed)
    regrets = [0.0] * len(legal)
    strategy_sum = [0.0] * len(legal)
    cumulative = []
    running = 0.0
    for weight in weights:
        running += weight; cumulative.append(running)

    def current_strategy() -> list[float]:
        positive = [max(0.0, value) for value in regrets]
        total = sum(positive)
        return [value / total for value in positive] if total > 0 else [1.0 / len(legal)] * len(legal)

    for _ in range(max(1, int(iterations))):
        strategy = current_strategy()
        for index, probability in enumerate(strategy):
            strategy_sum[index] += probability
        point = rng.random()
        particle_index = next((index for index, value in enumerate(cumulative) if point <= value), len(cumulative) - 1)
        utilities = [row[particle_index] for row in matrix]
        mixed_value = sum(probability * value for probability, value in zip(strategy, utilities))
        for index, value in enumerate(utilities):
            regrets[index] += value - mixed_value

    total_strategy = sum(strategy_sum)
    average_strategy = [value / total_strategy for value in strategy_sum]
    actions: list[CoordinatorAction] = []
    pure_expected: list[float] = []
    for index, card in enumerate(legal):
        weighted = [(value, weight) for value, weight in zip(matrix[index], weights)]
        mean = sum(value * weight for value, weight in weighted)
        downside = _cvar(weighted, cvar_alpha)
        robust_score = mean - max(0.0, float(risk_weight)) * max(0.0, mean - downside)
        positive = sum(weight for value, weight in weighted if value > 0)
        pure_expected.append(mean)
        actions.append(CoordinatorAction(
            card=card, probability=average_strategy[index], expected_margin=mean,
            worst_case_margin=min(matrix[index]), downside_cvar=downside,
            positive_probability=positive, regret=max(0.0, max(pure_expected + [mean]) - robust_score),
        ))
    actions.sort(
        key=lambda item: (item.probability, item.expected_margin - risk_weight * max(0.0, item.expected_margin - item.downside_cvar)),
        reverse=True,
    )
    mixed_expected = sum(
        average_strategy[action_index] * sum(matrix[action_index][p] * weights[p] for p in range(len(weights)))
        for action_index in range(len(legal))
    )
    best_expected = max(pure_expected)
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in average_strategy)
    return CoordinatorSearchResult(
        recommended_card=actions[0].card,
        actions=tuple(actions),
        iterations=max(1, int(iterations)),
        method="root_consistent_team_public_regret_matching_v1",
        guarantee="one_shared_root_policy_across_legal_hidden_particles_not_full_game_equilibrium",
        diagnostics={
            "particleCount": len(belief.particles),
            "effectiveSampleSize": belief.effective_sample_size,
            "exactContinuationShare": exact_count / (len(legal) * len(belief.particles)),
            "mixedExpectedMargin": mixed_expected,
            "bestPureExpectedMargin": best_expected,
            "externalRegretProxy": max(0.0, best_expected - mixed_expected),
            "strategyEntropy": entropy,
            "cvarAlpha": cvar_alpha,
            "riskWeight": risk_weight,
        },
    )
