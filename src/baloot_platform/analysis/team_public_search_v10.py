"""Hybrid observer-safe search for partnership Baloot.

Late positions are evaluated on the complete legal hidden-deal cohort with the
exact team solver.  Earlier positions use the strategy-consistent information-
set MCTS tree from V8.  The API states its guarantees explicitly: it is a
stronger baseline than independent determinization, but not an equilibrium
certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine

from .belief_v7 import build_particle_belief
from .ismcts_v8 import ISMCTSResult, analyze_information_set_mcts
from .models import InformationState, SignalObservation
from .search import exact_team_value


@dataclass(frozen=True, slots=True)
class TeamPublicAction:
    card: Card
    expected_round_margin: float
    positive_margin_probability: float
    exact_share: float
    visits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "card": self.card.to_dict(),
            "expectedRoundMargin": self.expected_round_margin,
            "positiveMarginProbability": self.positive_margin_probability,
            "exactShare": self.exact_share,
            "visits": self.visits,
        }


@dataclass(frozen=True, slots=True)
class TeamPublicSearchResult:
    recommended_card: Card
    actions: tuple[TeamPublicAction, ...]
    method: str
    guarantee: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendedCard": self.recommended_card.to_dict(),
            "actions": [item.to_dict() for item in self.actions],
            "method": self.method,
            "guarantee": self.guarantee,
            "diagnostics": self.diagnostics,
        }


def analyze_team_public_hybrid(
    info: InformationState,
    engine: BalootEngine,
    observations: tuple[SignalObservation, ...] = (),
    convention_profile: dict[str, Any] | None = None,
    *,
    particles: int = 256,
    exact_allocation_limit: int = 20_000,
    exact_remaining_threshold: int = 10,
    exact_search_max_nodes: int = 250_000,
    exact_search_max_seconds: float = 2.0,
    ismcts_iterations: int = 4_000,
    exploration: float = 1.15,
    seed: int = 1,
) -> TeamPublicSearchResult:
    belief = build_particle_belief(
        info,
        engine,
        observations,
        convention_profile,
        particles=particles,
        exact_allocation_limit=exact_allocation_limit,
        seed=seed,
    )
    remaining = sum(mask.bit_count() for mask in belief.particles[0].state.hands)
    legal = engine.legal_actions(belief.particles[0].state)
    if not legal:
        raise ValueError("no legal root action")

    if remaining <= exact_remaining_threshold:
        weights = belief.normalized_weights
        margins: dict[int, list[float]] = {card.id: [] for card in legal}
        exacts: dict[int, list[float]] = {card.id: [] for card in legal}
        for particle in belief.particles:
            working = particle.state.clone()
            for card in legal:
                token = engine.apply(
                    working,
                    card,
                    verify_identity=False,
                )
                try:
                    result = exact_team_value(
                        working,
                        engine,
                        info.observer.team,
                        max_nodes=exact_search_max_nodes,
                        max_seconds=exact_search_max_seconds,
                    )
                finally:
                    engine.undo(working, token)
                margins[card.id].append(float(result.value))
                exacts[card.id].append(float(result.exact))
        actions: list[TeamPublicAction] = []
        for card in legal:
            values = margins[card.id]
            mean = sum(value * weight for value, weight in zip(values, weights))
            positive = sum(weight for value, weight in zip(values, weights) if value > 0)
            exact_share = sum(value * weight for value, weight in zip(exacts[card.id], weights))
            actions.append(TeamPublicAction(card, mean, positive, exact_share))
        actions.sort(key=lambda item: (item.expected_round_margin, item.positive_margin_probability), reverse=True)
        return TeamPublicSearchResult(
            actions[0].card,
            tuple(actions),
            "legal_belief_exact_continuation_v2",
            "exact_continuation_per_legal_particle_not_equilibrium_across_unknown_future_information",
            {
                "particleCount": len(belief.particles),
                "effectiveSampleSize": belief.effective_sample_size,
                "remainingCards": remaining,
            },
        )

    result: ISMCTSResult = analyze_information_set_mcts(
        info,
        engine,
        observations,
        convention_profile,
        particles=particles,
        exact_allocation_limit=exact_allocation_limit,
        iterations=ismcts_iterations,
        exploration=exploration,
        seed=seed,
    )
    actions = tuple(
        TeamPublicAction(
            item.card,
            item.expected_round_margin,
            item.positive_margin_probability,
            0.0,
            item.visits,
        )
        for item in result.actions
    )
    return TeamPublicSearchResult(
        result.recommended_card,
        actions,
        "hybrid_information_set_team_search_v2",
        result.guarantee,
        {
            "particleCount": len(result.belief.particles),
            "effectiveSampleSize": result.belief.effective_sample_size,
            "iterations": result.iterations,
            "treeNodes": result.tree_nodes,
            "remainingCards": remaining,
        },
    )
