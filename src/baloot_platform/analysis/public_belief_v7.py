"""Public/team-information analysis built on a shared particle cohort."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine

from .belief_v7 import ParticleBelief, build_particle_belief
from .models import InformationState, SignalObservation
from .search import exact_team_value, heuristic_rollout_result


@dataclass(frozen=True, slots=True)
class PublicBeliefAction:
    card: Card
    expected_margin: float
    positive_margin_probability: float
    worst_decile: float
    best_share: float
    exact_share: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "card": self.card.to_dict(), "expectedRoundMargin": self.expected_margin,
            "positiveMarginProbability": self.positive_margin_probability,
            "worstDecile": self.worst_decile, "bestShare": self.best_share,
            "exactShare": self.exact_share,
        }


@dataclass(frozen=True, slots=True)
class PublicBeliefResult:
    recommended: Card
    actions: tuple[PublicBeliefAction, ...]
    belief: ParticleBelief
    method: str = "shared_particle_team_public_search_v1"

    def to_dict(self) -> dict[str, Any]:
        return {"recommendedCard": self.recommended.to_dict(),"actions":[a.to_dict() for a in self.actions],"belief":self.belief.to_dict(),"method":self.method}


def analyze_public_belief(
    info: InformationState,
    engine: BalootEngine,
    observations: tuple[SignalObservation, ...] = (),
    convention_profile: dict[str, Any] | None = None,
    *,
    particles: int = 128,
    exact_allocation_limit: int = 20_000,
    exact_search_max_remaining: int = 10,
    exact_search_max_nodes: int = 120_000,
    seed: int = 1,
) -> PublicBeliefResult:
    belief = build_particle_belief(info,engine,observations,convention_profile,particles=particles,exact_allocation_limit=exact_allocation_limit,seed=seed)
    legal = engine.legal_moves(belief.particles[0].state)
    weights = belief.normalized_weights
    values: dict[int,list[float]]={card.id:[] for card in legal}; exact:dict[int,list[float]]={card.id:[] for card in legal}
    for particle in belief.particles:
        state=particle.state
        for card in legal:
            branch=state.clone(); engine.play_card(branch,card)
            remaining=sum(mask.bit_count() for mask in branch.hands)
            if remaining<=exact_search_max_remaining:
                result=exact_team_value(branch,engine,info.observer.team,max_nodes=exact_search_max_nodes,max_seconds=1.0)
            else:
                result=heuristic_rollout_result(branch,engine,info.observer.team,seed=seed+card.id+remaining)
            values[card.id].append(float(result.margin)); exact[card.id].append(float(result.exact))
    action_rows=[]
    for card in legal:
        samples=values[card.id]
        mean=sum(v*w for v,w in zip(samples,weights))
        positive=sum(w for v,w in zip(samples,weights) if v>0)
        ordered=sorted(zip(samples,weights)); target=0.1; used=0.0; tail=0.0
        for value,weight in ordered:
            take=min(weight,target-used); tail+=value*take; used+=take
            if used>=target:break
        action_rows.append(PublicBeliefAction(card,mean,positive,tail/max(used,1e-12),0.0,sum(e*w for e,w in zip(exact[card.id],weights))))
    # best-share on the same particle cohort
    shares={card.id:0.0 for card in legal}
    for index,weight in enumerate(weights):
        best=max(values[card.id][index] for card in legal)
        winners=[card for card in legal if values[card.id][index]==best]
        for card in winners:shares[card.id]+=weight/len(winners)
    action_rows=[PublicBeliefAction(a.card,a.expected_margin,a.positive_margin_probability,a.worst_decile,shares[a.card.id],a.exact_share) for a in action_rows]
    action_rows.sort(key=lambda a:(a.expected_margin,a.positive_margin_probability),reverse=True)
    return PublicBeliefResult(action_rows[0].card,tuple(action_rows),belief)
