"""Strategy-consistent information-set Monte Carlo search for Baloot.

This is a genuine information-set tree baseline: actions are shared across all
sampled deals that produce the same player's information set. It therefore
avoids the root-level strategy fusion of independently solving every complete
deal and averaging the answers. It is not claimed to be an equilibrium solver;
team-public CFR/coordinator solving remains a replaceable future plugin.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable

from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat
from baloot_platform.core.models import RoundState

from .belief_v7 import ParticleBelief, build_particle_belief
from .models import InformationState, SignalObservation, public_projects_from_complete
from .search import terminal_margin


@dataclass(slots=True)
class _ActionStats:
    visits: int = 0
    total: float = 0.0
    squared: float = 0.0
    positive: int = 0

    def add(self, reward: float) -> None:
        self.visits += 1
        self.total += reward
        self.squared += reward * reward
        self.positive += int(reward > 0)

    @property
    def mean(self) -> float:
        return self.total / self.visits if self.visits else 0.0

    @property
    def variance(self) -> float:
        if self.visits < 2:
            return 0.0
        return max(0.0, (self.squared - self.total * self.total / self.visits) / (self.visits - 1))


@dataclass(frozen=True, slots=True)
class ISMCTSAction:
    card: Card
    visits: int
    visit_share: float
    expected_round_margin: float
    positive_margin_probability: float
    confidence_low: float
    confidence_high: float
    ekka: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "card": self.card.to_dict(),
            "visits": self.visits,
            "visitShare": self.visit_share,
            "expectedRoundMargin": self.expected_round_margin,
            "positiveMarginProbability": self.positive_margin_probability,
            "confidence95": [self.confidence_low, self.confidence_high],
            **({"ekka": True} if self.ekka else {}),
        }


@dataclass(frozen=True, slots=True)
class ISMCTSResult:
    observer: Seat
    recommended_card: Card
    actions: tuple[ISMCTSAction, ...]
    belief: ParticleBelief
    iterations: int
    tree_nodes: int
    seed: int
    common_cohort_size: int
    method: str = "information_set_mcts_v1"
    guarantee: str = "shared_strategy_by_information_set_not_equilibrium"
    recommended_ekka: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "observer": self.observer.name.lower(),
            "recommendedCard": self.recommended_card.to_dict(),
            "actions": [item.to_dict() for item in self.actions],
            "belief": self.belief.to_dict(),
            "iterations": self.iterations,
            "treeNodes": self.tree_nodes,
            "seed": self.seed,
            "commonCohortSize": self.common_cohort_size,
            "method": self.method,
            "guarantee": self.guarantee,
            **({"recommendedEkka": True} if self.recommended_ekka else {}),
        }


def information_set_key(state: RoundState) -> tuple[Any, ...]:
    """Canonical information-set key for the acting player.

    The key contains the acting player's own hand plus public state only. It
    intentionally excludes all opponents' unplayed cards.
    """
    if state.current_player is None:
        return ("terminal", tuple(state.match_score), len(state.completed_tricks))
    actor = state.current_player
    trick = tuple((int(play.seat), play.card.id, bool(play.ekka)) for play in (state.current_trick.plays if state.current_trick else ()))
    completed = tuple(
        (item.index, int(item.leader), int(item.winner), item.raw_points,
         tuple((int(play.seat), play.card.id, bool(play.ekka)) for play in item.plays))
        for item in state.completed_tricks
    )
    public_projects = tuple(sorted(
        (int(project.owner), project.kind, project.value,
         project.high_rank if project.high_rank is not None else -1,
         bool(project.declared), bool(project.scoring), bool(project.revealed),
         tuple(sorted(card.id for card in project.cards)) if project.revealed else ())
        for project in public_projects_from_complete(state, actor)
    ))
    return (
        int(actor),
        int(state.hands[int(actor)]),
        state.contract.value if state.contract else None,
        int(state.trump_suit) if state.trump_suit is not None else None,
        state.escalation,
        bool(state.locked),
        bool(state.trump_broken),
        int(state.dealer),
        int(state.buyer) if state.buyer is not None else None,
        int(state.declarer) if state.declarer is not None else None,
        tuple(state.match_score),
        completed,
        trick,
        public_projects,
        tuple(sorted(int(team) for team in state.baloot_teams)),
    )


def _sample_particle_at(
    belief: ParticleBelief,
    point: float,
) -> RoundState:
    running = 0.0
    weights = belief.normalized_weights
    for particle, weight in zip(belief.particles, weights):
        running += weight
        if point <= running:
            return particle.state.clone()
    return belief.particles[-1].state.clone()


def _sample_particle(
    belief: ParticleBelief,
    rng: random.Random,
) -> RoundState:
    return _sample_particle_at(belief, rng.random())


def _rollout(state: RoundState, engine: BalootEngine, observer: Seat, seed: int,
             cancel_check: Callable[[], bool] | None = None) -> float:
    state.events = []
    bot = HeuristicBot(engine, BotProfile(deterministic_seed=seed))
    while state.current_player is not None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("information-set search cancelled")
        card = bot.choose(state)
        engine.play_card(state, card)
        state.events = []
    return terminal_margin(state, observer.team)


def analyze_information_set_mcts(
    info: InformationState,
    engine: BalootEngine,
    observations: tuple[SignalObservation, ...] = (),
    convention_profile: dict[str, Any] | None = None,
    *,
    particles: int = 256,
    exact_allocation_limit: int = 20_000,
    iterations: int = 2_000,
    exploration: float = 1.15,
    maximum_tree_depth: int = 32,
    seed: int = 1,
    maximum_sampling_attempts: int | None = None,
    cancel_check: Callable[[], bool] | None = None,
    include_ekka: bool = False,
) -> ISMCTSResult:
    if info.current_player != info.observer:
        raise ValueError("information-set search currently requires the observer to act")
    belief = build_particle_belief(
        info, engine, observations, convention_profile,
        particles=max(1, int(particles)), exact_allocation_limit=max(1, int(exact_allocation_limit)),
        seed=int(seed),
        maximum_attempts=maximum_sampling_attempts, cancel_check=cancel_check,
    )
    from baloot_platform.core.models import LegalPlayActionV1
    def legal_actions(state):
        return (engine.legal_play_actions(state) if include_ekka else
                tuple(LegalPlayActionV1(card) for card in engine.legal_actions(state)))
    root_legal = legal_actions(belief.particles[0].state)
    if not root_legal:
        raise ValueError("no legal root action")
    root_ids = tuple(card.action_id for card in root_legal)
    for particle in belief.particles[1:]:
        ids = tuple(card.action_id for card in legal_actions(particle.state))
        if ids != root_ids:
            raise AssertionError("root legal actions differ across compatible hidden deals")

    requested_iterations = max(len(root_legal), int(iterations))
    rng = random.Random(int(seed))
    root_key = information_set_key(belief.particles[0].state)
    tree: dict[tuple[Any, ...], dict[int, _ActionStats]] = {
        root_key: {
            card.action_id: _ActionStats()
            for card in root_legal
        }
    }

    # Give every root action the same deterministic, belief-weighted hidden
    # cohort before adaptive tree selection begins. This prevents a move from
    # looking better merely because its first visits saw easier hidden deals.
    common_cohort_size = min(
        len(belief.particles),
        max(1, requested_iterations // len(root_legal)),
    )
    cohort_rng = random.Random(int(seed) ^ 0xB4100C)
    cohort_offset = cohort_rng.random() / common_cohort_size
    completed_simulations = 0
    for cohort_index in range(common_cohort_size):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("information-set search cancelled")
        point = cohort_offset + cohort_index / common_cohort_size
        cohort_state = _sample_particle_at(belief, point)
        for card in root_legal:
            branch = cohort_state.clone()
            engine.play_card(branch, card.card, ekka=card.ekka)
            reward = (
                terminal_margin(branch, info.observer.team)
                if branch.current_player is None
                else _rollout(
                    branch,
                    engine,
                    info.observer,
                    int(seed)
                    + cohort_index * 65_537
                    + card.action_id * 257, cancel_check,
                )
            )
            tree[root_key][card.action_id].add(reward)
            completed_simulations += 1

    for simulation in range(completed_simulations, requested_iterations):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("information-set search cancelled")
        state = _sample_particle(belief, rng)
        path: list[tuple[tuple[Any, ...], int]] = []
        expanded = False
        depth = 0
        while state.current_player is not None and depth < maximum_tree_depth:
            key = information_set_key(state)
            legal = legal_actions(state)
            node = tree.setdefault(key, {})
            for card in legal:
                node.setdefault(card.action_id, _ActionStats())
            unvisited = [card for card in legal if node[card.action_id].visits == 0]
            if unvisited:
                card = unvisited[rng.randrange(len(unvisited))]
                expanded = True
            else:
                parent_visits = max(1, sum(node[card.action_id].visits for card in legal))
                maximizing = state.current_player.team == info.observer.team
                def score(card: Card) -> tuple[float, int]:
                    stats = node[card.action_id]
                    # Normalize point margins for stable exploration while
                    # retaining raw margins in the reported values.
                    exploitation = max(-1.0, min(1.0, stats.mean / 200.0))
                    if not maximizing:
                        exploitation = -exploitation
                    bonus = float(exploration) * math.sqrt(math.log(parent_visits + 1.0) / stats.visits)
                    return exploitation + bonus, -card.action_id
                card = max(legal, key=score)
            path.append((key, card.action_id))
            engine.play_card(state, card.card, ekka=card.ekka)
            state.events = []
            depth += 1
            if expanded:
                break

        if state.current_player is None:
            reward = terminal_margin(state, info.observer.team)
        else:
            reward = _rollout(state, engine, info.observer, seed + simulation * 65_537 + depth, cancel_check)
        for key, card_id in path:
            tree[key][card_id].add(reward)

    root = tree[root_key]
    total_root_visits = sum(root[card.action_id].visits for card in root_legal)
    actions: list[ISMCTSAction] = []
    for card in root_legal:
        stats = root[card.action_id]
        if stats.visits <= 0:
            continue
        stderr = math.sqrt(stats.variance / stats.visits) if stats.visits > 1 else 0.0
        half = 1.96 * stderr
        actions.append(ISMCTSAction(
            card.card,
            stats.visits,
            stats.visits / max(1, total_root_visits),
            stats.mean,
            stats.positive / stats.visits,
            stats.mean - half,
            stats.mean + half,
            card.ekka,
        ))
    actions.sort(key=lambda item: (item.expected_round_margin, item.positive_margin_probability, item.visits), reverse=True)
    if not actions:
        raise RuntimeError("information-set search did not evaluate a root action")
    return ISMCTSResult(
        info.observer, actions[0].card, tuple(actions), belief,
        requested_iterations, len(tree), int(seed), common_cohort_size,
        method="information_set_mcts_actions_v2" if include_ekka else "information_set_mcts_v1",
        recommended_ekka=actions[0].ekka,
    )
