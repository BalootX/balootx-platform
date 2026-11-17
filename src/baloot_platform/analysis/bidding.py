from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from baloot_platform.core.bidding import _choose_action
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.cards import Card, FULL_DECK, cards_to_mask, mask_to_cards
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat, Suit
from baloot_platform.core.models import RoundState
from baloot_platform.core.rules import RuleBook

from .models import BidChoice, BidChoiceEvaluation
from .search import SearchResult, heuristic_rollout_result
from .search_backend import ExactSearchBackend




@dataclass(frozen=True, slots=True)
class BidAnalysisResult:
    actor: Seat
    recommended: BidChoice
    choices: tuple[BidChoiceEvaluation, ...]
    method: str
    hidden_deals: int
    difficulty: int
    debate_score: int
    explanation: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisionKind": "bid",
            "actor": self.actor.name.lower(),
            "recommendedAction": self.recommended.to_dict(),
            "candidates": [item.to_dict() for item in self.choices],
            "method": self.method,
            "hiddenDeals": self.hidden_deals,
            "difficulty": self.difficulty,
            "debateScore": self.debate_score,
            "explanation": list(self.explanation),
        }


def legal_bid_choices(machine: BiddingMachine) -> tuple[BidChoice, ...]:
    choices: list[BidChoice] = []
    for action in machine.legal_actions():
        if action != "hokum":
            choices.append(BidChoice(action))
            continue
        if machine.round_index == 1:
            choices.append(BidChoice("hokum", machine.purchase_suit))
        else:
            choices.extend(BidChoice("hokum", suit) for suit in Suit if suit != machine.purchase_suit)
    return tuple(choices)


def _sample_state(state: RoundState, actor: Seat, rng: random.Random) -> RoundState:
    if state.purchase_card is None:
        raise ValueError("purchase card is required")
    actor_cards = tuple(state.hand(actor))
    known_ids = {card.id for card in actor_cards}
    known_ids.add(state.purchase_card.id)
    unknown = [card for card in FULL_DECK if card.id not in known_ids]
    rng.shuffle(unknown)

    five: dict[Seat, tuple[Card, ...]] = {actor: actor_cards}
    cursor = 0
    for seat in Seat:
        if seat == actor:
            continue
        five[seat] = tuple(unknown[cursor:cursor + 5])
        cursor += 5
    undealt = tuple(unknown[cursor:])
    if len(undealt) != 11:
        raise AssertionError("bidding sample must leave eleven undealt cards")

    sampled = state.clone()
    sampled.initial_five = [cards_to_mask(five[Seat(index)]) for index in range(4)]
    sampled.hands = list(sampled.initial_five)
    sampled.initial_hands = [0, 0, 0, 0]
    sampled.undealt = undealt
    sampled.bid_history = []
    sampled.contract = None
    sampled.buyer = None
    sampled.declarer = None
    sampled.trump_suit = None
    sampled.current_player = None
    sampled.current_trick = None
    sampled.completed_tricks = []
    sampled.projects = []
    sampled.round_score = None
    sampled.events = []
    return sampled


def _finish_auction_start(
    machine: BiddingMachine,
    sampled: RoundState,
    rules: RuleBook,
    first_choice: BidChoice,
) -> tuple[RoundState, bool]:
    result = machine.submit(first_choice.action, trump=first_choice.trump_suit)
    five_cards = {seat: mask_to_cards(sampled.initial_five[int(seat)]) for seat in Seat}
    while result is None:
        action, trump = _choose_action(machine, five_cards, sampled.purchase_card, rules)
        result = machine.submit(action, trump=trump)
    if result.all_passed:
        return sampled, False
    engine = BalootEngine(rules)
    engine.complete_deal_from_bid_result(sampled, result)
    engine.prepare_projects(sampled)
    return sampled, True


def _finish_auction(
    machine: BiddingMachine,
    sampled: RoundState,
    rules: RuleBook,
    first_choice: BidChoice,
) -> tuple[RoundState, bool]:
    sampled, has_contract = _finish_auction_start(machine, sampled, rules, first_choice)
    if not has_contract:
        return sampled, False
    engine = BalootEngine(rules)
    bot = HeuristicBot(engine, BotProfile(sampled.seed + 71))
    while sampled.current_player is not None:
        engine.play_card(sampled, bot.choose(sampled))
    return sampled, True


def _cvar(values: list[float], fraction: float = 0.10) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    count = max(1, math.ceil(len(ordered) * fraction))
    return sum(ordered[:count]) / count


def analyze_bid(
    state: RoundState,
    machine: BiddingMachine,
    actor: Seat,
    rules: RuleBook,
    *,
    requested_samples: int = 48,
    seed: int = 1,
    cancel_check: Callable[[], bool] | None = None,
    exact_search_backend: ExactSearchBackend | None = None,
) -> BidAnalysisResult:
    """Compare all legal buy/pass choices using one observer-safe deal cohort.

    Only the actor's five cards, the public purchase card, public score, auction
    history, and active rules are fixed. Partner/opponent five-card hands and
    the eleven undealt cards are resampled for every hidden deal. Every legal
    choice is then evaluated on that exact same cohort.
    """
    if machine.finished or machine.actor != actor:
        raise ValueError("bid analysis requires the current auction actor")
    choices = legal_bid_choices(machine)
    if len(choices) < 2:
        raise ValueError("at least two legal bid choices are required")

    samples = max(2, int(requested_samples))
    margins: dict[str, list[float]] = defaultdict(list)
    wins: dict[str, list[float]] = defaultdict(list)
    contract_success: dict[str, list[float]] = defaultdict(list)
    best_flags: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(int(seed))

    # Build every (hidden deal, legal choice) state in a stable sample-major
    # order.  Contracted play starts are then evaluated in one native call;
    # all-pass choices remain explicit neutral outcomes.
    contracted_states: list[RoundState] = []
    contracted_seeds: list[int] = []
    outcome_slots: list[tuple[int, BidChoice, RoundState | None]] = []
    engine = BalootEngine(rules)
    for sample_index in range(samples):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("bid analysis cancelled")
        sampled_base = _sample_state(state, actor, rng)
        for choice in choices:
            sampled = sampled_base.clone()
            simulated_machine = copy.deepcopy(machine)
            started, has_contract = _finish_auction_start(
                simulated_machine, sampled, rules, choice
            )
            if not has_contract:
                outcome_slots.append((sample_index, choice, None))
                continue
            outcome_slots.append((sample_index, choice, started))
            contracted_states.append(started)
            contracted_seeds.append(int(started.seed) + 71)

    batch_method = getattr(exact_search_backend, "heuristic_rollout_batch", None)
    compact_batch_method = getattr(
        exact_search_backend, "heuristic_rollout_batch_compact", None
    )
    if (
        getattr(exact_search_backend, "supports_compact_rollout", False)
        and callable(compact_batch_method)
    ):
        batch_method = compact_batch_method
    if batch_method is not None and contracted_states:
        contracted_outcomes = batch_method(
            contracted_states,
            tuple(None for _ in contracted_states),
            contracted_seeds,
            engine,
            actor.team,
            cancel_check=cancel_check,
        )
        method = "paired_observer_safe_bid_native_batch_v2"
    else:
        reference_outcomes: list[SearchResult] = []
        for started, rollout_seed in zip(
            contracted_states, contracted_seeds, strict=True
        ):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("bid analysis cancelled")
            reference_outcomes.append(
                heuristic_rollout_result(started, engine, actor.team, rollout_seed)
            )
        contracted_outcomes = tuple(reference_outcomes)
        method = "paired_observer_safe_bid_monte_carlo_v1"

    outcome_cursor = 0
    per_sample_values: list[dict[str, float]] = [dict() for _ in range(samples)]
    for sample_index, choice, started in outcome_slots:
        if started is None:
            margin = 0.0
            team_win = 0.5
            success = 0.5
        else:
            outcome: SearchResult = contracted_outcomes[outcome_cursor]
            outcome_cursor += 1
            margin = float(outcome.own_points - outcome.opponent_points)
            team_win = float(outcome.own_points > outcome.opponent_points) + 0.5 * float(
                outcome.own_points == outcome.opponent_points
            )
            assert started.buyer is not None
            success = float(
                outcome.contract_succeeded
                if actor.team == started.buyer.team
                else not outcome.contract_succeeded
            )
            if outcome.match_winner == actor.team:
                margin += 200.0
            elif outcome.match_winner == actor.team.other:
                margin -= 200.0
        margins[choice.id].append(margin)
        wins[choice.id].append(team_win)
        contract_success[choice.id].append(success)
        per_sample_values[sample_index][choice.id] = margin

    if outcome_cursor != len(contracted_outcomes):
        raise AssertionError("bid rollout reduction did not consume every outcome")

    for per_sample in per_sample_values:
        best = max(per_sample.values())
        for choice in choices:
            best_flags[choice.id].append(float(per_sample[choice.id] == best))

    evaluations: list[BidChoiceEvaluation] = []
    for choice in choices:
        values = margins[choice.id]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        evaluations.append(
            BidChoiceEvaluation(
                choice,
                mean,
                sum(wins[choice.id]) / len(values),
                sum(contract_success[choice.id]) / len(values),
                _cvar(values),
                math.sqrt(max(0.0, variance)),
                sum(best_flags[choice.id]) / len(values),
                len(values),
            )
        )
    evaluations.sort(
        key=lambda item: (item.expected_match_margin, item.team_win_probability, item.best_share),
        reverse=True,
    )
    best, second = evaluations[0], evaluations[1]
    margin_gap = best.expected_match_margin - second.expected_match_margin
    win_gap = best.team_win_probability - second.team_win_probability
    closeness = max(0.0, 1.0 - abs(margin_gap) / 25.0)
    difficulty = max(1, min(100, round(35 + 12 * (len(evaluations) - 2) + 35 * closeness)))
    debate = max(1, min(100, round(100 - min(90.0, abs(margin_gap) * 3.0 + abs(win_gap) * 120.0))))
    return BidAnalysisResult(
        actor,
        best.choice,
        tuple(evaluations),
        method,
        samples,
        difficulty,
        debate,
        (
            f"Compared {len(evaluations)} legal buy/pass choices on the same {samples} hidden deals.",
            "Only the actor's five cards, public purchase card, score, and public auction history were fixed.",
            "Partner and opponents' cards plus the eleven undealt cards were resampled for every hidden deal.",
            f"Recommended {best.choice.id} by expected match-aware round margin.",
        ),
    )
