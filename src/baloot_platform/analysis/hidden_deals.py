from __future__ import annotations

import itertools
import math
from functools import lru_cache
from dataclasses import dataclass
from typing import Callable

from baloot_platform.core.cards import FULL_DECK, Card, cards_to_mask
from baloot_platform.core.enums import Contract, GamePhase, Seat, Suit
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.models import BidRecord, RoundState, Trick
from baloot_platform.core.shuffle import SplitMix64
from baloot_platform.core.projects import detect_projects, project_card_patterns
from baloot_platform.core.project_visibility import project_rank_visibility_policy
from .deductions import derive_hard_deductions, void_map
from .models import InformationState


@dataclass(frozen=True, slots=True)
class HiddenDealSet:
    states: tuple[RoundState, ...]
    method: str
    attempted: int
    coverage_complete: bool = False


def _should_stop_hidden_deal_generation(
    method: str,
    *,
    accepted_count: int,
    requested_samples: int,
) -> bool:
    """Bound sampled cohorts without truncating any exhaustive method."""

    return (
        not str(method).startswith("exhaustive_")
        and int(accepted_count) >= int(requested_samples)
    )


def _public_bid_history(info: InformationState) -> list[BidRecord]:
    return [
        BidRecord(
            int(item["round"]),
            int(item["turn"]),
            Seat.parse(item["seat"]),
            str(item["action"]),
            Contract.parse(item["contract"]) if item.get("contract") else None,
            Suit.parse(item["trumpSuit"]) if item.get("trumpSuit") else None,
            Seat.parse(item["bidder"]) if item.get("bidder") else None,
            Seat.parse(item["buyer"]) if item.get("buyer") else None,
        )
        for item in info.bid_history
    ]


def allocation_count(total: int, sizes: list[int]) -> int:
    remaining = total
    result = 1
    for size in sizes[:-1]:
        result *= math.comb(remaining, size)
        remaining -= size
    return result


def _public_project_signature(state: RoundState, observer: Seat) -> tuple[tuple, ...]:
    return _information_project_signature(InformationState.from_complete(state, observer))


def _information_project_signature(info: InformationState) -> tuple[tuple, ...]:
    return tuple(sorted((
        int(project.owner),
        project.kind,
        project.value,
        project.high_rank if project.high_rank is not None else -1,
        project.declared,
        project.scoring,
        project.revealed,
        tuple(sorted(card.id for card in project.cards)) if project.revealed else (),
    ) for project in info.projects))


def _build_start_reference(
    info: InformationState,
    current_hands: dict[Seat, tuple[Card, ...]],
    engine: BalootEngine,
) -> RoundState:
    state = RoundState(info.rules_hash, 0, 1, info.dealer, match_score=list(info.match_score),
                       project_rank_visibility=project_rank_visibility_policy(engine.rules))
    state.contract = info.contract
    state.bid_history = _public_bid_history(info)
    state.buyer = info.buyer
    state.declarer = info.declarer
    state.purchase_card = info.purchase_card
    state.trump_suit = info.trump_suit
    state.escalation = info.escalation
    state.escalation_history = [dict(x) for x in info.escalation_history]
    state.locked = info.locked
    state.first_leader = info.first_leader
    state.current_player = info.first_leader
    state.phase = GamePhase.PLAYING
    played_by_seat = {seat: [] for seat in Seat}
    for play in info.plays:
        played_by_seat[play.seat].append(play.card)
    initial = []
    initial_cards: dict[Seat, tuple[Card, ...]] = {}
    for seat in Seat:
        cards = (*current_hands[seat], *played_by_seat[seat])
        initial_cards[seat] = tuple(cards)
        initial.append(cards_to_mask(cards))
    state.initial_hands = list(initial)
    state.hands = list(initial)
    state.projects = detect_projects(initial_cards, info.contract, engine.rules, info.dealer)
    state.current_trick = Trick(0, info.first_leader)

    # Replay the entire public history under this hypothetical deal. The
    # authoritative state machine regenerates legal alternatives, declarations,
    # project comparison/reveal timing, Baloot, trick winners, and points.
    for public in info.plays:
        if state.current_player != public.seat:
            raise ValueError("public turn sequence is inconsistent")
        engine.play_card(state, public.card, ekka=public.ekka)

    # Mandatory declarations may already have happened on the current turn,
    # before its card; replaying cards alone does not reproduce that boundary.
    if (len(info.plays) < 4 and engine.project_domain.declaration_mode == "mandatory"
            and any(project.owner == state.current_player and project.declared for project in info.projects)):
        engine._announce_project_for_turn(state, state.current_player)

    state.match_score = list(info.match_score)
    if state.current_player != info.current_player:
        raise ValueError("hypothesis does not reach the expected current player")
    if _public_project_signature(state, info.observer) != _information_project_signature(info):
        raise ValueError("hypothesis does not reproduce the public project history")
    if tuple(sorted(state.baloot_teams, key=int)) != info.baloot_teams:
        raise ValueError("hypothesis does not reproduce public Baloot declarations")
    if state.trump_broken != info.trump_broken:
        raise ValueError("hypothesis does not reproduce trump-break state")
    return state


def _build_start(
    info: InformationState,
    current_hands: dict[Seat, tuple[Card, ...]],
    engine: BalootEngine,
) -> RoundState:
    """Reconstruct a compatible state with bounded authoritative replay.

    The first two tricks remain engine-replayed because they govern project
    declarations, reveals, and included Baloot. After that public trick facts
    are already frozen, so rebuilding them directly avoids replaying the same
    long history for every hidden-deal particle.
    """
    if len(info.plays) <= 8:
        return _build_start_reference(info, current_hands, engine)

    state = RoundState(
        info.rules_hash,
        0,
        1,
        info.dealer,
        match_score=list(info.match_score),
        project_rank_visibility=project_rank_visibility_policy(engine.rules),
    )
    state.contract = info.contract
    state.bid_history = _public_bid_history(info)
    state.buyer = info.buyer
    state.declarer = info.declarer
    state.purchase_card = info.purchase_card
    state.trump_suit = info.trump_suit
    state.escalation = info.escalation
    state.escalation_history = [dict(item) for item in info.escalation_history]
    state.locked = info.locked
    state.first_leader = info.first_leader
    state.current_player = info.first_leader
    state.phase = GamePhase.PLAYING
    played_by_seat = {seat: [] for seat in Seat}
    for play in info.plays:
        played_by_seat[play.seat].append(play.card)
    initial_cards = {
        seat: (*current_hands[seat], *played_by_seat[seat])
        for seat in Seat
    }
    state.initial_hands = [
        cards_to_mask(initial_cards[seat])
        for seat in Seat
    ]
    state.hands = list(state.initial_hands)
    state.projects = detect_projects(
        initial_cards,
        info.contract,
        engine.rules,
        info.dealer,
    )
    state.current_trick = Trick(0, info.first_leader)

    for public in info.plays[:8]:
        if state.current_player != public.seat:
            raise ValueError("public turn sequence is inconsistent")
        engine.play_card(state, public.card, ekka=public.ekka, emit_events=False)

    remaining = info.plays[8:]
    complete_count = len(remaining) // 4
    for offset in range(complete_count):
        plays = list(remaining[offset * 4 : offset * 4 + 4])
        index = 2 + offset
        trick = Trick(index, plays[0].seat, plays=plays)
        trick.winner = engine._trick_winner(trick, state)
        trick.raw_points = sum(
            engine.rules.points(play.card, info.contract, info.trump_suit)
            for play in plays
        )
        state.completed_tricks.append(trick)

    partial = list(remaining[complete_count * 4 :])
    if partial:
        state.current_trick = Trick(
            2 + complete_count,
            partial[0].seat,
            plays=partial,
        )
    else:
        previous = state.completed_tricks[-1]
        if previous.winner is None:
            raise ValueError("completed public trick is missing its winner")
        state.current_trick = Trick(2 + complete_count, previous.winner)
    state.hands = [cards_to_mask(current_hands[seat]) for seat in Seat]
    state.current_player = info.current_player
    state.match_score = list(info.match_score)
    state.baloot_teams = set(info.baloot_teams)
    state.trump_broken = info.trump_broken
    state.events = []
    state.validate_conservation()
    if _public_project_signature(state, info.observer) != _information_project_signature(info):
        raise ValueError("hypothesis does not reproduce the public project history")
    if tuple(sorted(state.baloot_teams, key=int)) != info.baloot_teams:
        raise ValueError("hypothesis does not reproduce public Baloot declarations")
    return state


def _passes_voids(hands: dict[Seat, tuple[Card, ...]], voids: dict) -> bool:
    for seat, suits in voids.items():
        if any(c.suit in suits for c in hands[seat]):
            return False
    return True


def _constraint_allocator(
    cards: tuple[Card, ...],
    seats: list[Seat],
    sizes: list[int],
    voids: dict,
    *,
    forced_owners: dict[int, Seat] | None = None,
):
    """Return a uniform constrained allocation counter/sampler.

    Cards are ordered by the number of eligible seats. Dynamic programming
    counts feasible completions for each remaining-capacity state. Sampling a
    seat in proportion to the completion count produces a uniform allocation
    over all assignments satisfying the hard suit-void constraints, avoiding
    the large waste of blind shuffle-and-reject sampling.
    """
    seat_index = {seat: index for index, seat in enumerate(seats)}
    ordered = tuple(sorted(
        cards,
        key=lambda card: (sum(card.suit not in voids.get(seat, set()) for seat in seats), card.id),
    ))
    forced_owners = forced_owners or {}
    allowed = tuple(tuple(
        seat_index[seat]
        for seat in seats
        if card.suit not in voids.get(seat, set())
        and (card.id not in forced_owners or forced_owners[card.id] == seat)
    ) for card in ordered)

    @lru_cache(maxsize=None)
    def count(index: int, capacities: tuple[int, ...]) -> int:
        if index == len(ordered):
            return int(all(value == 0 for value in capacities))
        remaining = len(ordered) - index
        if sum(capacities) != remaining or any(value < 0 for value in capacities):
            return 0
        total = 0
        for position in allowed[index]:
            if capacities[position] <= 0:
                continue
            next_caps = list(capacities)
            next_caps[position] -= 1
            total += count(index + 1, tuple(next_caps))
        return total

    initial = tuple(int(value) for value in sizes)

    def enumerate_allocations():
        hands: list[list[Card]] = [[] for _ in seats]
        def visit(index: int, capacities: tuple[int, ...]):
            if index == len(ordered):
                if all(value == 0 for value in capacities):
                    yield {seat: tuple(hands[position]) for position, seat in enumerate(seats)}
                return
            card = ordered[index]
            for position in allowed[index]:
                if capacities[position] <= 0:
                    continue
                next_caps = list(capacities)
                next_caps[position] -= 1
                next_tuple = tuple(next_caps)
                if count(index + 1, next_tuple) <= 0:
                    continue
                hands[position].append(card)
                yield from visit(index + 1, next_tuple)
                hands[position].pop()
        yield from visit(0, initial)

    def sample(rng: SplitMix64) -> dict[Seat, tuple[Card, ...]]:
        capacities = initial
        hands: list[list[Card]] = [[] for _ in seats]
        for index, card in enumerate(ordered):
            options: list[tuple[int, int, tuple[int, ...]]] = []
            total = 0
            for position in allowed[index]:
                if capacities[position] <= 0:
                    continue
                next_caps = list(capacities)
                next_caps[position] -= 1
                next_tuple = tuple(next_caps)
                completions = count(index + 1, next_tuple)
                if completions > 0:
                    options.append((position, completions, next_tuple))
                    total += completions
            if total <= 0:
                raise RuntimeError("hard deductions leave no feasible hidden allocation")
            draw = rng.randbelow(total)
            cursor = 0
            selected = options[-1]
            for option in options:
                cursor += option[1]
                if draw < cursor:
                    selected = option
                    break
            position, _completions, capacities = selected
            hands[position].append(card)
        return {seat: tuple(hands[position]) for position, seat in enumerate(seats)}

    return count(0, initial), enumerate_allocations, sample


def _project_forced_constraints(
    info: InformationState,
    engine: BalootEngine,
    unknown: tuple[Card, ...],
    seats: list[Seat],
    sizes: list[int],
) -> tuple[dict[int, Seat], ...]:
    """Build alternative hidden-card constraints from public project facts only.

    An unrevealed declaration identifies the project kind, but its rank may
    also be hidden. Each legal rule-derived pattern becomes an alternative
    constraint.  The trusted hidden hand is never consulted.  The full engine
    replay in ``_build_start`` remains the exact authority for declaration
    order, comparison, scoring, reveal timing, and every other interaction.
    """

    if not info.projects:
        return ({},)

    unknown_mask = cards_to_mask(unknown)
    played_masks = {seat: 0 for seat in Seat}
    for play in info.plays:
        played_masks[play.seat] |= 1 << play.card.id
    known_initial = {
        seat: info.known_unplayed[int(seat)] | played_masks[seat]
        for seat in Seat
    }
    known_elsewhere = {
        seat: cards_to_mask(
            card
            for other in Seat
            if other != seat
            for card in FULL_DECK
            if known_initial[other] & (1 << card.id)
        )
        for seat in Seat
    }
    capacity = {seat: sizes[index] for index, seat in enumerate(seats)}

    alternatives_by_owner: dict[Seat, list[tuple[int, ...]]] = {}
    for project in info.projects:
        patterns = (
            (cards_to_mask(project.cards),)
            if project.revealed
            else project_card_patterns(
                kind=project.kind,
                value=project.value,
                high_rank=project.high_rank,
                contract=info.contract,
                rules=engine.rules,
            )
        )
        viable: list[int] = []
        for pattern in patterns:
            if pattern & known_elsewhere[project.owner]:
                continue
            forced = pattern & unknown_mask
            if project.owner == info.observer:
                if forced:
                    continue
            elif forced.bit_count() > capacity[project.owner]:
                continue
            viable.append(pattern)
        if not viable:
            return ()
        alternatives_by_owner.setdefault(project.owner, []).append(tuple(sorted(set(viable))))

    owner_masks: dict[Seat, tuple[int, ...]] = {}
    for owner, requirements in alternatives_by_owner.items():
        combined: set[int] = set()
        for selected in itertools.product(*requirements):
            mask = 0
            for pattern in selected:
                mask |= pattern
            forced = mask & unknown_mask
            if owner == info.observer:
                if forced:
                    continue
            elif forced.bit_count() > capacity[owner]:
                continue
            combined.add(forced)
        if not combined:
            return ()
        owner_masks[owner] = tuple(sorted(combined))

    owners = tuple(sorted(owner_masks, key=int))
    constraints: list[dict[int, Seat]] = []
    for selected in itertools.product(*(owner_masks[owner] for owner in owners)):
        forced_owners: dict[int, Seat] = {}
        valid = True
        for owner, mask in zip(owners, selected, strict=True):
            for card in FULL_DECK:
                if not (mask & (1 << card.id)):
                    continue
                previous = forced_owners.get(card.id)
                if previous is not None and previous != owner:
                    valid = False
                    break
                forced_owners[card.id] = owner
            if not valid:
                break
        if valid:
            constraints.append(forced_owners)

    unique = {
        tuple(sorted((card_id, int(owner)) for card_id, owner in item.items())): item
        for item in constraints
    }
    return tuple(unique[key] for key in sorted(unique))


def _constraint_matches_allocation(
    constraint: dict[int, Seat],
    allocation: dict[Seat, tuple[Card, ...]],
) -> bool:
    owner_by_card = {
        card.id: seat
        for seat, cards in allocation.items()
        for card in cards
    }
    return all(owner_by_card.get(card_id) == owner for card_id, owner in constraint.items())


def generate_hidden_deals(
    info: InformationState,
    engine: BalootEngine,
    *,
    requested_samples: int,
    exact_allocation_limit: int,
    maximum_attempts: int,
    seed: int,
    cancel_check: Callable[[], bool] | None = None,
) -> HiddenDealSet:
    if cancel_check is not None and cancel_check():
        raise InterruptedError("hidden-deal generation cancelled")
    unknown = tuple(c for c in FULL_DECK if not (info.known_cards_mask & (1 << c.id)))
    seats = [seat for seat in Seat if seat != info.observer]
    sizes = [info.hand_sizes[int(seat)] - info.known_unplayed[int(seat)].bit_count() for seat in seats]
    if any(size < 0 for size in sizes) or sum(sizes) != len(unknown):
        raise ValueError("information-state card counts are inconsistent")
    deductions = derive_hard_deductions(info)
    voids = void_map(deductions)
    constraints = _project_forced_constraints(info, engine, unknown, seats, sizes)
    if not constraints:
        raise RuntimeError("public project history leaves no feasible hidden allocation")
    branches = []
    for constraint in constraints:
        count, enumerate_branch, sample_branch = _constraint_allocator(
            unknown,
            seats,
            sizes,
            voids,
            forced_owners=constraint,
        )
        if count > 0:
            branches.append((constraint, count, enumerate_branch, sample_branch))
    if not branches:
        raise RuntimeError("public facts leave no feasible hidden allocation")
    feasible_count = sum(item[1] for item in branches)

    accepted: list[RoundState] = []
    attempts = 0
    if feasible_count <= max(1, int(exact_allocation_limit)):
        method = (
            "exhaustive_constrained_hidden_deals"
            if len(constraints) == 1 and not constraints[0]
            else "exhaustive_project_conditioned_hidden_deals"
        )
        def exact_candidates():
            seen: set[tuple[tuple[int, ...], ...]] = set()
            for _constraint, _count, enumerate_branch, _sample_branch in branches:
                for allocation in enumerate_branch():
                    signature = tuple(tuple(sorted(card.id for card in allocation[seat])) for seat in seats)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    yield allocation
        candidates = exact_candidates()
    else:
        rng = SplitMix64(seed)
        method = (
            "constructive_uniform_hidden_deals"
            if len(constraints) == 1 and not constraints[0]
            else "constructive_uniform_project_conditioned_hidden_deals"
        )
        limit = max(int(requested_samples), int(maximum_attempts))
        seen: set[tuple[tuple[int, ...], ...]] = set()
        def candidates_gen():
            for _ in range(limit):
                draw = rng.randbelow(feasible_count)
                cursor = 0
                branch = branches[-1]
                for option in branches:
                    cursor += option[1]
                    if draw < cursor:
                        branch = option
                        break
                allocation = branch[3](rng)
                multiplicity = sum(
                    _constraint_matches_allocation(constraint, allocation)
                    for constraint, _count, _enumerate, _sample in branches
                )
                if multiplicity <= 0:
                    raise AssertionError("sampled allocation matches no project constraint")
                # Weighted branch selection samples an allocation once per
                # matching project pattern.  The reciprocal acceptance step
                # removes that multiplicity and preserves a uniform union.
                if multiplicity > 1 and rng.randbelow(multiplicity) != 0:
                    continue
                signature = tuple(tuple(sorted(card.id for card in allocation[seat])) for seat in seats)
                if signature in seen:
                    continue
                seen.add(signature)
                yield allocation
        candidates = candidates_gen()

    for allocation in candidates:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("hidden-deal generation cancelled")
        attempts += 1
        hands: dict[Seat, tuple[Card, ...]] = {}
        for seat in Seat:
            fixed = tuple(c for c in FULL_DECK if info.known_unplayed[int(seat)] & (1 << c.id))
            if seat == info.observer:
                hands[seat] = fixed
            else:
                hands[seat] = (*fixed, *allocation[seat])
        # The constructive allocator already enforces void deductions. Keep the
        # explicit check as a defensive invariant.
        if not _passes_voids(hands, voids):
            raise AssertionError("constructive allocation violated a hard void deduction")
        try:
            accepted.append(_build_start(info, hands, engine))
        except (ValueError, AssertionError):
            continue
        if _should_stop_hidden_deal_generation(
            method,
            accepted_count=len(accepted),
            requested_samples=requested_samples,
        ):
            break
    if not accepted:
        raise RuntimeError("no legal hidden deals could reproduce the public history")
    return HiddenDealSet(
        tuple(accepted),
        method,
        attempts,
        coverage_complete=method.startswith("exhaustive_"),
    )
