"""Engine-free CP-SAT play-tail construction from the published play Domain IR."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from itertools import permutations
from typing import Any, Callable, Iterable, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, GamePhase, Rank, Seat, Suit, Team
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.play_ir import PlayDomainIRV1
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.core.structural_ir import StructuralDomainIRV1

from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1
from .scenario_contracts import canonical_sha256
from .ortools_play_state_predicates import (
    CpSatPlayPrefixActionV1,
    CpSatPlayStatePredicateContextV1,
)
from .scenario_predicate_lowering_v2 import (
    CpSatActionVariablesV1,
    CpSatPredicateModelContextV1,
    CpSatTrickVariablesV1,
)


_LOWERING_ID = "ortools_play_tail_v1"
NATIVE_COMPACT_PARTNER_FEED_POLICY_ID = "observer_safe_partner_feed_compact_v1"
NATIVE_ESTABLISHED_CONTROL_POLICY_ID = "observer_safe_partner_feed_established_control_v1"
NATIVE_SECURE_CONTROL_POLICY_ID = "observer_safe_partner_feed_secure_control_v1"
NATIVE_GUARD_PROTECTION_POLICY_ID = "observer_safe_partner_feed_guard_protection_v1"
NATIVE_CLOSED_TRICK_FEED_POLICY_ID = "observer_safe_partner_feed_closed_trick_v1"
NATIVE_PARTNER_FEED_POLICY_IDS = frozenset({
    "observer_safe_partner_feed_v1", NATIVE_COMPACT_PARTNER_FEED_POLICY_ID,
    NATIVE_ESTABLISHED_CONTROL_POLICY_ID, NATIVE_SECURE_CONTROL_POLICY_ID,
    NATIVE_GUARD_PROTECTION_POLICY_ID,
    NATIVE_CLOSED_TRICK_FEED_POLICY_ID,
})
NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS = frozenset({
    "observer_safe_trick_control_v1",
    "observer_safe_tactical_category_v1",
    *NATIVE_PARTNER_FEED_POLICY_IDS,
})


def native_observer_safe_policy_identity_v1(policy_id: str, team: Team) -> dict[str, Any]:
    if type(policy_id) is not str or policy_id not in NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS or not isinstance(team, Team):
        raise ValueError("native visible policy requires a team and supported policy")
    core = {"schemaVersion": 1, "policyId": policy_id, "team": team.name.lower(),
            "informationScope": "actor_visible", "tieBreak": "lowest_legal_action_id"}
    return {**core, "sha256": canonical_sha256(core)}


def _observer_safe_play_score_v1(
    strength: int,
    points: int,
    *,
    partner_winning: bool | None = None,
    takes_control: bool | None = None,
    feed_partner: bool = False,
    secure_control: bool = False,
) -> int:
    """Frozen actor-visible trick-control policy score for runtime and CP-SAT."""

    if type(feed_partner) is not bool or type(secure_control) is not bool:
        raise ValueError("policy feed selection must be boolean")
    if (partner_winning is None) != (takes_control is None):
        raise ValueError("policy score requires both trick-control inputs or neither")
    if partner_winning is None:
        return 20_000 + 100 * strength + points
    if partner_winning and not takes_control:
        return 40_000 + (100 * points if feed_partner else -100 * points) - strength
    if not partner_winning and takes_control:
        return 50_000 + (100 if secure_control else -100) * strength - points
    if not partner_winning:
        return 30_000 - 100 * points - strength
    return 10_000 - 100 * strength - points


def _compact_observed_card_score_v1(model, context, card, strength, points,
        contract, trump, name, comparisons, *, priority):
    """Native control predicates plus four score rows; no hidden-hand inputs."""
    suit, incumbent, partner = context
    for key, variable, rhs, is_suit in ((('suit', int(card.suit)), suit, int(card.suit), True),
                                      (('strength', strength), incumbent, strength, False)):
        if key not in comparisons:
            flag = model.new_bool_var(f"{name}_{key[0]}_{key[1]}")
            model.add(variable == rhs if is_suit else variable < rhs).only_enforce_if(flag)
            model.add(variable != rhs if is_suit else variable >= rhs).only_enforce_if(flag.Not())
            comparisons[key] = flag
    follows, stronger = comparisons[('suit', int(card.suit))], comparisons[('strength', strength)]
    control = model.new_bool_var(f"{name}_control_{card.id}")
    if contract.effective == Contract.HOKUM and card.suit == trump:
        same_suit_control = model.new_bool_var(f"{name}_same_suit_control_{card.id}")
        model.add_min_equality(same_suit_control, [follows, stronger])
        model.add_max_equality(control, [same_suit_control, follows.Not()])
    else:
        model.add_min_equality(control, [follows, stronger])
    rows = [(has_partner, takes_control, priority(bool(has_partner), bool(takes_control)))
            for has_partner in (0, 1) for takes_control in (0, 1)]
    bounds = min(row[-1] for row in rows), max(row[-1] for row in rows)
    value = model.new_int_var(*bounds, f"{name}_card_{card.id}")
    model.add_allowed_assignments([partner, control, value], rows)
    return value, bounds


def _established_control_blockers_v1(card, *, domain, contract, trump, cards):
    """Finite rule-table blockers, not a hidden-hand or qualification decision."""
    return tuple(other for other in cards if (
        other.suit == card.suit
        and domain.strength(other, contract, trump) > domain.strength(card, contract, trump)
        or contract.effective == Contract.HOKUM and card.suit != trump and other.suit == trump))


def _established_control_bonus_v1(domain, contract, trump):
    scores = [_observer_safe_play_score_v1(domain.strength(Card(card), contract, trump),
        domain.points(Card(card), contract, trump)) for card in range(32)]
    return max(scores) - min(scores) + 1


def _native_established_lead_scores_v1(model, *, pool, owners, actor, played,
        domain, contract, trump, name):
    """Certainty uses only own-card membership and publicly played cards."""
    known = {}
    for index, card in enumerate(pool):
        owned = model.new_bool_var(f"{name}_own_{card.id}")
        model.add(owners[index] == actor).only_enforce_if(owned)
        model.add(owners[index] != actor).only_enforce_if(owned.Not())
        known[card.id] = model.new_bool_var(f"{name}_known_{card.id}")
        model.add_max_equality(known[card.id], [owned, played[index]])
    bonus = _established_control_bonus_v1(domain, contract, trump)
    values, bounds, masters = [], [], []
    for card in pool:
        blockers = [known[other.id] for other in _established_control_blockers_v1(
            card, domain=domain, contract=contract, trump=trump, cards=pool)]
        master = model.new_bool_var(f"{name}_master_{card.id}")
        masters.append(master)
        model.add_bool_and(blockers).only_enforce_if(master)
        model.add_bool_or([flag.Not() for flag in blockers]).only_enforce_if(master.Not())
        base = _observer_safe_play_score_v1(domain.strength(card, contract, trump),
            domain.points(card, contract, trump))
        value = model.new_int_var(base, base + bonus, f"{name}_score_{card.id}")
        model.add(value == base + bonus * master)
        values.append(value)
        bounds.append((base, base + bonus))
    return values, bounds, masters


def _guard_discard_bonus_v1(domain, contract, trump):
    costs = [100 * domain.points(Card(card), contract, trump)
             + domain.strength(Card(card), contract, trump) for card in range(32)]
    return max(costs) - min(costs) + 1


def _guard_discard_cards_v1(*, own_cards, public_played, pool, domain, contract, trump):
    """Policy reference only: retain the sole low guard of a vulnerable point card."""
    protected = set()
    for suit in Suit:
        hand = sorted((card for card in own_cards if card.suit == suit),
                      key=lambda card: domain.strength(card, contract, trump))
        if len(hand) != 2 or not domain.points(hand[1], contract, trump):
            continue
        if any(card.suit == suit and card not in own_cards and card not in public_played
               and domain.strength(card, contract, trump) > domain.strength(hand[1], contract, trump)
               for card in pool):
            protected.add(hand[0])
    return protected


def _native_guard_discard_flags_v1(model, *, pool, owners, actor, played,
        led_suit, partner_winning, domain, contract, trump, name,
        owned_flags=None, remaining_flags=None):
    """Reified own-hand/public-history policy; never consult opponent ownership."""
    remaining, unseen = {}, {}
    for index, card in enumerate(pool):
        if owned_flags is None:
            owned = model.new_bool_var(f"{name}_owned_{card.id}")
            model.add(owners[index] == actor).only_enforce_if(owned)
            model.add(owners[index] != actor).only_enforce_if(owned.Not())
        else:
            owned = owned_flags[index]
        if remaining_flags is None:
            remaining[card.id] = model.new_bool_var(f"{name}_remaining_{card.id}")
            model.add_min_equality(remaining[card.id], [owned, played[index].Not()])
        else:
            remaining[card.id] = remaining_flags[index]
        unseen[card.id] = model.new_bool_var(f"{name}_unseen_{card.id}")
        model.add_min_equality(unseen[card.id], [owned.Not(), played[index].Not()])
    flags = {}
    for suit in Suit:
        cards = [card for card in pool if card.suit == suit]
        pair = model.new_bool_var(f"{name}_pair_{int(suit)}")
        model.add(sum(remaining[card.id] for card in cards) == 2).only_enforce_if(pair)
        model.add(sum(remaining[card.id] for card in cards) != 2).only_enforce_if(pair.Not())
        discard = model.new_bool_var(f"{name}_discard_{int(suit)}")
        model.add(led_suit != int(suit)).only_enforce_if(discard)
        model.add(led_suit == int(suit)).only_enforce_if(discard.Not())
        top = model.new_int_var(-1, 7, f"{name}_top_{int(suit)}")
        unknown_top = model.new_int_var(-1, 7, f"{name}_unknown_top_{int(suit)}")
        model.add_max_equality(top, [(domain.strength(card, contract, trump) + 1)
                                    * remaining[card.id] - 1 for card in cards] or [-1])
        model.add_max_equality(unknown_top, [(domain.strength(card, contract, trump) + 1)
                                            * unseen[card.id] - 1 for card in cards] or [-1])
        valuable = model.new_bool_var(f"{name}_valuable_{int(suit)}")
        model.add_allowed_assignments([top, valuable], [(-1, 0), *[
            (domain.strength(card, contract, trump), int(domain.points(card, contract, trump) > 0))
            for card in cards]])
        vulnerable = model.new_bool_var(f"{name}_vulnerable_{int(suit)}")
        model.add(unknown_top > top).only_enforce_if(vulnerable)
        model.add(unknown_top <= top).only_enforce_if(vulnerable.Not())
        for card in cards:
            flags[card.id] = model.new_bool_var(f"{name}_guard_{card.id}")
            if contract.effective != Contract.HOKUM or suit != trump:
                low = model.new_bool_var(f"{name}_low_{card.id}")
                model.add(top > domain.strength(card, contract, trump)).only_enforce_if(low)
                model.add(top <= domain.strength(card, contract, trump)).only_enforce_if(low.Not())
                model.add_min_equality(flags[card.id], [remaining[card.id], low, pair,
                    discard, partner_winning.Not(), valuable, vulnerable])
            else:
                model.add(flags[card.id] == 0)
    return [flags[card.id] for card in pool]


def _enum_values(values: Iterable[Any] | None, enum_type: type) -> tuple[Any, ...]:
    if values is None:
        return ()
    result = tuple(
        enum_type(value) if isinstance(value, int) else enum_type.parse(value)
        for value in values
    )
    if len(set(result)) != len(result):
        raise ValueError("selector values must be unique")
    return tuple(sorted(result, key=int))


@dataclass(frozen=True, slots=True)
class TailPrefixActionV1:
    actor: Seat
    card: Card
    ekka: bool = False

    @classmethod
    def build(
        cls,
        actor: Seat | str | int,
        card: Card | str | int | dict,
        *,
        ekka: bool = False,
    ) -> "TailPrefixActionV1":
        if type(ekka) is not bool:
            raise TypeError("ekka must be a boolean")
        return cls(Seat.parse(actor), Card.parse(card), ekka)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.name.lower(),
            "card": self.card.code,
            "ekka": self.ekka,
        }


@dataclass(frozen=True, slots=True)
class PlayZeroObservationActionNoGoodV1:
    """One rejected visible opening hand/action pair."""

    actor: Seat
    actor_card_ids: tuple[int, ...]
    selected_card_id: int
    ekka: bool

    @classmethod
    def build(
        cls,
        *,
        actor: Seat | str | int,
        actor_cards: Iterable[Card | str | int | dict],
        selected_card: Card | str | int | dict,
        ekka: bool = False,
    ) -> "PlayZeroObservationActionNoGoodV1":
        if type(ekka) is not bool:
            raise TypeError("play-zero no-good Ekka must be a boolean")
        seat = Seat.parse(actor)
        card_ids = tuple(sorted(Card.parse(card).id for card in actor_cards))
        selected = Card.parse(selected_card).id
        if len(card_ids) != 8 or len(set(card_ids)) != 8:
            raise ValueError("play-zero no-good requires one eight-card actor hand")
        if selected not in card_ids:
            raise ValueError("play-zero no-good action must belong to the actor hand")
        return cls(seat, card_ids, selected, ekka)


def apply_play_zero_observation_action_no_good_v1(
    model: cp_model.CpModel,
    *,
    owner_of_card: Sequence[cp_model.IntVar],
    selected_card: cp_model.IntVar,
    ekka_lead: cp_model.IntVar,
    no_good: PlayZeroObservationActionNoGoodV1,
    name: str,
) -> None:
    """Forbid exactly the same actor-visible opening hand and chosen action."""

    if len(owner_of_card) != 32:
        raise ValueError("play-zero no-good requires 32 ownership variables")
    actor_cards = set(no_good.actor_card_ids)
    membership = tuple(
        model.new_bool_var(f"{name}_actor_owns_{card_id}")
        for card_id in range(32)
    )
    for card_id, owns in enumerate(membership):
        model.add(owner_of_card[card_id] == int(no_good.actor)).only_enforce_if(owns)
        model.add(owner_of_card[card_id] != int(no_good.actor)).only_enforce_if(
            owns.negated()
        )
    model.add_forbidden_assignments(
        (*membership, selected_card, ekka_lead),
        (
            (
                *(int(card_id in actor_cards) for card_id in range(32)),
                no_good.selected_card_id,
                int(no_good.ekka),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class TailActionConstraintV1:
    step: int
    actor: Seat | None = None
    team: Team | None = None
    card_ids: tuple[int, ...] = ()
    ranks: tuple[Rank, ...] = ()
    suits: tuple[Suit, ...] = ()
    actor_team_wins_trick: bool | None = None

    @classmethod
    def build(
        cls,
        *,
        step: int,
        actor: Seat | str | int | None = None,
        team: Team | str | int | None = None,
        cards: Iterable[Card | str | int | dict] | None = None,
        ranks: Iterable[Rank | str | int] | None = None,
        suits: Iterable[Suit | str | int] | None = None,
        actor_team_wins_trick: bool | None = None,
    ) -> "TailActionConstraintV1":
        if type(step) is not int or step < 0:
            raise ValueError("action constraint step must be a non-negative integer")
        if (
            actor_team_wins_trick is not None
            and type(actor_team_wins_trick) is not bool
        ):
            raise TypeError("actor_team_wins_trick must be a boolean")
        card_ids = tuple(sorted({Card.parse(card).id for card in cards or ()}))
        return cls(
            step,
            Seat.parse(actor) if actor is not None else None,
            Team(int(team))
            if team is not None and not isinstance(team, str)
            else (Team[str(team).strip().upper()] if isinstance(team, str) else None),
            card_ids,
            _enum_values(ranks, Rank),
            _enum_values(suits, Suit),
            actor_team_wins_trick,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "actor": self.actor.name.lower() if self.actor is not None else None,
            "team": self.team.name.lower() if self.team is not None else None,
            "cardIds": list(self.card_ids),
            "ranks": [rank.name.lower() for rank in self.ranks],
            "suits": [suit.name.lower() for suit in self.suits],
            "actorTeamWinsTrick": self.actor_team_wins_trick,
        }


@dataclass(frozen=True, slots=True)
class TailTrickWinnerConstraintV1:
    trick: int
    seat: Seat | None = None
    team: Team | None = None
    card_ids: tuple[int, ...] = ()
    ranks: tuple[Rank, ...] = ()
    suits: tuple[Suit, ...] = ()

    @classmethod
    def build(
        cls,
        *,
        trick: int,
        seat: Seat | str | int | None = None,
        team: Team | str | int | None = None,
        cards: Iterable[Card | str | int | dict] | None = None,
        ranks: Iterable[Rank | str | int] | None = None,
        suits: Iterable[Suit | str | int] | None = None,
    ) -> "TailTrickWinnerConstraintV1":
        if type(trick) is not int or trick < 0:
            raise ValueError("winner constraint trick must be a non-negative integer")
        parsed_team = (
            Team[str(team).strip().upper()]
            if isinstance(team, str)
            else Team(int(team))
            if team is not None
            else None
        )
        return cls(
            trick,
            Seat.parse(seat) if seat is not None else None,
            parsed_team,
            tuple(sorted({Card.parse(card).id for card in cards or ()})),
            _enum_values(ranks, Rank),
            _enum_values(suits, Suit),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trick": self.trick,
            "seat": self.seat.name.lower() if self.seat is not None else None,
            "team": self.team.name.lower() if self.team is not None else None,
            "cardIds": list(self.card_ids),
            "ranks": [rank.name.lower() for rank in self.ranks],
            "suits": [suit.name.lower() for suit in self.suits],
        }


@dataclass(frozen=True, slots=True)
class TailTrickCardCountConstraintV1:
    trick: int
    minimum: int
    maximum: int
    owner_team: Team | None = None
    card_ids: tuple[int, ...] = ()
    ranks: tuple[Rank, ...] = ()
    suits: tuple[Suit, ...] = ()

    @classmethod
    def build(
        cls,
        *,
        trick: int,
        minimum: int,
        maximum: int,
        owner_team: Team | str | int | None = None,
        cards: Iterable[Card | str | int | dict] | None = None,
        ranks: Iterable[Rank | str | int] | None = None,
        suits: Iterable[Suit | str | int] | None = None,
    ) -> "TailTrickCardCountConstraintV1":
        if type(trick) is not int or trick < 0:
            raise ValueError("card-count trick must be a non-negative integer")
        if (
            type(minimum) is not int
            or type(maximum) is not int
            or not 0 <= minimum <= maximum <= 4
        ):
            raise ValueError(
                "card-count range must satisfy 0 <= minimum <= maximum <= 4"
            )
        parsed_team = (
            Team[str(owner_team).strip().upper()]
            if isinstance(owner_team, str)
            else Team(int(owner_team))
            if owner_team is not None
            else None
        )
        return cls(
            trick,
            minimum,
            maximum,
            parsed_team,
            tuple(sorted({Card.parse(card).id for card in cards or ()})),
            _enum_values(ranks, Rank),
            _enum_values(suits, Suit),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trick": self.trick,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "ownerTeam": self.owner_team.name.lower()
            if self.owner_team is not None
            else None,
            "cardIds": list(self.card_ids),
            "ranks": [rank.name.lower() for rank in self.ranks],
            "suits": [suit.name.lower() for suit in self.suits],
        }


@dataclass(frozen=True, slots=True)
class OrToolsPlayTailResultV1:
    initial_hands: tuple[tuple[Card, ...], ...]
    plays: tuple[TailPrefixActionV1, ...]
    trick_winners: tuple[Seat, ...]
    trick_winning_cards: tuple[Card, ...]
    trick_raw_points: tuple[int, ...]
    prefix_length: int
    remaining_hands: tuple[tuple[Card, ...], ...]
    current_trick: tuple[TailPrefixActionV1, ...]
    next_actor: Seat | None
    legal_actions: tuple[Card, ...]
    legal_play_actions: tuple[TailPrefixActionV1, ...]
    legal_actions_by_step: tuple[tuple[Card, ...], ...]
    legal_play_actions_by_step: tuple[tuple[TailPrefixActionV1, ...], ...]
    raw_team_points: tuple[int, int] | None
    card_game_points: tuple[int, int] | None
    scoring_domain_hash: str | None
    model_hash: str
    evidence_hash: str
    solver_status: str
    sha256: str
    variable_count: int = 0
    constraint_count: int = 0

    def _prefix_fields(self, prefix_length: int):
        """Project solved state fields without constructing an evidence document."""
        if type(prefix_length) is not int or not 0 <= prefix_length <= len(self.plays):
            raise ValueError("prefix_length must lie inside the solved play trace")
        if prefix_length == self.prefix_length:
            return (self.remaining_hands, self.current_trick, self.next_actor,
                    self.legal_actions, self.legal_play_actions)
        played = {play.card.id for play in self.plays[:prefix_length]}
        remaining_hands = tuple(
            tuple(card for card in hand if card.id not in played)
            for hand in self.initial_hands
        )
        current_count = prefix_length % 4
        current_trick = (
            self.plays[prefix_length - current_count : prefix_length]
            if current_count
            else ()
        )
        next_actor = (
            self.plays[prefix_length].actor if prefix_length < len(self.plays) else None
        )
        legal_actions = (
            self.legal_actions_by_step[prefix_length]
            if prefix_length < len(self.plays)
            else ()
        )
        legal_play_actions = (
            self.legal_play_actions_by_step[prefix_length]
            if prefix_length < len(self.plays)
            else ()
        )
        return remaining_hands, current_trick, next_actor, legal_actions, legal_play_actions

    def at_prefix(self, prefix_length: int) -> "OrToolsPlayTailResultV1":
        """Project the solved complete trace to a decision without solving it again."""
        remaining_hands, current_trick, next_actor, legal_actions, legal_play_actions = self._prefix_fields(prefix_length)
        if prefix_length == self.prefix_length:
            return self
        evidence_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "sourceEvidenceHash": self.evidence_hash,
                "prefixLength": prefix_length,
                "remainingHands": [
                    [card.code for card in hand] for hand in remaining_hands
                ],
                "currentTrick": [play.to_dict() for play in current_trick],
                "nextActor": next_actor.name.lower()
                if next_actor is not None
                else None,
                "legalActions": [card.code for card in legal_actions],
                "legalPlayActions": [action.to_dict() for action in legal_play_actions],
            }
        )
        semantic_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "modelHash": self.model_hash,
                "evidenceHash": evidence_hash,
                "solverStatus": self.solver_status,
            }
        )
        return OrToolsPlayTailResultV1(
            self.initial_hands,
            self.plays,
            self.trick_winners,
            self.trick_winning_cards,
            self.trick_raw_points,
            prefix_length,
            remaining_hands,
            current_trick,
            next_actor,
            legal_actions,
            legal_play_actions,
            self.legal_actions_by_step,
            self.legal_play_actions_by_step,
            self.raw_team_points,
            self.card_game_points,
            self.scoring_domain_hash,
            self.model_hash,
            evidence_hash,
            self.solver_status,
            semantic_hash,
            self.variable_count,
            self.constraint_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "initialHands": {
                seat.name.lower(): [card.code for card in self.initial_hands[int(seat)]]
                for seat in Seat
            },
            "plays": [play.to_dict() for play in self.plays],
            "legalActionsByStep": [
                [card.code for card in actions]
                for actions in self.legal_actions_by_step
            ],
            "legalPlayActionsByStep": [
                [action.to_dict() for action in actions]
                for actions in self.legal_play_actions_by_step
            ],
            "tricks": [
                {
                    "index": index,
                    "winner": self.trick_winners[index].name.lower(),
                    "winningCard": self.trick_winning_cards[index].code,
                    "rawPoints": self.trick_raw_points[index],
                }
                for index in range(len(self.trick_winners))
            ],
            "decisionState": {
                "prefixLength": self.prefix_length,
                "remainingHands": {
                    seat.name.lower(): [
                        card.code for card in self.remaining_hands[int(seat)]
                    ]
                    for seat in Seat
                },
                "currentTrick": [play.to_dict() for play in self.current_trick],
                "nextActor": self.next_actor.name.lower()
                if self.next_actor is not None
                else None,
                "legalActions": [card.code for card in self.legal_actions],
                "legalPlayActions": [
                    action.to_dict() for action in self.legal_play_actions
                ],
            },
            "score": (
                {
                    "rawTeamPoints": list(self.raw_team_points),
                    "cardGamePoints": list(self.card_game_points),
                    "scoringDomainHash": self.scoring_domain_hash,
                }
                if self.raw_team_points is not None
                and self.card_game_points is not None
                else None
            ),
            "modelHash": self.model_hash,
            "evidenceHash": self.evidence_hash,
            "solverStatus": self.solver_status,
            "variableCount": self.variable_count,
            "constraintCount": self.constraint_count,
            "sha256": self.sha256,
        }


class PlayTailSolveStopped(ValueError):
    """Report a terminal CP-SAT solve status without masking input errors."""

    def __init__(
        self,
        solver_status: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        self.solver_status = solver_status
        core = {
            "schemaVersion": 1,
            "status": "incomplete",
            "stage": "play_tail_cp_sat",
            "exhaustionReason": solver_status,
            "evidence": dict(evidence or {}),
        }
        self.partial_receipt = {**core, "sha256": canonical_sha256(core)}
        super().__init__(
            f"play-tail model did not produce a legal witness: {solver_status}"
        )


@dataclass(frozen=True, slots=True)
class OrToolsPlayTailBuildV1:
    """Live tail model exposed for exact predicate lowering before one solve."""

    model: cp_model.CpModel
    play_state_context: CpSatPlayStatePredicateContextV1
    predicate_context: CpSatPredicateModelContextV1
    selected_card_literals: tuple[cp_model.IntVar, ...]
    selected_play_indexes: tuple[cp_model.IntVar, ...]
    initial_owner_variables: tuple[cp_model.IntVar, ...]
    ekka_lead_variables: tuple[cp_model.IntVar, ...]
    winner_team_variables: tuple[cp_model.IntVar, ...]
    trick_point_variables: tuple[cp_model.IntVar, ...]
    raw_team_variables: tuple[cp_model.IntVar, ...]
    card_game_point_variables: tuple[cp_model.IntVar, ...]
    request_hash: str
    _extract: Callable[[cp_model.CpSolver, int, str], OrToolsPlayTailResultV1]
    raw_team_prefix_variables: tuple[tuple[Any, ...], ...] = ()

    def native_legal_action_literals(self, step: int) -> dict[str, cp_model.IntVar]:
        """Expose the existing per-step legality variables, including Ekka."""
        extract = self._extract
        if extract.native_ekka_flags_by_step is None:
            raise ValueError("native play policy requires native action variables")
        if type(step) is not int or not 0 <= step < len(extract.legal_flags_by_step):
            raise ValueError("native play decision index is outside the compiled horizon")
        return {LegalPlayActionV1(card, ekka).code: literal
                for ekka, flags in ((False, extract.legal_flags_by_step[step]),
                                    (True, extract.native_ekka_flags_by_step[step]))
                for card, literal in zip(extract.pool, flags, strict=True)}

    def constrain_declarative_play_policies(self, *, team: Team, policies: Mapping[str, Mapping[str, Any]],
            start_step: int = 0, cancel_check: Callable[[], bool] | None = None) -> bool:
        """Bind the existing seeded first-legal policy using native legality variables."""
        from .ortools_proof import DeclarativeActionPolicyV1, ProofGraphEdgeV1, select_declarative_policy_action_v1

        if (not isinstance(team, Team) or not isinstance(policies, Mapping)
                or set(policies) != {seat.name.lower() for seat in Seat if seat.team == team}
                or type(start_step) is not int or not 0 <= start_step <= len(self._extract.actors)
                or self._extract.native_ekka_flags_by_step is None):
            raise ValueError("ranked play requires a complete team and native play horizon")
        parsed = {seat: DeclarativeActionPolicyV1.build(seat, value) for seat, value in sorted(policies.items())}
        if any(policy.choices or policy.materialization is None for policy in parsed.values()):
            raise ValueError("ranked play requires unmaterialized declarative policy templates")
        model = self.model
        for seat, policy in parsed.items():
            edges = [ProofGraphEdgeV1.build(action.action_id, action.code, "policy_order")
                for code in self.native_legal_action_literals(0)
                for action in (LegalPlayActionV1.from_code(code),)]
            ordered = []
            while edges:
                edge = select_declarative_policy_action_v1(policy, edges)
                ordered.append(LegalPlayActionV1.from_code(edge.action_code))
                edges.remove(edge)
            for step in range(start_step, len(self._extract.actors)):
                if cancel_check is not None and cancel_check():
                    return False
                name = f"ranked_play_{seat}_{step}"
                first = model.new_bool_var(name + "_actor")
                model.add(self._extract.actors[step] == int(Seat.parse(seat))).only_enforce_if(first)
                model.add(self._extract.actors[step] != int(Seat.parse(seat))).only_enforce_if(first.Not())
                available = self.native_legal_action_literals(step)
                # Policy steps are local to this tail; predicate slots use
                # absolute round positions and may be absent at this index.
                card_variable = self._extract.card_ids[step]
                ekka_variable = (self._extract.ekka_leads[step // 4]
                    if step % 4 == 0 else model.new_constant(0))
                for index, action in enumerate(ordered):
                    legal = available[action.code]
                    selected = model.new_bool_var(name + f"_selected_{index}")
                    model.add_min_equality(selected, [first, legal])
                    model.add(card_variable == action.card.id).only_enforce_if(selected)
                    model.add(ekka_variable == int(action.ekka)).only_enforce_if(selected)
                    following = model.new_bool_var(name + f"_remaining_{index}")
                    model.add_min_equality(following, [first, legal.Not()])
                    first = following
        return True

    def constrain_native_observer_safe_policy(self, *, team: Team, policy_id: str, start_step: int = 0,
            _override_matches: Mapping[int, Sequence[Any]] | None = None) -> None:
        """Bind one deterministic visible-trick policy, not an optimality claim.

        Scores share the production policy authority. Equal scores select the
        lowest legal action ID. Established-control and guard-protection also
        use own-card membership and public completed-card history.
        """
        extract, model = self._extract, self.model
        native_observer_safe_policy_identity_v1(policy_id, team)
        if (type(start_step) is not int or not 0 <= start_step < len(extract.actors)
                or extract.native_ekka_flags_by_step is None):
            raise ValueError("native visible policy requires a native play horizon")
        # Only the scenario owner supplies these full-history guards, after
        # binding and validating each replacement action on the same model.
        if _override_matches is not None and (not isinstance(_override_matches, Mapping)
                or any(type(step) is not int or not start_step <= step < len(extract.actors)
                       for step in _override_matches)):
            raise ValueError("native policy overrides must be within the future play horizon")
        for step in range(start_step, len(extract.actors)):
            name = f"visible_policy_{int(team)}_{step}"
            active = model.new_bool_var(name + "_active")
            model.add_allowed_assignments([extract.actors[step], active],
                [(int(seat), int(seat.team == team)) for seat in Seat])
            matches = (_override_matches or {}).get(step, ())
            if matches:
                fallback = model.new_bool_var(name + "_fallback")
                model.add_min_equality(fallback, [active, *(match.Not() for match in matches)])
                active = fallback
            values, bounds, masters = [], [], None
            context = extract.public_trick_context[step]
            comparisons = {}
            if context is None and policy_id == NATIVE_ESTABLISHED_CONTROL_POLICY_ID:
                values, bounds, masters = _native_established_lead_scores_v1(model,
                    pool=extract.pool, owners=extract.owner_vars, actor=extract.actors[step],
                    played=extract.public_played_by_step[step], domain=extract.domain,
                    contract=extract.contract, trump=extract.trump, name=name)
            for card in (() if context is None and policy_id == NATIVE_ESTABLISHED_CONTROL_POLICY_ID else extract.pool):
                strength = extract.domain.strength(card, extract.contract, extract.trump)
                points = extract.domain.points(card, extract.contract, extract.trump)
                def priority(partner=None, control=None):
                    score = _observer_safe_play_score_v1(strength, points,
                        partner_winning=partner, takes_control=control,
                        feed_partner=policy_id in NATIVE_PARTNER_FEED_POLICY_IDS and (
                            policy_id != NATIVE_CLOSED_TRICK_FEED_POLICY_ID or step % 4 == 3),
                        secure_control=policy_id == NATIVE_SECURE_CONTROL_POLICY_ID and step % 4 in (1, 2))
                    return (score + 9_999) // 10_000 if policy_id == "observer_safe_tactical_category_v1" else score
                if context is None:
                    value = priority()
                    bound = (value, value)
                else:
                    value, bound = _compact_observed_card_score_v1(model, context, card,
                        strength, points, extract.contract, extract.trump, name, comparisons, priority=priority)
                values.append(value)
                bounds.append(bound)
            if context is not None and policy_id == NATIVE_GUARD_PROTECTION_POLICY_ID:
                led_suit = model.new_int_var(0, 3, name + "_led_suit")
                model.add_element(extract.card_ids[step // 4 * 4],
                                  [int(Card(card).suit) for card in range(32)], led_suit)
                guards = _native_guard_discard_flags_v1(model, pool=extract.pool,
                    owners=extract.owner_vars, actor=extract.actors[step],
                    played=extract.public_played_by_step[step],
                    led_suit=led_suit, partner_winning=context[2],
                    domain=extract.domain, contract=extract.contract, trump=extract.trump, name=name,
                    owned_flags=extract.owned_by_step[step], remaining_flags=extract.remaining_by_step[step])
                bonus = _guard_discard_bonus_v1(extract.domain, extract.contract, extract.trump)
                values = [value - bonus * guard for value, guard in zip(values, guards, strict=True)]
                bounds = [(lower - bonus, upper) for lower, upper in bounds]
            legal = self.native_legal_action_literals(step)
            if masters is not None:
                # The control bonus already strictly dominates every non-master
                # score. Expose that implied choice directly for propagation.
                selected_master = model.new_bool_var(name + "_selected_master")
                model.add_element(self.selected_play_indexes[step], masters, selected_master)
                model.add(extract.winner_seats[step // 4] == extract.actors[step]).only_enforce_if(
                    [active, selected_master])
                for card, master in zip(extract.pool, masters, strict=True):
                    for ekka in (False, True):
                        action = LegalPlayActionV1(card, ekka)
                        model.add(selected_master >= master).only_enforce_if([active, legal[action.code]])
            if context is None and policy_id in {NATIVE_COMPACT_PARTNER_FEED_POLICY_ID,
                    NATIVE_SECURE_CONTROL_POLICY_ID, NATIVE_GUARD_PROTECTION_POLICY_ID,
                    NATIVE_CLOSED_TRICK_FEED_POLICY_ID}:
                # Static lead scores admit an exact first-legal chain. Following
                # scores still depend on the public trick, never hidden hands.
                ordered = sorted(((value * 64 - action.action_id, action)
                    for card, value in zip(extract.pool, values, strict=True)
                    for ekka in (False, True) for action in (LegalPlayActionV1(card, ekka),)),
                    key=lambda row: row[0], reverse=True)
                for index, (_, action) in enumerate(ordered):
                    available = legal[action.code]
                    model.add(extract.card_ids[step] == action.card.id).only_enforce_if([active, available])
                    model.add(extract.ekka_leads[step // 4] == int(action.ekka)).only_enforce_if([active, available])
                    following = model.new_bool_var(name + f"_remaining_{index}")
                    model.add_bool_and([active, available.Not()]).only_enforce_if(following)
                    model.add_bool_or([active.Not(), available]).only_enforce_if(following.Not())
                    active = following
                continue
            selected_score = model.new_int_var(min(row[0] for row in bounds), max(row[1] for row in bounds), name + "_score")
            model.add_element(self.selected_play_indexes[step], values, selected_score)
            ekka = extract.ekka_leads[step // 4] if step % 4 == 0 else 0
            selected_priority = selected_score * 64 - extract.card_ids[step] - 32 * ekka
            for card, value in zip(extract.pool, values, strict=True):
                for ekka in (False, True):
                    action = LegalPlayActionV1(card, ekka)
                    model.add(selected_priority >= value * 64 - action.action_id).only_enforce_if([active, legal[action.code]])

    @property
    def model_hash(self) -> str:
        return canonical_sha256(
            {
                "schemaVersion": 1,
                "requestHash": self.request_hash,
                "cpModelProtoHash": hashlib.sha256(
                    str(self.model.proto).encode("utf-8")
                ).hexdigest(),
            }
        )

    def solve(
        self,
        *,
        maximum_time_seconds: float = 10.0,
        maximum_conflicts: int = 250_000,
        random_seed: int = 0,
        solver: cp_model.CpSolver | None = None,
        telemetry_callback: Callable[[Mapping[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> OrToolsPlayTailResultV1:
        if type(maximum_time_seconds) not in {int, float} or maximum_time_seconds <= 0:
            raise ValueError("maximum_time_seconds must be positive")
        if type(maximum_conflicts) is not int or maximum_conflicts < 1:
            raise ValueError("maximum_conflicts must be positive")
        active_solver = solver if solver is not None else cp_model.CpSolver()
        active_solver.parameters.num_search_workers = 1
        active_solver.parameters.random_seed = random_seed & 0x7FFFFFFF
        # CP-SAT's deterministic single-worker portfolio is materially faster
        # for full-prefix lift models than forcing the authored owner/card
        # variable order.  The seed and worker count still make official solves
        # reproducible; model strategy remains a hint rather than a mandate.
        active_solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
        active_solver.parameters.max_time_in_seconds = float(maximum_time_seconds)
        active_solver.parameters.max_number_of_conflicts = maximum_conflicts
        model_hash = self.model_hash
        model_telemetry = {
            "modelHash": model_hash,
            "variableCount": len(self.model.proto.variables),
            "constraintCount": len(self.model.proto.constraints),
            "solverStatus": "running",
            "branches": 0,
            "conflicts": 0,
        }
        if telemetry_callback is not None:
            telemetry_callback(model_telemetry)
        status = _solve_cp_sat_with_cancel_v1(
            active_solver, self.model, cancel_check
        )
        solver_status = active_solver.status_name(status).lower()
        try:
            branches = int(active_solver.num_branches)
            conflicts = int(active_solver.num_conflicts)
            wall_time = float(active_solver.wall_time)
        except RuntimeError:
            if status != cp_model.UNKNOWN:
                raise
            branches, conflicts, wall_time = 0, 0, 0.0
        model_telemetry = {
            **model_telemetry,
            "solverStatus": solver_status,
            "branches": branches,
            "conflicts": conflicts,
            "wallTimeSeconds": wall_time,
        }
        if telemetry_callback is not None:
            telemetry_callback(model_telemetry)
        if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            raise PlayTailSolveStopped(solver_status, evidence=model_telemetry)
        return replace(
            self._extract(active_solver, status, model_hash),
            variable_count=len(self.model.proto.variables),
            constraint_count=len(self.model.proto.constraints),
        )


@dataclass(frozen=True, slots=True)
class _PlayTailExtractionV1:
    pool: tuple[Card, ...]
    owner_vars: tuple[Any, ...]
    selected_card_literals: tuple[Any, ...]
    selected_fragment: bool
    actors: tuple[Any, ...]
    card_ids: tuple[Any, ...]
    ekka_leads: tuple[Any, ...]
    prefix_length: int
    legal_flags_by_step: tuple[tuple[Any, ...], ...]
    native_ekka_flags_by_step: tuple[tuple[Any, ...], ...] | None
    domain: PlayDomainIRV1
    contract: Contract
    trump: Suit | None
    winner_seats: tuple[Any, ...]
    winner_cards: tuple[Any, ...]
    trick_points: tuple[Any, ...]
    raw_team_vars: tuple[Any, ...]
    game_team_vars: tuple[Any, ...]
    scoring_domain: ScoringDomainIRV1 | None
    public_trick_context: tuple[Any, ...]
    public_played_by_step: tuple[tuple[Any, ...], ...]
    owned_by_step: tuple[tuple[Any, ...], ...]
    remaining_by_step: tuple[tuple[Any, ...], ...]

    def __call__(
        self,
        solver: cp_model.CpSolver,
        status: int,
        model_hash: str,
    ) -> OrToolsPlayTailResultV1:
        solved_plays = tuple(
            TailPrefixActionV1(
                Seat(solver.value(self.actors[step])),
                Card(solver.value(self.card_ids[step])),
                bool(
                    step % 4 == 0
                    and solver.value(self.ekka_leads[step // 4])
                ),
            )
            for step in range(len(self.actors))
        )
        owned_cards = tuple(
            (card, solver.value(self.owner_vars[index]))
            for index, card in enumerate(self.pool)
            if not self.selected_fragment
            or solver.boolean_value(self.selected_card_literals[index])
        )
        initial_hands = tuple(
            tuple(card for card, owner in owned_cards if owner == int(seat))
            for seat in Seat
        )
        played_prefix_ids = {
            play.card.id for play in solved_plays[: self.prefix_length]
        }
        remaining_hands = tuple(
            tuple(
                card
                for card in initial_hands[int(seat)]
                if card.id not in played_prefix_ids
            )
            for seat in Seat
        )
        current_count = self.prefix_length % 4
        current_trick = (
            solved_plays[self.prefix_length - current_count : self.prefix_length]
            if current_count
            else ()
        )
        legal_actions_by_step = tuple(
            tuple(
                self.pool[index]
                for index, flag in enumerate(step_flags)
                if solver.boolean_value(flag)
            )
            for step_flags in self.legal_flags_by_step
        )

        def semantic_actions(step: int, cards_at_step: tuple[Card, ...]):
            actor = solved_plays[step].actor
            if self.native_ekka_flags_by_step is not None:
                return tuple(
                    [TailPrefixActionV1(actor, card) for card in cards_at_step]
                    + [
                        TailPrefixActionV1(actor, card, True)
                        for card, flag in zip(self.pool, self.native_ekka_flags_by_step[step], strict=True)
                        if solver.boolean_value(flag)
                    ]
                )
            played_before = {play.card.id for play in solved_plays[:step]}
            remaining = tuple(
                card
                for hand in initial_hands
                for card in hand
                if card.id not in played_before
            )
            return tuple(
                [TailPrefixActionV1(actor, card) for card in cards_at_step]
                + [
                    TailPrefixActionV1(actor, card, True)
                    for card in cards_at_step
                    if step % 4 == 0
                    and self.domain.ekka_enabled
                    and self.contract.effective == Contract.HOKUM
                    and card.suit != self.trump
                    and not any(
                        other.suit == card.suit
                        and self.domain.sun_strength[int(other.rank)]
                        > self.domain.sun_strength[int(card.rank)]
                        for other in remaining
                    )
                ]
            )

        legal_play_actions_by_step = tuple(
            semantic_actions(step, actions)
            for step, actions in enumerate(legal_actions_by_step)
        )
        if self.prefix_length < len(solved_plays):
            next_actor = Seat(solver.value(self.actors[self.prefix_length]))
            legal_actions = legal_actions_by_step[self.prefix_length]
            legal_play_actions = legal_play_actions_by_step[self.prefix_length]
        else:
            next_actor = None
            legal_actions = ()
            legal_play_actions = ()
        raw_team_points = (
            tuple(int(solver.value(value)) for value in self.raw_team_vars)
            if self.raw_team_vars
            else None
        )
        card_game_points = (
            tuple(int(solver.value(value)) for value in self.game_team_vars)
            if self.game_team_vars
            else None
        )
        trick_winners = tuple(Seat(solver.value(value)) for value in self.winner_seats)
        trick_winning_cards = tuple(Card(solver.value(value)) for value in self.winner_cards)
        trick_raw_points = tuple(solver.value(value) for value in self.trick_points)
        evidence_core = {
            "schemaVersion": 1,
            "initialHands": [[card.code for card in hand] for hand in initial_hands],
            "plays": [play.to_dict() for play in solved_plays],
            "legalActionsByStep": [
                [card.code for card in actions] for actions in legal_actions_by_step
            ],
            "legalPlayActionsByStep": [
                [action.to_dict() for action in actions]
                for actions in legal_play_actions_by_step
            ],
            "trickWinners": [seat.name.lower() for seat in trick_winners],
            "trickWinningCards": [card.code for card in trick_winning_cards],
            "trickRawPoints": list(trick_raw_points),
            "prefixLength": self.prefix_length,
            "remainingHands": [
                [card.code for card in hand] for hand in remaining_hands
            ],
            "currentTrick": [play.to_dict() for play in current_trick],
            "nextActor": next_actor.name.lower() if next_actor is not None else None,
            "legalActions": [card.code for card in legal_actions],
            "legalPlayActions": [action.to_dict() for action in legal_play_actions],
            "rawTeamPoints": list(raw_team_points)
            if raw_team_points is not None
            else None,
            "cardGamePoints": list(card_game_points)
            if card_game_points is not None
            else None,
            "scoringDomainHash": (
                self.scoring_domain.sha256 if self.scoring_domain is not None else None
            ),
        }
        evidence_hash = canonical_sha256(evidence_core)
        solver_status = solver.status_name(status).lower()
        semantic_core = {
            "schemaVersion": 1,
            "modelHash": model_hash,
            "evidenceHash": evidence_hash,
            "solverStatus": solver_status,
        }
        return OrToolsPlayTailResultV1(
            initial_hands,
            solved_plays,
            trick_winners,
            trick_winning_cards,
            trick_raw_points,
            self.prefix_length,
            remaining_hands,
            current_trick,
            next_actor,
            legal_actions,
            legal_play_actions,
            legal_actions_by_step,
            legal_play_actions_by_step,
            raw_team_points,
            card_game_points,
            self.scoring_domain.sha256 if self.scoring_domain is not None else None,
            model_hash,
            evidence_hash,
            solver_status,
            canonical_sha256(semantic_core),
        )


class OrToolsPlayTailStageV1:
    """Construct a legal bounded tail or exact complete round without runtime calls."""

    def __init__(self, *, native_conversion: bool = False):
        if type(native_conversion) is not bool:
            raise TypeError("native_conversion must be bool")
        self.native_conversion = native_conversion

    def solve(
        self,
        *,
        cards: Sequence[Card | str | int | dict],
        ownership: Mapping[
            Card | str | int, Seat | str | int | Sequence[Seat | str | int]
        ]
        | None,
        contract: Contract | str,
        trump: Suit | str | int | None,
        first_leader: Seat | str | int | None,
        play_domain: PlayDomainIRV1,
        structural_domain: StructuralDomainIRV1,
        scoring_domain: ScoringDomainIRV1 | None = None,
        prefix: Sequence[TailPrefixActionV1] = (),
        tail: Sequence[TailPrefixActionV1] = (),
        ekka_leads: Sequence[bool] | None = None,
        locked: bool = False,
        action_constraints: Sequence[TailActionConstraintV1] = (),
        winner_constraints: Sequence[TailTrickWinnerConstraintV1] = (),
        card_count_constraints: Sequence[TailTrickCardCountConstraintV1] = (),
        decision_prefix_length: int | None = None,
        decision_actor: Seat | str | int | None = None,
        decision_purchase_card: Card | str | int | dict | None = None,
        excluded_decision_status_vectors: Sequence[Sequence[int]] = (),
        exact_excluded_decision_status_vectors: Sequence[Sequence[int]] = (),
        minimum_decision_status_hamming_distance: int = 1,
        play_step_count: int | None = None,
        selected_play_count: int | None = None,
        policy_prefix_steps: int = 0,
        policy_id: str = "observer_safe_trick_control_v1",
        construction_plan_hash: str | None = None,
        model_configurator: Callable[[OrToolsPlayTailBuildV1], None] | None = None,
        maximum_time_seconds: float = 10.0,
        maximum_conflicts: int = 250_000,
        random_seed: int = 0,
        shared_model: cp_model.CpModel | None = None,
        shared_owner_of_card: Sequence[cp_model.IntVar] | None = None,
        solver: cp_model.CpSolver | None = None,
        telemetry_callback: Callable[[Mapping[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> OrToolsPlayTailResultV1:
        if (construction_plan_hash is None) != (model_configurator is None):
            raise ValueError(
                "construction_plan_hash and model_configurator must be supplied together"
            )
        if construction_plan_hash is not None and (
            len(construction_plan_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in construction_plan_hash
            )
        ):
            raise ValueError("construction_plan_hash must be canonical SHA-256")
        if type(maximum_time_seconds) not in {int, float} or maximum_time_seconds <= 0:
            raise ValueError("maximum_time_seconds must be positive")
        deadline = time.monotonic() + float(maximum_time_seconds)

        def stopped() -> bool:
            return time.monotonic() >= deadline or (cancel_check is not None and cancel_check())

        build_started = time.perf_counter()
        try:
            build = self.build_model(
                cards=cards,
                ownership=ownership,
                contract=contract,
                trump=trump,
                first_leader=first_leader,
                play_domain=play_domain,
                structural_domain=structural_domain,
                scoring_domain=scoring_domain,
                prefix=prefix,
                tail=tail,
                ekka_leads=ekka_leads,
                locked=locked,
                action_constraints=action_constraints,
                winner_constraints=winner_constraints,
                card_count_constraints=card_count_constraints,
                decision_prefix_length=decision_prefix_length,
                decision_actor=decision_actor,
                decision_purchase_card=decision_purchase_card,
                excluded_decision_status_vectors=excluded_decision_status_vectors,
                exact_excluded_decision_status_vectors=(
                    exact_excluded_decision_status_vectors
                ),
                minimum_decision_status_hamming_distance=(
                    minimum_decision_status_hamming_distance
                ),
                play_step_count=play_step_count,
                selected_play_count=selected_play_count,
                policy_prefix_steps=policy_prefix_steps,
                policy_id=policy_id,
                shared_model=shared_model,
                shared_owner_of_card=shared_owner_of_card,
                _export_predicate_contexts=model_configurator is not None,
                cancel_check=stopped,
            )
        except PlayTailSolveStopped as exc:
            if time.monotonic() >= deadline:
                raise PlayTailSolveStopped(
                    "unknown", evidence={"budgetExceeded": True, "modelBuildSeconds": time.perf_counter() - build_started}
                ) from exc
            raise
        model_build_seconds = time.perf_counter() - build_started
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PlayTailSolveStopped("unknown", evidence={"budgetExceeded": True, "modelBuildSeconds": model_build_seconds})
        configure_started = time.perf_counter()
        if model_configurator is not None:
            model_configurator(build)
        model_configure_seconds = time.perf_counter() - configure_started
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PlayTailSolveStopped("unknown", evidence={"budgetExceeded": True, "modelBuildSeconds": model_build_seconds, "modelConfigureSeconds": model_configure_seconds})
        def report_telemetry(evidence: Mapping[str, Any]) -> None:
            if telemetry_callback is not None:
                telemetry_callback({**evidence, "modelBuildSeconds": model_build_seconds, "modelConfigureSeconds": model_configure_seconds})
        return build.solve(
            maximum_time_seconds=remaining,
            maximum_conflicts=maximum_conflicts,
            random_seed=random_seed,
            solver=solver,
            telemetry_callback=report_telemetry if telemetry_callback is not None else None,
            cancel_check=stopped,
        )

    def build_model(
        self,
        *,
        cards: Sequence[Card | str | int | dict],
        ownership: Mapping[
            Card | str | int, Seat | str | int | Sequence[Seat | str | int]
        ]
        | None,
        contract: Contract | str,
        trump: Suit | str | int | None,
        first_leader: Seat | str | int | None,
        play_domain: PlayDomainIRV1,
        structural_domain: StructuralDomainIRV1,
        scoring_domain: ScoringDomainIRV1 | None = None,
        prefix: Sequence[TailPrefixActionV1] = (),
        tail: Sequence[TailPrefixActionV1] = (),
        ekka_leads: Sequence[bool] | None = None,
        locked: bool = False,
        action_constraints: Sequence[TailActionConstraintV1] = (),
        winner_constraints: Sequence[TailTrickWinnerConstraintV1] = (),
        card_count_constraints: Sequence[TailTrickCardCountConstraintV1] = (),
        decision_prefix_length: int | None = None,
        decision_actor: Seat | str | int | None = None,
        decision_purchase_card: Card | str | int | dict | None = None,
        excluded_decision_status_vectors: Sequence[Sequence[int]] = (),
        exact_excluded_decision_status_vectors: Sequence[Sequence[int]] = (),
        minimum_decision_status_hamming_distance: int = 1,
        play_step_count: int | None = None,
        selected_play_count: int | None = None,
        policy_prefix_steps: int = 0,
        policy_id: str = "observer_safe_trick_control_v1",
        shared_model: cp_model.CpModel | None = None,
        shared_owner_of_card: Sequence[cp_model.IntVar] | None = None,
        _export_predicate_contexts: bool = True,
        cancel_check: Callable[[], bool] | None = None,
    ) -> OrToolsPlayTailBuildV1:
        if cancel_check is not None and cancel_check():
            raise PlayTailSolveStopped("unknown", evidence={"cancelled": True})
        structural = structural_domain.require_integrity()
        domain = play_domain.require_integrity(structural)
        contract = Contract.parse(contract)
        trump = Suit.parse(trump) if trump is not None else None
        if contract.effective == Contract.HOKUM and trump is None:
            raise ValueError("Hokum play tails require trump")
        if contract.effective != Contract.HOKUM and trump is not None:
            raise ValueError("Sun/Ashkal play tails cannot carry trump")
        if type(locked) is not bool:
            raise TypeError("locked must be a boolean")
        pool = tuple(
            sorted((Card.parse(card) for card in cards), key=lambda card: card.id)
        )
        if not 4 <= len(pool) <= 32 or len(pool) % 4:
            raise ValueError(
                "play construction requires one through eight cards per seat"
            )
        if len({card.id for card in pool}) != len(pool):
            raise ValueError("play-tail cards must be unique")
        if play_step_count is not None and selected_play_count is not None:
            raise ValueError(
                "play_step_count and selected_play_count are mutually exclusive"
            )
        if selected_play_count is not None:
            if tuple(card.id for card in pool) != tuple(range(32)):
                raise ValueError("selected_play_count requires FULL_DECK")
            if selected_play_count not in {4, 8, 12, 16}:
                raise ValueError("selected_play_count must be 4, 8, 12, or 16")
        if first_leader is None and selected_play_count is None:
            raise ValueError("an unknown first leader requires selected_play_count")
        compiled_play_steps = (
            selected_play_count
            if selected_play_count is not None
            else len(pool)
            if play_step_count is None
            else play_step_count
        )
        if (
            type(compiled_play_steps) is not int
            or not 4 <= compiled_play_steps <= len(pool)
            or compiled_play_steps % 4 != 0
        ):
            raise ValueError("play_step_count must compile one or more complete tricks")
        if (
            type(policy_prefix_steps) is not int
            or policy_prefix_steps < 0
            or policy_prefix_steps > compiled_play_steps
            or (policy_prefix_steps and len(pool) != 32)
        ):
            raise ValueError("policy_prefix_steps requires a complete-deck model")
        if policy_id not in NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS:
            raise ValueError("play-tail policy_id is unsupported")
        decision_vectors = tuple(
            tuple(vector) for vector in excluded_decision_status_vectors
        )
        exact_decision_vectors = tuple(
            tuple(vector) for vector in exact_excluded_decision_status_vectors
        )
        all_decision_vectors = (*decision_vectors, *exact_decision_vectors)
        decision_purchase = (
            Card.parse(decision_purchase_card)
            if decision_purchase_card is not None
            else None
        )
        parsed_decision_actor = (
            Seat.parse(decision_actor) if decision_actor is not None else None
        )
        if all_decision_vectors:
            selective_boundary = selected_play_count is not None
            if len(pool) != 32 or (
                selective_boundary
                and (
                    type(decision_prefix_length) is not int
                    or not 0 <= decision_prefix_length < selected_play_count
                    or parsed_decision_actor is None
                    or decision_purchase is not None
                )
            ):
                raise ValueError(
                    "selective decision-state cuts require a known trick boundary"
                )
            if not selective_boundary and compiled_play_steps != len(pool):
                raise ValueError("decision-state cuts require a complete round")
            if not selective_boundary and (
                type(decision_prefix_length) is not int
                or not 0 <= decision_prefix_length < compiled_play_steps
            ):
                raise ValueError("decision-state cuts require a live decision step")
            if (
                not selective_boundary
                and decision_purchase is None
                and any(9 in vector for vector in all_decision_vectors)
            ):
                raise ValueError(
                    "purchase-aware decision-state cuts require the purchase card"
                )
            if (
                type(minimum_decision_status_hamming_distance) is not int
                or not 1 <= minimum_decision_status_hamming_distance <= 32
            ):
                raise ValueError("decision-state Hamming distance must be in 1..32")
            if any(
                len(vector) != 32
                or any(
                    type(status) is not int or not 0 <= status <= 9 for status in vector
                )
                for vector in all_decision_vectors
            ):
                raise ValueError(
                    "decision-state status vectors must contain 32 values in 0..9"
                )
        elif decision_prefix_length is not None and (
            type(decision_prefix_length) is not int
            or not 0 <= decision_prefix_length <= compiled_play_steps
        ):
            raise ValueError("decision_prefix_length lies outside the play trace")
        cards_per_seat = compiled_play_steps // 4
        owner_cards_per_seat = len(pool) // 4
        card_index = {card.id: index for index, card in enumerate(pool)}
        normalized_prefix = tuple(prefix)
        normalized_tail = tuple(tail)
        if any(not isinstance(play, TailPrefixActionV1) for play in normalized_prefix):
            raise TypeError("prefix actions must be TailPrefixActionV1 values")
        if any(not isinstance(play, TailPrefixActionV1) for play in normalized_tail):
            raise TypeError("tail actions must be TailPrefixActionV1 values")
        if len(normalized_prefix) > compiled_play_steps:
            raise ValueError("play prefix lies outside the compiled cards")
        if len(normalized_prefix) + len(normalized_tail) > compiled_play_steps:
            raise ValueError("fixed prefix and tail may not overlap")
        if len({play.card.id for play in normalized_prefix}) != len(normalized_prefix):
            raise ValueError("play-tail prefix cards must be unique")
        if any(play.card.id not in card_index for play in normalized_prefix):
            raise ValueError("play-tail prefix contains a card outside the tail")
        if len({play.card.id for play in normalized_tail}) != len(normalized_tail):
            raise ValueError("fixed tail cards must be unique")
        if any(play.card.id not in card_index for play in normalized_tail):
            raise ValueError("fixed tail contains a card outside the compiled cards")
        if {play.card.id for play in normalized_prefix}.intersection(
            play.card.id for play in normalized_tail
        ):
            raise ValueError("fixed prefix and tail cards must be disjoint")
        if len(pool) != 32 and scoring_domain is not None:
            raise ValueError("round scoring is only defined for complete 32-card play")
        if (shared_model is None) != (shared_owner_of_card is None):
            raise ValueError(
                "shared model and owner-of-card variables must be supplied together"
            )
        if shared_owner_of_card is not None and len(shared_owner_of_card) != 32:
            raise ValueError("shared owner-of-card variables must contain all 32 cards")

        supplied_ekka = tuple(ekka_leads) if ekka_leads is not None else None
        if supplied_ekka is not None and len(supplied_ekka) == owner_cards_per_seat:
            fixed_ekka = (
                supplied_ekka[-cards_per_seat:]
                if selected_play_count is not None
                else supplied_ekka[:cards_per_seat]
            )
        else:
            fixed_ekka = supplied_ekka
        if fixed_ekka is not None and (
            len(fixed_ekka) != cards_per_seat
            or any(type(value) is not bool for value in fixed_ekka)
        ):
            raise ValueError("ekka_leads must contain one boolean per tail trick")
        if fixed_ekka is not None and any(fixed_ekka) and (
            not domain.ekka_enabled or contract.effective != Contract.HOKUM
        ):
            raise ValueError("Ekka is only supported by enabled Hokum tails")

        allowed_owners: dict[int, tuple[Seat, ...]] = {
            card.id: tuple(Seat) for card in pool
        }
        declared_owners: set[int] = set()
        for raw_card, raw_seats in (ownership or {}).items():
            card = Card.parse(raw_card)
            if card.id not in card_index:
                raise ValueError("ownership contains a card outside the tail")
            if card.id in declared_owners:
                raise ValueError("ownership contains duplicate normalized cards")
            declared_owners.add(card.id)
            source = (
                raw_seats
                if isinstance(raw_seats, Sequence)
                and not isinstance(raw_seats, (str, bytes))
                else (raw_seats,)
            )
            seats = tuple(sorted({Seat.parse(seat) for seat in source}, key=int))
            if not seats:
                raise ValueError("every ownership domain must contain a seat")
            allowed_owners[card.id] = seats
        fixed_prefix_indexes = tuple(
            card_index[play.card.id] for play in normalized_prefix
        )

        for item in action_constraints:
            if (
                not isinstance(item, TailActionConstraintV1)
                or item.step >= compiled_play_steps
            ):
                raise ValueError("action constraint lies outside the tail")
        for item in (*winner_constraints, *card_count_constraints):
            if (
                not isinstance(
                    item, (TailTrickWinnerConstraintV1, TailTrickCardCountConstraintV1)
                )
                or item.trick >= compiled_play_steps // 4
            ):
                raise ValueError("trick constraint lies outside the tail")

        request_core = {
            **({"rawConversionAuthority": "native_arithmetic_v1"} if self.native_conversion else {}),
            "schemaVersion": 1,
            "loweringId": _LOWERING_ID,
            "playDomainHash": domain.sha256,
            "structuralDomainHash": structural.sha256,
            "cards": [card.code for card in pool],
            "ownership": {
                card.code: [seat.name.lower() for seat in allowed_owners[card.id]]
                for card in pool
            },
            "contract": contract.value,
            "trump": trump.name.lower() if trump is not None else None,
            "firstLeader": (
                Seat.parse(first_leader).name.lower()
                if first_leader is not None else None
            ),
            "prefix": [play.to_dict() for play in normalized_prefix],
            "tail": [play.to_dict() for play in normalized_tail],
            "ekkaLeads": list(fixed_ekka) if fixed_ekka is not None else None,
            "locked": locked,
            "actionConstraints": [item.to_dict() for item in action_constraints],
            "winnerConstraints": [item.to_dict() for item in winner_constraints],
            "cardCountConstraints": [item.to_dict() for item in card_count_constraints],
            "decisionPrefixLength": decision_prefix_length,
            "decisionActor": (
                parsed_decision_actor.name.lower()
                if parsed_decision_actor is not None
                else None
            ),
            "decisionPurchaseCard": (
                decision_purchase.code if decision_purchase is not None else None
            ),
            "excludedDecisionStatusHashes": [
                canonical_sha256(list(vector)) for vector in decision_vectors
            ],
            "exactExcludedDecisionStatusHashes": [
                canonical_sha256(list(vector)) for vector in exact_decision_vectors
            ],
            "minimumDecisionStatusHammingDistance": (
                minimum_decision_status_hamming_distance
            ),
            "playStepCount": compiled_play_steps,
            "selectedPlayCount": selected_play_count,
            **({"observerSafePolicyPrefixSteps": policy_prefix_steps,
                "observerSafePolicyId": policy_id} if policy_prefix_steps else {}),
            "scoringDomainHash": (
                scoring_domain.sha256 if scoring_domain is not None else None
            ),
            "sharedOwnerBinding": shared_owner_of_card is not None,
        }
        model = shared_model if shared_model is not None else cp_model.CpModel()
        serial = 0

        def name(prefix_value: str) -> str:
            nonlocal serial
            serial += 1
            if (
                serial % 256 == 0
                and cancel_check is not None
                and cancel_check()
            ):
                raise PlayTailSolveStopped(
                    "unknown", evidence={"cancelled": True}
                )
            return f"{prefix_value}_{serial}"

        fixed_literals: dict[int, bool] = {}

        def fixed(value: bool, prefix_value: str):
            variable_name = name(prefix_value)
            if self.native_conversion:
                result = model.new_constant(int(value))
                fixed_literals[result.index] = value
                fixed_literals[result.Not().index] = not value
                return result
            result = model.new_bool_var(variable_name)
            model.add(result == int(value))
            return result

        ekka_variables = tuple(
            fixed(
                bool(fixed_ekka[trick])
                if fixed_ekka is not None
                else False,
                "ekka_disabled",
            )
            if (
                fixed_ekka is not None
                or not domain.ekka_enabled
                or contract.effective != Contract.HOKUM
            )
            else model.new_bool_var(f"ekka_lead_{trick}")
            for trick in range(cards_per_seat)
        )

        equality_literals: dict[tuple[int, int], Any] = {}

        def eq_const(variable, value: int, prefix_value: str):
            key = (variable.index, value)
            if key in equality_literals:
                return equality_literals[key]
            domain_values = variable.proto.domain
            if self.native_conversion and len(domain_values) == 2 and domain_values[0] == domain_values[1]:
                result = fixed(domain_values[0] == value, prefix_value)
                equality_literals[key] = result
                return result
            result = model.new_bool_var(name(prefix_value))
            model.add(variable == value).only_enforce_if(result)
            model.add(variable != value).only_enforce_if(result.Not())
            equality_literals[key] = result
            return result

        def eq_var(left, right, prefix_value: str):
            result = model.new_bool_var(name(prefix_value))
            model.add(left == right).only_enforce_if(result)
            model.add(left != right).only_enforce_if(result.Not())
            return result

        greater_literals: dict[tuple[int, int], Any] = {}

        def gt_const(value: int, variable, prefix_value: str):
            key = (variable.index, value)
            if self.native_conversion and key in greater_literals:
                return greater_literals[key]
            result = model.new_bool_var(name(prefix_value))
            model.add(variable < value).only_enforce_if(result)
            model.add(variable >= value).only_enforce_if(result.Not())
            if self.native_conversion:
                greater_literals[key] = result
            return result

        def either(values: Iterable[Any], prefix_value: str):
            items = list(values)
            if self.native_conversion:
                if any(fixed_literals.get(item.index) is True for item in items):
                    return fixed(True, prefix_value)
                items = [item for item in items if fixed_literals.get(item.index) is not False]
            if not items:
                return fixed(False, prefix_value)
            if len(items) == 1:
                return items[0]
            result = model.new_bool_var(name(prefix_value))
            if self.native_conversion:
                model.add_bool_or(items).only_enforce_if(result)
                model.add_bool_and([item.Not() for item in items]).only_enforce_if(result.Not())
            else:
                model.add_max_equality(result, items)
            return result

        def together(values: Iterable[Any], prefix_value: str):
            items = list(values)
            if self.native_conversion:
                if any(fixed_literals.get(item.index) is False for item in items):
                    return fixed(False, prefix_value)
                items = [item for item in items if fixed_literals.get(item.index) is not True]
            if not items:
                return fixed(True, prefix_value)
            if len(items) == 1:
                return items[0]
            result = model.new_bool_var(name(prefix_value))
            if self.native_conversion:
                model.add_bool_and(items).only_enforce_if(result)
                model.add_bool_or([item.Not() for item in items]).only_enforce_if(result.Not())
            else:
                model.add_min_equality(result, items)
            return result

        def selected_from(values: Sequence[int], variable, prefix_value: str):
            if len(values) == 1:
                model.add(variable == values[0])
                return
            flags = [eq_const(variable, value, prefix_value) for value in values]
            model.add_bool_or(flags)

        owner_vars = (
            [shared_owner_of_card[card.id] for card in pool]
            if shared_owner_of_card is not None
            else [
                model.new_int_var_from_domain(
                    cp_model.Domain.from_values(
                        [int(seat) for seat in allowed_owners[card.id]]
                    ),
                    f"owner_{card.code}",
                )
                for card in pool
            ]
        )
        if shared_owner_of_card is not None:
            for card, owner in zip(pool, owner_vars, strict=True):
                model.add_allowed_assignments(
                    [owner], [(int(seat),) for seat in allowed_owners[card.id]]
                )
        for seat in Seat:
            model.add(
                sum(eq_const(owner, int(seat), "owner_count") for owner in owner_vars)
                == owner_cards_per_seat
            )

        play_indexes = [
            model.new_int_var(
                fixed_prefix_indexes[step] if self.native_conversion and step < len(fixed_prefix_indexes) else 0,
                fixed_prefix_indexes[step] if self.native_conversion and step < len(fixed_prefix_indexes) else len(pool) - 1,
                f"play_index_{step}",
            )
            for step in range(compiled_play_steps)
        ]
        model.add_all_different(play_indexes)
        if selected_play_count is None:
            selected_card_literals = [model.new_constant(1) for _ in pool]
        else:
            selected_card_literals = []
            for index in range(len(pool)):
                selected = model.new_bool_var(f"selected_card_{index}")
                model.add_max_equality(
                    selected,
                    [
                        eq_const(play_index, index, "selected_play_index")
                        for play_index in play_indexes
                    ],
                )
                selected_card_literals.append(selected)
            model.add(sum(selected_card_literals) == compiled_play_steps)
            for seat in Seat:
                selected_for_seat = []
                for index, owner in enumerate(owner_vars):
                    owned = eq_const(owner, int(seat), "selected_owner")
                    selected_for_seat.append(
                        together(
                            [selected_card_literals[index], owned],
                            "selected_for_seat",
                        )
                    )
                model.add(sum(selected_for_seat) == cards_per_seat)
        actors = [
            model.new_int_var(
                int(normalized_prefix[step].actor) if self.native_conversion and step < len(normalized_prefix) else 0,
                int(normalized_prefix[step].actor) if self.native_conversion and step < len(normalized_prefix) else 3,
                f"actor_{step}",
            )
            for step in range(compiled_play_steps)
        ]
        if parsed_decision_actor is not None:
            if (
                decision_prefix_length is None
                or decision_prefix_length >= compiled_play_steps
            ):
                raise ValueError("decision_actor requires a live decision step")
            model.add(actors[decision_prefix_length] == int(parsed_decision_actor))
        pool_ids = [card.id for card in pool]
        pool_suits = [int(card.suit) for card in pool]
        pool_ranks = [int(card.rank) for card in pool]
        pool_strengths = [domain.strength(card, contract, trump) for card in pool]
        pool_sun_strengths = [domain.sun_strength[int(card.rank)] for card in pool]
        pool_points = [domain.points(card, contract, trump) for card in pool]

        def selected_fields(values, maximum, field_name):
            # Expose authored card domains before lowering; keep every element
            # constraint below so ownership and rule-table bindings stay native.
            fields = []
            for step in range(compiled_play_steps):
                known = self.native_conversion and step < len(fixed_prefix_indexes)
                value = values[fixed_prefix_indexes[step]] if known else None
                known = known and 0 <= value <= maximum
                fields.append(model.new_int_var(value if known else 0,
                                                value if known else maximum,
                                                f"{field_name}_{step}"))
            return fields

        card_ids = selected_fields(pool_ids, 31, "card")
        suits = selected_fields(pool_suits, 3, "suit")
        ranks = selected_fields(pool_ranks, 7, "rank")
        strengths = selected_fields(pool_strengths, 7, "strength")
        sun_strengths = selected_fields(pool_sun_strengths, 7, "sun_strength")
        points = selected_fields(pool_points, 32, "points")
        if all_decision_vectors:
            assert decision_prefix_length is not None
            statuses = []
            relative_rows = [
                (owner, actor, (owner - actor) % 4)
                for owner in range(4)
                for actor in range(4)
            ]
            if selected_play_count is not None:
                assert parsed_decision_actor is not None
                decision_actor = model.new_constant(int(parsed_decision_actor))
                trick_start = decision_prefix_length - decision_prefix_length % 4
                for index in range(len(pool)):
                    status = model.new_int_var(0, 8, f"decision_status_{index}")
                    statuses.append(status)
                    relative_owner = model.new_int_var(
                        0, 3, f"decision_relative_owner_{index}"
                    )
                    model.add_allowed_assignments(
                        [owner_vars[index], decision_actor, relative_owner],
                        relative_rows,
                    )
                    played_steps = [
                        eq_const(
                            play_indexes[step], index,
                            f"decision_played_{index}_{step}",
                        )
                        for step in range(decision_prefix_length)
                    ]
                    remaining = together(
                        [
                            selected_card_literals[index],
                            either(played_steps, f"decision_played_{index}").Not(),
                        ],
                        f"decision_remaining_{index}",
                    )
                    model.add(status == relative_owner).only_enforce_if(remaining)
                    for step, played in enumerate(played_steps):
                        if step < trick_start:
                            model.add(status == 8).only_enforce_if(played)
                        else:
                            model.add_allowed_assignments(
                                [actors[step], status],
                                [
                                    (actor, 4 + (actor - int(parsed_decision_actor)) % 4)
                                    for actor in range(4)
                                ],
                            ).only_enforce_if(played)
                    model.add(status == 8).only_enforce_if(
                        selected_card_literals[index].Not()
                    )
            else:
                step_of_card = [
                    model.new_int_var(0, len(pool) - 1, f"step_of_card_{index}")
                    for index in range(len(pool))
                ]
                model.add_inverse(play_indexes, step_of_card)
                decision_actor = actors[decision_prefix_length]
                trick_start = decision_prefix_length - decision_prefix_length % 4
                for index, card in enumerate(pool):
                    status = model.new_int_var(0, 9, f"decision_status_{index}")
                    statuses.append(status)
                    remaining = model.new_bool_var(f"decision_remaining_{index}")
                    model.add(
                        step_of_card[index] >= decision_prefix_length
                    ).only_enforce_if(remaining)
                    model.add(
                        step_of_card[index] < decision_prefix_length
                    ).only_enforce_if(remaining.Not())
                    current = model.new_bool_var(f"decision_current_{index}")
                    if trick_start == decision_prefix_length:
                        model.add(current == 0)
                    else:
                        model.add(step_of_card[index] >= trick_start).only_enforce_if(
                            current
                        )
                        model.add(
                            step_of_card[index] < decision_prefix_length
                        ).only_enforce_if(current)
                        model.add(step_of_card[index] < trick_start).only_enforce_if(
                            [current.Not(), remaining.Not()]
                        )
                        model.add(current == 0).only_enforce_if(remaining)
                    relative_owner = model.new_int_var(
                        0, 3, f"decision_relative_owner_{index}"
                    )
                    model.add_allowed_assignments(
                        [owner_vars[index], decision_actor, relative_owner],
                        relative_rows,
                    )
                    model.add(status == relative_owner).only_enforce_if(remaining)
                    if trick_start < decision_prefix_length:
                        played_actor = model.new_int_var(
                            0, 3, f"decision_played_actor_{index}"
                        )
                        model.add_element(step_of_card[index], actors, played_actor)
                        relative_actor = model.new_int_var(
                            0, 3, f"decision_relative_actor_{index}"
                        )
                        model.add_allowed_assignments(
                            [played_actor, decision_actor, relative_actor],
                            relative_rows,
                        )
                        model.add(status == relative_actor + 4).only_enforce_if(current)
                    if decision_purchase is not None and card == decision_purchase:
                        model.add(status == 9).only_enforce_if(
                            [remaining.Not(), current.Not()]
                        )
                    else:
                        model.add(status == 8).only_enforce_if(
                            [remaining.Not(), current.Not()]
                        )

            suit_orders = tuple(permutations(range(4)))
            if contract.effective == Contract.HOKUM:
                assert trump is not None
                suit_orders = tuple(
                    order for order in suit_orders if order.index(int(trump)) == 0
                )
            cut_groups = (
                (decision_vectors, minimum_decision_status_hamming_distance),
                (exact_decision_vectors, 1),
            )
            for group_index, (vectors, distance) in enumerate(cut_groups):
                for cut_index, vector in enumerate(vectors):
                    for order_index, order in enumerate(suit_orders):
                        suit_map = {
                            original: canonical
                            for canonical, original in enumerate(order)
                        }
                        matches = [
                            eq_const(
                                statuses[index],
                                vector[suit_map[card.suit] * 8 + int(card.rank)],
                                (
                                    f"decision_cut_{group_index}_{cut_index}_"
                                    f"{order_index}"
                                ),
                            )
                            for index, card in enumerate(pool)
                        ]
                        required_distance = distance + int(
                            selected_play_count is not None and 9 in vector
                        )
                        model.add(sum(matches) <= 32 - required_distance)
        policy_lead_values = []
        policy_follow_tables = []
        policy_follow_bounds = []
        if policy_prefix_steps:
            for index, card in enumerate(pool):
                strength, card_points = pool_strengths[index], pool_points[index]
                lead_score = _observer_safe_play_score_v1(strength, card_points)
                policy_lead_values.append(
                    64 * ((lead_score + 9_999) // 10_000) - card.id
                    if policy_id == "observer_safe_tactical_category_v1"
                    else 100 * lead_score - card.id
                )
                rows = []
                for incumbent_suit in range(4):
                    for incumbent_strength in range(8):
                        # The current winner is always led suit or trump; no other suit can hold control.
                        takes_control = (
                            (int(card.suit) == incumbent_suit and strength > incumbent_strength)
                            or (
                                contract.effective == Contract.HOKUM
                                and card.suit == trump
                                and incumbent_suit != int(trump)
                            )
                        )
                        for partner_winning in (0, 1):
                            for context_flag in ((False, True) if policy_id in {
                                    NATIVE_SECURE_CONTROL_POLICY_ID, NATIVE_CLOSED_TRICK_FEED_POLICY_ID} else (False,)):
                                score = _observer_safe_play_score_v1(
                                    strength,
                                    card_points,
                                    partner_winning=bool(partner_winning),
                                    takes_control=takes_control,
                                    feed_partner=policy_id in NATIVE_PARTNER_FEED_POLICY_IDS and (
                                        policy_id != NATIVE_CLOSED_TRICK_FEED_POLICY_ID or context_flag),
                                    secure_control=policy_id == NATIVE_SECURE_CONTROL_POLICY_ID and context_flag,
                                )
                                value = (
                                    64 * ((score + 9_999) // 10_000) - card.id
                                    if policy_id == "observer_safe_tactical_category_v1"
                                    else 100 * score - card.id
                                )
                                context = (incumbent_suit, incumbent_strength, partner_winning)
                                if policy_id in {NATIVE_SECURE_CONTROL_POLICY_ID, NATIVE_CLOSED_TRICK_FEED_POLICY_ID}:
                                    context += (int(context_flag),)
                                rows.append((*context, value))
                policy_follow_tables.append(tuple(rows))
                policy_follow_bounds.append(
                    (min(row[-1] for row in rows), max(row[-1] for row in rows))
                )
        for step in range(compiled_play_steps):
            model.add_element(play_indexes[step], owner_vars, actors[step])
            model.add_element(play_indexes[step], pool_ids, card_ids[step])
            model.add_element(play_indexes[step], pool_suits, suits[step])
            model.add_element(play_indexes[step], pool_ranks, ranks[step])
            model.add_element(play_indexes[step], pool_strengths, strengths[step])
            model.add_element(
                play_indexes[step], pool_sun_strengths, sun_strengths[step]
            )
            model.add_element(play_indexes[step], pool_points, points[step])

        leaders: list[Any] = []
        winner_seats: list[Any] = []
        winner_teams: list[Any] = []
        winner_cards: list[Any] = []
        winner_ranks: list[Any] = []
        winner_suits: list[Any] = []
        trick_points: list[Any] = []
        legal_flags_by_step: list[list[Any]] = []
        public_trick_context = []
        unplayed_by_step: list[list[Any]] = []
        public_played_by_step: list[list[Any]] = []
        owned_by_step: list[list[Any]] = []
        remaining_by_step: list[list[Any]] = []

        def winner(upto_steps: Sequence[int], led_suit, prefix_value: str):
            eligible = []
            trump_flags = [
                eq_const(suits[step], int(trump), "winner_trump")
                if contract.effective == Contract.HOKUM
                else fixed(False, "winner_trump")
                for step in upto_steps
            ]
            any_trump = either(trump_flags, "winner_any_trump")
            effective_strengths = []
            for index, step in enumerate(upto_steps):
                follows = eq_var(suits[step], led_suit, "winner_follows")
                use = model.new_bool_var(name("winner_eligible"))
                if contract.effective == Contract.HOKUM:
                    model.add(use == trump_flags[index]).only_enforce_if(any_trump)
                    model.add(use == follows).only_enforce_if(any_trump.Not())
                else:
                    model.add(use == follows)
                effective = model.new_int_var(0, 7, name("winner_strength"))
                model.add(effective == strengths[step]).only_enforce_if(use)
                model.add(effective == 0).only_enforce_if(use.Not())
                eligible.append(use)
                effective_strengths.append(effective)
            maximum = model.new_int_var(0, 7, name("winner_max"))
            model.add_max_equality(maximum, effective_strengths)
            chosen = [model.new_bool_var(name("winner_choice")) for _ in upto_steps]
            model.add_exactly_one(chosen)
            for index, chosen_flag in enumerate(chosen):
                model.add(eligible[index] == 1).only_enforce_if(chosen_flag)
                model.add(effective_strengths[index] == maximum).only_enforce_if(
                    chosen_flag
                )
                if self.native_conversion and selected_play_count is None:
                    # Eligible cards share one suit and have unique ranks.
                    # Expose the reverse implication to avoid tied-winner search.
                    strict = model.add(effective_strengths[index] < maximum)
                    strict.with_name(f"winner_unique_rank_{chosen_flag.index}")
                    strict.only_enforce_if([eligible[index], chosen_flag.Not()])
            chosen_position = model.new_int_var(
                0, len(upto_steps) - 1, name("winner_position")
            )
            model.add(
                chosen_position
                == sum(index * flag for index, flag in enumerate(chosen))
            )
            seat = model.new_int_var(0, 3, name(prefix_value + "_seat"))
            team = model.new_int_var(0, 1, name(prefix_value + "_team"))
            card = model.new_int_var(0, 31, name(prefix_value + "_card"))
            rank = model.new_int_var(0, 7, name(prefix_value + "_rank"))
            suit = model.new_int_var(0, 3, name(prefix_value + "_suit"))
            strength = model.new_int_var(0, 7, name(prefix_value + "_strength"))
            model.add_element(
                chosen_position, [actors[step] for step in upto_steps], seat
            )
            model.add_allowed_assignments(
                [seat, team], [(int(value), int(value.team)) for value in Seat]
            )
            model.add_element(
                chosen_position, [card_ids[step] for step in upto_steps], card
            )
            model.add_element(
                chosen_position, [ranks[step] for step in upto_steps], rank
            )
            model.add_element(
                chosen_position, [suits[step] for step in upto_steps], suit
            )
            model.add_element(
                chosen_position, [strengths[step] for step in upto_steps], strength
            )
            return seat, team, card, rank, suit, strength

        for trick in range(compiled_play_steps // 4):
            base = trick * 4
            leader = model.new_int_var(0, 3, f"leader_{trick}")
            if trick == 0:
                if first_leader is not None:
                    model.add(leader == int(Seat.parse(first_leader)))
            else:
                model.add(leader == winner_seats[-1])
            leaders.append(leader)
            for position in range(4):
                step = base + position
                model.add_allowed_assignments(
                    [leader, actors[step]],
                    [
                        (int(seat), int(structural.offset(seat, position)))
                        for seat in Seat
                    ],
                )
                prior_steps = list(range(base, step))
                fixed_prior = frozenset(
                    fixed_prefix_indexes[: min(step, len(fixed_prefix_indexes))]
                )
                dynamic_prior = range(len(fixed_prior), step)
                available = []
                unplayed = []
                public_played = []
                own_flags = []
                known_actor = (
                    normalized_prefix[step].actor if step < len(normalized_prefix)
                    else structural.offset(Seat.parse(first_leader), position)
                    if self.native_conversion and trick == 0 and first_leader is not None
                    else None
                )
                for index, owner in enumerate(owner_vars):
                    fixed_owners = allowed_owners[pool[index].id]
                    owned = (
                        fixed(
                            fixed_owners[0] == known_actor,
                            "available_owner",
                        )
                        if known_actor is not None and len(fixed_owners) == 1
                        else eq_const(actors[step], int(fixed_owners[0]), "available_owner")
                        if self.native_conversion and len(fixed_owners) == 1
                        else eq_var(owner, actors[step], "available_owner")
                    )
                    own_flags.append(owned)
                    if index in fixed_prior:
                        already = fixed(True, "already_played_any")
                    elif self.native_conversion and selected_play_count is None and step > len(normalized_prefix):
                        # Carry native history forward instead of rebuilding a
                        # growing disjunction. Selected-tail absence is not play.
                        already = either(
                            [unplayed_by_step[-1][index].Not(),
                             eq_const(play_indexes[step - 1], index, "already_played")],
                            "already_played_any",
                        )
                    else:
                        already = either(
                            [
                                eq_const(
                                    play_indexes[prior], index, "already_played"
                                )
                                for prior in dynamic_prior
                            ],
                            "already_played_any",
                        )
                    public_played.append(already)
                    unplayed.append(
                        together([selected_card_literals[index], already.Not()], "unplayed")
                        if selected_play_count is not None
                        else already.Not()
                    )
                    # In the first modeled trick each seat acts once. Ownership
                    # plus the actor-order constraints already excludes cards
                    # played by earlier actors; later tricks still need history.
                    availability = [owned]
                    if not (self.native_conversion and trick == 0):
                        availability.append(already.Not())
                    if selected_play_count is not None:
                        availability.insert(0, selected_card_literals[index])
                    available.append(together(availability, "available"))
                unplayed_by_step.append(unplayed)
                public_played_by_step.append(public_played)
                owned_by_step.append(own_flags)
                remaining_by_step.append(available)

                if position == 0:
                    if (
                        contract.effective == Contract.HOKUM
                        and locked
                        and domain.locked_hokum_blocks_trump_lead
                    ):
                        non_trump_exists = either(
                            [
                                available[index]
                                for index, card in enumerate(pool)
                                if card.suit != trump
                            ],
                            "lead_non_trump_exists",
                        )
                        legal_flags = [
                            together(
                                [
                                    available[index],
                                    either(
                                        [
                                            non_trump_exists.Not(),
                                            fixed(
                                                card.suit != trump,
                                                "candidate_non_trump",
                                            ),
                                        ],
                                        "lead_allowed",
                                    ),
                                ],
                                "lead_legal",
                            )
                            for index, card in enumerate(pool)
                        ]
                    else:
                        legal_flags = available
                else:
                    led_suit = suits[base]
                    _, current_team, _, _, current_suit, current_strength = winner(
                        prior_steps, led_suit, "current_winner"
                    )
                    actor_team = model.new_int_var(0, 1, name("actor_team"))
                    model.add_allowed_assignments(
                        [actors[step], actor_team],
                        [(int(seat), int(seat.team)) for seat in Seat],
                    )
                    partner_winning = eq_var(
                        current_team, actor_team, "partner_winning"
                    )
                    if self.native_conversion:
                        held_suits = [either(
                            (available[index] for index, card in enumerate(pool) if card.suit == suit),
                            "held_suit") for suit in Suit]
                        follow_exists = model.new_bool_var(name("follow_exists"))
                        model.add_element(led_suit, held_suits, follow_exists)
                    else:
                        follow_exists = either(
                            [
                                together(
                                    [
                                        available[index],
                                        eq_const(led_suit, int(card.suit), "same_led"),
                                    ],
                                    "available_follow",
                                )
                                for index, card in enumerate(pool)
                            ],
                            "follow_exists",
                        )
                    led_is_trump = (
                        eq_const(led_suit, int(trump), "led_trump")
                        if contract.effective == Contract.HOKUM
                        else fixed(False, "led_trump")
                    )
                    winner_is_trump = (
                        eq_const(current_suit, int(trump), "winner_trump")
                        if contract.effective == Contract.HOKUM
                        else fixed(False, "winner_trump")
                    )
                    trump_exists = (
                        either(
                            [
                                available[index]
                                for index, card in enumerate(pool)
                                if card.suit == trump
                            ],
                            "trump_exists",
                        )
                        if contract.effective == Contract.HOKUM
                        else fixed(False, "trump_exists")
                    )
                    higher_exists = either(
                        [
                            together(
                                [
                                    available[index],
                                    gt_const(
                                        pool_strengths[index],
                                        current_strength,
                                        "higher",
                                    ),
                                ],
                                "higher_available",
                            )
                            for index, card in enumerate(pool)
                            if contract.effective == Contract.HOKUM
                            and card.suit == trump
                        ],
                        "higher_exists",
                    )
                    partner_led = model.new_bool_var(name("partner_led"))
                    partner_pairs = [
                        (int(seat), int(structural.partner(seat))) for seat in Seat
                    ]
                    model.add_allowed_assignments(
                        [actors[step], actors[base]], partner_pairs
                    ).only_enforce_if(partner_led)
                    model.add_forbidden_assignments(
                        [actors[step], actors[base]], partner_pairs
                    ).only_enforce_if(partner_led.Not())
                    lead_is_ace = eq_const(ranks[base], int(Rank.ACE), "lead_ace")
                    special_cut = (
                        together(
                            [
                                follow_exists.Not(),
                                trump_exists,
                                partner_winning,
                                fixed(position == 2, "third_hand"),
                                partner_led,
                                lead_is_ace.Not(),
                                ekka_variables[trick].Not(),
                            ],
                            "special_cut",
                        )
                        if contract.effective == Contract.HOKUM
                        else fixed(False, "special_cut")
                    )
                    opponent_void = together(
                        [follow_exists.Not(), trump_exists, partner_winning.Not()],
                        "opponent_void",
                    )
                    opponent_cut = together(
                        [opponent_void, winner_is_trump.Not()], "opponent_cut"
                    )
                    opponent_over = together(
                        [opponent_void, winner_is_trump, higher_exists], "opponent_over"
                    )
                    opponent_under = together(
                        [
                            opponent_void,
                            winner_is_trump,
                            higher_exists.Not(),
                            fixed(
                                not domain.allow_discard_when_cannot_overtrump,
                                "no_discard",
                            ),
                        ],
                        "opponent_under",
                    )
                    force_trump = either(
                        [special_cut, opponent_cut, opponent_over, opponent_under],
                        "force_trump",
                    )
                    follow_over = together(
                        [
                            follow_exists,
                            led_is_trump,
                            partner_winning.Not(),
                            higher_exists,
                        ],
                        "follow_over",
                    )
                    force_higher = either([opponent_over, follow_over], "force_higher")
                    legal_flags = []
                    for index, card in enumerate(pool):
                        same_led = eq_const(led_suit, int(card.suit), "candidate_led")
                        candidate_trump = fixed(
                            contract.effective == Contract.HOKUM and card.suit == trump,
                            "candidate_trump",
                        )
                        candidate_higher = gt_const(
                            pool_strengths[index], current_strength, "candidate_higher"
                        )
                        legal_flags.append(
                            together(
                                [
                                    available[index],
                                    either(
                                        [follow_exists.Not(), same_led], "follow_ok"
                                    ),
                                    either(
                                        [force_trump.Not(), candidate_trump], "trump_ok"
                                    ),
                                    either(
                                        [force_higher.Not(), candidate_higher],
                                        "higher_ok",
                                    ),
                                ],
                                "legal",
                            )
                        )
                legal_flags_by_step.append(legal_flags)
                public_trick_context.append(None if position == 0 else (current_suit, current_strength, partner_winning))
                chosen_legal = model.new_bool_var(name("chosen_legal"))
                model.add_element(play_indexes[step], legal_flags, chosen_legal)
                model.add(chosen_legal == 1)
                if step < policy_prefix_steps:
                    if position == 0:
                        policy_values = policy_lead_values
                        policy_min, policy_max = min(policy_values), max(policy_values)
                        if policy_id == NATIVE_ESTABLISHED_CONTROL_POLICY_ID:
                            scores, score_bounds, _ = _native_established_lead_scores_v1(model,
                                pool=pool, owners=owner_vars, actor=actors[step], played=public_played,
                                domain=domain, contract=contract, trump=trump, name=f"established_prefix_{step}")
                            policy_values = [100 * score - card.id for score, card in zip(scores, pool, strict=True)]
                            policy_min = min(100 * bound[0] - card.id for bound, card in zip(score_bounds, pool, strict=True))
                            policy_max = max(100 * bound[1] - card.id for bound, card in zip(score_bounds, pool, strict=True))
                        model.add(ekka_variables[trick] == 0)
                    else:
                        policy_values = []
                        for index, rows in enumerate(policy_follow_tables):
                            lower, upper = policy_follow_bounds[index]
                            value = model.new_int_var(lower, upper, name("policy_card_value"))
                            context = [current_suit, current_strength, partner_winning]
                            if policy_id == NATIVE_SECURE_CONTROL_POLICY_ID:
                                context.append(model.new_constant(int(position in (1, 2))))
                            elif policy_id == NATIVE_CLOSED_TRICK_FEED_POLICY_ID:
                                context.append(model.new_constant(int(position == 3)))
                            model.add_allowed_assignments(
                                [*context, value],
                                rows,
                            )
                            policy_values.append(value)
                        policy_min = min(bound[0] for bound in policy_follow_bounds)
                        policy_max = max(bound[1] for bound in policy_follow_bounds)
                        if policy_id == NATIVE_GUARD_PROTECTION_POLICY_ID:
                            guards = _native_guard_discard_flags_v1(model, pool=pool,
                                owners=owner_vars, actor=actors[step], played=public_played,
                                led_suit=suits[base], partner_winning=partner_winning,
                                domain=domain, contract=contract, trump=trump, name=f"guard_prefix_{step}",
                                owned_flags=own_flags, remaining_flags=available)
                            bonus = 100 * _guard_discard_bonus_v1(domain, contract, trump)
                            policy_values = [value - bonus * guard
                                for value, guard in zip(policy_values, guards, strict=True)]
                            policy_min -= bonus
                    selected_value = model.new_int_var(
                        policy_min, policy_max, name("policy_selected_value")
                    )
                    model.add_element(play_indexes[step], policy_values, selected_value)
                    for legal, value in zip(legal_flags, policy_values, strict=True):
                        model.add(selected_value >= value).only_enforce_if(legal)

            led_suit = suits[base]
            final_seat, final_team, final_card, final_rank, final_suit, _ = winner(
                list(range(base, base + 4)), led_suit, "trick_winner"
            )
            winner_seats.append(final_seat)
            winner_teams.append(final_team)
            winner_cards.append(final_card)
            winner_ranks.append(final_rank)
            winner_suits.append(final_suit)
            raw_points = model.new_int_var(0, 128, f"trick_points_{trick}")
            model.add(raw_points == sum(points[base : base + 4]))
            trick_points.append(raw_points)
            if contract.effective == Contract.HOKUM:
                model.add(suits[base] != int(trump)).only_enforce_if(
                    ekka_variables[trick]
                )
                for index, card in enumerate(pool):
                    same_suit = eq_const(suits[base], int(card.suit), "ekka_same_suit")
                    model.add(
                        domain.sun_strength[int(card.rank)] <= sun_strengths[base]
                    ).only_enforce_if(
                        (same_suit, unplayed_by_step[base][index], ekka_variables[trick])
                    )

        raw_team_vars: list[Any] = []
        raw_team_prefixes: list[list[Any]] = []
        game_team_vars: list[Any] = []
        if scoring_domain is not None and compiled_play_steps == len(pool):
            raw_maximum = scoring_domain.raw_card_totals[contract.effective.value]
            conversion = [
                scoring_domain.convert_raw(raw, contract)
                for raw in range(raw_maximum + 1)
            ] if not self.native_conversion else None
            for team in Team:
                raw_team = model.new_int_var(0, raw_maximum, f"raw_team_{int(team)}")
                contributions = []
                prefixes = [model.new_constant(0)]
                for trick, trick_raw in enumerate(trick_points):
                    won = eq_const(winner_teams[trick], int(team), "raw_team_won")
                    contribution = model.new_int_var(
                        0, 128, name("raw_team_contribution")
                    )
                    model.add(contribution == trick_raw).only_enforce_if(won)
                    model.add(contribution == 0).only_enforce_if(won.Not())
                    contributions.append(contribution)
                    if trick < len(trick_points) - 1:
                        running = model.new_int_var(0, raw_maximum, name("raw_team_prefix"))
                        model.add(running == prefixes[-1] + contribution)
                        prefixes.append(running)
                last_won = eq_const(winner_teams[-1], int(team), "last_trick_won")
                model.add(
                    raw_team
                    == sum(contributions)
                    + scoring_domain.last_trick_bonus_raw * last_won
                )
                if self.native_conversion:
                    from .ortools_scoring_stage import attach_raw_conversion_v1

                    game_team = attach_raw_conversion_v1(
                        model, raw_team, domain=scoring_domain, contract=contract,
                        name=f"game_team_{int(team)}",
                    )
                else:
                    game_team = model.new_int_var(
                        min(conversion), max(conversion), f"game_team_{int(team)}"
                    )
                    model.add_element(raw_team, conversion, game_team)
                raw_team_vars.append(raw_team)
                raw_team_prefixes.append([*prefixes, raw_team])
                game_team_vars.append(game_team)
            model.add(sum(raw_team_vars) == raw_maximum)

        for step, play in enumerate(normalized_prefix):
            model.add(actors[step] == int(play.actor))
            model.add(play_indexes[step] == card_index[play.card.id])
            model.add(card_ids[step] == play.card.id)
            if play.ekka and step % 4 != 0:
                raise ValueError("Ekka may be declared only on a trick lead")
            if step % 4 == 0:
                model.add(ekka_variables[step // 4] == int(play.ekka))
        tail_start = compiled_play_steps - len(normalized_tail)
        for offset, play in enumerate(normalized_tail):
            step = tail_start + offset
            model.add(actors[step] == int(play.actor))
            model.add(play_indexes[step] == card_index[play.card.id])
            model.add(card_ids[step] == play.card.id)
            if play.ekka and step % 4 != 0:
                raise ValueError("Ekka may be declared only on a trick lead")
            if step % 4 == 0:
                model.add(ekka_variables[step // 4] == int(play.ekka))

        def constrain_selector(variable, values: Sequence[int], label: str) -> None:
            if values:
                selected_from(values, variable, label)

        for item in action_constraints:
            step = item.step
            if item.actor is not None:
                model.add(actors[step] == int(item.actor))
            if item.team is not None:
                model.add_allowed_assignments(
                    [actors[step]],
                    [(int(seat),) for seat in Seat if seat.team == item.team],
                )
            constrain_selector(card_ids[step], item.card_ids, "action_card")
            constrain_selector(
                ranks[step], [int(value) for value in item.ranks], "action_rank"
            )
            constrain_selector(
                suits[step], [int(value) for value in item.suits], "action_suit"
            )
            if item.actor_team_wins_trick is not None:
                actor_team = model.new_int_var(0, 1, name("constraint_actor_team"))
                model.add_allowed_assignments(
                    [actors[step], actor_team],
                    [(int(seat), int(seat.team)) for seat in Seat],
                )
                if item.actor_team_wins_trick:
                    model.add(actor_team == winner_teams[step // 4])
                else:
                    model.add(actor_team != winner_teams[step // 4])

        for item in winner_constraints:
            trick = item.trick
            if item.seat is not None:
                model.add(winner_seats[trick] == int(item.seat))
            if item.team is not None:
                model.add(winner_teams[trick] == int(item.team))
            constrain_selector(winner_cards[trick], item.card_ids, "winner_card")
            constrain_selector(
                winner_ranks[trick], [int(value) for value in item.ranks], "winner_rank"
            )
            constrain_selector(
                winner_suits[trick], [int(value) for value in item.suits], "winner_suit"
            )

        for item in card_count_constraints:
            matches = []
            for step in range(item.trick * 4, item.trick * 4 + 4):
                selectors = []
                if item.owner_team is not None:
                    selectors.append(
                        either(
                            [
                                eq_const(actors[step], int(seat), "count_team")
                                for seat in Seat
                                if seat.team == item.owner_team
                            ],
                            "count_team_any",
                        )
                    )
                if item.card_ids:
                    selectors.append(
                        either(
                            [
                                eq_const(card_ids[step], value, "count_card")
                                for value in item.card_ids
                            ],
                            "count_card_any",
                        )
                    )
                if item.ranks:
                    selectors.append(
                        either(
                            [
                                eq_const(ranks[step], int(value), "count_rank")
                                for value in item.ranks
                            ],
                            "count_rank_any",
                        )
                    )
                if item.suits:
                    selectors.append(
                        either(
                            [
                                eq_const(suits[step], int(value), "count_suit")
                                for value in item.suits
                            ],
                            "count_suit_any",
                        )
                    )
                matches.append(together(selectors, "count_match"))
            model.add(sum(matches) >= item.minimum)
            model.add(sum(matches) <= item.maximum)

        tail_step_offset = 32 - (
            selected_play_count if selected_play_count is not None else len(pool)
        )
        context_prefix = (
            len(normalized_prefix)
            if decision_prefix_length is None
            else decision_prefix_length
        )
        context_prefix_absolute = tail_step_offset + context_prefix
        absent_slot = CpSatPlayPrefixActionV1(
            model.new_constant(0),
            model.new_constant(0),
            model.new_constant(0),
            model.new_constant(0),
        )
        play_slots = [absent_slot for _ in range(32)]
        action_variables: list[CpSatActionVariablesV1] = []
        for step in range(compiled_play_steps):
            absolute_step = tail_step_offset + step
            ekka = (
                ekka_variables[step // 4]
                if step % 4 == 0
                else model.new_constant(0)
            )
            play_slots[absolute_step] = CpSatPlayPrefixActionV1(
                card_ids[step],
                actors[step],
                ekka,
                model.new_constant(int(step < context_prefix)),
            )
            action_variables.append(
                CpSatActionVariablesV1(
                    f"PLAY-{absolute_step:02d}",
                    card_ids[step],
                    actors[step],
                    actors[step],
                    ekka,
                    model.new_constant(1),
                    None,
                )
            )

        trick_contexts: dict[int, CpSatTrickVariablesV1] = {}
        for relative_trick in range(compiled_play_steps // 4):
            absolute_trick = tail_step_offset // 4 + relative_trick
            base = relative_trick * 4
            winner_ekka = (
                together(
                    [
                        eq_var(
                            winner_cards[relative_trick],
                            card_ids[base],
                            "winner_lead",
                        ),
                        ekka_variables[relative_trick],
                    ],
                    "winner_ekka",
                )
                if _export_predicate_contexts
                else model.new_constant(0)
            )
            winner_action = CpSatActionVariablesV1(
                f"WINNER-{absolute_trick}",
                winner_cards[relative_trick],
                winner_seats[relative_trick],
                winner_seats[relative_trick],
                winner_ekka,
                model.new_constant(1),
                model.new_constant(1),
            )
            trick_contexts[absolute_trick] = CpSatTrickVariablesV1(
                absolute_trick,
                tuple(action_variables[base : base + 4]),
                winner_action,
            )

        current_size = context_prefix % 4
        current_start = context_prefix - current_size
        current_slots = []
        for position in range(4):
            relative_step = current_start + position
            if position < current_size and relative_step < compiled_play_steps:
                current_slots.append(
                    CpSatPlayPrefixActionV1(
                        card_ids[relative_step],
                        actors[relative_step],
                        (
                            ekka_variables[relative_step // 4]
                            if relative_step % 4 == 0
                            else model.new_constant(0)
                        ),
                        model.new_constant(1),
                    )
                )
            else:
                current_slots.append(absent_slot)

        remaining_sizes = {}
        context_cards_per_seat = (
            cards_per_seat if selected_play_count is not None
            else owner_cards_per_seat
        )
        completed_tricks = context_prefix // 4
        partial_steps = range(completed_tricks * 4, context_prefix)
        for seat in Seat:
            if _export_predicate_contexts:
                played_by_seat = [
                    eq_const(actors[step], int(seat), "context_played_by_seat")
                    for step in partial_steps
                ]
                remaining = model.new_int_var(
                    0, context_cards_per_seat, f"remaining_{seat.name}"
                )
                model.add(
                    remaining == context_cards_per_seat - completed_tricks - sum(played_by_seat)
                )
            else:
                remaining = model.new_constant(context_cards_per_seat - completed_tricks)
            remaining_sizes[seat] = remaining

        decision_legal = (
            legal_flags_by_step[context_prefix]
            if context_prefix < compiled_play_steps
            else [model.new_constant(0) for _ in pool]
        )
        legal_card_literals = {
            card_id: (
                decision_legal[card_index[card_id]]
                if card_id in card_index
                else model.new_constant(0)
            )
            for card_id in range(32)
        }
        def ekka_legal_literal(step, index):
            card = pool[index]
            if (
                step >= compiled_play_steps or step % 4
                or not domain.ekka_enabled or contract.effective != Contract.HOKUM
                or card.suit == trump
            ):
                return model.new_constant(0)
            stronger = either(
                [
                    unplayed_by_step[step][other]
                    for other, candidate in enumerate(pool)
                    if candidate.suit == card.suit
                    and domain.sun_strength[int(candidate.rank)] > domain.sun_strength[int(card.rank)]
                ],
                "ekka_stronger_available",
            )
            return together([legal_flags_by_step[step][index], stronger.Not()], "ekka_legal")

        native_ekka_flags = None
        if self.native_conversion:
            if not domain.ekka_enabled or contract.effective != Contract.HOKUM:
                disabled_ekka_row = (model.new_constant(0),) * len(pool)
                native_ekka_flags = (disabled_ekka_row,) * compiled_play_steps
            else:
                native_ekka_flags = tuple(
                    tuple(ekka_legal_literal(step, index) for index in range(len(pool)))
                    for step in range(compiled_play_steps)
                )
        legal_action_literals = {}
        for card_id in range(32):
            normal = legal_card_literals[card_id]
            legal_action_literals[(card_id, False)] = normal
            if _export_predicate_contexts and card_id in card_index:
                legal_action_literals[(card_id, True)] = (
                    native_ekka_flags[context_prefix][card_index[card_id]]
                    if native_ekka_flags is not None and context_prefix < compiled_play_steps
                    else ekka_legal_literal(context_prefix, card_index[card_id])
                )
            else:
                legal_action_literals[(card_id, True)] = model.new_constant(0)

        current_raw = {}
        completed_relative_tricks = context_prefix // 4
        context_raw_maximum = (
            scoring_domain.raw_card_totals[contract.effective.value]
            if scoring_domain is not None
            else sum(
                int(list(points.proto.domain)[-1])
                for points in trick_points[:completed_relative_tricks]
            )
        )
        for team in Team:
            if not _export_predicate_contexts:
                current_raw[team] = model.new_constant(0)
                continue
            contributions = []
            for trick in range(completed_relative_tricks):
                won = eq_const(winner_teams[trick], int(team), "context_raw_won")
                value = model.new_int_var(0, 128, name("context_raw_value"))
                model.add(value == trick_points[trick]).only_enforce_if(won)
                model.add(value == 0).only_enforce_if(won.Not())
                contributions.append(value)
            raw = model.new_int_var(
                0, context_raw_maximum, f"context_raw_{int(team)}"
            )
            model.add(raw == sum(contributions))
            current_raw[team] = raw

        current_game = {}
        for team in Team:
            if not _export_predicate_contexts or scoring_domain is None:
                current_game[team] = model.new_constant(0)
                continue
            raw_maximum = scoring_domain.raw_card_totals[contract.effective.value]
            bounded_raw = model.new_int_var(0, raw_maximum, f"bounded_raw_{int(team)}")
            model.add(bounded_raw == current_raw[team])
            if self.native_conversion:
                from .ortools_scoring_stage import attach_raw_conversion_v1

                converted = attach_raw_conversion_v1(
                    model, bounded_raw, domain=scoring_domain, contract=contract,
                    name=f"context_game_{int(team)}",
                )
            else:
                converted = model.new_int_var(0, 32, f"context_game_{int(team)}")
                model.add_element(
                    bounded_raw,
                    [
                        scoring_domain.convert_raw(raw, contract)
                        for raw in range(raw_maximum + 1)
                    ],
                    converted,
                )
            current_game[team] = converted

        play_state_context = CpSatPlayStatePredicateContextV1(
            model=model,
            context_id=f"play-tail-{canonical_sha256(request_core)[:16]}",
            play_slots=tuple(play_slots),
            prefix_length=model.new_constant(context_prefix_absolute),
            trick_index=model.new_constant(context_prefix_absolute // 4),
            next_actor=(
                actors[context_prefix]
                if context_prefix < compiled_play_steps
                else model.new_constant(4)
            ),
            remaining_hand_sizes=remaining_sizes,
            current_trick_slots=tuple(current_slots),
            current_trick_size=model.new_constant(current_size),
            current_trick_active=model.new_constant(int(current_size > 0)),
            current_trick_leader=(
                actors[current_start] if current_size else model.new_constant(0)
            ),
            current_trick_suit_led=(
                suits[current_start] if current_size else model.new_constant(0)
            ),
            trump_suit=model.new_constant(int(trump) if trump is not None else 4),
            completed_trick_literals={
                trick: model.new_constant(int(trick < context_prefix_absolute // 4))
                for trick in range(8)
            },
            completed_trick_winners={
                trick: (
                    winner_seats[trick - tail_step_offset // 4]
                    if tail_step_offset // 4
                    <= trick
                    < tail_step_offset // 4 + len(winner_seats)
                    else model.new_constant(0)
                )
                for trick in range(8)
            },
            legal_card_literals=legal_card_literals,
            legal_action_literals=legal_action_literals,
            phase=model.new_constant(
                tuple(GamePhase).index(
                    GamePhase.SCORED
                    if context_prefix_absolute == 32
                    else GamePhase.PLAYING
                )
            ),
            match_scores={team: model.new_constant(0) for team in Team},
            raw_points=current_raw,
            project_points={team: model.new_constant(0) for team in Team},
            game_points=current_game,
        )
        predicate_context = CpSatPredicateModelContextV1(
            model=model,
            context_id=play_state_context.context_id,
            tricks=trick_contexts,
            root_actions=(),
            proof_complete=model.new_constant(1),
        )
        model.add_decision_strategy(
            owner_vars + play_indexes, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE
        )
        extraction = _PlayTailExtractionV1(
            tuple(pool),
            tuple(owner_vars),
            tuple(selected_card_literals),
            selected_play_count is not None,
            tuple(actors),
            tuple(card_ids),
            tuple(ekka_variables),
            context_prefix,
            tuple(tuple(flags) for flags in legal_flags_by_step),
            native_ekka_flags,
            domain,
            contract,
            trump,
            tuple(winner_seats),
            tuple(winner_cards),
            tuple(trick_points),
            tuple(raw_team_vars),
            tuple(game_team_vars),
            scoring_domain,
            tuple(public_trick_context),
            tuple(tuple(flags) for flags in public_played_by_step),
            tuple(tuple(flags) for flags in owned_by_step),
            tuple(tuple(flags) for flags in remaining_by_step),
        )
        return OrToolsPlayTailBuildV1(
            model,
            play_state_context,
            predicate_context,
            tuple(selected_card_literals),
            tuple(play_indexes),
            tuple(owner_vars),
            tuple(ekka_variables),
            tuple(winner_teams),
            tuple(trick_points),
            tuple(raw_team_vars),
            tuple(game_team_vars),
            canonical_sha256(request_core),
            extraction,
            tuple(zip(*raw_team_prefixes)),
        )


__all__ = [
    "OrToolsPlayTailBuildV1",
    "OrToolsPlayTailResultV1",
    "OrToolsPlayTailStageV1",
    "PlayZeroObservationActionNoGoodV1",
    "PlayTailSolveStopped",
    "TailActionConstraintV1",
    "TailPrefixActionV1",
    "TailTrickCardCountConstraintV1",
    "TailTrickWinnerConstraintV1",
    "apply_play_zero_observation_action_no_good_v1",
]
