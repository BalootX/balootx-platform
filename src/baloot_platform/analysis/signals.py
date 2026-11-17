from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from baloot_platform.core.enums import Seat
from baloot_platform.core.models import PlayedCard
from baloot_platform.core.play_ir import PlayDomainIRV1
from .models import InformationState, SignalObservation


PRODUCTION_SIGNAL_KINDS = frozenset(
    {
        "high_expenditure",
        "high_then_low",
        "low_then_high",
        "minimum_winner",
        "partner_return",
    }
)
PROBABILITY_SIGNAL_KINDS = PRODUCTION_SIGNAL_KINDS


def _relationship(observer: Seat, actor: Seat) -> str:
    if actor == observer:
        return "self"
    if actor == observer.partner:
        return "partner"
    if actor == observer.next():
        return "right_opponent"
    return "left_opponent"


def _winner(
    plays: Sequence[PlayedCard],
    info: InformationState,
    play_domain: PlayDomainIRV1,
) -> PlayedCard:
    led = plays[0].card.suit
    trump = info.trump_suit if info.contract.effective.value == "hokum" else None

    def key(play: PlayedCard) -> tuple[int, int]:
        suit_priority = 2 if trump is not None and play.card.suit == trump else int(
            play.card.suit == led
        )
        return suit_priority, play_domain.strength(
            play.card, info.contract, info.trump_suit
        )

    return max(plays, key=key)


def detect_signals(
    info: InformationState,
    play_domain: PlayDomainIRV1 | None = None,
) -> tuple[SignalObservation, ...]:
    # Signal order and choice quality depend on the active contract's strength
    # table. Fail closed when that frozen rules authority is unavailable.
    if play_domain is None:
        return ()

    by_player_suit: dict[tuple, list] = defaultdict(list)
    observations: list[SignalObservation] = []
    for play in info.plays:
        by_player_suit[(play.seat, play.card.suit)].append(play)
    for (seat, _suit), plays in by_player_suit.items():
        if len(plays) < 2:
            continue
        first, second = plays[-2], plays[-1]
        first_strength = play_domain.strength(
            first.card, info.contract, info.trump_suit
        )
        second_strength = play_domain.strength(
            second.card, info.contract, info.trump_suit
        )
        if first_strength < second_strength:
            kind = "low_then_high"
            interpretations = ("possible encouragement or preserved strength", "could be forced, tactical, or deceptive")
        else:
            kind = "high_then_low"
            interpretations = ("possible shortness or discouragement", "could be forced, tactical, or deceptive")
        forcedness = "unknown"
        confidence = 0.35
        if first.seat == info.observer:
            alternatives = len(first.legal_cards)
            forcedness = (
                "forced"
                if alternatives <= 1
                else "near_forced"
                if alternatives == 2
                else "meaningful_choice"
                if alternatives <= 4
                else "highly_optional"
            )
            confidence = (
                0.15
                if alternatives <= 1
                else 0.42
                if alternatives == 2
                else 0.58
                if alternatives <= 4
                else 0.66
            )
        observations.append(SignalObservation(kind, seat, _relationship(info.observer, seat), (first.card, second.card), confidence, forcedness, interpretations, "A card-order pattern is an observation, not proof of intention."))

    # Choice-quality conventions are observer-safe only for the observer's own
    # recorded alternatives. Opponent alternatives are intentionally redacted.
    by_trick: dict[int, list[PlayedCard]] = defaultdict(list)
    for play in info.plays:
        by_trick[play.trick_index].append(play)
    for plays in by_trick.values():
        ordered = sorted(plays, key=lambda item: item.position)
        if len(ordered) != 4:
            continue
        played = next(
            (item for item in ordered if item.seat == info.observer), None
        )
        if played is None or not played.legal_cards:
            continue
        winning_alternatives = []
        for card in played.legal_cards:
            replaced = [
                PlayedCard(
                    item.seat,
                    card,
                    item.trick_index,
                    item.position,
                    item.legal_cards,
                    item.ekka,
                    item.baloot,
                )
                if item is played
                else item
                for item in ordered
            ]
            if _winner(replaced, info, play_domain).seat == info.observer:
                winning_alternatives.append(card)
        if not winning_alternatives or _winner(
            ordered, info, play_domain
        ).seat != info.observer:
            continue
        played_strength = play_domain.strength(
            played.card, info.contract, info.trump_suit
        )
        minimum_strength = min(
            play_domain.strength(card, info.contract, info.trump_suit)
            for card in winning_alternatives
        )
        kind = (
            "minimum_winner"
            if played_strength == minimum_strength
            else "high_expenditure"
        )
        interpretations = (
            ("possible strength preservation", "could still be tactical")
            if kind == "minimum_winner"
            else ("possible urgency or signal", "could be error or deception")
        )
        alternatives = len(played.legal_cards)
        forcedness = (
            "forced"
            if alternatives <= 1
            else "near_forced"
            if alternatives == 2
            else "meaningful_choice"
            if alternatives <= 4
            else "highly_optional"
        )
        observations.append(
            SignalObservation(
                kind,
                played.seat,
                "self",
                (played.card,),
                0.58,
                forcedness,
                interpretations,
                "Choice-quality conventions are soft evidence, never proof of intention.",
            )
        )
    # Partner suit return.
    completed = sorted(info.plays, key=lambda p: (p.trick_index, p.position))
    last_lead_by_seat = {}
    for play in completed:
        if play.position == 0:
            partner = play.seat.partner
            if partner in last_lead_by_seat and last_lead_by_seat[partner] == play.card.suit:
                observations.append(SignalObservation("partner_return", play.seat, _relationship(info.observer, play.seat), (play.card,), 0.48, "unknown", ("possible support for partner's suit", "could simply be tactically best"), "Return conventions vary by partnership and table."))
            last_lead_by_seat[play.seat] = play.card.suit
    return tuple(observations)


__all__ = [
    "PROBABILITY_SIGNAL_KINDS",
    "PRODUCTION_SIGNAL_KINDS",
    "detect_signals",
]
