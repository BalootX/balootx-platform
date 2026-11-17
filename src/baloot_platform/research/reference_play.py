from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Rank, Seat, Suit
from baloot_platform.core.play_ir import PlayDomainIRV1
from baloot_platform.core.structural_ir import StructuralDomainIRV1


@dataclass(frozen=True, slots=True)
class ReferencePlayV1:
    seat: Seat
    card: Card
    ekka: bool = False


class ReferencePlayInterpreterV1:
    """Pure play-legality and trick interpreter independent of BalootEngine."""

    def __init__(
        self,
        domain: PlayDomainIRV1,
        structural_domain: StructuralDomainIRV1,
    ):
        if not isinstance(domain, PlayDomainIRV1):
            raise TypeError("reference play requires PlayDomainIRV1")
        if not isinstance(structural_domain, StructuralDomainIRV1):
            raise TypeError("reference play requires StructuralDomainIRV1")
        self.structural_domain = structural_domain.require_integrity()
        self.domain = domain.require_integrity(self.structural_domain)

    @classmethod
    def from_rules(cls, rules: Mapping[str, Any]) -> "ReferencePlayInterpreterV1":
        structural = StructuralDomainIRV1.from_rules(rules)
        return cls(
            PlayDomainIRV1.from_rules(
                rules,
                structural_domain=structural,
            ),
            structural,
        )

    def resolve_first_leader(self, *, dealer: Seat, buyer: Seat) -> Seat:
        dealer = Seat.parse(dealer)
        buyer = Seat.parse(buyer)
        if self.domain.first_leader == "dealer_right":
            order = self.structural_domain.play_order
            return order[(order.index(dealer) + 1) % len(order)]
        if self.domain.first_leader == "buyer":
            return buyer
        if self.domain.first_leader == "configured":
            return self.domain.configured_first_leader
        raise AssertionError("validated play domain has an invalid first leader")

    def strength(self, card: Card, contract: Contract, trump: Suit | None) -> int:
        card = Card.parse(card)
        effective = Contract.parse(contract).effective
        if effective == Contract.HOKUM and trump is not None and card.suit == trump:
            table = self.domain.hokum_trump_strength
        elif effective == Contract.HOKUM:
            table = self.domain.hokum_non_trump_strength
        else:
            table = self.domain.sun_strength
        return table[int(card.rank)]

    def points(self, card: Card, contract: Contract, trump: Suit | None) -> int:
        card = Card.parse(card)
        effective = Contract.parse(contract).effective
        if effective == Contract.HOKUM and trump is not None and card.suit == trump:
            table = self.domain.hokum_trump_points
        elif effective == Contract.HOKUM:
            table = self.domain.hokum_non_trump_points
        else:
            table = self.domain.sun_points
        return table[int(card.rank)]

    def trick_winner(
        self,
        plays: Sequence[ReferencePlayV1],
        *,
        contract: Contract,
        trump: Suit | None,
    ) -> Seat:
        if not plays:
            raise ValueError("cannot resolve empty trick")
        led = plays[0].card.suit
        effective_trump = trump if Contract.parse(contract).effective == Contract.HOKUM else None
        if effective_trump is not None and any(play.card.suit == effective_trump for play in plays):
            eligible = [play for play in plays if play.card.suit == effective_trump]
        else:
            eligible = [play for play in plays if play.card.suit == led]
        return max(
            eligible,
            key=lambda play: self.strength(play.card, contract, trump),
        ).seat

    def trick_raw_points(
        self,
        plays: Sequence[ReferencePlayV1],
        *,
        contract: Contract,
        trump: Suit | None,
    ) -> int:
        return sum(self.points(play.card, contract, trump) for play in plays)

    def legal_moves(
        self,
        hand: Sequence[Card],
        plays: Sequence[ReferencePlayV1],
        *,
        actor: Seat,
        contract: Contract,
        trump: Suit | None,
        locked: bool = False,
    ) -> tuple[Card, ...]:
        cards = tuple(sorted((Card.parse(card) for card in hand), key=lambda card: card.id))
        actor = Seat.parse(actor)
        contract = Contract.parse(contract)
        if not plays:
            if (
                self.domain.locked_hokum_blocks_trump_lead
                and contract.effective == Contract.HOKUM
                and locked
                and trump is not None
            ):
                non_trump = tuple(card for card in cards if card.suit != trump)
                if non_trump:
                    return non_trump
            return cards

        led = plays[0].card.suit
        followers = tuple(card for card in cards if card.suit == led)
        effective_trump = trump if contract.effective == Contract.HOKUM else None
        winner = self.trick_winner(plays, contract=contract, trump=trump)
        winning_play = next(play for play in plays if play.seat == winner)

        if followers and self.domain.must_follow_suit:
            if (
                self.domain.must_overtrump_when_possible
                and effective_trump is not None
                and led == effective_trump
                and not self._same_team(winner, actor)
            ):
                higher = tuple(
                    card
                    for card in followers
                    if self.strength(card, contract, trump)
                    > self.strength(winning_play.card, contract, trump)
                )
                if higher:
                    return higher
            return followers

        if contract.effective != Contract.HOKUM or effective_trump is None:
            return cards
        trumps = tuple(card for card in cards if card.suit == effective_trump)
        if not self.domain.must_trump_when_void or not trumps:
            return cards
        if self._same_team(winner, actor) and self.domain.partner_winning_exception == "position_aware_ace_or_ekka":
            position = len(plays)
            partner_led = plays[0].seat == self.structural_domain.partner(actor)
            partner_ace = plays[0].card.rank == Rank.ACE
            if position == 2 and partner_led and not (plays[0].ekka or partner_ace):
                return trumps
            return cards
        if winning_play.card.suit == effective_trump and self.domain.must_overtrump_when_possible:
            higher = tuple(
                card
                for card in trumps
                if self.strength(card, contract, trump)
                > self.strength(winning_play.card, contract, trump)
            )
            if higher:
                return higher
            return cards if self.domain.allow_discard_when_cannot_overtrump else trumps
        return trumps

    def _same_team(self, left: Seat, right: Seat) -> bool:
        left = Seat.parse(left)
        right = Seat.parse(right)
        return left == right or self.structural_domain.partner(left) == right

    def can_declare_ekka(
        self,
        *,
        card: Card,
        remaining_cards: Sequence[Card],
        contract: Contract,
        trump: Suit | None,
    ) -> bool:
        if not self.domain.ekka_enabled:
            return False
        card = Card.parse(card)
        if Contract.parse(contract).effective != Contract.HOKUM or trump is None or card.suit == trump:
            return False
        strength = self.strength(card, Contract.SUN, None)
        return not any(
            other.suit == card.suit
            and self.strength(other, Contract.SUN, None) > strength
            for other in (Card.parse(value) for value in remaining_cards)
        )
