from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from baloot_platform.core.enums import Seat
from baloot_platform.core.structural_ir import StructuralDomainIRV1

_MASK64 = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class StructuralDealLayoutV1:
    dealer: Seat
    buyer: Seat | None
    initial_packets: tuple[tuple[int, Seat, tuple[int, ...]], ...]
    final_packets: tuple[tuple[Seat, tuple[int, ...]], ...]
    purchase_position: int
    undealt_positions: tuple[int, ...]
    initial_positions: tuple[tuple[int, ...], ...]
    final_positions: tuple[tuple[int, ...], ...] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dealer": self.dealer.name.lower(),
            "buyer": self.buyer.name.lower() if self.buyer is not None else None,
            "initialPackets": [
                {
                    "packet": packet,
                    "seat": seat.name.lower(),
                    "positions": list(positions),
                }
                for packet, seat, positions in self.initial_packets
            ],
            "finalPackets": [
                {"seat": seat.name.lower(), "positions": list(positions)}
                for seat, positions in self.final_packets
            ],
            "purchasePosition": self.purchase_position,
            "undealtPositions": list(self.undealt_positions),
            "initialPositions": [list(value) for value in self.initial_positions],
            "finalPositions": (
                [list(value) for value in self.final_positions]
                if self.final_positions is not None
                else None
            ),
        }


class ReferenceStructuralInterpreterV1:
    """Independent, direct interpreter for structural Baloot semantics."""

    def __init__(self, domain: StructuralDomainIRV1):
        if not isinstance(domain, StructuralDomainIRV1):
            raise TypeError(
                "ReferenceStructuralInterpreterV1 requires StructuralDomainIRV1"
            )
        self.domain = domain.require_integrity()

    @classmethod
    def from_rules(cls, rules: Mapping[str, Any]) -> "ReferenceStructuralInterpreterV1":
        return cls(StructuralDomainIRV1.from_rules(rules))

    def _offset(self, seat: Seat, steps: int) -> Seat:
        return self.domain.offset(seat, steps)

    def dealer_right(self, dealer: Seat) -> Seat:
        return self._offset(dealer, 1)

    def dealer_left(self, dealer: Seat) -> Seat:
        return self._offset(dealer, -1)

    def next_dealer(self, dealer: Seat) -> Seat:
        return self.domain.next_dealer(dealer)

    def partner(self, seat: Seat) -> Seat:
        return self.domain.partner(seat)

    def annulment_actor(self, dealer: Seat) -> Seat:
        return self.domain.annulment_actor(dealer)

    def layout(self, *, dealer: Seat, buyer: Seat | None = None) -> StructuralDealLayoutV1:
        dealer = Seat.parse(dealer)
        buyer = Seat.parse(buyer) if buyer is not None else None
        order = self.domain.recipient_order(dealer)
        by_seat: dict[Seat, list[int]] = {seat: [] for seat in Seat}
        initial: list[tuple[int, Seat, tuple[int, ...]]] = []
        cursor = 0
        for packet, size in enumerate(self.domain.initial_packets, 1):
            for seat in order:
                positions = tuple(range(cursor, cursor + size))
                cursor += size
                by_seat[seat].extend(positions)
                initial.append((packet, seat, positions))
        purchase_position = cursor
        cursor += self.domain.exposed_purchase_cards
        undealt = tuple(range(cursor, self.domain.card_count))
        final_packets: list[tuple[Seat, tuple[int, ...]]] = []
        final_positions: tuple[tuple[int, ...], ...] | None = None
        if buyer is not None:
            completed = {seat: list(by_seat[seat]) for seat in Seat}
            completed[buyer].append(purchase_position)
            for seat in order:
                count = (
                    self.domain.final_cards_for_buyer
                    if seat == buyer
                    else self.domain.final_cards_for_others
                )
                positions = tuple(range(cursor, cursor + count))
                cursor += count
                completed[seat].extend(positions)
                final_packets.append((seat, positions))
            if cursor != self.domain.card_count or any(
                len(completed[seat]) != self.domain.cards_per_player
                for seat in Seat
            ):
                raise ValueError(
                    "reference final deal must consume the complete governed deck"
                )
            final_positions = tuple(
                tuple(completed[Seat(index)])
                for index in range(self.domain.players)
            )
        return StructuralDealLayoutV1(
            dealer,
            buyer,
            tuple(initial),
            tuple(final_packets),
            purchase_position,
            undealt,
            tuple(
                tuple(by_seat[Seat(index)])
                for index in range(self.domain.players)
            ),
            final_positions,
        )

    @staticmethod
    def _splitmix64_next(state: int) -> tuple[int, int]:
        state = (state + 0x9E3779B97F4A7C15) & _MASK64
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
        return state, (value ^ (value >> 31)) & _MASK64

    def shuffle_observation(
        self,
        *,
        seed: int,
        dealer: Seat,
        cut_position: int | None,
        skip_cut: bool = False,
    ) -> dict[str, Any]:
        if skip_cut and not self.domain.allow_no_cut:
            return {"accepted": False}
        if self.domain.shuffle_algorithm != "splitmix64_fisher_yates_v1":
            raise ValueError(f"unsupported shuffle: {self.domain.shuffle_algorithm}")
        state = int(seed) & _MASK64
        cards = list(range(self.domain.card_count))
        for index in range(len(cards) - 1, 0, -1):
            while True:
                state, value = self._splitmix64_next(state)
                limit = (1 << 64) - ((1 << 64) % (index + 1))
                if value < limit:
                    other = value % (index + 1)
                    break
            cards[index], cards[other] = cards[other], cards[index]
        apply_cut = self.domain.cut_enabled and not skip_cut
        if apply_cut:
            position = cut_position
            if position is None:
                state, value = self._splitmix64_next(state)
                position = 5 + value % 23
            position = self.domain.validate_cut_position(position)
            cards = cards[int(position):] + cards[:int(position)]
        else:
            position = None
        cutter = (
            self.domain.cutter_for(dealer).name.lower() if apply_cut else None
        )
        return {
            "accepted": True,
            "cutApplied": apply_cut,
            "cutPosition": position,
            "cutter": cutter,
            "deckOrder": cards,
        }

    def annulling_observation(
        self,
        hand_ranks: Sequence[str],
        *,
        dealer: Seat = Seat.EAST,
        seat: Seat = Seat.NORTH,
    ) -> dict[str, Any]:
        allowed = self.domain.can_declare_annulment(
            dealer=dealer,
            seat=seat,
        ) and self.domain.can_annul_ranks(hand_ranks)
        return {
            "eligible": allowed,
            "dealerAdvances": self.domain.annulling_dealer_advances if allowed else None,
        }

    def annulment_reason_observation(self, reason: str) -> dict[str, Any]:
        return {
            "reason": str(reason),
            "accepted": self.domain.accepts_annulment_reason(reason),
            "declarationWindow": self.domain.annulling_declaration_window,
        }

    def all_pass_observation(self, *, dealer: Seat) -> dict[str, Any]:
        return {
            "redeal": self.domain.all_pass_redeal,
            "nextDealer": (
                self.domain.next_dealer(dealer).name.lower()
                if self.domain.all_pass_redeal
                else None
            ),
        }

    def semantic_observation(self, *, dealer: Seat) -> dict[str, Any]:
        return self.domain.semantic_observation(dealer=dealer)
