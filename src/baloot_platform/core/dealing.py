from __future__ import annotations

from dataclasses import dataclass
from .enums import Seat
from .structural_ir import StructuralDomainIRV1


@dataclass(frozen=True, slots=True)
class DealPacketV1:
    stage: str
    packet: int
    seat: Seat
    positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DealLayoutV1:
    """Authoritative mapping from a post-cut deck to Baloot recipients."""

    dealer: Seat
    buyer: Seat | None
    initial_packets: tuple[DealPacketV1, ...]
    final_packets: tuple[DealPacketV1, ...]
    purchase_position: int
    undealt_positions: tuple[int, ...]
    initial_positions: tuple[tuple[int, ...], ...]
    final_hand_positions: tuple[tuple[int, ...], ...] | None

    @classmethod
    def build(
        cls,
        dealing: StructuralDomainIRV1,
        *,
        dealer: Seat,
        buyer: Seat | None = None,
    ) -> "DealLayoutV1":
        if not isinstance(dealing, StructuralDomainIRV1):
            raise TypeError(
                "DealLayoutV1 requires StructuralDomainIRV1; raw mappings are unsupported"
            )
        dealing.require_integrity()
        dealer = Seat.parse(dealer)
        buyer = Seat.parse(buyer) if buyer is not None else None
        packets = dealing.initial_packets
        if not packets or any(value <= 0 for value in packets):
            raise ValueError("initial deal packets must be positive and non-empty")
        order = dealing.recipient_order(dealer)
        cursor = 0
        initial: list[DealPacketV1] = []
        by_seat: dict[Seat, list[int]] = {seat: [] for seat in Seat}
        for packet_index, packet_size in enumerate(packets, 1):
            for seat in order:
                positions = tuple(range(cursor, cursor + packet_size))
                cursor += packet_size
                by_seat[seat].extend(positions)
                initial.append(DealPacketV1("initial", packet_index, seat, positions))
        initial_count = sum(packets)
        if any(len(by_seat[seat]) != initial_count for seat in Seat):
            raise ValueError("initial deal layout does not give every seat the same count")
        purchase_position = cursor
        cursor += dealing.exposed_purchase_cards
        if cursor != purchase_position + 1:
            raise ValueError("the supported Baloot layout requires one purchase card")
        undealt_positions = tuple(range(cursor, dealing.card_count))
        final_packets: list[DealPacketV1] = []
        final_positions: tuple[tuple[int, ...], ...] | None = None
        if buyer is not None:
            buyer_count = dealing.final_cards_for_buyer
            other_count = dealing.final_cards_for_others
            completed = {seat: list(by_seat[seat]) for seat in Seat}
            completed[buyer].append(purchase_position)
            for seat in order:
                count = buyer_count if seat == buyer else other_count
                positions = tuple(range(cursor, cursor + count))
                cursor += count
                completed[seat].extend(positions)
                final_packets.append(DealPacketV1("final", 1, seat, positions))
            if cursor != dealing.card_count or any(
                len(completed[seat]) != dealing.cards_per_player for seat in Seat
            ):
                raise ValueError("final deal layout must consume 32 cards into four hands")
            final_positions = tuple(
                tuple(completed[Seat(index)])
                for index in range(dealing.players)
            )
        elif cursor + len(undealt_positions) != dealing.card_count:
            raise ValueError("initial deal layout does not consume a 32-card deck")
        allocated = [
            position
            for positions in by_seat.values()
            for position in positions
        ]
        allocated.append(purchase_position)
        allocated.extend(undealt_positions)
        if sorted(allocated) != list(dealing.card_ids):
            raise AssertionError(
                "initial deal layout must allocate every deck position exactly once"
            )
        return cls(
            dealer,
            buyer,
            tuple(initial),
            tuple(final_packets),
            purchase_position,
            undealt_positions,
            tuple(
                tuple(by_seat[Seat(index)])
                for index in range(dealing.players)
            ),
            final_positions,
        )

    def positions_for(self, seat: Seat, *, final: bool) -> tuple[int, ...]:
        if final:
            if self.final_hand_positions is None:
                raise ValueError("final positions require an assumed buyer")
            return self.final_hand_positions[int(seat)]
        return self.initial_positions[int(seat)]

    def purchase_invariant_undealt_groups(
        self, dealing: StructuralDomainIRV1,
    ) -> tuple[tuple[int, ...], ...]:
        """Position permutations preserving hands for every purchase recipient."""
        completions = tuple(
            self.build(dealing, dealer=self.dealer, buyer=buyer) for buyer in Seat
        )
        groups: dict[tuple[int, ...], list[int]] = {}
        for position in self.undealt_positions:
            signature = tuple(
                next(int(seat) for seat in Seat if position in completed.positions_for(seat, final=True))
                for completed in completions
            )
            groups.setdefault(signature, []).append(position)
        return tuple(tuple(positions) for positions in groups.values())
