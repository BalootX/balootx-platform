from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Seat
from baloot_platform.core.structural_ir import StructuralDomainIRV1

from .contracts import canonical_sha256


_MASK64 = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class StructuralPlanV1:
    structural_domain: StructuralDomainIRV1
    dealer: Seat
    solver_status: str
    structural_domain_ir_hash: str
    request_accepted: bool
    cut_applied: bool
    cut_position: int | None
    cutter: Seat | None
    deck_order: tuple[int, ...]
    initial_packets: tuple[tuple[int, Seat, tuple[int, ...]], ...]
    final_packets: tuple[tuple[Seat, tuple[int, ...]], ...]
    purchase_position: int
    undealt_positions: tuple[int, ...]
    initial_positions: tuple[tuple[int, ...], ...]
    final_positions: tuple[tuple[int, ...], ...] | None
    annulling_eligible: bool
    annulment_actor: Seat
    annulling_dealer_advances: bool | None
    model_constraint_count: int
    sha256: str

    def semantic_observation(self) -> dict[str, Any]:
        if not self.request_accepted:
            return {"accepted": False}
        return {
            **self.structural_domain.semantic_observation(dealer=self.dealer),
            "accepted": True,
            "cutApplied": self.cut_applied,
            "cutPosition": self.cut_position,
            "cutter": self.cutter.name.lower() if self.cutter is not None else None,
            "deckOrder": list(self.deck_order),
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
            "annullingEligible": self.annulling_eligible,
            "annullingDealerAdvances": self.annulling_dealer_advances,
            "allPassRedeal": {
                "redeal": self.structural_domain.all_pass_redeal,
                "nextDealer": self.structural_domain.next_dealer(
                    self.dealer
                ).name.lower(),
            },
        }


class OrToolsCpSatStructuralCompilerV1:
    """CP-SAT lowering for deck allocation and governed deal structure."""

    @staticmethod
    def available() -> bool:
        try:
            from ortools.sat.python import cp_model  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _randbelow(state: int, n: int) -> tuple[int, int]:
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            state = (state + 0x9E3779B97F4A7C15) & _MASK64
            value = state
            value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
            value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
            value = (value ^ (value >> 31)) & _MASK64
            if value < limit:
                return state, value % n

    @classmethod
    def _shuffle(
        cls, domain: StructuralDomainIRV1, seed: int
    ) -> tuple[list[int], int]:
        state = int(seed) & _MASK64
        cards = list(domain.card_ids)
        for index in range(len(cards) - 1, 0, -1):
            state, other = cls._randbelow(state, index + 1)
            cards[index], cards[other] = cards[other], cards[index]
        return cards, state

    def compile(
        self,
        *,
        rules: StructuralDomainIRV1,
        dealer: Seat,
        buyer: Seat | None,
        seed: int,
        cut_position: int | None,
        skip_cut: bool = False,
        annulling_hand_ranks: Sequence[str] = (),
    ) -> StructuralPlanV1:
        from ortools.sat.python import cp_model

        if not isinstance(rules, StructuralDomainIRV1):
            raise TypeError(
                "OrToolsCpSatStructuralCompilerV1 requires StructuralDomainIRV1; "
                "raw rule mappings are unsupported"
            )
        domain = rules.require_integrity()
        dealer = Seat.parse(dealer)
        buyer = Seat.parse(buyer) if buyer is not None else None
        accepted = not (skip_cut and not domain.allow_no_cut)
        deck, shuffle_state = self._shuffle(domain, seed)
        apply_cut = accepted and domain.cut_enabled and not skip_cut
        position: int | None = None
        if apply_cut:
            if cut_position is None:
                _state, generated = self._randbelow(shuffle_state, 23)
                position = 5 + generated
            else:
                position = domain.validate_cut_position(cut_position)
            deck = deck[position:] + deck[:position]

        layout = DealLayoutV1.build(domain, dealer=dealer, buyer=buyer)
        initial = tuple(
            (packet.packet, packet.seat, packet.positions)
            for packet in layout.initial_packets
        )
        final_packets = tuple(
            (packet.seat, packet.positions) for packet in layout.final_packets
        )
        final_positions = layout.final_hand_positions

        model = cp_model.CpModel()
        cards = [
            model.new_int_var(0, domain.card_count - 1, f"card_at_{index}")
            for index in range(domain.card_count)
        ]
        model.add_all_different(cards)
        for index, card in enumerate(deck):
            model.add(cards[index] == card)
        accepted_var = model.new_bool_var("request_accepted")
        model.add(accepted_var == int(accepted))
        cut_applied_var = model.new_bool_var("cut_applied")
        model.add(cut_applied_var == int(apply_cut))
        all_pass_redeal = model.new_bool_var("all_pass_redeal")
        model.add(all_pass_redeal == int(domain.all_pass_redeal))

        play_order = [
            model.new_int_var(0, domain.players - 1, f"play_order_{index}")
            for index in range(domain.players)
        ]
        model.add_all_different(play_order)
        for index, seat in enumerate(domain.play_order):
            model.add(play_order[index] == int(seat))
        partners = [
            model.new_int_var(0, domain.players - 1, f"partner_{index}")
            for index in range(domain.players)
        ]
        for index in range(domain.players):
            model.add(partners[index] == int(domain.partner(Seat(index))))
            model.add_element(partners[index], partners, index)
        next_dealer = model.new_int_var(
            0, domain.players - 1, "next_dealer"
        )
        model.add(next_dealer == int(domain.next_dealer(dealer)))
        annulment_actor = model.new_int_var(
            0, domain.players - 1, "annulment_actor"
        )
        model.add(annulment_actor == int(domain.annulment_actor(dealer)))
        annulment_next_dealer = model.new_int_var(
            0, domain.players - 1, "annulment_next_dealer"
        )
        model.add(
            annulment_next_dealer
            == int(domain.next_dealer_after_annulment(dealer))
        )

        position_owners = [
            model.new_int_var(-1, domain.players - 1, f"position_owner_{position}")
            for position in range(domain.card_count)
        ]
        governed_positions = (
            layout.final_hand_positions
            if layout.final_hand_positions is not None
            else layout.initial_positions
        )
        owner_by_position = [-1] * domain.card_count
        for seat in Seat:
            for position_value in governed_positions[int(seat)]:
                owner_by_position[position_value] = int(seat)
        for position_index, owner in enumerate(owner_by_position):
            model.add(position_owners[position_index] == owner)
        expected_count = (
            domain.cards_per_player
            if layout.final_hand_positions is not None
            else sum(domain.initial_packets)
        )
        for seat in Seat:
            memberships = []
            for position_index, owner in enumerate(position_owners):
                member = model.new_bool_var(
                    f"position_{position_index}_owned_by_{int(seat)}"
                )
                model.add(owner == int(seat)).only_enforce_if(member)
                model.add(owner != int(seat)).only_enforce_if(member.Not())
                memberships.append(member)
            model.add(sum(memberships) == expected_count)
        if buyer is not None:
            model.add(position_owners[layout.purchase_position] == int(buyer))

        allowed_rank_names = set(domain.annulling_eligible_ranks)
        rank_allowed = []
        for index, rank in enumerate(annulling_hand_ranks):
            literal = model.new_bool_var(f"annul_rank_{index}_allowed")
            model.add(literal == int(str(rank) in allowed_rank_names))
            rank_allowed.append(literal)
        eligible = model.new_bool_var("annulling_eligible")
        expected_annul = domain.can_annul_ranks(annulling_hand_ranks)
        model.add(eligible == int(expected_annul))
        if rank_allowed and expected_annul:
            model.add_bool_and(rank_allowed)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        status = solver.solve(model)
        if status != cp_model.OPTIMAL:
            raise ValueError("structural CP-SAT lowering was not proven optimal")
        solved_deck = tuple(solver.value(value) for value in cards)
        domain.validate_deck_order(solved_deck)
        cutter = domain.cutter_for(dealer) if apply_cut else None
        constraint_count = len(model.Proto().constraints)
        core = {
            "schemaVersion": 1,
            "solverStatus": "optimal",
            "structuralDomainIrHash": domain.sha256,
            "requestAccepted": bool(solver.value(accepted_var)),
            "cutApplied": apply_cut,
            "cutPosition": position,
            "cutter": cutter.name.lower() if cutter is not None else None,
            "deckOrder": list(solved_deck),
            "domainSemantics": domain.semantic_observation(dealer=dealer),
            "initialPackets": [
                [packet, seat.name.lower(), list(positions)]
                for packet, seat, positions in initial
            ],
            "finalPackets": [
                [seat.name.lower(), list(positions)] for seat, positions in final_packets
            ],
            "purchasePosition": layout.purchase_position,
            "undealtPositions": list(layout.undealt_positions),
            "initialPositions": [list(value) for value in layout.initial_positions],
            "finalPositions": (
                [list(value) for value in final_positions]
                if final_positions is not None
                else None
            ),
            "annullingEligible": bool(solver.value(eligible)),
            "annulmentActor": Seat(solver.value(annulment_actor)).name.lower(),
            "annulmentNextDealer": Seat(
                solver.value(annulment_next_dealer)
            ).name.lower(),
            "annullingDealerAdvances": (
                domain.annulling_dealer_advances if expected_annul else None
            ),
            "modelConstraintCount": constraint_count,
        }
        return StructuralPlanV1(
            domain,
            dealer,
            "optimal",
            domain.sha256,
            bool(solver.value(accepted_var)),
            apply_cut,
            position,
            cutter,
            solved_deck,
            tuple(initial),
            tuple(final_packets),
            layout.purchase_position,
            layout.undealt_positions,
            layout.initial_positions,
            final_positions,
            bool(solver.value(eligible)),
            Seat(solver.value(annulment_actor)),
            domain.annulling_dealer_advances if expected_annul else None,
            constraint_count,
            canonical_sha256(core),
        )
