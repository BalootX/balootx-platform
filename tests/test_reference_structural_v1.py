from __future__ import annotations

import copy

from baloot_platform.config import build_bundle
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.reference_structural import (
    ReferenceStructuralInterpreterV1,
)


def _runtime_layout(value: DealLayoutV1) -> dict:
    return {
        "dealer": value.dealer.name.lower(),
        "buyer": value.buyer.name.lower() if value.buyer is not None else None,
        "initialPackets": [
            {
                "packet": packet.packet,
                "seat": packet.seat.name.lower(),
                "positions": list(packet.positions),
            }
            for packet in value.initial_packets
        ],
        "finalPackets": [
            {"seat": packet.seat.name.lower(), "positions": list(packet.positions)}
            for packet in value.final_packets
        ],
        "purchasePosition": value.purchase_position,
        "undealtPositions": list(value.undealt_positions),
        "initialPositions": [list(item) for item in value.initial_positions],
        "finalPositions": (
            [list(item) for item in value.final_hand_positions]
            if value.final_hand_positions is not None
            else None
        ),
    }


def test_independent_reference_layout_matches_runtime_for_every_rotation(root):
    rules = build_bundle(root).rules
    reference = ReferenceStructuralInterpreterV1.from_rules(rules.data)
    for dealer in Seat:
        assert reference.next_dealer(dealer) == dealer.next()
        assert reference.partner(dealer) == dealer.partner
        for buyer in (None, *tuple(Seat)):
            expected = reference.layout(dealer=dealer, buyer=buyer).to_dict()
            actual = _runtime_layout(
                DealLayoutV1.build(
                    reference.domain,
                    dealer=dealer,
                    buyer=buyer,
                )
            )
            assert expected == actual


def test_independent_reference_shuffle_matches_runtime_cut_and_no_cut(root):
    rules = build_bundle(root).rules
    reference = ReferenceStructuralInterpreterV1.from_rules(rules.data)
    engine = BalootEngine(rules)
    for skip_cut in (False, True):
        state = engine.create_round(seed=1905, dealer=Seat.EAST)
        engine.shuffle_and_initial_deal(state, cut_position=11, skip_cut=skip_cut)
        expected = reference.shuffle_observation(
            seed=1905,
            dealer=Seat.EAST,
            cut_position=11,
            skip_cut=skip_cut,
        )
        assert expected == {
            "accepted": True,
            "cutApplied": state.shuffle.cut_position is not None,
            "cutPosition": state.shuffle.cut_position,
            "cutter": (
                state.shuffle.cutter.name.lower()
                if state.shuffle.cutter is not None
                else None
            ),
            "deckOrder": list(state.shuffle.deck_order),
        }

    required = copy.deepcopy(rules.data)
    required["dealing"]["allow_no_cut"] = False
    required_rules = RuleBook(required)
    assert ReferenceStructuralInterpreterV1.from_rules(
        required_rules.data
    ).shuffle_observation(
        seed=1905,
        dealer=Seat.EAST,
        cut_position=11,
        skip_cut=True,
    ) == {"accepted": False}


def test_independent_reference_annulling_uses_enabled_ranks_and_dealer_policy(root):
    data = copy.deepcopy(build_bundle(root).rules.data)
    data["dealing"]["annulling"]["enabled"] = True
    reference = ReferenceStructuralInterpreterV1.from_rules(RuleBook(data).data)
    ranks = ("seven", "eight", "nine", "seven", "eight")
    assert reference.annulling_observation(ranks) == {
        "eligible": True,
        "dealerAdvances": True,
    }
    data["dealing"]["annulling"]["eligible_ranks"] = ["seven", "eight"]
    assert ReferenceStructuralInterpreterV1.from_rules(
        RuleBook(data).data
    ).annulling_observation(ranks) == {
        "eligible": False,
        "dealerAdvances": None,
    }
