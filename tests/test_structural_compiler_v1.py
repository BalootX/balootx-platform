from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from baloot_platform.config import build_bundle
from baloot_platform.core.bidding import BidResult
from baloot_platform.core.cards import Card, cards_to_mask
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import GamePhase, Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.shuffle import governed_preordered_deck
from baloot_platform.core.structural_ir import StructuralDomainIRV1
from baloot_platform.research.reference_structural import (
    ReferenceStructuralInterpreterV1,
)
from baloot_platform.research.structural_compiler import (
    OrToolsCpSatStructuralCompilerV1,
)


_STRUCTURAL_LEAVES = {
    "rules.deck.suits",
    "rules.deck.ranks",
    "rules.deck.players",
    "rules.deck.cards_per_player",
    "rules.seating.play_order",
    "rules.seating.dealer_rotation",
    "rules.seating.partnerships.north",
    "rules.seating.partnerships.west",
    "rules.seating.partnerships.south",
    "rules.seating.partnerships.east",
    "rules.dealing.shuffle_algorithm",
    "rules.dealing.cut_enabled",
    "rules.dealing.cutter",
    "rules.dealing.allow_no_cut",
    "rules.dealing.initial_packets",
    "rules.dealing.exposed_purchase_cards",
    "rules.dealing.final_cards_for_buyer",
    "rules.dealing.final_cards_for_others",
    "rules.dealing.first_recipient",
    "rules.dealing.all_pass_redeal",
    "rules.dealing.annulling.enabled",
    "rules.dealing.annulling.names",
    "rules.dealing.annulling.eligible_ranks",
    "rules.dealing.annulling.declaration_window",
    "rules.dealing.annulling.dealer_advances",
}


def _behavior(value: dict) -> dict:
    return {
        key: child
        for key, child in value.items()
        if key not in {"structuralDomainIrHash", "deck", "seating", "dealing"}
    }


def _runtime_annulling(rules: RuleBook, ranks: tuple[str, ...]) -> dict:
    cards = tuple(
        Card.of(suit, rank)
        for suit, rank in zip(
            (0, 1, 2, 3, 0),
            ({"seven": 0, "eight": 1, "nine": 2}[value] for value in ranks),
            strict=True,
        )
    )
    engine = BalootEngine(rules)
    state = engine.create_round(seed=1, dealer=Seat.EAST)
    state.phase = GamePhase.INITIAL_DEALT
    state.initial_five[int(Seat.NORTH)] = cards_to_mask(cards)
    state.hands[int(Seat.NORTH)] = cards_to_mask(cards)
    eligible = engine.can_annul_initial_deal(state, Seat.NORTH)
    advances = None
    if eligible:
        engine.annul_initial_deal(state, Seat.NORTH)
        advances = state.events[-1].data["dealerAdvances"]
    return {"eligible": eligible, "dealerAdvances": advances}


@pytest.mark.skipif(
    not OrToolsCpSatStructuralCompilerV1.available(),
    reason="OR-Tools optimization extra is not installed",
)
def test_cp_sat_structural_plan_matches_reference_and_runtime(root):
    rules = build_bundle(root).rules
    reference = ReferenceStructuralInterpreterV1(rules.structural_domain)
    engine = BalootEngine(rules, structural_domain=reference.domain)
    plan = OrToolsCpSatStructuralCompilerV1().compile(
        rules=reference.domain,
        dealer=Seat.EAST,
        buyer=Seat.SOUTH,
        seed=1905,
        cut_position=11,
        annulling_hand_ranks=("seven", "eight", "nine", "seven", "eight"),
    )
    assert plan.solver_status == "optimal"
    assert plan.structural_domain_ir_hash == engine.structural_domain.sha256
    assert plan.structural_domain_ir_hash == reference.domain.sha256
    assert engine.structural_domain is rules.structural_domain
    assert reference.domain is rules.structural_domain
    assert plan.structural_domain is rules.structural_domain
    assert plan.model_constraint_count > 100
    assert sorted(plan.deck_order) == list(range(32))
    assert plan.deck_order == tuple(
        reference.shuffle_observation(
            seed=1905,
            dealer=Seat.EAST,
            cut_position=11,
        )["deckOrder"]
    )
    expected_layout = reference.layout(dealer=Seat.EAST, buyer=Seat.SOUTH).to_dict()
    actual = plan.semantic_observation()
    for key in (
        "initialPackets",
        "finalPackets",
        "purchasePosition",
        "undealtPositions",
        "initialPositions",
        "finalPositions",
    ):
        assert actual[key] == expected_layout[key]


def test_runtime_rejects_a_structural_ir_from_different_rules(root):
    rules = build_bundle(root).rules
    changed = copy.deepcopy(rules.data)
    changed["dealing"]["initial_packets"] = [2, 3]
    other = RuleBook(changed)
    foreign = other.structural_domain
    with pytest.raises(ValueError, match="structural domain does not match"):
        BalootEngine(rules, structural_domain=foreign)


@pytest.mark.skipif(
    not OrToolsCpSatStructuralCompilerV1.available(),
    reason="OR-Tools optimization extra is not installed",
)
@pytest.mark.parametrize(
    "path,value",
    [
        (("cut_enabled",), False),
        (("allow_no_cut",), False),
        (("initial_packets",), [2, 3]),
        (("annulling", "enabled"), True),
        (("annulling", "eligible_ranks"), ["seven", "eight"]),
        (("annulling", "dealer_advances"), False),
    ],
)
def test_supported_structural_counterfactuals_change_cp_sat_output(root, path, value):
    baseline_data = copy.deepcopy(build_bundle(root).rules.data)
    mutated_data = copy.deepcopy(baseline_data)
    cursor = mutated_data["dealing"]
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value
    if path[0] == "annulling" and path[-1] != "enabled":
        baseline_data["dealing"]["annulling"]["enabled"] = True
        mutated_data["dealing"]["annulling"]["enabled"] = True
    baseline = RuleBook(baseline_data)
    mutated = RuleBook(mutated_data)
    kwargs = {
        "dealer": Seat.EAST,
        "buyer": Seat.SOUTH,
        "seed": 1905,
        "cut_position": 11,
        "skip_cut": path == ("allow_no_cut",),
        "annulling_hand_ranks": ("seven", "eight", "nine", "seven", "eight"),
    }
    first = OrToolsCpSatStructuralCompilerV1().compile(
        rules=baseline.structural_domain, **kwargs
    ).semantic_observation()
    second = OrToolsCpSatStructuralCompilerV1().compile(
        rules=mutated.structural_domain, **kwargs
    ).semantic_observation()
    assert _behavior(first) != _behavior(second)

    if path[0] == "annulling":
        first_reference = ReferenceStructuralInterpreterV1.from_rules(
            baseline.data
        ).annulling_observation(kwargs["annulling_hand_ranks"])
        second_reference = ReferenceStructuralInterpreterV1.from_rules(
            mutated.data
        ).annulling_observation(kwargs["annulling_hand_ranks"])
        assert first_reference == _runtime_annulling(
            baseline, kwargs["annulling_hand_ranks"]
        )
        assert second_reference == _runtime_annulling(
            mutated, kwargs["annulling_hand_ranks"]
        )


def test_structural_targets_reject_raw_or_forged_parallel_authorities(root):
    rules = build_bundle(root).rules
    with pytest.raises(TypeError, match="raw rule mappings are unsupported"):
        OrToolsCpSatStructuralCompilerV1().compile(
            rules=rules.data,  # type: ignore[arg-type]
            dealer=Seat.EAST,
            buyer=Seat.SOUTH,
            seed=1,
            cut_position=11,
        )

    forged = replace(
        rules.structural_domain,
        initial_packets=(2, 3),
    )
    with pytest.raises(ValueError, match="integrity mismatch"):
        BalootEngine(rules, structural_domain=forged)
    with pytest.raises(ValueError, match="integrity mismatch"):
        ReferenceStructuralInterpreterV1(forged)
    with pytest.raises(ValueError, match="integrity mismatch"):
        DealLayoutV1.build(forged, dealer=Seat.EAST)
    with pytest.raises(ValueError, match="integrity mismatch"):
        OrToolsCpSatStructuralCompilerV1().compile(
            rules=forged,
            dealer=Seat.EAST,
            buyer=Seat.SOUTH,
            seed=1,
            cut_position=11,
        )


@pytest.mark.skipif(
    not OrToolsCpSatStructuralCompilerV1.available(),
    reason="OR-Tools optimization extra is not installed",
)
def test_automatic_cut_and_every_dealer_buyer_layout_have_three_target_parity(root):
    rules = build_bundle(root).rules
    domain = rules.structural_domain
    reference = ReferenceStructuralInterpreterV1(domain)
    for dealer in Seat:
        runtime = BalootEngine(rules)
        state = runtime.create_round(seed=1905, dealer=dealer)
        runtime.shuffle_and_initial_deal(state)
        expected_shuffle = reference.shuffle_observation(
            seed=1905,
            dealer=dealer,
            cut_position=None,
        )
        assert state.shuffle is not None
        assert list(state.shuffle.deck_order) == expected_shuffle["deckOrder"]
        assert state.shuffle.cut_position == expected_shuffle["cutPosition"]
        assert runtime.next_dealer(dealer) == reference.next_dealer(dealer)
        for seat in Seat:
            assert runtime.partner(seat) == reference.partner(seat)
        for buyer in Seat:
            plan = OrToolsCpSatStructuralCompilerV1().compile(
                rules=domain,
                dealer=dealer,
                buyer=buyer,
                seed=1905,
                cut_position=None,
            )
            expected = reference.layout(dealer=dealer, buyer=buyer).to_dict()
            actual = plan.semantic_observation()
            assert plan.deck_order == tuple(expected_shuffle["deckOrder"])
            for key in (
                "initialPackets",
                "finalPackets",
                "purchasePosition",
                "undealtPositions",
                "initialPositions",
                "finalPositions",
            ):
                assert actual[key] == expected[key]


@pytest.mark.parametrize("cut_position", (-1, 0, 32))
def test_invalid_cut_boundaries_fail_in_every_target(root, cut_position):
    rules = build_bundle(root).rules
    reference = ReferenceStructuralInterpreterV1(rules.structural_domain)
    state = BalootEngine(rules).create_round(seed=1)
    with pytest.raises(ValueError, match="cut position"):
        BalootEngine(rules).shuffle_and_initial_deal(
            state, cut_position=cut_position
        )
    with pytest.raises(ValueError, match="cut position"):
        reference.shuffle_observation(
            seed=1,
            dealer=Seat.EAST,
            cut_position=cut_position,
        )
    with pytest.raises(ValueError, match="cut position"):
        OrToolsCpSatStructuralCompilerV1().compile(
            rules=rules.structural_domain,
            dealer=Seat.EAST,
            buyer=Seat.SOUTH,
            seed=1,
            cut_position=cut_position,
        )


def test_card_uniqueness_redeal_and_annulment_are_runtime_governed(root):
    rules = build_bundle(root).rules
    engine = BalootEngine(rules)
    with pytest.raises(ValueError, match="every governed card exactly once"):
        governed_preordered_deck(
            tuple(range(31)) + (30,),
            seed=1,
            dealer=Seat.EAST,
            structural_domain=rules.structural_domain,
        )

    state = engine.create_round(seed=11, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=11)
    assert engine.complete_deal_from_bid_result(state, BidResult((), True)) is False
    assert state.events[-1].type == "all_pass_redeal"
    assert state.events[-1].data["nextDealer"] == engine.next_dealer(
        Seat.EAST
    ).name.lower()

    data = copy.deepcopy(rules.data)
    data["dealing"]["annulling"]["enabled"] = True
    governed = RuleBook(data)
    ranks = ("seven", "eight", "nine", "seven", "eight")
    cards = tuple(
        Card.of(suit, rank)
        for suit, rank in zip(
            (0, 1, 2, 3, 0),
            ({"seven": 0, "eight": 1, "nine": 2}[value] for value in ranks),
            strict=True,
        )
    )
    annul_state = BalootEngine(governed).create_round(seed=1)
    annul_state.phase = GamePhase.INITIAL_DEALT
    annul_state.initial_five[int(Seat.NORTH)] = cards_to_mask(cards)
    annul_state.hands[int(Seat.NORTH)] = cards_to_mask(cards)
    annul_state.initial_five[int(Seat.WEST)] = cards_to_mask(cards)
    annul_state.hands[int(Seat.WEST)] = cards_to_mask(cards)
    assert BalootEngine(governed).can_annul_initial_deal(
        annul_state, Seat.WEST
    ) is False
    with pytest.raises(ValueError, match="unsupported annulment reason"):
        BalootEngine(governed).annul_initial_deal(
            annul_state, Seat.NORTH, reason="invented"
        )

    no_advance_data = copy.deepcopy(data)
    no_advance_data["dealing"]["annulling"]["dealer_advances"] = False
    no_advance_engine = BalootEngine(RuleBook(no_advance_data))
    no_advance_state = no_advance_engine.create_round(
        seed=2,
        dealer=Seat.EAST,
    )
    no_advance_state.phase = GamePhase.INITIAL_DEALT
    no_advance_state.initial_five[int(Seat.NORTH)] = cards_to_mask(cards)
    no_advance_state.hands[int(Seat.NORTH)] = cards_to_mask(cards)
    no_advance_engine.annul_initial_deal(no_advance_state, Seat.NORTH)
    assert no_advance_state.events[-1].data["nextDealer"] == "east"
    assert no_advance_engine.next_round_dealer(no_advance_state) == Seat.EAST


def test_literal_structural_leaf_inventory_is_exact(root):
    rules = build_bundle(root).rules

    def leaves(value, prefix):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from leaves(child, f"{prefix}.{key}")
        else:
            yield prefix

    actual = {
        primitive
        for section in ("deck", "seating", "dealing")
        for primitive in leaves(rules.data[section], f"rules.{section}")
    }
    assert actual == _STRUCTURAL_LEAVES
    assert len(actual) == 25


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda data: data["deck"].update(players=5),
            "deck.players",
        ),
        (
            lambda data: data["deck"].update(
                suits=["spades", "hearts", "diamonds", "spades"]
            ),
            "deck.suits",
        ),
        (
            lambda data: data["seating"].update(
                play_order=["north", "east", "south", "west"]
            ),
            "seating.play_order",
        ),
        (
            lambda data: data["seating"]["partnerships"].update(
                north="east"
            ),
            "seating.partnerships",
        ),
        (
            lambda data: data["dealing"].update(initial_packets=[4, 2]),
            "dealing.initial_packets",
        ),
        (
            lambda data: data["dealing"].update(
                exposed_purchase_cards=2
            ),
            "dealing.exposed_purchase_cards",
        ),
        (
            lambda data: data["dealing"].update(all_pass_redeal=False),
            "dealing.all_pass_redeal",
        ),
        (
            lambda data: data["dealing"]["annulling"].update(
                eligible_ranks=["seven", "seven"]
            ),
            "eligible_ranks",
        ),
        (
            lambda data: data["dealing"].update(invented=True),
            "unknown structural fields at dealing",
        ),
    ],
)
def test_invalid_structural_profiles_fail_at_the_single_ir_gate(
    root, mutate, message
):
    payload = copy.deepcopy(build_bundle(root).rules.data)
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        StructuralDomainIRV1.from_rules(payload)
    with pytest.raises(ValueError, match=message):
        RuleBook(payload)
