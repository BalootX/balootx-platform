from __future__ import annotations

import copy

import pytest

from baloot_platform.core.cards import Card, cards_to_mask
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, GamePhase, Seat, Suit
from baloot_platform.core.models import PlayedCard, RoundState, Trick
from baloot_platform.core.play_ir import PlayDomainIRV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.research import (
    OrToolsCpSatPlayCompilerV1,
    ReferencePlayInterpreterV1,
    ReferencePlayV1,
)


def _runtime_state(rules, hand, plays, *, actor=Seat.SOUTH):
    state = RoundState(rules.sha256, 1, 1, Seat.EAST)
    state.contract = Contract.HOKUM
    state.buyer = Seat.SOUTH
    state.declarer = Seat.SOUTH
    state.trump_suit = Suit.HEARTS
    state.first_leader = Seat.NORTH
    state.phase = GamePhase.PLAYING
    state.hands = [0, 0, 0, 0]
    state.hands[int(actor)] = cards_to_mask(hand)
    state.initial_hands = list(state.hands)
    trick = Trick(0, Seat.NORTH)
    for ordinal, play in enumerate(plays):
        trick.plays.append(
            PlayedCard(play.seat, play.card, 0, ordinal, (), play.ekka)
        )
        state.initial_hands[int(play.seat)] |= 1 << play.card.id
    state.current_trick = trick
    state.current_player = actor
    return state


def test_reference_card_tables_match_runtime_and_cp_sat(rules):
    reference = ReferencePlayInterpreterV1.from_rules(rules.data)
    probe = OrToolsCpSatPlayCompilerV1.configuration_probe(rules)
    for card_id in range(32):
        card = Card(card_id)
        for contract, trump in (
            (Contract.SUN, None),
            (Contract.ASHKAL, None),
            (Contract.HOKUM, Suit.HEARTS),
        ):
            assert reference.strength(card, contract, trump) == rules.strength(
                card, contract, trump
            )
            assert reference.points(card, contract, trump) == rules.points(
                card, contract, trump
            )
    assert probe["tables"] == {
        key: [int(value) for value in rules.data["card_rules"][key]]
        for key in (
            "sun_strength",
            "hokum_non_trump_strength",
            "hokum_trump_strength",
            "sun_points",
            "hokum_non_trump_points",
            "hokum_trump_points",
        )
    }


def test_reference_runtime_and_cp_sat_match_discard_switch(rules):
    plays = (
        ReferencePlayV1(Seat.NORTH, Card.parse("S7")),
        ReferencePlayV1(Seat.WEST, Card.parse("H9")),
    )
    hand = (Card.parse("H7"), Card.parse("C7"))
    reference = ReferencePlayInterpreterV1.from_rules(rules.data)
    expected = {card.code for card in reference.legal_moves(
        hand,
        plays,
        actor=Seat.SOUTH,
        contract=Contract.HOKUM,
        trump=Suit.HEARTS,
    )}
    runtime = {
        card.code
        for card in BalootEngine(rules).legal_moves(
            _runtime_state(rules, hand, plays)
        )
    }
    probe = OrToolsCpSatPlayCompilerV1.configuration_probe(rules)
    assert expected == runtime == {"H7", "C7"}
    assert probe["unableToOvertrumpActions"] == ["trump", "discard"]


def test_reference_first_leader_matches_runtime_policy(rules):
    reference = ReferencePlayInterpreterV1.from_rules(rules.data)
    engine = BalootEngine(rules)
    for dealer in Seat:
        for buyer in Seat:
            assert reference.resolve_first_leader(
                dealer=dealer, buyer=buyer
            ) == engine.dealer_right(dealer)


def test_cp_sat_first_leader_uses_governed_context(rules):
    expected = OrToolsCpSatPlayCompilerV1.resolve_first_leader(
        rules=rules,
        dealer=Seat.EAST,
        buyer=Seat.SOUTH,
    )
    assert expected == Seat.NORTH


def test_shared_play_domain_is_frozen_across_runtime_reference_and_cp_sat(rules):
    domain = PlayDomainIRV1.from_rules(rules.data)
    engine = BalootEngine(rules, play_domain=domain)
    reference = ReferencePlayInterpreterV1(domain, rules.structural_domain)
    assert engine.play_domain is domain
    assert reference.domain is domain
    assert rules.play_domain.sha256 == domain.sha256
    assert OrToolsCpSatPlayCompilerV1.configuration_probe(rules)["sha256"]


def test_runtime_rejects_a_play_domain_from_different_rules(rules):
    changed = copy.deepcopy(rules.data)
    changed["play"]["allow_discard_when_cannot_overtrump"] = False
    foreign = PlayDomainIRV1.from_rules(changed)
    with pytest.raises(ValueError, match="play domain does not match"):
        BalootEngine(rules, play_domain=foreign)


@pytest.mark.parametrize(
    ("policy", "configured", "expected"),
    (
        ("dealer_right", "south", Seat.NORTH),
        ("buyer", "south", Seat.SOUTH),
        ("configured", "west", Seat.WEST),
    ),
)
def test_first_leader_variants_share_one_domain(
    rules, policy: str, configured: str, expected: Seat
):
    changed = copy.deepcopy(rules.data)
    changed["play"]["first_leader"] = policy
    changed["play"]["configured_first_leader"] = configured
    changed_rules = RuleBook(changed)
    reference = ReferencePlayInterpreterV1(
        changed_rules.play_domain,
        changed_rules.structural_domain,
    )
    engine = BalootEngine(changed_rules)
    compiled = OrToolsCpSatPlayCompilerV1.resolve_first_leader(
        rules=changed_rules,
        dealer=Seat.EAST,
        buyer=Seat.SOUTH,
    )
    assert reference.resolve_first_leader(
        dealer=Seat.EAST, buyer=Seat.SOUTH
    ) == expected
    assert engine.resolve_first_leader(
        dealer=Seat.EAST, buyer=Seat.SOUTH
    ) == expected
    assert compiled == expected
