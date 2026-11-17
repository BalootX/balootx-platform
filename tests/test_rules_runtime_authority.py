from __future__ import annotations

import copy

import pytest

from baloot_platform.config import build_bundle
from baloot_platform.core.bidding import _choose_action, run_bidding
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.projects import detect_projects
from baloot_platform.core.rules import RuleBook


def _copy(rules: RuleBook) -> dict:
    return copy.deepcopy(rules.data)


def test_every_distributed_mode_is_accepted_by_runtime_authority(root):
    for mode in ("full", "sun_only", "hokum_only"):
        assert build_bundle(root, mode=mode).rules.id


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update({"unknown": {}}), "unknown rule sections"),
        (lambda data: data["seating"].update(play_order=["north", "east", "south", "west"]), "seating.play_order"),
        (lambda data: data["dealing"].update(first_recipient="dealer_left"), "dealing.first_recipient"),
        (lambda data: data["play"].update(must_follow_suit=False), "play.must_follow_suit"),
        (lambda data: data["projects"].update(declaration_timing="before_play"), "projects.declaration_timing"),
        (lambda data: data["scoring"]["conversion"].update(sun="floor"), "scoring.conversion.sun"),
        (lambda data: data["scoring"]["raw_card_totals"].update(sun=129), "scoring.raw_card_totals.sun"),
        (lambda data: data["bidding"]["heuristics"].update(sun_rank_weights=[1.0]), "sun_rank_weights"),
    ],
)
def test_unsupported_rule_variants_fail_closed(rules, mutate, message):
    data = _copy(rules)
    mutate(data)
    with pytest.raises(ValueError, match=message):
        RuleBook(data)


def test_fallback_bidding_weights_are_governed(rules):
    purchase = Card.parse("S8")
    cards = (
        Card.parse("SA"), Card.parse("HA"), Card.parse("DA"),
        Card.parse("CA"), Card.parse("S7"),
    )
    hands = {seat: cards for seat in Seat}
    machine = BiddingMachine(
        Seat.EAST,
        purchase.suit,
        AuctionDomainIRV1.from_rules(rules.data),
    )
    action, _trump = _choose_action(machine, hands, purchase, rules)
    assert action == "sun"

    data = _copy(rules)
    heuristic = data["bidding"]["heuristics"]
    heuristic["sun_rank_weights"] = [0.0] * 8
    heuristic["hokum_trump_rank_weights"] = [0.0] * 8
    heuristic["sun_extra_suit_card_weight"] = 0.0
    heuristic["hokum_extra_trump_card_weight"] = 0.0
    heuristic["hokum_off_trump_ace_ten_weight"] = 0.0
    governed = RuleBook(data)
    action, _trump = _choose_action(machine, hands, purchase, governed)
    assert action == "pass"


def test_preselected_mode_uses_its_declared_default(root):
    rules = build_bundle(root, mode="sun_only").rules
    purchase = Card.parse("S8")
    cards = (
        Card.parse("S7"), Card.parse("H7"), Card.parse("D7"),
        Card.parse("C7"), Card.parse("H8"),
    )
    result = run_bidding(
        {seat: cards for seat in Seat},
        purchase,
        Seat.EAST,
        rules,
    )
    assert result.contract == Contract.SUN
    assert result.buyer == Seat.NORTH


def test_optional_cut_and_required_cut_are_enforced(rules):
    engine = BalootEngine(rules)
    optional = engine.create_round(seed=91)
    engine.shuffle_and_initial_deal(optional, skip_cut=True)
    assert optional.shuffle is not None and optional.shuffle.cutter is None

    data = _copy(rules)
    data["dealing"]["allow_no_cut"] = False
    required_rules = RuleBook(data)
    required = BalootEngine(required_rules).create_round(seed=91)
    with pytest.raises(ValueError, match="requires a cut"):
        BalootEngine(required_rules).shuffle_and_initial_deal(required, skip_cut=True)


def test_project_kind_and_minimum_length_switches_govern_detection(rules):
    sequence_hand = (
        Card.parse("S7"), Card.parse("S8"), Card.parse("S9"),
        Card.parse("HA"), Card.parse("DA"), Card.parse("CA"),
        Card.parse("H7"), Card.parse("D7"),
    )
    hands = {seat: sequence_hand for seat in Seat}
    assert any(project.kind == "siri" for project in detect_projects(hands, Contract.SUN, rules, Seat.EAST))

    data = _copy(rules)
    data["projects"]["sequences"]["minimum_length"] = 4
    governed = RuleBook(data)
    assert not any(
        project.kind == "siri"
        for project in detect_projects(hands, Contract.SUN, governed, Seat.EAST)
    )

    data = _copy(rules)
    data["projects"]["sequences"]["enabled"] = False
    data["projects"]["four_of_kind"]["enabled"] = False
    governed = RuleBook(data)
    assert detect_projects(hands, Contract.SUN, governed, Seat.EAST) == []
