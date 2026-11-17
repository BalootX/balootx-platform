from dataclasses import replace

import pytest

from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.posterior import deal_likelihood
from baloot_platform.analysis.signals import (
    PROBABILITY_SIGNAL_KINDS,
    detect_signals,
)
from baloot_platform.challenge.perspective_pipeline import build_observer_projection
from baloot_platform.core.cards import Card, cards_to_mask
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Rank, Seat, Suit
from baloot_platform.core.models import PlayedCard, Trick


def _state(rules):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=8123, dealer=Seat.WEST)
    engine.shuffle_and_initial_deal(state)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.SOUTH,
        forced_trump=Suit.HEARTS,
    )
    engine.prepare_projects(state)
    return state


def _information(rules) -> InformationState:
    return InformationState.from_complete(_state(rules), Seat.SOUTH)


def _trick(card: Card) -> tuple[PlayedCard, ...]:
    legal = (
        Card.of(Suit.HEARTS, Rank.TEN),
        Card.of(Suit.HEARTS, Rank.ACE),
        Card.of(Suit.HEARTS, Rank.JACK),
    )
    return (
        PlayedCard(Seat.EAST, Card.of(Suit.HEARTS, Rank.SEVEN), 0, 0),
        PlayedCard(Seat.NORTH, Card.of(Suit.HEARTS, Rank.EIGHT), 0, 1),
        PlayedCard(Seat.WEST, Card.of(Suit.HEARTS, Rank.QUEEN), 0, 2),
        PlayedCard(Seat.SOUTH, card, 0, 3, legal),
    )


@pytest.mark.parametrize(
    ("contract", "trump", "expected"),
    (
        (Contract.HOKUM, Suit.HEARTS, "high_then_low"),
        (Contract.HOKUM, Suit.CLUBS, "low_then_high"),
        (Contract.SUN, None, "low_then_high"),
    ),
)
def test_order_signal_uses_contract_and_trump_strength(
    rules, contract, trump, expected
):
    info = _information(rules)
    plays = (
        PlayedCard(Seat.SOUTH, Card.of(Suit.HEARTS, Rank.JACK), 0, 0),
        PlayedCard(Seat.SOUTH, Card.of(Suit.HEARTS, Rank.ACE), 1, 0),
    )
    observations = detect_signals(
        replace(info, contract=contract, trump_suit=trump, plays=plays),
        rules.play_domain,
    )
    assert [item.kind for item in observations] == [expected]


def test_order_signal_fails_closed_without_rule_strength_authority(rules):
    info = _information(rules)
    plays = (
        PlayedCard(Seat.SOUTH, Card.of(Suit.HEARTS, Rank.JACK), 0, 0),
        PlayedCard(Seat.SOUTH, Card.of(Suit.HEARTS, Rank.ACE), 1, 0),
    )
    assert detect_signals(replace(info, plays=plays)) == ()


def test_observer_projection_passes_frozen_play_strength_authority(rules):
    state = _state(rules)
    state.hands = [0, 0, 0, 0]
    state.current_trick = Trick(
        0,
        Seat.SOUTH,
        [
            PlayedCard(Seat.SOUTH, Card.parse("HJ"), 0, 0),
            PlayedCard(Seat.SOUTH, Card.parse("HA"), 1, 0),
        ],
    )

    class Store:
        play_domain = rules.play_domain

        @staticmethod
        def get_decision(_decision_id):
            return {"decisionKind": "play"}

        @staticmethod
        def load_decision_payload(_decision_id):
            return {"state": state}

    projection = build_observer_projection(Store(), "DEC-SIGNAL", Seat.SOUTH)
    assert [
        item["kind"] for item in projection.information_state["signals"]
    ] == ["high_then_low"]


def test_enabled_choice_quality_conventions_have_production_detectors(rules):
    info = _information(rules)
    minimum = detect_signals(
        replace(info, plays=_trick(Card.of(Suit.HEARTS, Rank.TEN))),
        rules.play_domain,
    )
    expensive = detect_signals(
        replace(info, plays=_trick(Card.of(Suit.HEARTS, Rank.JACK))),
        rules.play_domain,
    )
    assert [item.kind for item in minimum] == ["minimum_winner"]
    assert [item.kind for item in expensive] == ["high_expenditure"]
    assert minimum[0].relationship == expensive[0].relationship == "self"


def test_hidden_opponent_alternatives_cannot_produce_choice_quality_signal(rules):
    info = _information(rules)
    opponent_trick = tuple(
        replace(item, seat=Seat.WEST, legal_cards=())
        if item.seat == Seat.SOUTH
        else item
        for item in _trick(Card.of(Suit.HEARTS, Rank.JACK))
    )
    observations = detect_signals(
        replace(info, plays=opponent_trick), rules.play_domain
    )
    assert not {
        item.kind for item in observations
    } & {"minimum_winner", "high_expenditure"}


def test_every_enabled_convention_has_a_reachable_production_detector(
    rules, convention
):
    info = _information(rules)
    cases = (
        (
            PlayedCard(Seat.SOUTH, Card.parse("S7"), 0, 0),
            PlayedCard(Seat.SOUTH, Card.parse("S8"), 1, 1),
        ),
        (
            PlayedCard(Seat.SOUTH, Card.parse("SA"), 0, 0),
            PlayedCard(Seat.SOUTH, Card.parse("S10"), 1, 1),
        ),
        (
            PlayedCard(Seat.SOUTH, Card.parse("D7"), 0, 0),
            PlayedCard(Seat.NORTH, Card.parse("D8"), 1, 0),
        ),
        _trick(Card.of(Suit.HEARTS, Rank.TEN)),
        _trick(Card.of(Suit.HEARTS, Rank.JACK)),
    )
    produced = {
        observation.kind
        for plays in cases
        for observation in detect_signals(
            replace(info, plays=plays), rules.play_domain
        )
    }
    enabled = {
        kind
        for kind, value in convention["signals"].items()
        if value["enabled"] is True
    }
    assert enabled == PROBABILITY_SIGNAL_KINDS
    assert produced >= enabled


def test_posterior_strength_interpretations_are_sun_hokum_aware(rules):
    high_card = Card.parse("HJ")
    cases = (
        (
            "low_then_high",
            (
                PlayedCard(
                    Seat.SOUTH,
                    Card.parse("H7"),
                    0,
                    0,
                    (Card.parse("H7"), Card.parse("D8"), Card.parse("S9")),
                ),
                PlayedCard(Seat.SOUTH, Card.parse("H10"), 1, 0),
            ),
            (high_card,),
            {"encourage_suit": 1.2, "no_signal_or_deception": 0.8},
        ),
        (
            "partner_return",
            (
                PlayedCard(Seat.NORTH, Card.parse("H8"), 0, 0),
                PlayedCard(
                    Seat.SOUTH,
                    Card.parse("H7"),
                    1,
                    0,
                    (Card.parse("H7"), Card.parse("D8"), Card.parse("S9")),
                ),
            ),
            (high_card,),
            {"support_partner_suit": 1.2, "tactical_return": 0.8},
        ),
    )
    for kind, plays, remaining, ratios in cases:
        state = _state(rules)
        state.hands = [0, 0, 0, 0]
        state.hands[int(Seat.SOUTH)] = cards_to_mask(remaining)
        state.current_trick = Trick(0, plays[0].seat, list(plays))
        observed = InformationState.from_complete(state, Seat.SOUTH)
        observation = next(
            item
            for item in detect_signals(observed, rules.play_domain)
            if item.kind == kind
        )
        profile = {
            "apply_to_probability": True,
            "signals": {
                kind: {"enabled": True, "likelihood_ratios": ratios}
            },
        }
        hokum_weight = deal_likelihood(
            state,
            observed,
            (observation,),
            profile,
            play_domain=rules.play_domain,
        )
        sun_weight = deal_likelihood(
            state,
            replace(observed, contract=Contract.SUN, trump_suit=None),
            (observation,),
            profile,
            play_domain=rules.play_domain,
        )
        assert hokum_weight > 1.0 > sun_weight


@pytest.mark.parametrize(
    ("card", "ratios", "comparison"),
    (
        (
            Card.parse("H10"),
            {"preserve_strength": 1.8},
            "greater",
        ),
        (
            Card.parse("HJ"),
            {"urgency_or_signal": 1.8, "error_or_deception": 0.5},
            "greater",
        ),
        (
            Card.parse("HA"),
            {"urgency_or_signal": 1.8, "error_or_deception": 0.5},
            "less",
        ),
    ),
)
def test_choice_quality_observations_are_soft_probability_evidence(
    rules, card, ratios, comparison
):
    state = _state(rules)
    state.hands = [0, 0, 0, 0]
    state.current_trick = Trick(
        0,
        Seat.EAST,
        list(_trick(card)),
    )
    info = InformationState.from_complete(state, Seat.SOUTH)
    observation = detect_signals(info, rules.play_domain)[0]
    injected_probability_control = {
        "apply_to_probability": True,
        "signals": {
            observation.kind: {
                "enabled": True,
                "likelihood_ratios": ratios,
            }
        },
    }
    before = state.to_dict(reveal_hands=True)
    weight = deal_likelihood(
        state,
        info,
        (observation,),
        injected_probability_control,
        play_domain=rules.play_domain,
    )
    assert (weight > 1.0) if comparison == "greater" else (weight < 1.0)
    assert state.to_dict(reveal_hands=True) == before
