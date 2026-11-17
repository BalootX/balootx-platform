from __future__ import annotations

import pytest

from baloot_platform.core.cards import FULL_DECK
from baloot_platform.core.enums import Rank, Seat
from baloot_platform.research.ortools_play_state_predicates import (
    OrToolsPlayStatePredicateLoweringV1,
)
from baloot_platform.research.ortools_play_tail import (
    OrToolsPlayTailStageV1,
    PlayTailSolveStopped,
)
from baloot_platform.research.scenario_contracts import (
    ScenarioExpressionV1,
    ScenarioPredicateV1,
)
from baloot_platform.research.scenario_predicate_lowering_v2 import (
    ScenarioPredicateLoweringRegistryV2,
)


def _ownership():
    return {card: Seat(card.id // 8) for card in FULL_DECK}


def _build(rules, count: int):
    return OrToolsPlayTailStageV1().build_model(
        cards=FULL_DECK,
        ownership=_ownership(),
        contract="sun",
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        selected_play_count=count,
        decision_prefix_length=0,
    )


@pytest.mark.parametrize("count", [4, 8, 12, 16])
def test_selective_tail_chooses_balanced_legal_cards_from_full_deck(
    rules, count: int
) -> None:
    build = _build(rules, count)
    result = build.solve(maximum_time_seconds=10)

    assert len(build.selected_card_literals) == 32
    assert len(build.selected_play_indexes) == count
    assert len(result.plays) == count
    assert all(len(hand) == count // 4 for hand in result.initial_hands)
    assert {play.card for play in result.plays} == {
        card for hand in result.initial_hands for card in hand
    }


def test_live_tail_contexts_constrain_action_trick_and_state_variables(rules) -> None:
    build = _build(rules, 12)
    action_expression = ScenarioExpressionV1.build(
        {
            "op": "all",
            "children": [
                {
                    "predicateId": "PRED-TAIL-ACE",
                    "kind": "play.action_matches",
                    "version": 1,
                    "hardness": "hard",
                    "arguments": {
                        "trickIndex": 5,
                        "position": 0,
                        "actingSeat": "north",
                        "selector": {"ranks": ["ace"]},
                    },
                }
            ],
        }
    )
    action_root, _ = ScenarioPredicateLoweringRegistryV2().lower_expression(
        action_expression, build.predicate_context
    )
    build.model.add(action_root == 1)
    state_predicate = ScenarioPredicateV1.build(
        {
            "predicateId": "PRED-TAIL-TRICK",
            "kind": "play.trick_index",
            "version": 1,
            "hardness": "hard",
            "arguments": {"minimum": 5, "maximum": 5},
        }
    )
    state_lowering = OrToolsPlayStatePredicateLoweringV1().lower_predicate(
        state_predicate, build.play_state_context
    )
    build.model.add(state_lowering.literal == 1)

    result = build.solve(maximum_time_seconds=10)

    assert result.plays[0].actor == Seat.NORTH
    assert result.plays[0].card.rank == Rank.ACE


def test_selective_tail_applies_governed_decision_state_distance(rules) -> None:
    first = _build(rules, 8).solve(maximum_time_seconds=10)
    first_vector = [8] * 32
    for seat in Seat:
        for card in first.initial_hands[int(seat)]:
            first_vector[card.id] = (int(seat) - int(Seat.NORTH)) % 4

    second = OrToolsPlayTailStageV1().solve(
        cards=FULL_DECK,
        ownership=_ownership(),
        contract="sun",
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        selected_play_count=8,
        decision_prefix_length=0,
        decision_actor=Seat.NORTH,
        excluded_decision_status_vectors=(first_vector,),
        minimum_decision_status_hamming_distance=4,
        maximum_time_seconds=10,
    )
    second_vector = [8] * 32
    for seat in Seat:
        for card in second.initial_hands[int(seat)]:
            second_vector[card.id] = (int(seat) - int(Seat.NORTH)) % 4

    assert sum(
        left != right
        for left, right in zip(first_vector, second_vector, strict=True)
    ) >= 4


def test_four_trick_tail_excludes_exact_state_after_completed_prior_tricks(rules) -> None:
    first = _build(rules, 16).solve(maximum_time_seconds=10)
    decision = first.at_prefix(8)
    assert decision.next_actor is not None
    vector = [8] * 32
    for seat in Seat:
        for card in decision.remaining_hands[int(seat)]:
            vector[card.id] = (int(seat) - int(decision.next_actor)) % 4
    with pytest.raises(PlayTailSolveStopped, match="infeasible"):
        OrToolsPlayTailStageV1().solve(
            cards=FULL_DECK,
            ownership=_ownership(),
            contract="sun",
            trump=None,
            first_leader=Seat.NORTH,
            play_domain=rules.play_domain,
            structural_domain=rules.structural_domain,
            selected_play_count=16,
            decision_prefix_length=8,
            decision_actor=decision.next_actor,
            tail=first.plays,
            exact_excluded_decision_status_vectors=(vector,),
            maximum_time_seconds=10,
        )


def test_four_trick_tail_enforces_observer_safe_policy_in_lead_in(rules) -> None:
    result = OrToolsPlayTailStageV1().solve(
        cards=FULL_DECK,
        ownership=_ownership(),
        contract="sun",
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        selected_play_count=16,
        decision_prefix_length=8,
        policy_prefix_steps=8,
        policy_id="observer_safe_tactical_category_v1",
        maximum_time_seconds=10,
    )
    assert len(result.plays) == 16
    assert len(result.at_prefix(8).remaining_hands[0]) == 2


def test_selective_tail_can_choose_unknown_first_leader(rules) -> None:
    result = OrToolsPlayTailStageV1().solve(
        cards=FULL_DECK,
        ownership=_ownership(),
        contract="sun",
        trump=None,
        first_leader=None,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        selected_play_count=16,
        decision_prefix_length=8,
        maximum_time_seconds=10,
    )
    assert len(result.plays) == 16
    assert result.plays[0].actor in Seat

    with pytest.raises(ValueError, match="unknown first leader"):
        OrToolsPlayTailStageV1().build_model(
            cards=FULL_DECK,
            ownership=_ownership(),
            contract="sun",
            trump=None,
            first_leader=None,
            play_domain=rules.play_domain,
            structural_domain=rules.structural_domain,
        )


def test_selected_tail_is_closed_and_legacy_step_count_remains_separate(rules) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        OrToolsPlayTailStageV1().build_model(
            cards=FULL_DECK,
            ownership=_ownership(),
            contract="sun",
            trump=None,
            first_leader=Seat.NORTH,
            play_domain=rules.play_domain,
            structural_domain=rules.structural_domain,
            play_step_count=12,
            selected_play_count=12,
        )
    with pytest.raises(ValueError, match="4, 8, 12, or 16"):
        _build(rules, 20)
