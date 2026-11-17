from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Rank, Seat
from baloot_platform.research.ortools_flagship_tail import (
    OrToolsFlagshipTailSearchV1,
    find_flagship_tail_variants,
)


def _winner(cards: list[tuple[Seat, Card]], rules) -> Seat:
    lead = cards[0][1].suit
    return max(
        (item for item in cards if item[1].suit == lead),
        key=lambda item: rules.play_domain.strength(item[1], Contract.SUN, None),
    )[0]


def _assert_legal_tail(candidate, rules) -> None:
    hands: dict[Seat, list[Card]] = defaultdict(list)
    for play in candidate.plays:
        hands[play.actor].append(play.card)
    assert sorted(len(hand) for hand in hands.values()) == [3, 3, 3, 3]

    for trick_index in (6, 7, 8):
        plays = [play for play in candidate.plays if play.trick_index == trick_index]
        assert [play.ordinal for play in plays] == [1, 2, 3, 4]
        lead_suit = plays[0].card.suit
        for play in plays:
            if any(card.suit == lead_suit for card in hands[play.actor]):
                assert play.card.suit == lead_suit
            hands[play.actor].remove(play.card)
        assert _winner([(play.actor, play.card) for play in plays], rules) == (
            candidate.trick_winners[trick_index - 6]
        )
    assert not any(hands.values())


def test_flagship_tail_is_deterministic_and_honestly_unqualified(rules):
    first = OrToolsFlagshipTailSearchV1(rules=rules).solve(maximum_seconds=10)
    repeated = OrToolsFlagshipTailSearchV1(rules=rules).solve(maximum_seconds=10)

    assert first.status == repeated.status == "tail_candidate"
    assert first.solver_status == repeated.solver_status == "optimal"
    assert first.candidate is not None
    assert repeated.candidate is not None
    assert first.candidate.candidate_hash == repeated.candidate.candidate_hash
    assert first.candidate.exact_no_good == repeated.candidate.exact_no_good
    assert first.model_hash == repeated.model_hash
    payload = first.candidate.to_dict()
    assert payload["requiredLift"]["status"] == "not_run"
    assert payload["requiredProof"]["status"] == "not_run"
    assert payload["requiredProof"]["requestedWinningRootActionCount"] == 1
    assert "qualified" not in payload


def test_flagship_tail_encodes_requested_cards_play_and_scoring(rules):
    result = OrToolsFlagshipTailSearchV1(rules=rules).solve(maximum_seconds=10)
    candidate = result.candidate
    assert candidate is not None
    _assert_legal_tail(candidate, rules)

    assert candidate.contract == Contract.SUN
    assert candidate.buyer.team == candidate.target_team.other
    assert candidate.root_action.rank == Rank.ACE
    assert len(candidate.root_legal_actions) == 3
    assert set(candidate.root_legal_actions) == {
        play.card for play in candidate.plays if play.actor == candidate.acting_seat
    }
    trick_six_lead = next(
        play.card
        for play in candidate.plays
        if play.trick_index == 6 and play.ordinal == 1
    )
    assert all(card.suit != trick_six_lead.suit for card in candidate.root_legal_actions)
    assert candidate.trick_winners[0].team == candidate.target_team.other
    assert candidate.trick_winners[1].team == candidate.target_team
    assert candidate.trick_winners[2].team == candidate.target_team

    for trick_index in (7, 8):
        winner = candidate.trick_winners[trick_index - 6]
        winning_play = next(
            play
            for play in candidate.plays
            if play.trick_index == trick_index and play.actor == winner
        )
        assert winning_play.card.rank in {Rank.SEVEN, Rank.EIGHT, Rank.NINE}

    final_aces = [
        play
        for play in candidate.plays
        if play.trick_index == 8 and play.card.rank == Rank.ACE
    ]
    assert len(final_aces) == 2
    assert all(play.actor.team == candidate.buyer.team for play in final_aces)
    assert candidate.project_game_points[int(candidate.buyer.team)] == (
        rules.project_domain.sequence_value_for(Contract.SUN, "five_plus")
    )
    totals = tuple(
        candidate.final_game_points[index] + candidate.project_game_points[index]
        for index in range(2)
    )
    assert totals[int(candidate.target_team)] > totals[int(candidate.buyer.team)]
    assert sum(candidate.final_raw_points) == 130
    point_value = rules.play_domain.sun_points
    points_by_trick = {
        trick: sum(
            point_value[int(play.card.rank)]
            for play in candidate.plays
            if play.trick_index == trick
        )
        for trick in (6, 7, 8)
    }
    tail_points = sum(points_by_trick.values())
    assert sum(candidate.prefix_raw_points) == 120 - tail_points
    assert candidate.final_raw_points[int(candidate.buyer.team)] == (
        candidate.prefix_raw_points[int(candidate.buyer.team)] + points_by_trick[6]
    )
    assert candidate.final_raw_points[int(candidate.target_team)] == (
        candidate.prefix_raw_points[int(candidate.target_team)]
        + points_by_trick[7]
        + points_by_trick[8]
        + 10
    )


def test_exact_no_good_and_suit_symmetry_exclusion_find_new_tail(rules):
    first = OrToolsFlagshipTailSearchV1(rules=rules).solve(maximum_seconds=10).candidate
    assert first is not None
    second_result = OrToolsFlagshipTailSearchV1(
        rules=rules,
        excluded_exact_tails=[first.exact_no_good, first.exact_no_good],
        excluded_suit_symmetry_keys=[first.suit_symmetry_key],
    ).solve(maximum_seconds=10)
    second = second_result.candidate

    assert second is not None
    assert second.exact_no_good != first.exact_no_good
    assert second.exact_no_good_hash != first.exact_no_good_hash
    assert second.suit_symmetry_key != first.suit_symmetry_key
    assert len(second_result.exact_no_good_hashes) == 1


def test_five_tail_variants_are_exactly_distinct_and_not_suit_symmetric(rules):
    variants = find_flagship_tail_variants(
        rules=rules,
        count=5,
        maximum_seconds_per_variant=10,
    )

    assert len(variants) == 5
    assert len({candidate.candidate_hash for candidate in variants}) == 5
    assert len({candidate.exact_no_good_hash for candidate in variants}) == 5
    assert len({candidate.suit_symmetry_key for candidate in variants}) == 5
    assert all(candidate.status == "tail_candidate" for candidate in variants)


@pytest.mark.parametrize(
    "tail",
    [tuple(range(11)), (*range(11), 10), (*range(11), 32)],
)
def test_invalid_exact_tail_no_goods_fail_closed(rules, tail):
    with pytest.raises(ValueError, match="tail no-good"):
        OrToolsFlagshipTailSearchV1(rules=rules, excluded_exact_tails=[tail])


@pytest.mark.parametrize("count", [0, 65, True, 1.5])
def test_variant_count_is_governed(rules, count):
    with pytest.raises(ValueError, match="variant count"):
        find_flagship_tail_variants(rules=rules, count=count)


def test_new_tail_constructor_has_no_engine_or_runtime_research_import(root: Path):
    source = (
        root / "src/baloot_platform/research/ortools_flagship_tail.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "core.engine",
        "BalootEngine",
        "HeuristicBot",
        "scenario_verifier",
        "proof_service",
    )
    assert all(token not in source for token in forbidden)
