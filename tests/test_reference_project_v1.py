from __future__ import annotations

import random

from baloot_platform.core.cards import FULL_DECK
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.projects import project_candidates
from baloot_platform.research import ReferenceProjectInterpreterV1


def _fingerprint(projects):
    return [
        (
            project.owner,
            project.kind,
            tuple(card.id for card in project.cards),
            project.value,
            project.high_rank,
            project.category_priority,
        )
        for project in projects
    ]


def test_reference_project_candidates_match_runtime_across_fixed_deals(rules):
    reference = ReferenceProjectInterpreterV1.from_rules(rules.data)
    for seed in range(12):
        deck = list(FULL_DECK)
        random.Random(seed).shuffle(deck)
        hands = {
            seat: tuple(deck[int(seat) * 8 : (int(seat) + 1) * 8])
            for seat in Seat
        }
        for contract in (Contract.SUN, Contract.HOKUM, Contract.ASHKAL):
            for seat in Seat:
                assert _fingerprint(
                    reference.candidates(seat, hands[seat], contract)
                ) == _fingerprint(
                    project_candidates(seat, hands[seat], contract, rules)
                )


def test_reference_project_selection_matches_runtime_compiler_fixture(rules):
    reference = ReferenceProjectInterpreterV1.from_rules(rules.data)
    hand = tuple(FULL_DECK[:8])
    candidates = reference.candidates(Seat.NORTH, hand, Contract.SUN)
    selected = reference.select(candidates)
    assert selected
    assert reference.group_score(selected) >= (0, 0, 0, -32)
