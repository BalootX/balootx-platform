from types import SimpleNamespace

import pytest

from baloot_platform.research.ortools_research_executor import _future_card_count_admissible_actions_v1
from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioPredicateV1


@pytest.mark.parametrize("plays,actor,actions,trick,count,expected", [
    ([], "north", ["CA", "SK"], 7, 2, ("SK",)),
    ([('north', 'SK'), ('west', 'S7')], "south", ["SA", "S9"], 0, 1, ("SA",)),
    ([('north', 'SK'), ('west', 'S7'), ('south', 'S9')], "east", ["S8", "SJ"], 0, 1, ()),
])
def test_native_witness_constraints_filter_before_selection_and_survive_partial_tricks(
    plays, actor, actions, trick, count, expected,
):
    hands = {"north": ["SK", "H10", "DQ", "C10", "CJ", "CQ", "CK", "CA"],
        "west": ["S7", "S10", "H9", "D10", "DK", "DA", "C8", "C9"],
        "south": ["S9", "SQ", "SA", "H7", "HJ", "HQ", "HK", "C7"],
        "east": ["S8", "SJ", "H8", "HA", "D7", "D8", "D9", "DJ"]}
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "dealer": "east",
        "deckOrder": [11, 21, 27, 0, 22, 23, 2, 5, 7, 1, 16, 17, 28, 31, 25, 26,
                      8, 24, 18, 20, 6, 29, 30, 3, 10, 19, 12, 13, 14, 4, 9, 15],
        "matchScore": [0, 0], "auctionActions": [{"actor": "north", "action": "sun"}],
        "escalationActions": [], "projectDeclarations": [],
        "playActions": [{"actor": seat, "card": card} for seat, card in plays]})
    predicate = ScenarioPredicateV1.build({"predicateId": "PRED-PRESERVED-ACES",
        "version": 1, "hardness": "hard", "kind": "play.trick_card_count", "arguments": {
            "trickIndex": trick, "selector": {"ranks": ["ace"],
                "owner": {"relation": "team", "team": "north_south"}},
            "minimum": count, "maximum": count}})
    normalized = SimpleNamespace(next_actor=actor, legal_actions=[{"code": code} for code in actions])
    before = prefix.to_dict()
    assert _future_card_count_admissible_actions_v1(hands, prefix, normalized, (predicate,),
        maximum_seconds=5) == expected
    assert prefix.to_dict() == before
