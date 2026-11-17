from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_exact_state import _native_project_visibility_projection
from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1


@pytest.mark.parametrize("scope,expected", [("scoring_team", [1, 1, 0]), ("comparison_winner", [1, 0, 0])])
def test_native_reveal_scope_preserves_scoring_without_disclosing_teammate(scope, expected):
    model = cp_model.CpModel()
    projects = [SimpleNamespace(
        project_id=str(index), owner=seat, eligible=model.new_constant(1),
        declared=model.new_constant(1), declaration_order=model.new_constant(index),
        comparison_winner=model.new_constant(int(index == 0)),
        awarded=model.new_constant(int(index < 2)),
    ) for index, seat in enumerate((Seat.NORTH, Seat.SOUTH, Seat.WEST))]
    result = _native_project_visibility_projection(
        model=model, fragment=SimpleNamespace(projects=projects),
        seen_first={seat: model.new_constant(1) for seat in Seat},
        owner_seen=lambda *args: model.new_constant(1), prefix_length=8,
        declaration_seats=tuple(Seat), reveal_scope=scope,
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    assert [solver.value(result[4][str(i)]) for i in range(3)] == [1, 1, 0]
    assert [solver.value(result[5][str(i)]) for i in range(3)] == expected


def test_public_project_binder_does_not_remove_hidden_scoring_teammate(monkeypatch):
    model = cp_model.CpModel()
    projects = [SimpleNamespace(
        owner=seat, kind="siri", value=4, high_rank=rank,
        cards=tuple(map(Card.parse, cards.split())), declared=model.new_constant(1),
        comparison_winner=model.new_constant(int(index == 0)), awarded=model.new_constant(1),
    ) for index, (seat, rank, cards) in enumerate((
        (Seat.NORTH, 2, "H7 H8 H9"), (Seat.SOUTH, 1, "D7 D8 D9"))) ]
    compiler = object.__new__(OrToolsActorInformationSetCompilerV1)
    compiler.rules = SimpleNamespace(project_domain=SimpleNamespace(reveal_scope="comparison_winner"))
    monkeypatch.setattr(compiler, "_history", lambda state: [SimpleNamespace(actor=seat) for seat in tuple(Seat) * 2])
    state = {"projects": [{"owner": p.owner.name.lower(), "kind": p.kind,
                           "value": p.value, "highRank": p.high_rank,
                           "declarationOrder": index, "revealed": index == 0,
                           "cards": [card.to_dict() for card in p.cards] if index == 0 else None}
                          for index, p in enumerate(projects)]}
    compiler._bind_public_project_choices(model, SimpleNamespace(projects=projects), state)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL


@pytest.mark.parametrize("native", [False, True])
def test_project_world_projection_reveals_only_order_winner(root, native):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v6.yaml")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=native)
    hands = {seat: tuple(map(Card.parse, cards.split())) for seat, cards in (
        (Seat.NORTH, "H7 H8 H9 SA D10 CJ DQ C7"),
        (Seat.WEST, "D7 D8 D9 S10 HA C10 SQ C8"),
        (Seat.SOUTH, "S7 S8 S9 HQ HK DA C9 CA"),
        (Seat.EAST, "SJ DJ SK H10 HJ CK DK CQ"),
    )}
    projects = [{"owner": seat.name.lower(), "kind": "siri", "value": 4,
                 "highRank": 2, "declarationOrder": order}
                for order, seat in enumerate((Seat.NORTH, Seat.WEST, Seat.SOUTH), start=1)]
    history = [[(Seat.NORTH, "SA"), (Seat.WEST, "S10"), (Seat.SOUTH, "S7"), (Seat.EAST, "SJ")],
               [(Seat.NORTH, "C7"), (Seat.WEST, "C8"), (Seat.SOUTH, "C9"), (Seat.EAST, "CQ")]]
    state = {"contract": "sun", "dealer": "east", "currentPlayer": "east",
             "projects": projects, "currentTrick": None,
             "completedTricks": [{"plays": [{"seat": seat.name.lower(), "card": Card.parse(card).to_dict()}
                                             for seat, card in trick]} for trick in history]}
    rows, _ = compiler._projects(state, hands)
    assert {row["owner"] for row in rows if row["scoring"]} == {"north", "south"}
    assert {row["owner"] for row in rows if row["revealed"]} == {"north"}
    assert all(row["cards"] is None for row in rows if row["owner"] != "north")


@pytest.mark.parametrize("native", [False, True])
def test_ratified_scoring_team_reveals_both_scorers_and_neither_loser(root, native):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v7.yaml")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=native)
    hands = {seat: tuple(map(Card.parse, cards.split())) for seat, cards in (
        (Seat.NORTH, "S7 S8 S9 HA SQ D10 C10 DA"),
        (Seat.WEST, "H7 H8 H9 SA SK DJ CJ CA"),
        (Seat.SOUTH, "D7 D8 D9 HK HQ DQ CQ H10"),
        (Seat.EAST, "C7 C8 C9 SJ HJ DK CK S10"),
    )}
    projects = [{"owner": seat.name.lower(), "kind": "siri", "value": 4,
                 "highRank": 2, "declarationOrder": order}
                for order, seat in enumerate((Seat.NORTH, Seat.WEST, Seat.SOUTH, Seat.EAST), start=1)]
    history = [[(Seat.NORTH, "HA"), (Seat.WEST, "H7"), (Seat.SOUTH, "HQ"), (Seat.EAST, "HJ")],
               [(Seat.NORTH, "C10"), (Seat.WEST, "CJ"), (Seat.SOUTH, "CQ"), (Seat.EAST, "C7")]]
    state = {"contract": "sun", "dealer": "east", "currentPlayer": "south",
             "projects": projects, "currentTrick": None,
             "completedTricks": [{"plays": [{"seat": seat.name.lower(), "card": Card.parse(card).to_dict()}
                                             for seat, card in trick]} for trick in history]}
    rows, _ = compiler._projects(state, hands)
    assert {row["owner"] for row in rows if row["scoring"]} == {"north", "south"}
    assert {row["owner"] for row in rows if row["revealed"]} == {"north", "south"}
    assert all(row["cards"] for row in rows if row["scoring"])
    assert all(row["cards"] is None for row in rows if not row["scoring"])
