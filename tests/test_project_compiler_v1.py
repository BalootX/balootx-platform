from __future__ import annotations

import copy

import pytest

from baloot_platform.core.cards import FULL_DECK, Card
from baloot_platform.core.enums import Contract, Rank, Seat
from baloot_platform.core.project_ir import ProjectDomainIRV1
from baloot_platform.core.projects import project_candidates, score_project_team
from baloot_platform.core.rules import RuleBook
from baloot_platform.research import OrToolsCpSatProjectCompilerV1
from baloot_platform.research.reference_project import ReferenceProjectInterpreterV1
from baloot_platform.service.container import build_container


def _trace(seed: int):
    return build_container().research.simulator.simulate(seed).final_trace.trusted_final_state


@pytest.mark.parametrize("seed", [0, 1, 5, 17])
def test_fixed_trace_matches_project_selection_declaration_and_score(seed: int):
    container = build_container()
    state = _trace(seed)
    plan = OrToolsCpSatProjectCompilerV1().verify_trace(
        state, rules=container.research.simulator.rules
    )
    assert plan.solver_status == "optimal"
    assert list(plan.project_points) == state["roundScore"]["projectPoints"]
    assert list(plan.baloot_points) == state["roundScore"]["balootPoints"]
    assert {team.name.lower() for team in plan.baloot_teams} == set(
        state["balootTeams"]
    )
    second_trick_order = [play["seat"] for play in state["completedTricks"][1]["plays"]]
    assert all(project.revealed for project in plan.projects if project.scoring)
    assert all(not project.revealed for project in plan.projects if not project.scoring)
    for reveal in plan.reveal_schedule:
        assert second_trick_order[reveal.play_position] == reveal.owner.name.lower()
        assert reveal.trick_index == 1


def test_fixed_project_compilation_is_deterministic():
    container = build_container()
    state = _trace(1)
    compiler = OrToolsCpSatProjectCompilerV1()
    first = compiler.verify_trace(state, rules=container.research.simulator.rules)
    second = compiler.verify_trace(state, rules=container.research.simulator.rules)
    assert first == second


def test_trace_project_tamper_is_rejected():
    container = build_container()
    state = copy.deepcopy(_trace(1))
    state["projects"][0]["value"] += 1
    with pytest.raises(AssertionError, match="project selection"):
        OrToolsCpSatProjectCompilerV1().verify_trace(
            state, rules=container.research.simulator.rules
        )


def test_four_hundred_is_compiled_from_the_governed_project_rules():
    container = build_container()
    aces = [card for card in FULL_DECK if card.rank == Rank.ACE]
    remainder = [card for card in FULL_DECK if card.rank != Rank.ACE]
    hands = {
        Seat.NORTH: tuple(aces + remainder[:4]),
        Seat.WEST: tuple(remainder[4:12]),
        Seat.SOUTH: tuple(remainder[12:20]),
        Seat.EAST: tuple(remainder[20:28]),
    }
    plan = OrToolsCpSatProjectCompilerV1().compile(
        hands=hands,
        contract=Contract.SUN,
        dealer=Seat.EAST,
        trump=None,
        rules=container.research.simulator.rules,
    )
    four_hundred = [project for project in plan.projects if project.kind == "four_hundred"]
    assert len(four_hundred) == 1
    assert four_hundred[0].owner == Seat.NORTH
    assert four_hundred[0].declared is True
    assert four_hundred[0].scoring is True
    assert plan.project_points[int(Seat.NORTH.team)] >= four_hundred[0].value
    assert four_hundred[0].value == int(
        container.research.simulator.rules.data["projects"]["four_of_kind"]
        ["values_game_points"]["sun"]["ace"]
    )


def test_project_disable_influences_runtime_reference_and_cp_sat(rules):
    north = tuple(
        Card.parse(code)
        for code in ("S7", "S8", "S9", "H7", "D9", "C10", "HQ", "DK")
    )
    remainder = tuple(card for card in FULL_DECK if card not in north)
    hands = {
        Seat.NORTH: north,
        Seat.WEST: remainder[:8],
        Seat.SOUTH: remainder[8:16],
        Seat.EAST: remainder[16:24],
    }
    payload = copy.deepcopy(rules.data)
    payload["projects"]["enabled"] = False
    disabled_rules = RuleBook(payload)
    domain = ProjectDomainIRV1.from_rules(disabled_rules.data)
    reference = ReferenceProjectInterpreterV1(domain)

    assert project_candidates(
        Seat.NORTH, north, Contract.SUN, disabled_rules
    ) == ()
    assert reference.candidates(Seat.NORTH, north, Contract.SUN) == ()
    plan = OrToolsCpSatProjectCompilerV1().compile(
        hands=hands,
        contract=Contract.SUN,
        dealer=Seat.EAST,
        trump=None,
        rules=disabled_rules,
    )
    assert plan.solver_status == "optimal"
    assert plan.projects == ()


def test_project_domain_rejects_legacy_and_incomplete_semantic_inputs(rules):
    with pytest.raises(TypeError, match="compiled ProjectDomainIRV1"):
        ReferenceProjectInterpreterV1(rules.data)
    with pytest.raises(TypeError):
        score_project_team([])

    payload = copy.deepcopy(rules.data)
    del payload["projects"]["sequences"]["values_game_points"]["sun"]["three"]
    with pytest.raises(ValueError, match="incomplete or invalid project domain"):
        ProjectDomainIRV1.from_rules(payload)
