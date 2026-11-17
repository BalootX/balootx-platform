from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from baloot_platform.core.cards import FULL_DECK, Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_project_model import (
    OrToolsProjectLifecycleModelV1,
)
from baloot_platform.research.ortools_research_executor import (
    GenericCompleteTraceCompilerV1,
    _project_count_projection_is_infeasible,
)
from baloot_platform.research.ortools_scenario_model import OrToolsScenarioModelV1
from baloot_platform.research.scenario_contracts import ResearchScenarioV1


_AWARDED_PARTNER_DECK = (
    10, 13, 23, 5, 9, 0, 7, 22, 31, 6, 27, 20, 1, 12, 4, 18,
    25, 24, 28, 8, 11, 14, 16, 17, 3, 19, 29, 26, 15, 2, 21, 30,
)


def _hands_from_deck(rules):
    layout = DealLayoutV1.build(
        rules.structural_domain,
        dealer=Seat.EAST,
        buyer=Seat.NORTH,
    )
    return {
        seat: tuple(
            Card(_AWARDED_PARTNER_DECK[position])
            for position in layout.positions_for(seat, final=True)
        )
        for seat in Seat
    }


def _hands_with_north(codes: tuple[str, ...]):
    north = tuple(Card.parse(code) for code in codes)
    remainder = tuple(card for card in FULL_DECK if card not in north)
    return {
        Seat.NORTH: north,
        Seat.WEST: remainder[:8],
        Seat.SOUTH: remainder[8:16],
        Seat.EAST: remainder[16:],
    }


def _sun_awarded_hundred_search() -> ResearchScenarioV1:
    return ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    {
                        "predicateId": "PRED-SUN",
                        "kind": "auction.contract",
                        "version": 1,
                        "hardness": "hard",
                        "arguments": {"contract": "sun"},
                    },
                    {
                        "predicateId": "PRED-AWARDED-HUNDRED",
                        "kind": "project.exists",
                        "version": 1,
                        "hardness": "hard",
                        "arguments": {
                            "team": "north_south",
                            "kind": "hundred_sequence",
                            "value": 20,
                            "state": "awarded",
                        },
                    },
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {"trickIndex": 6}},
            "outcome": {
                "predicateId": "PRED-REPORT-ROUND",
                "kind": "outcome.team_round",
                "version": 1,
                "hardness": "report",
                "arguments": {"team": "north_south", "status": "won"},
            },
        }
    )


def test_second_trick_project_comparison_requires_observed_leader(rules):
    with pytest.raises(ValueError, match="second-trick leader"):
        OrToolsProjectLifecycleModelV1().solve(
            hands=_hands_from_deck(rules), contract=Contract.SUN,
            dealer=Seat.EAST, trump=None,
            domain=replace(rules.project_domain, tie_break_order="second_trick_reveal"),
        )


def test_cp_sat_project_lifecycle_keeps_all_states_distinct(rules):
    result = OrToolsProjectLifecycleModelV1().solve(
        hands=_hands_from_deck(rules),
        contract=Contract.SUN,
        dealer=Seat.EAST,
        trump=None,
        domain=rules.project_domain,
    )

    north_hundred = next(
        project
        for project in result.projects
        if project.owner == Seat.NORTH and project.kind == "hundred_sequence"
    )
    south_siri = next(
        project
        for project in result.projects
        if project.owner == Seat.SOUTH and project.kind == "siri"
    )
    west_selected = [
        project
        for project in result.projects
        if project.owner == Seat.WEST and project.selected
    ]

    assert result.solver_status == "optimal"
    assert result.winning_team == Team.NORTH_SOUTH
    assert result.combined_awarded_value == (24, 0)
    assert result.eligible_count == 9
    assert result.selected_count == 4
    assert result.declared_count == 1
    assert result.awarded_count == 2
    assert north_hundred.eligible and north_hundred.selected
    assert north_hundred.declared and north_hundred.comparison_winner
    assert north_hundred.awarded
    assert south_siri.eligible and south_siri.selected and south_siri.awarded
    assert not south_siri.declared and not south_siri.comparison_winner
    assert len(west_selected) == 2
    assert not (west_selected[0].card_mask & west_selected[1].card_mask)
    assert all(not project.awarded for project in west_selected)


def test_cp_sat_project_evidence_and_model_hashes_are_deterministic(rules):
    hands = _hands_from_deck(rules)
    reversed_hands = {
        seat: tuple(reversed(cards)) for seat, cards in reversed(tuple(hands.items()))
    }
    model = OrToolsProjectLifecycleModelV1()
    first = model.solve(
        hands=hands,
        contract="sun",
        dealer="east",
        trump=None,
        domain=rules.project_domain,
    )
    second = model.solve(
        hands=reversed_hands,
        contract="sun",
        dealer="east",
        trump=None,
        domain=rules.project_domain,
    )

    assert first.model_hash == second.model_hash
    assert first.evidence_hash == second.evidence_hash
    assert first.projects == second.projects


def test_cp_sat_project_stage_derives_baloot_and_blocked_baloot(rules):
    model = OrToolsProjectLifecycleModelV1()
    eligible = model.solve(
        hands=_hands_with_north(
            ("HQ", "HK", "S7", "C8", "D9", "S10", "CJ", "DA")
        ),
        contract=Contract.HOKUM,
        dealer=Seat.EAST,
        trump=Suit.HEARTS,
        domain=rules.project_domain,
    )
    blocked = model.solve(
        hands=_hands_with_north(
            ("HQ", "SQ", "DQ", "CQ", "HK", "S7", "D8", "C9")
        ),
        contract=Contract.HOKUM,
        dealer=Seat.EAST,
        trump=Suit.HEARTS,
        domain=rules.project_domain,
    )

    assert eligible.baloot_seats == frozenset({Seat.NORTH})
    assert eligible.baloot_teams == frozenset({Team.NORTH_SOUTH})
    assert eligible.baloot_points == (rules.project_domain.baloot_value, 0)
    assert blocked.winning_team == Team.NORTH_SOUTH
    assert any(
        project.owner == Seat.NORTH
        and project.kind == "hundred_four_kind"
        and project.awarded
        for project in blocked.projects
    )
    assert blocked.baloot_seats == frozenset()
    assert blocked.baloot_points == (0, 0)


def test_cp_sat_project_stage_counts_baloot_inside_official_hundred(root: Path):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    result = OrToolsProjectLifecycleModelV1().solve(
        hands=_hands_with_north(
            ("H9", "H10", "HJ", "HQ", "HK", "S7", "D8", "C9")
        ),
        contract=Contract.HOKUM,
        dealer=Seat.EAST,
        trump=Suit.HEARTS,
        domain=rules.project_domain,
    )

    assert any(
        project.owner == Seat.NORTH
        and project.kind == "hundred_sequence"
        and project.selected
        for project in result.projects
    )
    assert result.baloot_points == (rules.project_domain.baloot_value, 0)


def test_cp_sat_project_stage_handles_disabled_projects_and_rejects_bad_inputs(rules):
    disabled = replace(rules.project_domain, enabled=False)
    result = OrToolsProjectLifecycleModelV1().solve(
        hands=_hands_from_deck(rules),
        contract="sun",
        dealer="east",
        trump=None,
        domain=disabled,
    )
    assert result.projects == ()
    assert result.combined_awarded_value == (0, 0)

    unsupported = replace(rules.project_domain, only_winning_team_scores=False)
    with pytest.raises(ValueError, match="unsupported"):
        OrToolsProjectLifecycleModelV1().solve(
            hands=_hands_from_deck(rules),
            contract="sun",
            dealer="east",
            trump=None,
            domain=unsupported,
        )
    duplicate = _hands_from_deck(rules)
    duplicate[Seat.NORTH] = (*duplicate[Seat.NORTH][:-1], duplicate[Seat.WEST][0])
    with pytest.raises(ValueError, match="every card exactly once"):
        OrToolsProjectLifecycleModelV1().solve(
            hands=duplicate,
            contract="sun",
            dealer="east",
            trump=None,
            domain=rules.project_domain,
        )
    with pytest.raises(ValueError, match="requires trump"):
        OrToolsProjectLifecycleModelV1().solve(
            hands=_hands_from_deck(rules),
            contract="hokum",
            dealer="east",
            trump=None,
            domain=rules.project_domain,
        )


def test_complete_trace_constructs_the_hard_awarded_hundred_in_sun(rules, monkeypatch):
    original_solve = OrToolsScenarioModelV1.solve
    legacy_calls = []

    def capture_legacy_solve(model, **kwargs):
        assert "solver" not in kwargs
        legacy_calls.append(1)
        return original_solve(model, **kwargs)

    monkeypatch.setattr(OrToolsScenarioModelV1, "solve", capture_legacy_solve)
    attempt = GenericCompleteTraceCompilerV1(rules=rules)(
        _sun_awarded_hundred_search(),
        (),
        seed=0,
        maximum_seconds=15,
    )

    assert attempt.status == "feasible", attempt.reason
    assert legacy_calls
    assert attempt.candidate is not None
    assert attempt.candidate.trace["contract"] == "sun"
    assert any(
        project["team"] == "north_south"
        and project["kind"] == "hundred_sequence"
        and project["value"] == 20
        and project["awarded"]
        for project in attempt.candidate.trace["projects"]["projects"]
    )
    project_receipt = attempt.candidate.stage_receipts["projects"]
    assert project_receipt.model_hash == attempt.candidate.trace["projects"][
        "modelHash"
    ]
    assert project_receipt.semantic_hash != "0" * 64


def test_contradictory_project_construction_is_exactly_infeasible(rules):
    payload = _sun_awarded_hundred_search().to_dict()
    payload.pop("sha256")
    payload["constraints"] = {
        "op": "all",
        "children": [
            {
                "predicateId": "PRED-SUN",
                "kind": "auction.contract",
                "version": 1,
                "hardness": "hard",
                "arguments": {"contract": "sun"},
            },
            {
                "predicateId": "PRED-PROJECT-PRESENT",
                "kind": "project.count",
                "version": 1,
                "hardness": "hard",
                "arguments": {
                    "team": "north_south",
                    "minimum": 1,
                    "maximum": 32,
                    "state": "eligible",
                },
            },
            {
                "predicateId": "PRED-PROJECT-ABSENT",
                "kind": "project.count",
                "version": 1,
                "hardness": "hard",
                "arguments": {
                    "team": "north_south",
                    "minimum": 0,
                    "maximum": 0,
                    "state": "eligible",
                },
            },
        ],
    }
    scenario = ResearchScenarioV1.build(payload)
    assert _project_count_projection_is_infeasible(
        scenario,
        tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.kind == "project.count"
        ),
        rules,
        maximum_conflicts=250_000,
    )

    alternative_payload = dict(payload)
    alternative_payload["constraints"] = dict(payload["constraints"])
    alternative_payload["constraints"]["op"] = "any"
    alternative = ResearchScenarioV1.build(alternative_payload)
    assert not _project_count_projection_is_infeasible(
        alternative,
        tuple(
            predicate
            for predicate in alternative.predicates()
            if predicate.kind == "project.count"
        ),
        rules,
        maximum_conflicts=250_000,
    )

    attempt = GenericCompleteTraceCompilerV1(rules=rules)(
        scenario,
        (),
        seed=0,
        maximum_seconds=15,
    )

    assert attempt.status == "infeasible", attempt.reason
    assert attempt.reason == "deal_auction_project_infeasible"


def test_new_project_stage_has_no_runtime_research_dependency():
    source = (
        Path(__file__).parents[1]
        / "src/baloot_platform/research/ortools_project_model.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "core.engine",
        "core.projects",
        "project_compiler",
        "reference_project",
        "HeuristicBot",
        "proof_service",
        "RuleBook",
    ):
        assert forbidden not in source
