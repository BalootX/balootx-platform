from types import SimpleNamespace
from unittest.mock import patch

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
from baloot_platform.research.ortools_research_executor import (
    DefaultStageContextCompilerV1,
    OrToolsResearchCompilerV1,
    OrToolsStageContextBindingV1,
    OrToolsUnifiedResearchExecutorV1,
)
from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ResearchScenarioV1
from baloot_platform.research.scenario_adjudication_predicates_v3 import (
    shared_scenario_predicate_catalog_lowering_manifest_v3,
)
from test_research_ortools_nonplay_complete_graph import (
    _escalation_prefix,
    _optional_project_prefix,
)


@pytest.mark.parametrize("decision_kind", ["bid", "escalation", "project", "project_mandatory0", "project_mandatory3"])
def test_native_nonplay_game_points_and_observer_visibility(root, rules, decision_kind):
    prefix = _escalation_prefix()
    decision = {"kind": decision_kind}
    if decision_kind == "bid":
        raw = prefix.to_dict()
        raw.pop("sha256")
        raw["auctionActions"] = []
        prefix = ExactTracePrefixV1.build(raw)
    elif decision_kind == "escalation":
        decision["selector"] = {"actingSeat": "west"}
    elif decision_kind == "project":
        rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
        prefix = _optional_project_prefix()
        decision["selector"] = {"owner": "east"}
    elif decision_kind.startswith("project_mandatory"):
        rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")
        prefix = _optional_project_prefix()
        decision = {"kind": "project", "selector": {"owner": "north" if decision_kind.endswith("0") else "east"}}

    with patch.object(BalootEngine, "__init__", side_effect=AssertionError("native path used runtime")):
        if decision_kind.startswith("project_mandatory"):
            raw = prefix.to_dict()
            raw.pop("sha256")
            raw.update(projectActions=[], playActions=[])
            prefix = ExactTracePrefixV1.build(raw)
            normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
            actions = (("north", "S7"), ("west", "S10"), ("south", "SK"))
            for actor, card in actions[:int(decision_kind[-1])]:
                project = normalizer.normalize(prefix,
                    decision_point={"kind": "project", "selector": {"owner": actor}}, maximum_seconds=15)
                assert len(project.legal_actions) == 1
                prefix = normalizer.append_normalized_action(prefix, project, project.legal_actions[0]["code"])
                play = normalizer.normalize(prefix, decision_point={"kind": "play"}, maximum_seconds=15)
                prefix = normalizer.append_normalized_action(prefix, play, card)
        normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
            prefix, decision_point=decision, maximum_seconds=15,
        )
        candidate = OrToolsUnifiedResearchExecutorV1._candidate_from_normalized_exact(
            SimpleNamespace(rules=rules), normalized, prefix,
        )
        context = DefaultStageContextCompilerV1(
            rules, required_stages=("state", "projects") if decision_kind.startswith("project") else ("state",),
            native_cp_sat_authority=True,
            visibility_requests=[(Seat.NORTH, card) for card in range(32)],
        )(candidate, None)
        project_context = context.get("projects")
        context = context["state"].context
    if project_context is not None:
        assert project_context.context.model is context.model
    assert context.context_id.startswith(f"candidate-native-{decision['kind']}-")

    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST)
    public = {prefix.deck_order[layout.purchase_position]}
    public.update(Card.parse(row["card"]).id for row in prefix.to_dict()["playActions"])
    own = {Card.parse(code).id for code in normalized.state["hands"]["north"]}
    checks = [
        (kind, {"team": team, "minimum": value, "maximum": value}, value == 0)
        for kind in ("state.current_raw_points", "state.current_game_points", "state.current_project_points", "state.match_score")
        for team in ("north_south", "west_east") for value in (0, 1)
    ]
    expected_actor = (
        decision["selector"]["owner"] if decision["kind"] == "project"
        else {"bid": "north", "escalation": "west"}[decision["kind"]]
    )
    checks.extend(("state.next_actor", {"seat": seat}, seat == expected_actor)
                  for seat in ("north", "west", "south", "east"))
    checks.append(("state.legal_action_count", {"minimum": 0, "maximum": 0}, False))
    if decision_kind.startswith("project_mandatory"):
        assert len(normalized.legal_actions) == 1
        checks.append(("state.legal_action_count", {"minimum": 1, "maximum": 1}, True))
    checks.extend(
        ("state.observer_card_visibility",
         {"observer": "north", "card": card, "visibility": visibility},
         (card in own | public) == (visibility == "visible"))
        for card in range(32) for visibility in ("visible", "hidden")
    )
    manifest = shared_scenario_predicate_catalog_lowering_manifest_v3()
    claims = []
    for kind, arguments, expected in checks:
        predicate = ResearchScenarioV1.build({
            "schemaVersion": 1, "origin": {"kind": "search"},
            "constraints": {"predicateId": "PRED-NONPLAY", "kind": kind,
                            "version": 1, "hardness": "hard", "arguments": arguments},
            "decisionPoint": decision,
            "outcome": {"predicateId": "PRED-REPORT", "kind": "outcome.team_round",
                        "version": 1, "hardness": "report",
                        "arguments": {"team": "north_south", "status": "won"}},
        }).constraints.predicates()[0]
        claims.append((manifest.lower_predicate(predicate, context), expected))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(context.model) == cp_model.OPTIMAL
    for (claim, expected), (kind, arguments, _) in zip(claims, checks, strict=True):
        assert solver.boolean_value(claim.literal) == expected, (kind, arguments)
        assert bool(claim.extract(solver)["satisfied"]) == expected, (kind, arguments)
    if project_context is not None:
        fragment = project_context.context.projects
        for project in fragment.projects:
            declared_seats = {row["actor"] for row in prefix.to_dict().get("projectActions", ())}
            if project.owner.name.lower() not in declared_seats:
                assert not solver.boolean_value(project.declared)
            assert not solver.boolean_value(project.comparison_winner)
            assert not solver.boolean_value(project.awarded)
        assert all(solver.value(value) == 0 for value in fragment.combined_awarded_value)


@pytest.fixture
def native_bid_input_context(rules):
    raw = _escalation_prefix().to_dict()
    raw.pop("sha256")
    raw["auctionActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15,
    )
    candidate = OrToolsUnifiedResearchExecutorV1._candidate_from_normalized_exact(
        SimpleNamespace(rules=rules), normalized, prefix,
    )
    bindings = DefaultStageContextCompilerV1(
        rules, required_stages=("state",), native_cp_sat_authority=True,
    )(candidate, None)
    return candidate, bindings


@pytest.mark.parametrize("operator,minimum,maximum,expected_miss", [
    ("all", None, None, True),
    ("any", None, None, False),
    ("not", None, None, False),
    ("count", 1, 1, False),
    ("count", 2, 2, True),
])
def test_native_input_projection_defers_future_truth(native_bid_input_context, operator, minimum, maximum, expected_miss):
    candidate, bindings = native_bid_input_context
    known_false = {"predicateId": "PRED-KNOWN", "kind": "state.next_actor", "version": 1,
                   "hardness": "hard", "arguments": {"seat": "west"}}
    future = {"predicateId": "PRED-FUTURE", "kind": "outcome.team_round", "version": 1,
              "hardness": "hard", "arguments": {"team": "north_south", "status": "won"}}
    expression = {"op": operator, "children": [known_false, future]}
    if operator == "not":
        known_true = {**known_false, "arguments": {"seat": "north"}}
        expression = {"op": "all", "children": [known_true, {"op": "not", "children": [future]}]}
    if minimum is not None:
        expression.update(minimum=minimum, maximum=maximum)
    scenario = ResearchScenarioV1.build({
        "schemaVersion": 1, "origin": {"kind": "search"}, "constraints": expression,
        "decisionPoint": {"kind": "bid"},
        "outcome": {**future, "predicateId": "PRED-REPORT", "hardness": "report"},
    })
    compiler = OrToolsResearchCompilerV1(native_cp_sat_authority=True)
    assert compiler.preflight(scenario).supported
    result = compiler.reject_input_projection(scenario, candidate, bindings, maximum_seconds=5)
    assert (result is not None) == expected_miss
    if result is not None:
        assert result.status == "target_miss" and not result.satisfied
    bad = dict(bindings)
    bad["state"] = OrToolsStageContextBindingV1("state", "f" * 64, bindings["state"].context)
    with pytest.raises(ValueError, match="does not own candidate evidence"):
        compiler.reject_input_projection(scenario, candidate, bad, maximum_seconds=5)


def test_native_input_projection_unknown_never_qualifies(native_bid_input_context, monkeypatch):
    candidate, bindings = native_bid_input_context
    scenario = ResearchScenarioV1.build({
        "schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"predicateId": "PRED-KNOWN", "kind": "state.next_actor", "version": 1,
                        "hardness": "hard", "arguments": {"seat": "north"}},
        "decisionPoint": {"kind": "bid"},
        "outcome": {"predicateId": "PRED-REPORT", "kind": "outcome.team_round", "version": 1,
                    "hardness": "report", "arguments": {"team": "north_south", "status": "won"}},
    })
    monkeypatch.setattr("baloot_platform.research.ortools_research_executor._solve_cp_sat_with_cancel_v1",
                        lambda *_args: cp_model.UNKNOWN)
    result = OrToolsResearchCompilerV1(native_cp_sat_authority=True).reject_input_projection(
        scenario, candidate, bindings, maximum_seconds=5,
    )
    assert result is None


def test_native_project_input_projection_defers_complete_play(root):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")
    raw = _optional_project_prefix().to_dict()
    raw.pop("sha256")
    raw.update(projectActions=[], playActions=[])
    prefix = ExactTracePrefixV1.build(raw)
    decision = {"kind": "project", "selector": {"owner": "north"}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point=decision, maximum_seconds=15,
    )
    candidate = OrToolsUnifiedResearchExecutorV1._candidate_from_normalized_exact(
        SimpleNamespace(rules=rules), normalized, prefix,
    )
    bindings = DefaultStageContextCompilerV1(
        rules, required_stages=("state",), native_cp_sat_authority=True,
    )(candidate, None)
    assert 7 not in bindings["playPrefix"].context.tricks
    scenario = ResearchScenarioV1.build({
        "schemaVersion": 1, "origin": {"kind": "search"}, "decisionPoint": decision,
        "constraints": {"op": "all", "children": [
            {"predicateId": "PRED-KNOWN", "kind": "state.next_actor", "version": 1,
             "hardness": "hard", "arguments": {"seat": "north"}},
            {"predicateId": "PRED-FUTURE-PLAY", "kind": "play.trick_card_count", "version": 1,
             "hardness": "hard", "arguments": {"trickIndex": 7, "selector": {"ranks": ["ace"]},
                                                 "minimum": 0, "maximum": 4}},
        ]},
        "outcome": {"predicateId": "PRED-REPORT", "kind": "outcome.team_round", "version": 1,
                    "hardness": "report", "arguments": {"team": "north_south", "status": "won"}},
    })
    compiler = OrToolsResearchCompilerV1(native_cp_sat_authority=True)
    assert compiler.preflight(scenario).supported
    assert compiler.reject_input_projection(scenario, candidate, bindings, maximum_seconds=5) is None
