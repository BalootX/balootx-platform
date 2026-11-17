from __future__ import annotations

import pytest
from ortools.sat.python import cp_model

import baloot_platform.research.ortools_tail_first_candidate as candidate_module
from baloot_platform.research.ortools_play_tail import OrToolsPlayTailStageV1
from baloot_platform.research.scenario_contracts import ResearchScenarioV1, canonical_sha256
from baloot_platform.research.tail_first_stage_planner import TailFirstStagePlannerV1
from test_research_ortools_flagship_search_recipe import (
    _exact_unique_winner_proof,
    _leaf,
    _nearest_feasible_changed_scenario,
)


@pytest.mark.real_clock
@pytest.mark.parametrize("case", [
    "invariant", "nontrivial", "not", "any", "count", "unknown", "cancelled",
    "winner", "action",
])
def test_early_count_only_omits_native_proven_required_invariant(rules, monkeypatch, case):
    raw = _nearest_feasible_changed_scenario().to_dict()
    raw.pop("sha256")
    raw["constraints"].pop("sha256")
    early = _leaf("PRED-EARLY-CARD-COUNT", "play.trick_card_count", {
        "trickIndex": 0,
        "selector": {"ranks": [
            "seven", "eight", "nine", "jack", "queen", "king", "ten", "ace",
        ]},
        "minimum": 4, "maximum": 4,
    })
    if case == "nontrivial":
        early["arguments"]["selector"] = {"ranks": ["ace"]}
    elif case in {"winner", "action"}:
        early = _leaf("PRED-EARLY-CARD-COUNT", (
            "play.trick_winner_matches" if case == "winner" else "play.action_matches"
        ), {"trickIndex": 0, "selector": {"ranks": ["ace"]}, **(
            {"team": "north_south"} if case == "winner" else
            {"position": 0, "actingSeat": "north"}
        )})
    expression = early
    if case == "not":
        expression = {"op": "not", "children": [early]}
    elif case in {"any", "count"}:
        expression = {"op": case, "children": [early, _leaf(
            "PRED-ALTERNATIVE", "state.phase", {"state": "playing"},
        )], **({"minimum": 1, "maximum": 1} if case == "count" else {})}
    raw["constraints"]["children"].append(expression)
    scenario = ResearchScenarioV1.build(raw)
    before = scenario.to_dict()
    authored = next(predicate for predicate in scenario.predicates()
                    if predicate.predicate_id == early["predicateId"])
    plan = TailFirstStagePlannerV1().plan(scenario)
    assert plan.tail_length == 12

    invariant_models = []
    statuses = []
    plans = {}
    cancelled = False
    solve = candidate_module._solve_cp_sat_with_cancel_v1
    hash_value = candidate_module.canonical_sha256

    def cancel_check():
        nonlocal cancelled
        result, cancelled = cancelled, False
        return result

    def invariant_solve(solver, model, cancel_check):
        nonlocal cancelled
        invariant_models.append(model)
        assert solver.parameters.num_search_workers == 1
        assert solver.parameters.max_number_of_conflicts == 250_000
        assert 0 < solver.parameters.max_time_in_seconds <= 15
        if case == "unknown":
            return cp_model.UNKNOWN
        status = solve(solver, model, cancel_check)
        statuses.append(status)
        cancelled = case == "cancelled"
        return status

    def record_plan(document):
        result = hash_value(document)
        if "stagePlanHash" in document and "predicateIds" in document:
            plans[result] = document
        return result

    class Inspected(Exception):
        pass

    def inspect_tail(_stage, **kwargs):
        assert scenario.to_dict() == before
        assert authored in scenario.predicates()
        construction = plans[kwargs["construction_plan_hash"]]
        if case == "invariant":
            assert kwargs.get("selected_play_count") == 12
            assert "play_step_count" not in kwargs
            assert authored.predicate_id not in construction["predicateIds"]
            assert statuses == [cp_model.INFEASIBLE]
            invariant, = construction["nativeConstructionInvariants"]
            assert invariant["predicateId"] == authored.predicate_id
            assert invariant["predicateHash"] == authored.sha256
            assert invariant["solverStatus"] == "infeasible"
            assert invariant["modelHash"] == canonical_sha256({
                "proto": str(invariant_models[0].proto),
            })
            assert invariant["sha256"] == canonical_sha256({
                key: value for key, value in invariant.items() if key != "sha256"
            })
        else:
            assert kwargs.get("play_step_count") == 32
            assert "selected_play_count" not in kwargs
            assert authored.predicate_id in construction["predicateIds"]
            assert "nativeConstructionInvariants" not in construction
        if case in {"not", "any", "count", "winner", "action"}:
            assert not invariant_models
        elif case == "nontrivial":
            assert statuses == [cp_model.OPTIMAL]
        raise Inspected

    monkeypatch.setattr(candidate_module, "_solve_cp_sat_with_cancel_v1", invariant_solve)
    monkeypatch.setattr(candidate_module, "canonical_sha256", record_plan)
    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", inspect_tail)
    compiler = candidate_module.OrToolsTailFirstCandidateCompilerV1(
        rules=rules, plan=plan, proof_request=_exact_unique_winner_proof(),
        maximum_proof_nodes=100_000, maximum_proof_depth=12,
        maximum_line_count_limbs=64, maximum_conflicts=250_000,
        maximum_prefix_lift_attempts=1, native_cp_sat_scoring=True,
        cancel_check=cancel_check,
    )
    with pytest.raises(Inspected):
        compiler(scenario, (), seed=0, maximum_seconds=15, maximum_conflicts=250_000)
