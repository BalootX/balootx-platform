from dataclasses import replace
from types import SimpleNamespace
import time

import pytest
import yaml

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat, Team
from baloot_platform.research import ortools_information_set_compiler as module
from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, ScenarioProofRequestV1


@pytest.mark.real_clock
@pytest.mark.parametrize("case,admissible,expected_action,expected_calls,reason", [
    ("legacy", None, "SA", [None], None),
    ("best_admissible", ("SA",), "SA", [None], None),
    ("equal_best", ("D7", "C7"), "D7", [None, None], None),
    ("strictly_worse", ("C7",), None, [None, None], "no_optimal_witness_action"),
    ("empty", (), None, [None], "no_optimal_witness_action"),
    ("query_limit", ("D7", "C7"), None, [None], "maximum_queries"),
    ("cancelled", ("D7",), None, [None, None], "cancelled"),
    ("time_limit", ("D7",), None, [None, None], "maximum_seconds"),
    ("missing_best_root", ("D7",), None, [None], "native_root_policy_missing"),
    ("audit", None, "C7", [None, "C7"], None),
    ("audit_hint", ("SA",), None, [], "invalid"),
    ("illegal", ("H7",), None, [], "invalid"),
    ("duplicate", ("C7", "C7"), None, [], "invalid"),
    ("not_tuple", ["C7"], None, [], "invalid"),
])
def test_witness_selection_preserves_unrestricted_best_and_shared_budget(
    root, monkeypatch, case, admissible, expected_action, expected_calls, reason,
):
    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]["proofBudgets"]["profiles"]["bounded_v1"]
    budget = ScenarioProofBudgetV1.build("bounded_v1", profile)
    native = replace(budget.native_observed, maximum_queries=2 if case == "query_limit" else 10,
                     maximum_iterations=20)
    budget = replace(budget, native_observed=native)
    request_raw = dict(schemaVersion=1, mode="bounded", visibility="actor_information_set",
        strategyClass="pure_deterministic",
        claim="winning_root_action_count", targetTeam="north_south", utility="awarded_point_margin",
        budgetProfile="bounded_v1", horizon={"unit": "plies", "value": 1},
        boundaryEvaluator="current_raw_point_margin_v1")
    if case in {"audit", "audit_hint"}:
        request_raw["selectedAction"] = "C7"
    request = ScenarioProofRequestV1.build(request_raw)
    actions = tuple(SimpleNamespace(actor=Seat.NORTH, card=Card.parse(card), ekka=False)
                    for card in ("SA", "C7", "D7"))
    worlds = (("world-1", SimpleNamespace(start=SimpleNamespace(expected_legal_actions=actions)), None),
              ("world-2", SimpleNamespace(start=SimpleNamespace(expected_legal_actions=actions)), None))
    exact = SimpleNamespace(state_hash="state", exact_state_id="exact-state", next_actor="north",
                            require_integrity=lambda: None)
    compiler = object.__new__(module.OrToolsActorInformationSetCompilerV1)
    compiler.rules = SimpleNamespace(sha256="rules")
    clock, calls = [0.0], []
    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: clock[0], process_time=lambda: 0.0))

    def prepare(_exact, _request, **kwargs):
        assert _exact is exact and _request is request
        assert "admissible_root_actions" not in kwargs
        return worlds, {"worldCount": 2}

    def solve(actual_worlds, target, **kwargs):
        assert actual_worlds is worlds and target == Team.NORTH_SOUTH
        allowed = kwargs.get("admissible_root_actions")
        if allowed is not None:
            assert allowed == tuple(sorted(admissible))
        index = len(calls)
        assert kwargs["native_limits"].maximum_queries == native.maximum_queries - 2 * index
        assert kwargs["native_limits"].maximum_iterations == native.maximum_iterations - 3 * index
        assert kwargs["maximum_seconds"] == pytest.approx(seconds - clock[0])
        action = kwargs["selected_root_action"]
        calls.append(action)
        clock[0] += 0.25
        chosen = ("D7" if "D7" in allowed else allowed[0]) if allowed else (
            action or ("H7" if case == "missing_best_root" else "SA"))
        proof = {"scope": {"utilityScope": {"boundaryEvaluator": "current_raw_point_margin_v1"},
                           "rootActor": {"observationHash": "observation"}, "selectedRootAction": action,
                           **({"admissibleRootActions": list(allowed)} if allowed else {})},
                 "policy": {"observation": chosen}, "value": 1 if chosen == "C7" else 10,
                 "proofMethod": "counterexample_guided_native_v1",
                 "resources": {"usage": {"counterexampleQueries": 2, "masterIterations": 3}}}
        return {**proof, "sha256": module.canonical_sha256(proof)}

    monkeypatch.setattr(compiler, "_prepare_observed_play_worlds", prepare)
    monkeypatch.setattr(compiler, "_solve_observed_worlds", solve)
    seconds = 0.4 if case == "time_limit" else 4.0
    context = dict(budget=budget, maximum_seconds=seconds,
                   cancel_check=lambda: case == "cancelled" and len(calls) >= 2)

    def run():
        return compiler._compile_observed_play_selection(exact, request,
            admissible_root_actions=admissible, **context)

    if reason == "invalid":
        with pytest.raises(ValueError, match="witness action hints"):
            run()
    elif reason:
        with pytest.raises(module.InformationSetCompilationStopped) as error:
            run()
        assert error.value.reason == reason
        cause = error.value.partial_receipt["evidence"]["causeReceipt"]
        assert cause["stage"] == "native_observed_selected_action"
        if reason == "no_optimal_witness_action":
            evidence = cause["evidence"]
            assert evidence["admissibleRootActions"] == sorted(admissible)
            assert len(evidence["proofs"]) == len(expected_calls)
            assert evidence["proofs"][evidence["bestProofHash"]]["value"] == 10
    else:
        receipt = run()
        artifact = receipt["rootArtifact"]
        assert artifact["selectedAction"] == expected_action
        if admissible is not None:
            assert artifact["admissibleRootMethod"] == "joint_native_policy_domain_v1"
        assert receipt["actionQuality"]["bestValue"] == 10
        assert receipt["actionQuality"]["actualValue"] == (1 if case == "audit" else 10)
        assert receipt["actionQuality"]["actualActionOptimal"] is (case != "audit")
        assert artifact["usage"] == {"counterexampleQueries": 2 * len(expected_calls),
                                     "masterIterations": 3 * len(expected_calls)}
        for document in (receipt, artifact):
            if admissible is None:
                assert "admissibleRootActions" not in document
            else:
                assert document["admissibleRootActions"] == sorted(admissible)
            assert document["sha256"] == module.canonical_sha256({
                key: value for key, value in document.items() if key != "sha256"})
        assert calls == expected_calls
        calls.clear()
        clock[0] = 0.0
        assert compiler._verify_observed_play_selection(receipt, exact, request, **context) == receipt
    assert calls == expected_calls


@pytest.mark.real_clock
@pytest.mark.parametrize("allowed,expected", [(None, 10), (("C7", "D7"), 10), (("C7",), 1)])
def test_grouped_root_domain_keeps_full_legal_catalog_and_cuts(allowed, expected):
    observation = "root"
    actions = ("C7", "D7", "SA")
    catalog = module._Catalog({("target", observation): actions}, 1, (0, 10))
    cuts = tuple(module._CounterexampleCut(value, ((observation, action),))
                 for action, value in (("SA", 10), ("C7", 1), ("D7", 10)))
    proof = module.OrToolsActorInformationSetProofV1()
    options = dict(allowed_actions={observation: allowed}) if allowed is not None else {}
    result = proof._solve_master("scope", "policy_value", catalog, cuts,
        time.monotonic() + 2, None, canonicalize_policy=False, **options)
    assert result.status == "optimal" and result.value == expected
    assert dict(result.policy)[observation] in (allowed or actions)
    assert catalog.domains[("target", observation)] == actions
    assert len(cuts) == 3 and any(cut.target_actions == ((observation, "SA"),) for cut in cuts)
    baseline = proof._solve_master("scope", "policy_value", catalog, cuts,
        time.monotonic() + 2, None, canonicalize_policy=False)
    assert (result.model_hash == baseline.model_hash) is (allowed is None)
