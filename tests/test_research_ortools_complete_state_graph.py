from __future__ import annotations

import ast
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ortools")

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.research import ortools_complete_state_graph
from baloot_platform.research.ortools_complete_state_graph import (
    CompleteStateGraphSeedV1,
    INTERNAL_RAW_POINT_MARGIN_UTILITY_V1,
    OrToolsCompleteStateGraphCompilerV1,
    _transposition_core_v1,
)
from baloot_platform.research.ortools_exact_state import (
    OrToolsExactStateNormalizerV1,
    OrToolsExactNormalizationReceiptV1,
    OrToolsNormalizedExactStateV1,
)
from baloot_platform.research.ortools_play_tail import (
    OrToolsPlayTailStageV1,
    TailPrefixActionV1,
)
from baloot_platform.research.ortools_proof import OrToolsScenarioGraphProofV1
from baloot_platform.research.ortools_tail_first_candidate import (
    _raw_root_state_key_v1,
    _raw_root_uniqueness_key_v1,
    _seed_with_target_raw_v1,
)
from baloot_platform.research.scenario_contracts import (
    ExactTracePrefixV1,
    ScenarioProofRequestV1,
    canonical_json,
    canonical_sha256,
)


@pytest.mark.real_clock
@pytest.mark.parametrize("disclosure,stop", [(kind, None) for kind in ("baloot", "projects", "both", "neither")]
    + [("both", "cancelled"), ("both", "maximum_seconds")])
def test_observed_policy_reuses_shared_native_prefix_projection(rules, disclosure, stop, monkeypatch):
    from baloot_platform.core.project_ir import ProjectDeclarationActionV1
    from baloot_platform.research import ortools_information_set_compiler as module

    plays = tuple(TailPrefixActionV1.build(seat, card) for seat, card in (
        ("north", "D7"), ("west", "H7"), ("south", "S7")))
    checkpoint = SimpleNamespace(next_actor=Seat.SOUTH, legal_play_actions=(plays[-1],),
        remaining_hands={}, current_trick=(), trick_winners=(), trick_raw_points=(),
        project_checkpoint=None, native_play=None)
    calls = []
    clock = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def solve(path, **kwargs):
        calls.append(path)
        if stop == "maximum_seconds":
            clock[0] = 10.0
        return checkpoint, plays[:2]

    branch = tuple(ProjectDeclarationActionV1(seat, "pass") for seat in (Seat.SOUTH, Seat.EAST))
    oracle = SimpleNamespace(base_length=1, project_continuation_actions=branch,
        public_baloot_binding=object() if disclosure in {"baloot", "both"} else None,
        project_disclosure_bindings=(object(),) if disclosure in {"projects", "both"} else (),
        solve=solve, _project=lambda *_: checkpoint)
    witness = SimpleNamespace(plays=plays, at_prefix=lambda _: checkpoint)
    context = (object(), SimpleNamespace(project_actions=(None, None)),
        {"sequences": [[action.to_dict() for action in branch]]})
    compiler = module.OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    arguments = dict(known_domains={}, project_context=context, maximum_seconds=10,
        cancel_check=lambda: bool(calls) and stop == "cancelled",
        _project_observer=lambda _: "project-observation")
    if stop is not None:
        with pytest.raises(module.PlayTailSolveStopped) as stopped:
            compiler._observed_policy_bindings({}, oracle, witness, Team.NORTH_SOUTH, {}, {}, **arguments)
        assert stopped.value.solver_status == stop
        assert calls == [(plays[1],)]
        return
    bindings, domains, witnessed = compiler._observed_policy_bindings({}, oracle, witness,
        Team.NORTH_SOUTH, {}, {}, **arguments)
    assert calls == [(plays[1],)]
    assert bindings == {} and set(domains.values()) == {("pass",), ("S7",)}
    assert {action for _, action in witnessed} == {"pass", "S7"}


@pytest.mark.parametrize("conflict", ["project", "terminal", "boundary", None])
def test_batch_checks_evidence_before_deduplicating_paths(monkeypatch, conflict):
    module = ortools_complete_state_graph
    model = module.cp_model.CpModel()
    auxiliary = model.new_bool_var("project_evidence")
    oracle = object.__new__(module._CpSatPathOracle)
    oracle.public_baloot_binding = None
    oracle.build = SimpleNamespace(model=model,
        _extract=lambda *_: SimpleNamespace(plays=(), at_prefix=lambda _: None))
    oracle._project = lambda *_: None
    oracle.base_length = 0
    oracle.start = SimpleNamespace(cards=(Card(0),))
    oracle.model_hash = "fixture"
    oracle.project_checkpoints = {0: object()} if conflict == "project" else {}
    oracle.terminal_utility_variable = auxiliary if conflict == "terminal" else None
    oracle.boundary_utilities = (auxiliary,) if conflict == "boundary" else ()
    monkeypatch.setattr(module, "_extract_native_project_prefix_projection",
        lambda **kwargs: ((), kwargs["solver"].value(auxiliary)))
    if conflict is None:
        assert oracle.enumerate_paths(maximum_nodes=10, maximum_seconds=5, cancelled=None) is None
        assert oracle.results == {(): None}
    else:
        with pytest.raises(ValueError, match="inconsistent .* evidence"):
            oracle.enumerate_paths(maximum_nodes=10, maximum_seconds=5, cancelled=None)


def test_batch_checks_response_once_and_reads_every_solution(monkeypatch):
    module = ortools_complete_state_graph
    model = module.cp_model.CpModel()
    flag = model.new_bool_var("varying_solution")
    response_checks = []
    original = module.cp_model.CpSolverSolutionCallback.has_response

    def has_response(callback):
        response_checks.append(True)
        return original(callback)

    monkeypatch.setattr(module.cp_model.CpSolverSolutionCallback, "has_response", has_response)
    observed = []

    def extract(callback, *_):
        value = callback.value(flag)
        for _ in range(4):
            assert callback.value(flag + 3) == value + 3
            assert callback.boolean_value(flag) == bool(value)
            assert callback.boolean_value(~flag) == (not value)
        observed.append(value)
        return SimpleNamespace(plays=(), at_prefix=lambda _: None)

    oracle = object.__new__(module._CpSatPathOracle)
    oracle.public_baloot_binding = None
    oracle.build = SimpleNamespace(model=model, _extract=extract)
    oracle._project = lambda *_: None
    oracle.base_length = 0
    oracle.start = SimpleNamespace(cards=(Card(0),))
    oracle.model_hash = "fixture"
    oracle.project_checkpoints = {}
    oracle.terminal_utility_variable = None
    oracle.boundary_utilities = ()
    assert oracle.enumerate_paths(maximum_nodes=10, maximum_seconds=5, cancelled=None) is None
    assert sorted(observed) == [0, 1]
    assert len(response_checks) == len(observed)


@pytest.mark.real_clock
@pytest.mark.parametrize("reason", ["cancelled", "maximum_seconds"])
def test_batch_projection_respects_stop_before_publishing_results(monkeypatch, reason):
    module = ortools_complete_state_graph
    clock = [100.0]
    stopped = [False]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    oracle = object.__new__(module._CpSatPathOracle)
    oracle.public_baloot_binding = None
    model = module.cp_model.CpModel()
    model.new_constant(0)
    oracle.build = SimpleNamespace(model=model,
        _extract=lambda *_: SimpleNamespace(plays=(), at_prefix=lambda _: None))
    def project(*_):
        if reason == "cancelled":
            stopped[0] = True
        else:
            clock[0] += 6
        return None
    oracle._project = project
    oracle.base_length = 0
    oracle.start = SimpleNamespace(cards=(Card(0),))
    oracle.model_hash = "fixture"
    oracle.project_checkpoints = {}
    oracle.terminal_utility_variable = None
    oracle.boundary_utilities = ()
    oracle.results = None
    assert oracle.enumerate_paths(maximum_nodes=10, maximum_seconds=5,
        cancelled=lambda: stopped[0]) == reason
    assert oracle.results is None


@pytest.mark.real_clock
@pytest.mark.parametrize("setup_seconds", [2.0, 5.0, 6.0])
def test_batch_solver_budget_excludes_elapsed_setup(monkeypatch, setup_seconds):
    module = ortools_complete_state_graph
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    model = module.cp_model.CpModel()
    model.new_constant(0)
    clone = model.clone
    def slow_clone():
        clock[0] += setup_seconds
        return clone()
    monkeypatch.setattr(model, "clone", slow_clone)
    oracle = object.__new__(module._CpSatPathOracle)
    oracle.public_baloot_binding = None
    oracle.build = SimpleNamespace(model=model)
    oracle.base_length = 0
    oracle.start = SimpleNamespace(cards=(Card(0),))
    seen = []
    def solve(solver, *_):
        seen.append(solver.parameters.max_time_in_seconds)
        return module.cp_model.UNKNOWN
    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", solve)
    assert oracle.enumerate_paths(maximum_nodes=10, maximum_seconds=5,
        cancelled=None) == "maximum_seconds"
    assert seen == ([3.0] if setup_seconds == 2.0 else [])
    assert oracle.results is None


def _request(
    *,
    mode: str = "exact",
    threshold: int = 0,
    claim: str = "outcome_reachable",
    selected_action: str | None = None,
    line_universe: str | None = None,
) -> ScenarioProofRequestV1:
    value = {
        "schemaVersion": 1,
        "mode": mode,
        "visibility": "complete_state",
        "claim": claim,
        "targetTeam": "north_south",
        "utility": "awarded_point_margin",
        "successThreshold": threshold,
        "budgetProfile": "test",
    }
    if selected_action is not None:
        value["selectedAction"] = selected_action
    if line_universe is not None:
        value["lineUniverse"] = line_universe
    if mode == "bounded":
        value["horizon"] = {"unit": "plies", "value": 1}
        value["boundaryEvaluator"] = "current_raw_point_margin_v1"
    return ScenarioProofRequestV1.build(value)


def _normalized_state(rules) -> OrToolsNormalizedExactStateV1:
    hands = {
        "north": [Card.parse("H7").to_dict(), Card.parse("C7").to_dict()],
        "west": [Card.parse("S8").to_dict(), Card.parse("HA").to_dict()],
        "south": [Card.parse("S7").to_dict(), Card.parse("SA").to_dict()],
        "east": [Card.parse("S9").to_dict(), Card.parse("C8").to_dict()],
    }
    completed = [
        {"winner": "north", "rawPoints": 20},
        {"winner": "west", "rawPoints": 18},
        {"winner": "south", "rawPoints": 30},
        {"winner": "east", "rawPoints": 30},
        {"winner": "north", "rawPoints": 0},
        {"winner": "west", "rawPoints": 0},
    ]
    state = {
        "rulesHash": rules.sha256,
        "contract": "sun",
        "trumpSuit": None,
        "buyer": "west",
        "escalation": "normal",
        "escalationHistory": [],
        "locked": False,
        "matchScore": [0, 0],
        "hands": hands,
        "currentPlayer": "south",
        "currentTrick": {
            "index": 6,
            "leader": "south",
            "plays": [],
            "winner": None,
            "rawPoints": 0,
        },
        "completedTricks": completed,
        "projects": [],
        "projectWinningTeam": None,
        "balootTeams": [],
    }
    state_hash = canonical_sha256(state)
    legal = tuple(
        {
            "actionId": card.id,
            "code": card.code,
            "card": card.to_dict(),
            "ekka": False,
        }
        for card in sorted(
            (Card.parse("S7"), Card.parse("SA")),
            key=lambda item: item.id,
        )
    )
    identity = canonical_sha256({"stateHash": state_hash, "rulesHash": rules.sha256})
    return OrToolsNormalizedExactStateV1(
        f"STATE-{identity[:20].upper()}",
        "0" * 64,
        rules.sha256,
        state_hash,
        "playing",
        "play",
        "south",
        legal,
        state,
        0,
        0,
        "1" * 64,
        identity,
        OrToolsExactNormalizationReceiptV1({"sha256": "2" * 64}),
    )


def _compiler(rules, *, native_conversion=False) -> OrToolsCompleteStateGraphCompilerV1:
    return OrToolsCompleteStateGraphCompilerV1(
        play_domain=rules.play_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        native_conversion=native_conversion,
        structural_domain=rules.structural_domain,
        project_domain=rules.project_domain,
    )


@pytest.mark.parametrize("audited_action,project_points", [("SJ", 5), ("S9", 7)])
@pytest.mark.parametrize("tie_rule", ["buyer_makes", "buyer_loses"])
@pytest.mark.parametrize("target", list(Team))
@pytest.mark.parametrize("utility,utility_encoding", [
    ("round_win", None), ("contract_success", None),
    ("contract_success", "signed_win_loss_v1"),
])
def test_round_win_companion_uses_scoring_winner_not_zero_margin(
    rules, audited_action, project_points, tie_rule, target, utility, utility_encoding,
):
    from ortools.sat.python import cp_model
    from baloot_platform.core.models import Project, RoundState, Trick
    from baloot_platform.core.project_ir import ProjectDomainIRV1
    from baloot_platform.core.scoring import score_round
    from baloot_platform.research.proof_service import _terminal_utility
    from baloot_platform.research.result_authority import native_round_win_quality_request_v1
    from baloot_platform.research.ortools_research_executor import _policy_terminal_utility_v1

    data = copy.deepcopy(rules.data)
    data["scoring"]["contract"]["buyer_tie_outcome"] = tie_rule
    scoring = ScoringDomainIRV1.from_rules(data)
    model = cp_model.CpModel()
    oracle = object.__new__(ortools_complete_state_graph._CpSatPathOracle)
    oracle.build = SimpleNamespace(model=model, request_hash="0" * 64,
        winner_team_variables=(model.new_constant(int(Team.WEST_EAST)),),
        trick_point_variables=(model.new_constant(0),))
    oracle.start = SimpleNamespace(contract=Contract.SUN, buyer=Seat.NORTH,
        escalation="normal", last_escalating_team=None, base_raw=(65, 55),
        base_tricks=(4, 3), project_points=(project_points, project_points),
        baloot_points=(0, 0), starting_match_score=(0, 0),
        terminal_project_domain=None, exact_state_id=f"STATE-TIE-{audited_action}",
        proof_state_projection_hash="1" * 64)
    oracle.project_lifecycle = None
    oracle.play_stage = SimpleNamespace(native_conversion=True)
    fragment, value = oracle._attach_terminal_scoring(
        scoring_domain=scoring, target=target, utility=utility, utility_encoding=utility_encoding)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL

    state = RoundState(rules.sha256, 1, 1, Seat.EAST)
    state.contract = Contract.SUN
    state.buyer = state.declarer = Seat.NORTH
    state.completed_tricks = [Trick(i, seat, winner=seat, raw_points=points)
        for i, (seat, points) in enumerate([
            (Seat.NORTH, 65), (Seat.WEST, 55), (Seat.NORTH, 0), (Seat.WEST, 0),
            (Seat.NORTH, 0), (Seat.WEST, 0), (Seat.NORTH, 0), (Seat.WEST, 0)])]
    state.projects = [Project(seat, "test", (), project_points, 0, 0, scoring=True)
        for seat in (Seat.NORTH, Seat.WEST)]
    state.round_score = score_round(state, scoring, ProjectDomainIRV1.from_rules(data))
    request = ScenarioProofRequestV1.build({**_request().to_dict(), "sha256": "",
        "targetTeam": target.name.lower()})
    companion_request = native_round_win_quality_request_v1(request)
    expected = 1 if state.round_score.winner == target else -1
    signed_request = ScenarioProofRequestV1.build({**request.to_dict(), "sha256": "",
        "utility": "contract_success", "utilityEncoding": "signed_win_loss_v1"})
    assert solver.value(fragment.round_winner) == int(state.round_score.winner)
    assert _terminal_utility(state, companion_request) == expected
    assert _terminal_utility(state, signed_request) == expected
    assert signed_request.to_dict()["utilityEncoding"] == "signed_win_loss_v1"
    legacy_contract_request = ScenarioProofRequestV1.build({**request.to_dict(), "sha256": "",
        "utility": "contract_success"})
    assert "utilityEncoding" not in legacy_contract_request.to_dict()
    assert _terminal_utility(state, legacy_contract_request) == (1 if expected == 1 else 0)
    selected_request = (companion_request if utility == "round_win" else
        signed_request if utility_encoding is not None else legacy_contract_request)
    assert solver.value(value) == _terminal_utility(state, selected_request)
    assert _policy_terminal_utility_v1(
        {"roundWinner": state.round_score.winner.name.lower()}, selected_request
    ) == solver.value(value)
    legacy_core = legacy_contract_request.to_dict()
    legacy_core.pop("sha256")
    assert legacy_contract_request.sha256 == canonical_sha256(legacy_core)
    assert signed_request.sha256 != legacy_contract_request.sha256
    assert companion_request.sha256 != request.sha256
    assert request.utility == "awarded_point_margin" and request.success_threshold == 0
    if tie_rule == "buyer_makes":
        assert state.round_score.awarded_points == [13 + project_points] * 2
        assert _terminal_utility(state, request) == 0
        assert (solver.value(value) == 1) is (target == Seat.NORTH.team)
        if target == Team.WEST_EAST and utility == "round_win":
            from baloot_platform.research.ortools_information_set_proof import (
                FiniteInformationNodeV1, FiniteInformationSetProblemV1,
                FiniteInformationWorldV1, OrToolsActorInformationSetProofV1,
            )
            game_problem = FiniteInformationSetProblemV1.build(problem_id=f"audited-{audited_action}-winner",
                success_threshold=0, worlds=(FiniteInformationWorldV1.build("actual", {
                    "best": FiniteInformationNodeV1.terminal(1),
                    audited_action: FiniteInformationNodeV1.terminal(solver.value(value)),
                }),))
            game_proof = OrToolsActorInformationSetProofV1().prove(game_problem)
            assert game_proof.winning_root_action_count == 1
            assert next(row for row in game_proof.roots if row.root_action == audited_action).satisfies is False


def test_basic_policy_companion_is_observer_safe_and_distinct_from_exact_proof(rules):
    from baloot_platform.research.ortools_research_executor import _observer_safe_play_policy_artifact_v1
    from baloot_platform.research.result_authority import native_basic_policy_quality_request_v1
    request = native_basic_policy_quality_request_v1(_request())
    state = _normalized_state(rules)
    _, first = _observer_safe_play_policy_artifact_v1(rules, state, request,
        actual_action=None, maximum_seconds=2, cancel_check=None)
    hidden_changed = copy.deepcopy(state.state)
    hidden_changed["hands"]["north"], hidden_changed["hands"]["west"] = (
        hidden_changed["hands"]["west"], hidden_changed["hands"]["north"])
    _, second = _observer_safe_play_policy_artifact_v1(rules,
        replace(state, state=hidden_changed), request,
        actual_action=None, maximum_seconds=2, cancel_check=None)
    assert first == second
    assert first["proof"]["completionStatus"] == "bounded_complete"
    assert first["proofModel"]["actionQuality"]["policyId"] == "observer_safe_trick_control_v1"
    assert request.sha256 != _request().sha256


@pytest.mark.parametrize("tie_rule", ["buyer_makes", "buyer_loses"])
def test_tie_comparison_freezes_only_the_tie_leaf_without_mutating_source(rules, tie_rule):
    from baloot_platform.research.result_authority import native_tie_rule_profile_v1
    original = copy.deepcopy(rules.data)
    variant = native_tie_rule_profile_v1(rules, tie_rule)
    expected = copy.deepcopy(original)
    expected["scoring"]["contract"]["buyer_tie_outcome"] = tie_rule
    assert rules.data == original
    assert variant.data == expected
    assert variant.sha256 == canonical_sha256(expected)


def _tie_allocation_artifact():
    return {"proof": {"hiddenWorlds": {"count": 2, "complete": True}}, "allocationModel": {
        "observer": "west", "pool": [0, 1], "handSizes": {"north": 1, "south": 1},
        "fixedOwnership": {}, "acceptedWorlds": 2, "terminalSolverStatus": "infeasible",
        "reachability": "deal_auction_escalation_project_play_cp_sat_v1",
        "auctionDomainIrHash": "a" * 64, "escalationDomainIrHash": "b" * 64}}


def test_tie_universe_binding_ignores_rule_bound_state_ids_but_not_card_ownership():
    from baloot_platform.research.result_authority import native_tie_allocation_universe_hash_v1
    source = _tie_allocation_artifact()
    variant = copy.deepcopy(source)
    variant["allocationModel"]["exactStateId"] = "STATE-OTHER-RULE"
    assert native_tie_allocation_universe_hash_v1(source) == native_tie_allocation_universe_hash_v1(variant)
    variant["allocationModel"]["fixedOwnership"] = {"0": "north"}
    assert native_tie_allocation_universe_hash_v1(source) != native_tie_allocation_universe_hash_v1(variant)
    variant["proof"]["hiddenWorlds"]["complete"] = False
    with pytest.raises(ValueError, match="exhaustive"):
        native_tie_allocation_universe_hash_v1(variant)


@pytest.mark.parametrize("tamper", ["source", "prefix", "baseline", "snapshot", "worlds"])
def test_tie_companion_readback_rejects_rehashed_substitutions(rules, tamper):
    from baloot_platform.research.result_authority import (
        _validate_tie_rule_quality_companions_v1, native_tie_rule_profile_v1,
        native_tie_allocation_universe_hash_v1,
    )
    source = _tie_allocation_artifact()
    source["request"] = _request().to_dict()
    digest = canonical_sha256(source)
    prefix_hash = "c" * 64
    comparison = {"sourceArtifactHash": digest, "sourcePrefixHash": prefix_hash, "baselineAction": "SJ"}
    for tie_rule in ("buyer_makes", "buyer_loses"):
        variant = native_tie_rule_profile_v1(rules, tie_rule)
        comparison[tie_rule] = {"margin": copy.deepcopy(source), "roundWin": copy.deepcopy(source),
            "rulesSnapshot": variant.data, "rulesHash": variant.sha256, "exactStateId": "STATE-TEST",
            "allocationUniverseHash": native_tie_allocation_universe_hash_v1(source)}
    if tamper in {"source", "prefix", "baseline"}:
        comparison[{"source": "sourceArtifactHash", "prefix": "sourcePrefixHash", "baseline": "baselineAction"}[tamper]] = "different"
    elif tamper == "snapshot":
        comparison["buyer_makes"]["rulesSnapshot"]["profile"]["id"] = "unrelated-profile"
    else:
        comparison["buyer_makes"]["allocationUniverseHash"] = "d" * 64
    comparison["sha256"] = canonical_sha256(comparison)
    evidence = {"artifacts": {digest: source}, "decisions": [{"proofArtifactHash": digest,
        "sourcePrefixHash": prefix_hash, "decisionKind": "play"}], "tieRuleEvidence": {digest: comparison},
        "baselineEvidence": {digest: {"proof": {"selectedAction": "SJ"}}}}
    with pytest.raises(ValueError, match="tie-rule companion"):
        _validate_tie_rule_quality_companions_v1(evidence, rules=rules,
            execution_budget=SimpleNamespace(video_interestingness={"schemaVersion": 4}),
            remaining=lambda: 1, cancel_check=None, private_recall=None)


@pytest.mark.real_clock
@pytest.mark.parametrize("baseline_policy_id", [
    "observer_safe_trick_control_v1",
    "observer_safe_partner_feed_established_control_v1",
])
def test_tie_companions_real_native_prefix_producer_and_readback(rules, root, baseline_policy_id, monkeypatch):
    """Both frozen scoring variants certify the same legal hidden-card universe."""
    import time
    import yaml
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.decision_selection import evaluate_round_interestingness_v1
    from baloot_platform.research.ortools_research_executor import (
        _editorial_play_quality_artifact_v1, _observer_safe_play_policy_artifact_v1,
    )
    from baloot_platform.research.result_authority import (
        _validate_basic_policy_quality_companions_v1, _validate_tie_rule_quality_companions_v1,
        native_basic_policy_quality_request_v1, native_round_win_quality_request_v1,
        native_tie_allocation_universe_hash_v1, native_tie_rule_profile_v1,
    )
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, ScenarioQualityExecutionBudgetV1

    fixture = json.loads((root / "examples/research-scenario-v4-unique-saving.json").read_text(encoding="utf-8"))
    prefix = ExactTracePrefixV1.build(fixture["exactTracePrefix"])
    profiles = yaml.safe_load((root / "configs/challenge/progressive_mining_ortools_v3_1.yaml").read_text())[
        "scenarioResearchV4"]["proofBudgets"]["profiles"]
    profile_id = "actor_information_set_exact_v1"
    budget = ScenarioProofBudgetV1.build(profile_id, profiles[profile_id])
    policy = {"schemaVersion": 4, "maximumFinalMargin": 40, "minimumPointSwing": 20,
        "maximumFeaturedTrick": 7, "minimumMeaningfulDecisions": 1, "minimumConsistentWorlds": 2,
        "minimumBaselineRegret": 20, "minimumNextBestRegret": 20,
        "baselinePolicyId": baseline_policy_id, "allowTieRuleOnlyLessons": False}
    execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
        "maximumWallSeconds": budget.maximum_wall_seconds, "maximumEvidenceBytes": budget.maximum_cache_bytes,
        "maximumEscalationDecisions": 16, "videoInterestingness": policy})
    deadline = time.monotonic() + budget.maximum_wall_seconds
    cpu_deadline = time.process_time() + budget.maximum_cpu_seconds

    def remaining():
        seconds = min(deadline - time.monotonic(), cpu_deadline - time.process_time())
        if seconds <= 0:
            raise TimeoutError("real tie companion fixture exhausted its shared parent budget")
        return seconds

    source_rules = native_tie_rule_profile_v1(rules, "buyer_makes")
    state = OrToolsExactStateNormalizerV1(source_rules, native_cp_sat=True).normalize_play_or_project(
        prefix, maximum_seconds=remaining())
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "exact",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "winning_root_action_count", "targetTeam": Seat.parse(state.next_actor).team.name.lower(),
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": profile_id})
    options = dict(budget=budget, nodes=budget.maximum_nodes, depth=budget.maximum_depth,
        worlds=budget.maximum_hidden_worlds, remaining_seconds=remaining,
        declared_seconds=budget.maximum_wall_seconds, cancel_check=None, private_recall=None,
        graph_workers=1, worker_cpu_callback=None)
    source = _editorial_play_quality_artifact_v1(source_rules, state, prefix, request, **options)
    assert source["proof"]["completionStatus"] == "exact_complete"
    digest = canonical_sha256(source)
    baseline_request = native_basic_policy_quality_request_v1(request)
    _, baseline = _observer_safe_play_policy_artifact_v1(source_rules, state, baseline_request,
        actual_action=None, policy_id=policy["baselinePolicyId"], maximum_seconds=remaining(),
        cancel_check=None)
    baseline = {**baseline, "sourcePrefix": prefix.to_dict()}
    universe = native_tie_allocation_universe_hash_v1(source)
    comparison = {"sourceArtifactHash": digest, "sourcePrefixHash": prefix.sha256,
        "baselineAction": baseline["proof"]["selectedAction"]}
    for tie_rule in ("buyer_makes", "buyer_loses"):
        variant = native_tie_rule_profile_v1(source_rules, tie_rule)
        variant_state = (state if tie_rule == "buyer_makes" else
            OrToolsExactStateNormalizerV1(variant, native_cp_sat=True).normalize_play_or_project(
                prefix, maximum_seconds=remaining()))
        margin = source if tie_rule == "buyer_makes" else _editorial_play_quality_artifact_v1(
            variant, variant_state, prefix, request, **options)
        winner = _editorial_play_quality_artifact_v1(variant, variant_state, prefix,
            native_round_win_quality_request_v1(request), **options)
        assert winner["proof"]["completionStatus"] == margin["proof"]["completionStatus"] == "exact_complete"
        assert native_tie_allocation_universe_hash_v1(winner) == universe
        assert native_tie_allocation_universe_hash_v1(margin) == universe
        comparison[tie_rule] = {"margin": margin, "roundWin": winner,
            "rulesSnapshot": variant.data, "rulesHash": variant.sha256,
            "exactStateId": variant_state.exact_state_id, "allocationUniverseHash": universe}
    comparison["sha256"] = canonical_sha256(comparison)
    row = {"proofArtifactHash": digest, "sourcePrefixHash": prefix.sha256,
        "decisionKind": "play", "exactStateId": state.exact_state_id, "actor": state.next_actor,
        "targetTeam": request.target_team, "action": source["proof"]["selectedAction"],
        "playIndex": len(prefix.play_actions), "proofBudgetHash": budget.sha256}
    group = {"artifacts": {digest: source}, "decisions": [row],
        "baselineEvidence": {digest: baseline}, "tieRuleEvidence": {digest: comparison},
        "roundWinEvidence": {digest: comparison["buyer_makes"]["roundWin"]},
        "proofBudgetProfiles": {profile_id: profiles[profile_id]}, "limits": {
            "maximumDecisionSeconds": budget.maximum_wall_seconds, "maximumNodesPerDecision": budget.maximum_nodes,
            "maximumDepth": budget.maximum_depth, "maximumWorldsPerDecision": budget.maximum_hidden_worlds}}
    _validate_basic_policy_quality_companions_v1(group, rules=source_rules,
        execution_budget=execution, remaining=remaining, cancel_check=None)
    substituted = copy.deepcopy(group)
    other_policy_id = ("observer_safe_partner_feed_established_control_v1"
        if baseline_policy_id == "observer_safe_trick_control_v1" else "observer_safe_trick_control_v1")
    _, other_baseline = _observer_safe_play_policy_artifact_v1(source_rules, state, baseline_request,
        actual_action=None, policy_id=other_policy_id, maximum_seconds=remaining(), cancel_check=None)
    substituted["baselineEvidence"][digest] = {**other_baseline, "sourcePrefix": prefix.to_dict()}
    with pytest.raises(ValueError, match="basic policy companion differs from observer-safe selection"):
        _validate_basic_policy_quality_companions_v1(substituted, rules=source_rules,
            execution_budget=execution, remaining=remaining, cancel_check=None)
    calls, verified = [], set()
    reverify = authority.reverify_native_exact_observed_play_artifact_v1
    def tracked(artifact, **context):
        calls.append(canonical_sha256(artifact))
        return reverify(artifact, **context)
    monkeypatch.setattr(authority, "reverify_native_exact_observed_play_artifact_v1", tracked)
    for companion, factory in (({**source, "sourcePrefix": prefix.to_dict()}, lambda request: request),
            (comparison["buyer_makes"]["roundWin"], native_round_win_quality_request_v1)):
        authority._validate_raw_point_quality_companions_v1(
            {**group, "roundWinEvidence": {digest: companion}}, rules=source_rules,
            execution_budget=execution, remaining=remaining, cancel_check=None, private_recall=None,
            evidence_key="roundWinEvidence", request_factory=factory, verified_artifacts=verified)
    for _ in range(2):
        _validate_tie_rule_quality_companions_v1(group, rules=source_rules,
            execution_budget=execution, remaining=remaining, cancel_check=None, private_recall=None,
            verified_artifacts=verified)
    assert len(calls) == len(set(calls)) == len(verified) == 4
    corrupted = copy.deepcopy(group)
    proof = corrupted["tieRuleEvidence"][digest]["buyer_makes"]["roundWin"]["proof"]
    proof["rootActions"][0]["value"] += 1
    proof["sha256"] = canonical_sha256({key: value for key, value in proof.items() if key != "sha256"})
    changed = corrupted["tieRuleEvidence"][digest]
    changed["sha256"] = canonical_sha256({key: value for key, value in changed.items() if key != "sha256"})
    before = set(verified)
    with pytest.raises(ValueError, match="fresh native proof"):
        _validate_tie_rule_quality_companions_v1(corrupted, rules=source_rules,
            execution_budget=execution, remaining=remaining, cancel_check=None, private_recall=None,
            verified_artifacts=verified)
    assert verified == before
    assert len(calls) == 5
    # A complete proof alone cannot establish editorial acceptance.
    verdict = evaluate_round_interestingness_v1({"postAuction": {"projectPlay": group}},
        {"awardedPoints": [18, 18]}, policy, featured_state_id=state.exact_state_id)
    values = {row["action"]: row["value"] for row in source["proof"]["rootActions"]}
    baseline_action = baseline["proof"]["selectedAction"]
    regret = values[row["action"]] - values[baseline_action]
    if regret < policy["minimumBaselineRegret"]:
        assert verdict["passed"] is False
        assert "interestingness_baseline_regret_insufficient" in verdict["blockers"]
    elif verdict["passed"]:
        assert baseline_action != row["action"]
        assert source["proof"]["hiddenWorlds"]["count"] >= policy["minimumConsistentWorlds"]
    missing_baseline = copy.deepcopy(group)
    missing_baseline.pop("baselineEvidence")
    verdict = evaluate_round_interestingness_v1({"postAuction": {"projectPlay": missing_baseline}},
        {"awardedPoints": [18, 18]}, policy, featured_state_id=state.exact_state_id)
    assert verdict["passed"] is False
    assert "interestingness_baseline_evidence_unavailable" in verdict["blockers"]
    missing = copy.deepcopy(group)
    missing.pop("tieRuleEvidence")
    verdict = evaluate_round_interestingness_v1({"postAuction": {"projectPlay": missing}},
        {"awardedPoints": [18, 18]}, policy, featured_state_id=state.exact_state_id)
    assert "interestingness_tie_rule_evidence_unavailable" in verdict["blockers"]


def _seed(rules, *, project_points=(0, 0)) -> CompleteStateGraphSeedV1:
    exact = _normalized_state(rules)
    state = exact.state
    completed = state["completedTricks"]
    base_raw = [0, 0]
    base_tricks = [0, 0]
    for trick in completed:
        team = Seat.parse(trick["winner"]).team
        base_raw[int(team)] += trick["rawPoints"]
        base_tricks[int(team)] += 1
    return CompleteStateGraphSeedV1.build(
        exact_state_id=exact.exact_state_id,
        rules_hash=rules.sha256,
        remaining_hands=tuple(
            tuple(Card.parse(card) for card in state["hands"][seat.name.lower()])
            for seat in Seat
        ),
        current_trick=(),
        first_leader=Seat.SOUTH,
        next_actor=Seat.SOUTH,
        legal_actions=tuple(
            TailPrefixActionV1.build(Seat.SOUTH, item["card"])
            for item in exact.legal_actions
        ),
        contract=Contract.SUN,
        trump=None,
        buyer=Seat.WEST,
        locked=False,
        base_raw=(base_raw[0], base_raw[1]),
        base_tricks=(base_tricks[0], base_tricks[1]),
        awarded_project_points=project_points,
        baloot_points=(0, 0),
        escalation="normal",
        last_escalating_team=None,
        starting_match_score=(0, 0),
    )


@pytest.mark.parametrize("target", ["north_south", "west_east"])
@pytest.mark.parametrize("prefix_length", [2, 3])
@pytest.mark.parametrize("tactic", ["small_card_big_gain", "large_card_sacrifice", "partner_enablement", "entry_unblocking"])
def test_paired_raw_payoffs_include_partial_trick_and_last_bonus(rules, target, prefix_length, tactic):
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3
    from baloot_platform.tactics import TacticalDetectorProfileV1

    # Legal tail kernel; full-round prefix reachability is tested separately.
    actor = Seat.EAST if prefix_length == 3 else Seat.SOUTH
    actions = ("C9", "DA") if prefix_length == 3 else ("H8",)
    source = CompleteStateGraphSeedV1.build(exact_state_id="STATE-PAIRED-RAW", rules_hash=rules.sha256,
        remaining_hands=tuple(tuple(Card.parse(card) for card in hand.split())
            for hand in ("CA", "C7", "C8" if prefix_length == 3 else "H8 C8", "C9 DA")),
        current_trick=tuple(TailPrefixActionV1.build(seat, card) for seat, card in
            (("north", "H7"), ("west", "HA"), ("south", "H8"))[:prefix_length]),
        first_leader=Seat.NORTH, next_actor=actor,
        legal_actions=tuple(TailPrefixActionV1.build(actor, card) for card in actions),
        contract=Contract.SUN, trump=None, buyer=Seat.NORTH, locked=False,
        base_raw=(40, 47), base_tricks=(3, 3), awarded_project_points=(0, 0), baloot_points=(0, 0),
        escalation="normal", last_escalating_team=None, starting_match_score=(0, 0))
    raw = _request(claim="winning_root_action_count", selected_action=actions[-1]).to_dict()
    raw.pop("sha256")
    request = ScenarioProofRequestV1.build({**raw, "targetTeam": target})
    compilation = _compiler(rules, native_conversion=True).compile(source, request,
        expected_rules_hash=rules.sha256, maximum_nodes=2000, maximum_depth=8, maximum_seconds=15)
    assert compilation.complete, compilation.receipt.to_dict()
    scenario = ResearchScenarioV1.build({"schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"predicateId": "PRED-PAIRED-RAW", "kind": "tactic.verified", "version": 1,
            "hardness": "report", "arguments": {"kind": tactic, "minimumScore": 0,
                "minimumEvidence": 1, "causalRequired": True}},
        "decisionPoint": {"kind": "play", "selector": {}},
        "outcome": {"predicateId": "PRED-ROUND", "kind": "outcome.team_round", "version": 1,
            "hardness": "report", "arguments": {"team": target, "status": "won"}}})
    configure = _native_proof_tactic_configurer_v3(scenario=scenario,
        candidate=SimpleNamespace(trace={}, state={"contract": "sun", "trumpSuit": None,
            "currentTrick": {"leader": "north", "plays": [{"actor": action.actor.name.lower(), "card": action.card.code}
                                       for action in source.current_trick]}}),
        graph=compilation.graph, request=request, rules=rules,
        manifest=shared_scenario_predicate_catalog_lowering_manifest_v3(),
        tactical_detector_profile=TacticalDetectorProfileV1.defaults())
    proof = OrToolsScenarioGraphProofV1().prove(compilation.graph, request, tactic_configure=configure)
    assert proof.completion_status == "exact_complete"
    observed = proof.native_qualification["PRED-PAIRED-RAW"]["observed"]
    if tactic in {"partner_enablement", "entry_unblocking"} and prefix_length == 2:
        assert observed["inapplicabilityReason"] == "required_alternative_absent"
        assert observed["verified"] is False
        return
    assert observed["pairedRawPayoffUnit"] == "raw_card_points"
    assert observed["causalProven"] is False
    if tactic in {"partner_enablement", "entry_unblocking"}:
        assert observed["applicable"] is False  # Both roots already give West the first trick.
        assert observed["mechanismScope"] == "paired_canonical_continuations"
    receipts = proof.model_document()["pairedRawPayoffVariables"]
    assert {row["action"] for row in receipts} == set(actions)
    for index, row in enumerate(receipts):
        expected = ((11, 32, -21) if row["action"] == "C9" else (22, 21, 1)
                    if prefix_length == 3 else (22, 21, -1))
        assert (observed[f"root{index}Trick7RawPoints"], observed[f"root{index}Trick8RawPoints"],
                observed[f"root{index}ContinuationRawMargin"]) == expected
        assert observed[f"root{index}FirstTrickRawMargin"] == (expected[0] if prefix_length == 3 else -expected[0])
        assert observed[f"root{index}ImmediateRawMargin"] == (expected[0] if prefix_length == 3 else 0)
        assert observed[f"root{index}Trick7WinnerSeat"] == int(Seat.WEST)
        assert observed[f"root{index}Trick8WinnerSeat"] == int(Seat.NORTH)
        if tactic == "entry_unblocking":
            assert observed[f"root{index}Trick7LeaderSeat"] == int(Seat.NORTH)
            assert observed[f"root{index}Trick7LeadCard"] == Card.parse("H7").id
            assert observed[f"root{index}Trick8LeaderSeat"] == int(Seat.WEST)
            assert observed[f"root{index}Trick8LeadCard"] == Card.parse("C7").id
            assert observed[f"root{index}Trick8CardRawMargin"] == observed[f"root{index}Trick8RawMargin"] + 10
        assert row["currentTrickCards"] == [action.card.id for action in source.current_trick]
        assert row["lastTrickBonusRaw"] == 10
    assert {row.value for row in proof.root_actions} != {-21, 1}


@pytest.mark.parametrize("metric", ["raw_card_points", "awarded_point_margin"])
@pytest.mark.parametrize("target", ["north_south", "west_east"])
@pytest.mark.parametrize("kernel", ["keep_ace", "unblock_partner"])
def test_native_delayed_payoff_producer_certifies_legal_tails(rules, metric, target, kernel):
    measured_seconds = 900.0  # Sulaiman p99 542.366s; capped at the 900s file lane.
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3
    from baloot_platform.tactics import TacticalDetectorProfileV1
    from baloot_platform.research.result_authority import _validate_native_tactical_profile_v1

    # Kernel legality only; the exact-input compiler owns full-round prefix reachability.
    keep_ace = kernel == "keep_ace"
    # Keep CA under HA, or sacrifice S10 under SA to unblock partner's S9 and DA.
    hands = ("CK", "C7 CA", "C8", "C9") if keep_ace else ("S10 S7 C7", "S8 D7", "S9 DA", "C8 C9")
    prefix = (("south", "H7"), ("east", "H8"), ("north", "HA")) if keep_ace else (
        ("west", "SA"), ("south", "SJ"), ("east", "D8"))
    actor = Seat.WEST if keep_ace else Seat.NORTH
    source = CompleteStateGraphSeedV1.build(exact_state_id="STATE-DELAYED-" + kernel.upper(), rules_hash=rules.sha256,
        remaining_hands=tuple(tuple(Card.parse(card) for card in hand.split())
            for hand in hands),
        current_trick=tuple(TailPrefixActionV1.build(seat, card) for seat, card in prefix),
        first_leader=Seat.SOUTH if keep_ace else Seat.WEST, next_actor=actor,
        legal_actions=tuple(TailPrefixActionV1.build(actor.name.lower(), card)
            for card in (("C7", "CA") if keep_ace else ("S10", "S7"))),
        contract=Contract.SUN, trump=None, buyer=Seat.NORTH, locked=False,
        base_raw=(47, 47) if keep_ace else (49, 37), base_tricks=(3, 3) if keep_ace else (2, 3),
        awarded_project_points=(0, 0), baloot_points=(0, 0),
        escalation="normal", last_escalating_team=None, starting_match_score=(0, 0))
    profile = TacticalDetectorProfileV1.defaults().to_dict()
    profile.pop("sha256")
    profile = TacticalDetectorProfileV1.build({**profile,
        "nativeDelayedPayoff": {"gainMetric": metric, "minimumGain": 2}})
    cases = (("small_card_big_gain", "C7", True), ("large_card_sacrifice", "CA", False)) if keep_ace else (
        ("large_card_sacrifice", "S10", True), ("small_card_big_gain", "S7", False))
    for tactic, selected, expected in cases:
        raw = _request(claim="winning_root_action_count", selected_action=selected).to_dict()
        raw.pop("sha256")
        request = ScenarioProofRequestV1.build({**raw, "targetTeam": target})
        compilation = _compiler(rules, native_conversion=True).compile(source, request,
            expected_rules_hash=rules.sha256, maximum_nodes=2000, maximum_depth=12,
            maximum_seconds=measured_seconds)
        assert compilation.complete, compilation.receipt.to_dict()
        scenario = ResearchScenarioV1.build({"schemaVersion": 1, "origin": {"kind": "search"},
            "constraints": {"predicateId": "PRED-DELAYED", "kind": "tactic.verified", "version": 1,
                "hardness": "hard" if expected else "report", "arguments": {"kind": tactic, "minimumScore": 100,
                    "minimumEvidence": 1, "causalRequired": True}},
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": {"predicateId": "PRED-ROUND", "kind": "outcome.team_round", "version": 1,
                "hardness": "report", "arguments": {"team": target, "status": "won"}}})
        configure = _native_proof_tactic_configurer_v3(scenario=scenario,
            candidate=SimpleNamespace(trace={}, state={"contract": "sun", "trumpSuit": None,
                "currentTrick": {"leader": source.first_leader.name.lower(), "plays": [{"actor": action.actor.name.lower(), "card": action.card.code}
                    for action in source.current_trick]}}),
            graph=compilation.graph, request=request, rules=rules,
            manifest=shared_scenario_predicate_catalog_lowering_manifest_v3(native_delayed_payoff=True),
            tactical_detector_profile=profile)
        proof = OrToolsScenarioGraphProofV1().prove(compilation.graph, request, tactic_configure=configure)
        assert proof.completion_status == "exact_complete"
        evidence = proof.native_qualification["PRED-DELAYED"]
        assert evidence["satisfied"] is expected
        observed = evidence["observed"]
        assert observed["causalProven"] is expected
        assert observed["detectorVersion"] == "3.1.0"
        if expected:
            assert observed["nativeGain"] == ((30 if keep_ace else 2) if metric == "raw_card_points"
                else (32 if keep_ace else 28) * compilation.graph.utility_scale)
            assert observed["laterCardMarginGain"] == (19 if keep_ace else 12)
            assert observed["entry"] or observed["partnerEnablement"]
        assert all(row["delayedControl"] for row in proof.model_document()["pairedRawPayoffVariables"])
        snapshot = {"bundleHash": "b" * 64, "components": {"challenge": {
            "payload": {"tactical_detectors": {key: value for key, value in profile.to_dict().items()
                if key != "sha256"}}}}}
        snapshot["sha256"] = canonical_sha256(snapshot)
        _validate_native_tactical_profile_v1({"evidence": [evidence]}, proof.to_dict(), snapshot,
            bundle_hash=snapshot["bundleHash"], snapshot_hash=snapshot["sha256"],
            proof_graph=compilation.graph.to_dict(), proof_request=request.to_dict(),
            proof_model=proof.model_document(), backend_version="3.1.126", scenario=scenario.to_dict())


@pytest.mark.parametrize("projects", [(0, 0), (20, 0)])
@pytest.mark.parametrize("compact", [False, True])
def test_native_graph_retains_project_counterfactual_per_terminal(rules, projects, compact, monkeypatch):
    measured_seconds = 900.0  # Sulaiman p99 542.366s; capped at the 900s file lane.
    compiler = _compiler(rules, native_conversion=True)
    request = _request()
    limits = dict(expected_rules_hash=rules.sha256, maximum_nodes=2000,
                  maximum_depth=8, maximum_seconds=measured_seconds)
    source = _seed(rules, project_points=projects)
    assert CompleteStateGraphSeedV1.from_dict(source.to_dict()) == source
    for mutation in ("unknown", "locked", "hash", "points"):
        changed = copy.deepcopy(source.to_dict())
        if mutation == "unknown":
            changed["unowned"] = True
        elif mutation == "locked":
            changed["locked"] = 1
        elif mutation == "hash":
            changed["proofStateProjectionHash"] = "0" * 64
        else:
            changed["baseRaw"][0] = True
        with pytest.raises(ValueError, match="proof seed"):
            CompleteStateGraphSeedV1.from_dict(changed)
    paired = compiler.compile(source, request, **limits,
                              project_counterfactual=True, _compact_forced=compact)
    assert paired.complete and paired.project_counterfactual_graph is not None
    assert canonical_sha256(paired.model_document) == paired.receipt.model_hash
    assert paired.model_document["projectCounterfactualGraphHash"] == paired.project_counterfactual_graph.sha256
    ordinary = compiler.compile(source, request, **limits)
    neutral = compiler.compile(_seed(rules), request, **limits)
    assert ordinary.complete and neutral.complete
    prover = OrToolsScenarioGraphProofV1()

    def capture(bindings):
        return {"PROJECT": SimpleNamespace(
            literal=bindings.proof_complete_variable,
            extract=lambda solver: {"satisfied": True, "deltas": [
                solver.value(value) for value in bindings.root_project_consequence_variables]},
        )}

    receipt = paired.receipt.to_dict()
    receipt.pop("elapsedSeconds")
    receipt.pop("sha256")
    evidence = {"model": paired.model_document, "receipt": receipt, "seed": source.to_dict(),
                "request": request.to_dict(), "graph": paired.graph.to_dict(),
                "counterfactualGraph": paired.project_counterfactual_graph.to_dict()}
    proof = prover.prove(paired.graph, request, tactic_configure=capture,
                         native_root_only=True, project_counterfactual_graph=paired.project_counterfactual_graph,
                         project_compilation_evidence=(evidence,))
    assert proof.model_document()["projectCompilationEvidence"] == [evidence]
    assert canonical_sha256(proof.model_document()) == proof.model_hash
    actual = prover.prove(ordinary.graph, request)
    baseline = prover.prove(neutral.graph, request)
    assert proof.root_actions == actual.root_actions
    assert prover.prove(paired.project_counterfactual_graph, request).root_actions == baseline.root_actions
    expected = [left.value - right.value for left, right in zip(actual.root_actions, baseline.root_actions, strict=True)]
    assert proof.native_qualification["PROJECT"]["deltas"] == expected
    assert any(expected) if projects != (0, 0) else not any(expected)
    def reject_eager(*args, **kwargs):
        raise AssertionError("native exact compilation must not pre-enumerate above the node tier")
    monkeypatch.setattr(ortools_complete_state_graph._CpSatPathOracle, "enumerate_paths", reject_eager)
    stopped = compiler.compile(source, request, **{**limits, "maximum_nodes": 1},
                               project_counterfactual=True)
    assert not stopped.complete and stopped.project_counterfactual_graph is None


@pytest.mark.parametrize("contract,node_limit,eager", [
    (Contract.SUN, 16, True), (Contract.SUN, 15, False), (Contract.HOKUM, 16, False),
])
def test_native_graph_path_forecast_counts_only_available_ekka(rules, monkeypatch, contract, node_limit, eager):
    compiler = _compiler(rules, native_conversion=True)
    source = _seed(rules)
    start = compiler._parse_start(source, expected_rules_hash=rules.sha256)
    # Synthetic forecast fixture; the existing parity fixtures verify legal roots.
    start = replace(start, contract=contract, trump=Suit.CLUBS if contract == Contract.HOKUM else None)
    monkeypatch.setattr(compiler, "_parse_start", lambda *_args, **_kwargs: start)
    calls = []

    def stop_enumeration(oracle, **kwargs):
        calls.append(kwargs["maximum_nodes"])
        assert bool(oracle.binary_ekka_steps) is (contract == Contract.HOKUM)
        return "forecast_fixture_stop"

    def stop_query(*_args, **_kwargs):
        raise ortools_complete_state_graph._CompilationStopped("forecast_fixture_stop")

    monkeypatch.setattr(ortools_complete_state_graph._CpSatPathOracle, "enumerate_paths", stop_enumeration)
    monkeypatch.setattr(ortools_complete_state_graph._CpSatPathOracle, "solve", stop_query)
    compiled = compiler.compile(source, _request(), expected_rules_hash=rules.sha256,
        maximum_nodes=node_limit, maximum_depth=8, maximum_seconds=10)
    assert calls == ([node_limit] if eager else [])
    assert not compiled.complete and compiled.receipt.exhaustion_reason == "forecast_fixture_stop"


def test_native_trump_extraction_is_verified_on_a_legal_hokum_tail(rules) -> None:
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.tactics import TacticalDetectorProfileV1

    # This exercises the native tail kernel, not full-prefix reachability.
    source = CompleteStateGraphSeedV1.build(exact_state_id='STATE-TRUMP-EXTRACTION', rules_hash=rules.sha256,
        remaining_hands=tuple(tuple(Card.parse(code) for code in hand.split())
            for hand in ('S9 C7', 'SJ C8', 'SA CA', 'S7 C9')), current_trick=(),
        first_leader=Seat.NORTH, next_actor=Seat.NORTH,
        legal_actions=tuple(TailPrefixActionV1.build('north', card) for card in ('S9', 'C7')),
        contract=Contract.HOKUM, trump=Suit.SPADES, buyer=Seat.NORTH, locked=False,
        base_raw=(64, 32), base_tricks=(3, 3), awarded_project_points=(0, 0), baloot_points=(0, 0),
        escalation='normal', last_escalating_team=None, starting_match_score=(0, 0))
    request = _request(claim='selected_action_is_unique_winner', selected_action='S9')
    compilation = _compiler(rules, native_conversion=True).compile(source, request,
        expected_rules_hash=rules.sha256, maximum_nodes=10000, maximum_depth=8, maximum_seconds=15)
    assert compilation.complete, compilation.receipt.to_dict()
    scenario = ResearchScenarioV1.build({'schemaVersion': 1, 'origin': {'kind': 'search'},
        'constraints': {'predicateId': 'PRED-LEGAL-EXTRACTION', 'kind': 'tactic.verified', 'version': 1,
            'hardness': 'hard', 'arguments': {'kind': 'trump_control', 'minimumScore': 100,
                'minimumEvidence': 1, 'causalRequired': True}},
        'decisionPoint': {'kind': 'play', 'selector': {}},
        'outcome': {'predicateId': 'PRED-REPORT', 'kind': 'outcome.team_round', 'version': 1,
            'hardness': 'report', 'arguments': {'team': 'north_south', 'status': 'won'}}})
    configure = _native_proof_tactic_configurer_v3(scenario=scenario,
        candidate=SimpleNamespace(trace={}, state={'contract': 'hokum', 'trumpSuit': 'spades'}),
        graph=compilation.graph, request=request, rules=rules,
        manifest=shared_scenario_predicate_catalog_lowering_manifest_v3(),
        tactical_detector_profile=TacticalDetectorProfileV1.defaults())
    proof = OrToolsScenarioGraphProofV1().prove(compilation.graph, request, tactic_configure=configure)
    assert proof.completion_status == 'exact_complete' and proof.claim_satisfied
    values = {row.action_code: row.value for row in proof.root_actions}
    # The frozen buyer-makes rule is a win at zero awarded-point margin.
    assert values == {'S9': 0, 'C7': -16}
    assert [row.action_code for row in proof.root_actions if row.satisfies] == ['S9']
    evidence = proof.native_qualification['PRED-LEGAL-EXTRACTION']
    assert evidence['satisfied'] and evidence['observed']['trumpExtraction']


def test_hokum_root_legal_actions_match_tail_including_ekka(rules) -> None:
    source = _seed(rules)
    expected = (
        TailPrefixActionV1.build("south", "S7"),
        TailPrefixActionV1.build("south", "SA"),
        TailPrefixActionV1.build("south", "SA", ekka=True),
    )
    seed = CompleteStateGraphSeedV1.build(
        exact_state_id=source.exact_state_id,
        rules_hash=source.rules_hash,
        remaining_hands=source.remaining_hands,
        current_trick=source.current_trick,
        first_leader=source.first_leader,
        next_actor=source.next_actor,
        legal_actions=expected,
        contract=Contract.HOKUM,
        trump=Suit.CLUBS,
        buyer=source.buyer,
        locked=source.locked,
        base_raw=(65, 65),
        base_tricks=source.base_tricks,
        awarded_project_points=source.awarded_project_points,
        baloot_points=source.baloot_points,
        escalation=source.escalation,
        last_escalating_team=source.last_escalating_team,
        starting_match_score=source.starting_match_score,
    )
    compiler = _compiler(rules, native_conversion=True)
    start = compiler._parse_seed_start(seed, expected_rules_hash=rules.sha256)
    oracle = compiler._path_oracle(
        start, target=Team.NORTH_SOUTH, utility="awarded_point_margin"
    )
    result, _ = oracle.solve((), maximum_seconds=10, cancelled=None)
    assert {(action.card.code, action.ekka) for action in result.legal_play_actions} == {
        (action.card.code, action.ekka) for action in expected
    }


def test_hokum_locked_state_changes_root_legal_actions(rules) -> None:
    source = _seed(rules)
    hands = list(source.remaining_hands)
    hands[int(Seat.NORTH)] = (Card.parse("H7"), Card.parse("S7"))
    hands[int(Seat.SOUTH)] = (Card.parse("C7"), Card.parse("SA"))
    expected = (
        TailPrefixActionV1.build("south", "C7"),
        TailPrefixActionV1.build("south", "SA"),
    )
    seed = CompleteStateGraphSeedV1.build(
        exact_state_id=source.exact_state_id,
        rules_hash=source.rules_hash,
        remaining_hands=tuple(hands),
        current_trick=(),
        first_leader=Seat.SOUTH,
        next_actor=Seat.SOUTH,
        legal_actions=expected,
        contract=Contract.HOKUM,
        trump=Suit.SPADES,
        buyer=source.buyer,
        locked=True,
        base_raw=(58, 58),
        base_tricks=source.base_tricks,
        awarded_project_points=source.awarded_project_points,
        baloot_points=source.baloot_points,
        escalation="double",
        last_escalating_team=Team.NORTH_SOUTH,
        starting_match_score=source.starting_match_score,
    )
    compiler = _compiler(rules, native_conversion=True)
    start = compiler._parse_seed_start(seed, expected_rules_hash=rules.sha256)
    oracle = compiler._path_oracle(
        start, target=Team.NORTH_SOUTH, utility="awarded_point_margin"
    )
    result, _ = oracle.solve((), maximum_seconds=10, cancelled=None)
    assert {action.card.code for action in result.legal_play_actions} == {"C7"}


def test_raw_root_cache_key_ignores_only_prefix_raw_split(rules) -> None:
    seed = _seed(rules)
    changed_raw = _seed_with_target_raw_v1(seed, Team.NORTH_SOUTH, 0)
    assert _raw_root_state_key_v1(seed, Team.NORTH_SOUTH) == (
        _raw_root_state_key_v1(changed_raw, Team.NORTH_SOUTH)
    )
    assert _raw_root_state_key_v1(seed, Team.NORTH_SOUTH) != (
        _raw_root_state_key_v1(replace(seed, locked=True), Team.NORTH_SOUTH)
    )
    scoring_changes = replace(
        changed_raw, awarded_project_points=(20, 0), baloot_points=(0, 2),
        escalation="double", last_escalating_team=Team.NORTH_SOUTH,
        starting_match_score=(100, 110), base_tricks=(2, 4),
    )
    raw_key = _raw_root_uniqueness_key_v1(seed, Team.NORTH_SOUTH)
    assert raw_key == _raw_root_uniqueness_key_v1(scoring_changes, Team.NORTH_SOUTH)
    # Awarded-score probes must NOT share the looser raw-ranking key.
    assert _raw_root_state_key_v1(seed, Team.NORTH_SOUTH) != (
        _raw_root_state_key_v1(scoring_changes, Team.NORTH_SOUTH)
    )
    for changed in (
        replace(seed, locked=True),
        replace(seed, legal_actions=seed.legal_actions[:1]),
        replace(seed, current_trick=(TailPrefixActionV1.build("north", "H7"),)),
        replace(seed, base_tricks=(2, 3)),
    ):
        assert raw_key != _raw_root_uniqueness_key_v1(changed, Team.NORTH_SOUTH)
    assert raw_key != _raw_root_uniqueness_key_v1(seed, Team.WEST_EAST)


def test_raw_root_ranking_is_invariant_under_prefix_raw_split(rules) -> None:
    seed = _seed(rules)
    request = _request(claim="winning_root_action_count")
    request_core = request.to_dict()
    request_core.pop("sha256")
    request_core.update({
        "utility": INTERNAL_RAW_POINT_MARGIN_UTILITY_V1,
        "successThreshold": 0.0,
    })
    request = replace(
        request,
        utility=INTERNAL_RAW_POINT_MARGIN_UTILITY_V1,
        success_threshold=0.0,
        sha256=canonical_sha256(request_core),
    )
    root_values = []
    for target_raw in (0, 1, 1):
        changed = _seed_with_target_raw_v1(seed, Team.NORTH_SOUTH, target_raw)
        if len(root_values) == 2:
            changed = CompleteStateGraphSeedV1.build(**({
                name: getattr(changed, name)
                for name in changed.__dataclass_fields__
                if name != "proof_state_projection_hash"
            } | {
                "awarded_project_points": (20, 0), "escalation": "double",
                "last_escalating_team": Team.NORTH_SOUTH,
                "starting_match_score": (100, 110), "base_tricks": (2, 4),
            }))
        graph = _compiler(rules, native_conversion=True).compile(
            changed, request, expected_rules_hash=rules.sha256,
            maximum_nodes=2000, maximum_depth=8, maximum_seconds=10,
        )
        assert graph.complete and graph.graph is not None
        proof = OrToolsScenarioGraphProofV1().prove(
            graph.graph, request, maximum_seconds=10,
            maximum_line_count_limbs=64,
        )
        assert proof.completion_status == "exact_complete"
        root_values.append({item.action_code: item.value for item in proof.root_actions})
    assert root_values[0].keys() == root_values[1].keys()
    assert all(root_values[1][action] - value == 2
               for action, value in root_values[0].items())
    assert root_values[2] == root_values[1]


@pytest.mark.real_clock
def test_graph_deadline_includes_model_construction(rules, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(ortools_complete_state_graph, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    compiler = _compiler(rules)
    def build(*args, **kwargs):
        clock[0] = 2.0
        return SimpleNamespace(model_hash="constructed", solve=lambda *a, **k: pytest.fail("expired model must not solve"))
    monkeypatch.setattr(compiler, "_path_oracle", build)
    result = compiler.compile(_normalized_state(rules), _request(), expected_rules_hash=rules.sha256,
        maximum_nodes=2000, maximum_depth=8, maximum_seconds=1)
    assert result.graph is None
    assert result.receipt.exhaustion_reason == "maximum_seconds"
    assert result.receipt.elapsed_seconds == 2.0
    assert result.receipt.explored_node_count == 0


@pytest.mark.real_clock
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("reason", ["cancelled", "maximum_seconds"])
def test_graph_construction_stop_preserves_incomplete_receipt(rules, monkeypatch, reason, compact):
    clock = [0.0]
    cancelled = [False]
    monkeypatch.setattr(ortools_complete_state_graph, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    compiler = _compiler(rules)
    calls = []
    def build(*args, **kwargs):
        calls.append("build")
        cancelled[0] = reason == "cancelled"
        clock[0] = 2.0 if reason == "maximum_seconds" else 0.0
        kwargs["cancel_check"]()
        pytest.fail("construction continued after cancellation/deadline")
    monkeypatch.setattr(compiler.play_stage, "build_model", build)
    result = compiler.compile(_normalized_state(rules), _request(), expected_rules_hash=rules.sha256,
        maximum_nodes=2000, maximum_depth=8, maximum_seconds=1, cancelled=lambda: cancelled[0],
        _compact_forced=compact)
    assert calls == ["build"]
    assert result.graph is None
    assert result.receipt.status == "incomplete"
    assert result.receipt.exhaustion_reason == reason
    assert result.receipt.explored_node_count == 0
    assert result.decision_observation_ids == {}
    assert result.sequence_pack is None


def _compile(rules, request: ScenarioProofRequestV1, *, native_conversion=False, **budgets):
    return _compiler(rules, native_conversion=native_conversion).compile(
        _normalized_state(rules),
        request,
        expected_rules_hash=rules.sha256,
        maximum_nodes=budgets.get("maximum_nodes", 2_000),
        maximum_depth=budgets.get("maximum_depth", 8),
        maximum_seconds=budgets.get("maximum_seconds", 30.0),
        cancelled=budgets.get("cancelled"),
        decision_observer=budgets.get("decision_observer"),
    )


@pytest.mark.real_clock
@pytest.mark.parametrize("stop", ["expired", "cancelled", "during_solve", "unknown"])
@pytest.mark.parametrize("query", [False, True])
def test_path_oracle_preserves_budget_stop_reason(rules, monkeypatch, stop, query):
    compiler = _compiler(rules, native_conversion=True)
    start = compiler._parse_start(_seed(rules), expected_rules_hash=rules.sha256)
    oracle = compiler._path_oracle(
        start, target=Team.NORTH_SOUTH, utility="awarded_point_margin",
    )
    cancelled = stop == "cancelled"
    calls = []
    if not query:
        clock = iter((0.0, 0.0001))
        monkeypatch.setattr(ortools_complete_state_graph, "time",
                            SimpleNamespace(monotonic=lambda: next(clock)))

    def solve(solver, model, cancel_check):
        nonlocal cancelled
        calls.append(solver.parameters.max_time_in_seconds)
        cancelled = stop == "during_solve"
        return ortools_complete_state_graph.cp_model.UNKNOWN

    monkeypatch.setattr(ortools_complete_state_graph, "_solve_cp_sat_with_cancel_v1", solve)
    with pytest.raises(ortools_complete_state_graph.PlayTailSolveStopped) as caught:
        method = oracle.find_path_below if query else oracle.solve
        method(
            0 if query else (), maximum_seconds=0 if stop == "expired" else (5 if query else 0.0005),
            cancelled=lambda: cancelled,
            **({"conditional_actions": {}} if query else {}),
        )
    expected = {"expired": "maximum_seconds", "during_solve": "cancelled"}.get(stop, stop)
    assert caught.value.solver_status == expected
    if query:
        assert len(calls) == int(stop not in {"expired", "cancelled"})
        assert all(0 < seconds <= 5 for seconds in calls)
    else:
        assert calls == ([] if stop in {"expired", "cancelled"} else [0.0004])


@pytest.mark.parametrize("contract", [Contract.SUN, Contract.HOKUM])
def test_native_complement_feasibility_matches_exact_upper(rules, contract):
    compiler = _compiler(rules, native_conversion=True)
    start = compiler._parse_start(_seed(rules), expected_rules_hash=rules.sha256)
    # Synthetic continuation fixture, not an exact-state admission receipt.
    start = replace(start, contract=contract, trump=Suit.SPADES if contract == Contract.HOKUM else None)
    if contract == Contract.HOKUM:
        # The trump nine remains in hand; the already-played trump jack
        # contributes 20 rather than 2 to this synthetic accumulated score.
        start = replace(start, base_raw=(start.base_raw[0] + 18, start.base_raw[1]))
    target = compiler._path_oracle(start, target=Team.NORTH_SOUTH, utility="awarded_point_margin")
    opponent = compiler._path_oracle(start, target=Team.WEST_EAST, utility="awarded_point_margin")
    arguments = dict(conditional_actions={}, maximum_seconds=10, cancelled=None)
    _, exact_upper, _ = target.find_path_below(target.query_utility_bounds()[1] + 1,
        **arguments, maximize_utility=True)
    assert exact_upper is not None, "fixture must have a legal terminal continuation"
    for threshold in (exact_upper - 1, exact_upper, exact_upper + 1):
        witness, value, _ = opponent.find_path_below(1 - threshold, **arguments, feasibility_only=True)
        assert (witness is None) == (exact_upper < threshold)
        if witness is not None:
            assert -value >= threshold


def test_native_path_feasibility_matches_exhaustive_paths(rules, monkeypatch):
    compiler = _compiler(rules, native_conversion=True)
    start = compiler._parse_start(_seed(rules), expected_rules_hash=rules.sha256)
    oracle = compiler._path_oracle(start, target=Team.NORTH_SOUTH, utility="awarded_point_margin")
    assert oracle.enumerate_paths(maximum_nodes=2000, maximum_seconds=10, cancelled=None) is None
    length = max(map(len, oracle.results))
    values = [state.terminal_utility for path, state in oracle.results.items() if len(path) == length]
    before = str(oracle.build.model.proto)
    arguments = dict(conditional_actions={}, maximum_seconds=10, cancelled=None)
    for threshold in (min(values), min(values) + 1, max(values) + 1):
        found, value, identity = oracle.find_path_below(threshold, **arguments, feasibility_only=True)
        assert (found is not None) == any(v < threshold for v in values)
        if found is not None:
            assert value in values and value < threshold
        assert identity != oracle.find_path_below(threshold, **arguments)[2]
    assert str(oracle.build.model.proto) == before
    for invalid in (1, None, "true"):
        with pytest.raises(TypeError, match="feasibility selection"):
            oracle.find_path_below(0, **arguments, feasibility_only=invalid)
    with pytest.raises(ValueError, match="cannot maximize"):
        oracle.find_path_below(0, **arguments, feasibility_only=True, maximize_utility=True)
    from ortools.sat.python import cp_model
    from baloot_platform.research.ortools_play_tail import PlayTailSolveStopped
    monkeypatch.setattr(ortools_complete_state_graph, "_solve_cp_sat_with_cancel_v1",
        lambda *args: cp_model.UNKNOWN)
    with pytest.raises(PlayTailSolveStopped):
        oracle.find_path_below(0, **arguments, feasibility_only=True)


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
def test_native_adverse_query_matches_exhaustive_paths_without_mutating_model(rules, bounded, maximize, monkeypatch):
    compiler = _compiler(rules, native_conversion=True)
    start = compiler._parse_start(_seed(rules), expected_rules_hash=rules.sha256)
    oracle = compiler._path_oracle(start, target=Team.NORTH_SOUTH, utility="awarded_point_margin",
        **({"play_step_count": 4, "boundary_evaluator": "current_raw_point_margin_v1"} if bounded else {}))
    assert oracle.enumerate_paths(maximum_nodes=2000, maximum_seconds=10, cancelled=None) is None
    length = max(map(len, oracle.results))
    endpoints = {path: state.boundary_utility if bounded else state.terminal_utility
                 for path, state in oracle.results.items() if len(path) == length}
    lower, upper = oracle.query_utility_bounds()
    assert all(lower <= value <= upper for value in endpoints.values())
    witness = next(iter(endpoints))
    original_model = str(oracle.build.model.proto)
    import baloot_platform.research.ortools_complete_state_graph as graph_module
    original_solve = graph_module._solve_cp_sat_with_cancel_v1
    seen_limits = []
    def limited(solver, *args):
        seen_limits.append(solver.parameters.max_number_of_conflicts)
        return original_solve(solver, *args)
    with monkeypatch.context() as patch:
        patch.setattr(graph_module, "_solve_cp_sat_with_cancel_v1", limited)
        oracle.find_path_below(upper + 1, conditional_actions={}, maximum_seconds=10,
            cancelled=None, maximum_conflicts=5000)
    assert seen_limits == [5000]
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="conflict limit"):
            oracle.find_path_below(upper + 1, conditional_actions={}, maximum_seconds=10,
                cancelled=None, maximum_conflicts=invalid)
    for guards in ({}, {witness[:1]: witness[1]}, {(): witness[0], witness[:1]: witness[1]},
                   {witness[:1]: None}, {witness[:2]: None, (): witness[0]}, {(): None}):
        eligible = {path: value for path, value in endpoints.items()
                    if all(path[:len(prefix)] != prefix or (action is not None and path[len(prefix)] == action)
                           for prefix, action in guards.items())}
        minimum = min(eligible.values(), default=upper)
        for threshold in (minimum, minimum + 1, upper + 1):
            found, solved_utility, query_hash = oracle.find_path_below(threshold, conditional_actions=guards,
                maximum_seconds=10, cancelled=None, maximize_utility=maximize)
            assert (found is not None) == any(value < threshold for value in eligible.values())
            if found is not None:
                assert found.model_hash == query_hash
                assert eligible[tuple(found.plays[oracle.base_length:])] == solved_utility
                assert solved_utility == (max if maximize else min)(
                    value for value in eligible.values() if value < threshold)
                exhaustive = oracle.results
                oracle.results = None
                with monkeypatch.context() as patch:
                    patch.setattr("baloot_platform.research.ortools_complete_state_graph._solve_cp_sat_with_cancel_v1",
                                  lambda *args, **kwargs: pytest.fail("queried witness was solved again"))
                    projected, prefix = oracle.solve(tuple(found.plays[oracle.base_length:]),
                        maximum_seconds=10, cancelled=None)
                oracle.results = exhaustive
                assert prefix == found.plays
                assert projected.remaining_hands == found.at_prefix(len(prefix)).remaining_hands
            else:
                assert solved_utility is None
            assert str(oracle.build.model.proto) == original_model
        _, _, reordered_hash = oracle.find_path_below(upper + 1,
            conditional_actions=dict(reversed(tuple(guards.items()))), maximum_seconds=10,
            cancelled=None, maximize_utility=maximize)
        assert reordered_hash == query_hash
    _, _, minimum_hash = oracle.find_path_below(upper + 1, conditional_actions={},
        maximum_seconds=10, cancelled=None)
    _, _, maximum_hash = oracle.find_path_below(upper + 1, conditional_actions={},
        maximum_seconds=10, cancelled=None, maximize_utility=True)
    assert minimum_hash != maximum_hash
    for invalid in (None, 0, 1, "true"):
        with pytest.raises(TypeError, match="objective selection"):
            oracle.find_path_below(upper + 1, conditional_actions={}, maximum_seconds=10,
                cancelled=None, maximize_utility=invalid)
    for status in (graph_module.cp_model.UNKNOWN, graph_module.cp_model.FEASIBLE):
        with monkeypatch.context() as patch:
            patch.setattr(graph_module, "_solve_cp_sat_with_cancel_v1", lambda *args: status)
            with pytest.raises(graph_module.PlayTailSolveStopped):
                oracle.find_path_below(upper + 1, conditional_actions={}, maximum_seconds=10,
                    cancelled=None, maximize_utility=maximize)
    with pytest.raises(ValueError, match="actor disagrees"):
        oracle.find_path_below(1000, conditional_actions={():
            replace(witness[0], actor=Seat((int(witness[0].actor) + 1) % 4))},
            maximum_seconds=10, cancelled=None)
    for invalid_seconds in (float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="finite"):
            oracle.find_path_below(0, conditional_actions={},
                maximum_seconds=invalid_seconds, cancelled=None)
    assert str(oracle.build.model.proto) == original_model


@pytest.mark.parametrize("target", tuple(Team))
def test_uniform_native_bounds_skip_only_score_invariant_worlds(root, rules, target, monkeypatch):
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1)
    from baloot_platform.research.scenario_contracts import NativeObservedProofLimitsV1

    compiler = _compiler(rules, native_conversion=True)
    raw = json.loads((root / "examples/research-scenario-v4-golden.json").read_text(encoding="utf-8"))
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(raw["exactTracePrefix"]), decision_point={"kind": "play"}, maximum_seconds=30)
    start = compiler._parse_start(exact, expected_rules_hash=rules.sha256)
    oracle = compiler._path_oracle(start, target=target, utility="awarded_point_margin")
    assert oracle.enumerate_paths(maximum_nodes=2000, maximum_seconds=10, cancelled=None) is None
    by_action = {}
    for path, row in oracle.results.items():
        if row.terminal_utility is not None:
            by_action.setdefault(path[0].card.code, set()).add(row.terminal_utility)
    selected = next(action for action, values in sorted(by_action.items()) if len(values) == 1)
    expected = next(iter(by_action[selected]))
    # Different private recall worlds; every continuation is invariant in both.
    other_actor = next(seat.name.lower() for seat in Seat if seat.name.lower() != exact.next_actor)
    worlds = [(exact.state, oracle, {}), (exact.state, oracle, {other_actor: ("0" * 64,)})]
    service = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    limits = NativeObservedProofLimitsV1.build({"schemaVersion": 1,
        "maximumIterations": 10, "maximumQueries": 100, "maximumObservations": 1000,
        "maximumBoundPrefixes": 1000, "maximumCachedPrefixesPerBranch": 1000})
    originals = [str(oracle.build.model.proto) for _, oracle, _ in worlds]
    with monkeypatch.context() as patch:
        for _, oracle, _ in worlds:
            patch.setattr(oracle, "enumerate_paths", lambda **_: pytest.fail("uniform proof enumerated paths"))
        result = service._solve_observed_worlds(worlds, target, native_limits=limits,
            maximum_seconds=30, selected_root_action=selected)
        assert result["proofMethod"] == "uniform_native_path_bounds_v1"
        assert result["strategyCoverage"] == "all_legal_continuations_value_invariant"
        assert result["value"] == expected and result["worldCount"] == 2
        assert len(result["queries"]) == result["resources"]["usage"]["counterexampleQueries"] == 4
        assert all(not cut["targetActions"] for cut in result["cuts"])
        for query in result["queries"]:
            assert query["solverStatus"] == "optimal"
            assert query["sha256"] == canonical_sha256({k: v for k, v in query.items() if k != "sha256"})
        assert result["sha256"] == canonical_sha256({k: v for k, v in result.items() if k != "sha256"})
        repeated = service._solve_observed_worlds(list(reversed(worlds)), target, native_limits=limits,
            maximum_seconds=30, selected_root_action=selected)
        assert repeated == result
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            service._solve_observed_worlds(worlds, target,
                native_limits=replace(limits, maximum_queries=3), maximum_seconds=30, selected_root_action=selected)
        assert stopped.value.reason == "maximum_queries"
        assert len(stopped.value.partial_receipt["evidence"]["queryHashes"]) == 3
        original_query = oracle.find_path_below
        def incomplete_maximum(*args, **kwargs):
            if kwargs.get("maximize_utility"):
                raise ortools_complete_state_graph.PlayTailSolveStopped("unknown", evidence={
                    "queryModelHash": "f" * 64, "baseModelHash": oracle.model_hash, "objective": "maximum"})
            return original_query(*args, **kwargs)
        patch.setattr(oracle, "find_path_below", incomplete_maximum)
        with pytest.raises(InformationSetCompilationStopped) as incomplete:
            service._solve_observed_worlds(worlds, target, native_limits=limits,
                maximum_seconds=30, selected_root_action=selected)
        assert incomplete.value.reason == "unknown"
        partial = incomplete.value.partial_receipt["evidence"]
        assert len(partial["queries"]) == partial["resources"]["usage"]["counterexampleQueries"] == 2
        assert partial["queries"][-1]["status"] == "incomplete"
        assert "value" not in partial["queries"][-1]
    assert [str(oracle.build.model.proto) for _, oracle, _ in worlds] == originals


@pytest.mark.parametrize("prefix_length,ekka,expected", [
    (19, False, 1296), (19, True, 10368),
    (18, False, 5184), (16, True, 663552), (15, False, 331776), (0, False, 1000001),
])
def test_remaining_path_forecast_uses_card_orders_and_ekka(prefix_length, ekka, expected):
    from baloot_platform.research.ortools_information_set_compiler import _remaining_native_path_bound_v1

    cards = tuple(Card(index) for index in range(32))
    ownership = {card: Seat(index % 4) for index, card in enumerate(cards)}
    root_ekka = ekka and prefix_length % 4 == 0
    oracle = SimpleNamespace(start=SimpleNamespace(cards=cards, ownership=ownership,
        base_prefix=tuple(TailPrefixActionV1(ownership[card], card, False) for card in cards[:prefix_length]),
        expected_legal_actions=(TailPrefixActionV1(ownership[cards[prefix_length]], cards[prefix_length], root_ekka),)),
        base_length=prefix_length,
        binary_ekka_steps=set(range(0, 32, 4)) if ekka else set())
    selected = cards[prefix_length].code + ("+ekka" if root_ekka else "")
    assert _remaining_native_path_bound_v1(oracle, selected, 1000000) == expected
    if expected <= 1000000:
        assert _remaining_native_path_bound_v1(oracle, selected, expected - 1) == expected


def test_midtrick_late_round_uses_existing_native_exhaustive_proof(root, rules, monkeypatch):
    measured_seconds = 900.0  # Sulaiman p99 542.366s; capped at the 900s file lane.
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import NativeObservedProofLimitsV1

    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    raw["playActions"] = raw["playActions"][:19]
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(raw), decision_point={"kind": "play"}, maximum_seconds=measured_seconds)
    compiler = _compiler(rules, native_conversion=True)
    target = Seat.parse(exact.next_actor).team
    start = compiler._parse_start(exact, expected_rules_hash=rules.sha256)
    oracle = compiler._path_oracle(start, target=target, utility="awarded_point_margin")
    assert len(start.cards) == 16 and oracle.base_length == 3
    selected = next(action for action in start.expected_legal_actions if action.card.code == "DJ")
    assert oracle.enumerate_paths(maximum_nodes=20000, maximum_seconds=measured_seconds, cancelled=None,
        root_action=selected.card.code) is None
    assert len({node.terminal_utility for node in oracle.results.values()
        if node.terminal_utility is not None}) > 1

    def minimax(path):
        node = oracle.results[path]
        if node.terminal_utility is not None:
            return node.terminal_utility
        return (max if node.next_actor.team == target else min)(
            minimax((*path, action)) for action in node.legal_play_actions)

    expected = minimax((selected,))
    observer = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    monkeypatch.setattr(observer, "_find_observed_policy_counterexample",
        lambda *args, **kwargs: pytest.fail("small residual tail reached CEGIS"))
    limits = NativeObservedProofLimitsV1.build(dict(schemaVersion=1, maximumIterations=100,
        maximumQueries=20000, maximumObservations=20000, maximumBoundPrefixes=20000,
        maximumCachedPrefixesPerBranch=20000))
    proof = observer._solve_observed_worlds([(exact.state, oracle, {})], target,
        native_limits=limits, maximum_seconds=measured_seconds, selected_root_action=selected.card.code)
    assert proof["proofMethod"] == "exhaustive_native_paths_v1" and proof["value"] == expected
    assert proof["scope"]["utilityScope"]["cardCount"] == 16
    assert proof["sha256"] == canonical_sha256({key: value for key, value in proof.items() if key != "sha256"})
    # Each world's 1,296 orders fit, but their sum exceeds this shared budget.
    class CegisFallbackExpected(Exception):
        pass

    def stop_cegis(*args, **kwargs):
        raise CegisFallbackExpected

    monkeypatch.setattr(observer, "_find_observed_policy_counterexample", stop_cegis)
    other_actor = next(seat.name.lower() for seat in Seat if seat.name.lower() != exact.next_actor)
    with pytest.raises(CegisFallbackExpected):
        observer._solve_observed_worlds([(exact.state, oracle, {}),
            (exact.state, oracle, {other_actor: ("0" * 64,)})], target,
            native_limits=replace(limits, maximum_queries=2000), maximum_seconds=measured_seconds,
            selected_root_action=selected.card.code)


def test_nonuniform_native_bounds_keep_exhaustive_fallback_and_budget(root, rules, monkeypatch):
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1)
    from baloot_platform.research.scenario_contracts import NativeObservedProofLimitsV1

    raw = json.loads((root / "examples/research-scenario-v4-unique-saving.json").read_text(encoding="utf-8"))
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(raw["exactTracePrefix"]), decision_point={"kind": "play"}, maximum_seconds=30)
    compiler = _compiler(rules, native_conversion=True)
    target = Seat.parse(exact.next_actor).team
    start = compiler._parse_start(exact, expected_rules_hash=rules.sha256)
    oracle = compiler._path_oracle(start, target=target, utility="awarded_point_margin")
    assert oracle.enumerate_paths(maximum_nodes=2000, maximum_seconds=10, cancelled=None) is None
    endpoints = {}
    for path, node in oracle.results.items():
        if node.terminal_utility is not None:
            endpoints.setdefault(path[0], set()).add(node.terminal_utility)
    action = next(action for action, values in endpoints.items() if len(values) > 1)
    def minimax(path):
        node = oracle.results[path]
        if node.terminal_utility is not None:
            return node.terminal_utility
        return (max if node.next_actor.team == target else min)(
            minimax((*path, child)) for child in node.legal_play_actions)
    expected = minimax((action,))
    limits = NativeObservedProofLimitsV1.build({"schemaVersion": 1,
        "maximumIterations": 10, "maximumQueries": 1000, "maximumObservations": 1000,
        "maximumBoundPrefixes": 1000, "maximumCachedPrefixesPerBranch": 1000})
    result = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._solve_observed_worlds(
        [(exact.state, oracle, {})], target, native_limits=limits,
        maximum_seconds=30, selected_root_action=action.card.code)
    assert result["proofMethod"] == "exhaustive_native_paths_v1" and result["value"] == expected
    assert [query["kind"] for query in result["queries"][:2]] == ["native_path_extremum_v1"] * 2
    assert result["queries"][0]["value"] < result["queries"][1]["value"]
    assert result["resources"]["usage"]["counterexampleQueries"] == len(result["queryHashes"])
    with monkeypatch.context() as patch:
        patch.setattr("baloot_platform.research.ortools_information_set_proof.OrToolsActorInformationSetProofV1._solve_master",
            lambda *args, **kwargs: pytest.fail("exhausted query budget launched another master"))
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._solve_observed_worlds(
                [(exact.state, oracle, {})], target, native_limits=replace(limits, maximum_queries=2),
                maximum_seconds=30, selected_root_action=action.card.code)
    assert stopped.value.reason == "maximum_queries"
    partial = stopped.value.partial_receipt["evidence"]
    assert len(partial["queries"]) == 2 and partial["masterHashes"] == []


@pytest.mark.parametrize("trump", [None, Suit.CLUBS])
@pytest.mark.parametrize("length", [2, 3])
@pytest.mark.parametrize("previously_broken", [False, True])
def test_dynamic_observation_retains_new_public_trump_cut(rules, trump, length, previously_broken):
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    state = copy.deepcopy(_normalized_state(rules).state)
    state["trumpBroken"] = previously_broken
    state["trumpSuit"] = trump.name.lower() if trump is not None else None
    state["contract"] = "hokum" if trump is not None else "sun"
    owners = {Card.parse(card): seat for seat in Seat for card in state["hands"][seat.name.lower()]}
    path = tuple(TailPrefixActionV1.build(seat, card) for seat, card in
        ((Seat.SOUTH, "S7"), (Seat.EAST, "S9"), (Seat.NORTH, "C7")))[:length]
    build = OrToolsPlayTailStageV1().build_model(cards=tuple(owners), ownership=owners,
        contract=state["contract"], trump=trump, first_leader=Seat.SOUTH,
        play_domain=rules.play_domain, structural_domain=rules.structural_domain, prefix=path)
    result = build.solve(maximum_time_seconds=5).at_prefix(length)
    original = copy.deepcopy(state)
    observer = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    view = observer._dynamic_state(state, SimpleNamespace(base_prefix=(), first_leader=Seat.SOUTH,
        trump=trump), result, path)
    assert view["trumpBroken"] is (previously_broken or (trump is not None and length == 3))
    assert state == original


def test_dynamic_observation_shares_read_only_history_without_changing_hash(root, rules):
    from baloot_platform.core.exact_state_view import canonical_observer_information_hash_v1
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(raw), decision_point={"kind": "play"}, maximum_seconds=30)
    observer = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    start = observer.graph_compiler._parse_start(exact, expected_rules_hash=rules.sha256)
    oracle = observer.graph_compiler._path_oracle(start, target=Seat.parse(exact.next_actor).team,
        utility="awarded_point_margin")
    original = copy.deepcopy(exact.state)
    for action in start.expected_legal_actions:
        path = (action,)
        result, _ = oracle.solve(path, maximum_seconds=10, cancelled=None)
        view = observer._dynamic_state(exact.state, start, result, path)
        assert view["events"] is exact.state["events"]
        assert view["projects"] is exact.state["projects"]
        for field in ("hands", "currentTrick", "completedTricks"):
            assert view[field] is not exact.state[field]
        expected = canonical_observer_information_hash_v1(
            complete_state=copy.deepcopy(view), acting_seat=result.next_actor.name.lower())
        assert observer._path_observation_id(exact.state, start, result, path, {}) == expected
        assert exact.state == original


def test_native_query_deltas_retain_linear_history(tmp_path):
    from itertools import islice, permutations
    import time
    from baloot_platform.research.ortools_complete_state_graph import iter_native_path_query_guards_v1
    from baloot_platform.research.scenario_contracts import canonical_json

    started = time.monotonic()
    history, cumulative_guard_bytes, expanded_bytes = [], 0, 0
    for index, cards in enumerate(islice(permutations(range(32), 3), 2327)):
        addition = {"prefix": [TailPrefixActionV1.build(seat, card).to_dict()
            for seat, card in zip((Seat.NORTH, Seat.WEST, Seat.SOUTH), cards, strict=True)],
            "action": TailPrefixActionV1.build(Seat.EAST, next(card for card in range(32) if card not in cards)).to_dict()}
        row = {"schemaVersion": 2, "kind": "native_observed_path_query_v2",
            "baseModelHash": "a" * 64, "queryModelHash": canonical_sha256(index),
            "threshold": 1, "status": "witness_found", "utility": 0,
            "previousQueryReceiptHash": history[-1]["sha256"] if history else None,
            "conditionalActionCount": index + 1, "conditionalActionsAdded": [addition]}
        history.append({**row, "sha256": canonical_sha256(row)})
        cumulative_guard_bytes += len(canonical_json(addition).encode("utf-8"))
        legacy = {key: value for key, value in history[-1].items() if key not in {
            "kind", "previousQueryReceiptHash", "conditionalActionCount", "conditionalActionsAdded"}}
        legacy.update(schemaVersion=1, conditionalActions=[])
        # Exact serialized-size arithmetic; never allocate the quadratic legacy history.
        expanded_bytes += len(canonical_json(legacy).encode("utf-8")) + cumulative_guard_bytes + index
    compact_bytes = len(canonical_json(history).encode("utf-8"))
    for _, guards in iter_native_path_query_guards_v1(history, maximum_bindings=2327):
        pass
    assert len(guards) == sum(len(row["conditionalActionsAdded"]) for row in history) == 2327
    assert expanded_bytes > 100 * compact_bytes
    altered = copy.deepcopy(history[-1])
    altered["conditionalActionsAdded"] = history[0]["conditionalActionsAdded"]
    altered["sha256"] = canonical_sha256({k: v for k, v in altered.items() if k != "sha256"})
    with pytest.raises(ValueError, match="repeated or replaced"):
        list(iter_native_path_query_guards_v1([*history[:-1], altered], maximum_bindings=2328))
    with pytest.raises(ValueError, match="binding cap"):
        list(iter_native_path_query_guards_v1(history, maximum_bindings=2326))
    report = {"kind": "synthetic_query_retention_benchmark_v1", "queries": len(history),
        "compactBytes": compact_bytes, "expandedLegacyBytes": expanded_bytes,
        "elapsedSeconds": time.monotonic() - started, "strategicProof": False}
    (tmp_path / "query-retention-benchmark.json").write_text(json.dumps(report, sort_keys=True), encoding="utf-8")


def test_native_minimum_counterexample_receipt_roundtrip(root, rules):
    from baloot_platform.research.scenario_contracts import NativeObservedProofLimitsV1
    from baloot_platform.research.ortools_information_set_compiler import (
        OrToolsActorInformationSetCompilerV1, InformationSetCompilationStopped)
    from baloot_platform.research.ortools_complete_state_graph import iter_native_path_query_guards_v1
    legacy = {"schemaVersion": 1, "maximumIterations": 100, "maximumQueries": 100,
        "maximumObservations": 1000, "maximumBoundPrefixes": 1000,
        "maximumCachedPrefixesPerBranch": 1000}
    assert NativeObservedProofLimitsV1.build(legacy).to_dict() == legacy
    configured = {**legacy, "schemaVersion": 2, "counterexampleObjective": "minimum_utility"}
    assert NativeObservedProofLimitsV1.build(configured).to_dict() == configured
    for invalid in (None, True, "unknown", "feasibility"):
        with pytest.raises(ValueError):
            NativeObservedProofLimitsV1.build({**configured, "counterexampleObjective": invalid})
    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(raw), decision_point={"kind": "play"}, maximum_seconds=30)
    compiler = _compiler(rules, native_conversion=True)
    target = Seat.parse(exact.next_actor).team
    oracle = compiler._path_oracle(compiler._parse_start(exact, expected_rules_hash=rules.sha256),
        target=target, utility="awarded_point_margin")
    observer = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    def query(threshold, **kwargs):
        return observer._find_observed_policy_counterexample(exact.state, oracle, target, {}, {},
            threshold=threshold, known_domains={}, maximum_queries=100,
            maximum_prefixes=1000, maximum_seconds=10,
            counterexample_objective="minimum_utility", **kwargs)
    for threshold in (1000, -1000):
        result = query(threshold)
        for receipt, guards in iter_native_path_query_guards_v1(result["queries"], maximum_bindings=1000):
            assert receipt["schemaVersion"] == 3
            witness, utility, model_hash = oracle.find_path_below(threshold,
                conditional_actions=guards, maximum_seconds=10, cancelled=None, feasibility_only=False)
            assert model_hash == receipt["queryModelHash"]
            assert utility == receipt["utility"]
            assert (witness is None) == (receipt["status"] == "infeasible")
            altered = {**receipt, "counterexampleObjective": "feasibility"}
            altered["sha256"] = canonical_sha256({k: v for k, v in altered.items() if k != "sha256"})
            with pytest.raises(ValueError, match="objective"):
                list(iter_native_path_query_guards_v1([altered], maximum_bindings=1000))
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        query(1000, cancel_check=lambda: True)
    assert stopped.value.reason == "cancelled"


@pytest.mark.real_clock
def test_native_witness_policy_binding_keeps_unknown_observations_unbound(root, rules, monkeypatch):
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.ortools_complete_state_graph import iter_native_path_query_guards_v1
    compiler = _compiler(rules, native_conversion=True)
    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(raw), decision_point={"kind": "play"}, maximum_seconds=30)
    start = compiler._parse_start(exact, expected_rules_hash=rules.sha256)
    target = Seat.parse(exact.next_actor).team
    oracle = compiler._path_oracle(start, target=target, utility="awarded_point_margin")
    witness, _, _ = oracle.find_path_below(1000, conditional_actions={}, maximum_seconds=10, cancelled=None)
    observer = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    state = exact.state
    bindings, domains, _ = observer._observed_policy_bindings(
        state, oracle, witness, target, {}, {}, known_domains={})
    assert bindings == {} and domains
    policy = {observation: actions[0] for observation, actions in domains.items()}
    bindings, known_domains, _ = observer._observed_policy_bindings(
        state, oracle, witness, target, policy, {}, known_domains=domains)
    assert domains == known_domains and () in bindings
    next_witness, _, _ = oracle.find_path_below(1000, conditional_actions=bindings,
        maximum_seconds=10, cancelled=None)
    assert next_witness.plays[oracle.base_length] == bindings[()]
    repeated, _, _ = observer._observed_policy_bindings(
        state, oracle, next_witness, target, policy, {}, known_domains=domains)
    assert repeated[()] == bindings[()]
    with pytest.raises(ValueError, match="illegal action"):
        observer._observed_policy_bindings(state, oracle, witness, target,
            {observation: "invalid" for observation in domains}, {}, known_domains=domains)
    prior_domains = dict(domains)
    conflicting = {**domains, next(iter(domains)): ("invalid",)}
    conflict_before = dict(conflicting)
    with pytest.raises(ValueError, match="observation_action_domain_mismatch"):
        observer._observed_policy_bindings(state, oracle, witness, target, policy, {},
            known_domains=conflicting)
    assert conflicting == conflict_before and domains == prior_domains
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    def query(policy_value, threshold=1000, **limits):
        return observer._find_observed_policy_counterexample(state, oracle, target, policy_value, {},
            threshold=threshold, known_domains=domains, maximum_queries=limits.get("queries", 100),
            maximum_prefixes=limits.get("prefixes", 1000), maximum_seconds=limits.get("seconds", 10),
            maximum_evidence_bytes=limits.get("evidence"),
            cancel_check=limits.get("cancel"))
    pending = query({})
    assert pending["status"] == "policy_extension_required"
    absent = query({}, threshold=-1000)
    assert absent["status"] == "no_counterexample" and absent["witness"] is None
    assert len(absent["queryHashes"]) == 1
    with pytest.raises(InformationSetCompilationStopped) as oversized:
        query({}, threshold=-1000, evidence=1)
    assert oversized.value.reason == "maximum_evidence_bytes"
    rejected = oversized.value.partial_receipt["evidence"]
    assert len(rejected["queryHashes"]) == 1 and rejected["queries"] == []
    assert rejected["queryEvidenceBytes"] == 0
    assert rejected["causeReceipt"]["candidateBytes"] > 1
    with pytest.raises(ValueError, match="evidence limit"):
        query({}, evidence=True)
    for _ in range(50):
        outcome = query(policy)
        if outcome["status"] != "policy_extension_required":
            break
        policy.update({observation: actions[0] for observation, actions in outcome["domains"].items()})
    assert outcome["status"] == "counterexample"
    selected, _, witnessed_actions = observer._observed_policy_bindings(state, oracle, outcome["witness"], target,
        policy, {}, known_domains=outcome["domains"])
    assert all(outcome["witness"].plays[oracle.base_length + len(prefix)] == action
               for prefix, action in selected.items())
    assert outcome["utility"] < 1000
    for result in (pending, absent, outcome):
        assert tuple(row["queryModelHash"] for row in result["queries"]) == result["queryHashes"]
        # Reconstruct the query from retained inputs, not a saved truth constant.
        legacy = []
        for receipt, guards in iter_native_path_query_guards_v1(result["queries"], maximum_bindings=1000):
            assert receipt["baseModelHash"] == oracle.model_hash
            found, utility, model_hash = oracle.find_path_below(receipt["threshold"],
                conditional_actions=guards, maximum_seconds=10, cancelled=None, feasibility_only=True)
            assert model_hash == receipt["queryModelHash"]
            assert (found is None) == (receipt["status"] == "infeasible")
            assert utility == receipt["utility"]
            old = {key: value for key, value in receipt.items() if key not in {
                "sha256", "kind", "previousQueryReceiptHash", "conditionalActionCount", "conditionalActionsAdded"}}
            old.update(schemaVersion=1, conditionalActions=[{"prefix": [step.to_dict() for step in prefix],
                "action": choice.to_dict() if choice is not None else None} for prefix, choice in guards.items()])
            legacy.append({**old, "sha256": canonical_sha256(old)})
        assert list(iter_native_path_query_guards_v1(legacy, maximum_bindings=1000))[-1][1] == guards
        for field, value in (("previousQueryReceiptHash", "0" * 64), ("conditionalActionCount", -1),
                              ("baseModelHash", "forged"), ("unknown", True)):
            altered = copy.deepcopy(result["queries"])
            altered[-1][field] = value
            altered[-1]["sha256"] = canonical_sha256({k: v for k, v in altered[-1].items() if k != "sha256"})
            with pytest.raises(ValueError, match="native query"):
                list(iter_native_path_query_guards_v1(altered, maximum_bindings=1000))
    assert outcome["targetActions"] == witnessed_actions
    assert all(policy[observation] == action for observation, action in witnessed_actions)
    _, _, with_unreached = observer._observed_policy_bindings(state, oracle, outcome["witness"], target,
        {**policy, "unreached": "CA"}, {},
        known_domains={**outcome["domains"], "unreached": ("CA",)})
    assert with_unreached == witnessed_actions
    _, _, pending_actions = observer._observed_policy_bindings(state, oracle, pending["witness"], target,
        {}, {}, known_domains=pending["domains"])
    assert pending["targetActions"] == pending_actions and pending_actions
    assert absent["targetActions"] == ()
    prefix_observations = {}
    changed_policy = {observation: actions[-1] for observation, actions in pending["domains"].items()}
    expected_bindings, _, _ = observer._observed_policy_bindings(state, oracle, pending["witness"], target,
        changed_policy, {}, known_domains=pending["domains"], _prefix_observations=prefix_observations)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(observer, "_path_observation_id", lambda *args: pytest.fail("cached observation was rebuilt"))
        cached_bindings, cached_domains, cached_actions = observer._observed_policy_bindings(
            state, oracle, pending["witness"], target, changed_policy, {},
            known_domains=pending["domains"], _known_prefix_observations=prefix_observations)
        assert cached_bindings == expected_bindings and cached_domains == pending["domains"]
        assert cached_actions == pending_actions
    recorded_bindings = []
    query_path = oracle.find_path_below
    def record_query(*args, **kwargs):
        recorded_bindings.append(dict(kwargs["conditional_actions"]))
        return query_path(*args, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(oracle, "find_path_below", record_query)
        observer._find_observed_policy_counterexample(state, oracle, target, changed_policy, {},
            threshold=1000, known_domains=pending["domains"], maximum_queries=100,
            maximum_prefixes=1000, maximum_seconds=10, _prefix_observations=prefix_observations)
    assert recorded_bindings[0] == expected_bindings and expected_bindings
    assert len(prefix_observations) <= 1000
    from baloot_platform.research.ortools_information_set_proof import (
        OrToolsActorInformationSetProofV1, _Catalog, _CounterexampleCut)
    import time
    pending_cut = _CounterexampleCut(pending["utility"], pending["targetActions"])
    pending_catalog = _Catalog({("target", observation): actions
                               for observation, actions in pending["domains"].items()},
                              len(pending["domains"]), (pending["utility"], pending["utility"] + 1))
    pending_master = OrToolsActorInformationSetProofV1()._solve_master(
        oracle.model_hash, "new-observation-cut", pending_catalog, [pending_cut],
        time.monotonic() + 10, None, fixed_actions=dict(pending_actions))
    assert pending_master.status == "optimal" and pending_master.value == pending["utility"]
    # This tests one native cut, not convergence or a whole-game proof.
    cut = _CounterexampleCut(outcome["utility"], outcome["targetActions"])
    catalog = _Catalog({("target", observation): actions
                        for observation, actions in outcome["domains"].items()},
                       len(outcome["domains"]), (outcome["utility"], outcome["utility"] + 1))
    scope_hash = canonical_sha256({"nativeModel": oracle.model_hash,
        "queryHashes": outcome["queryHashes"], "utilityBounds": catalog.utilities})
    master = OrToolsActorInformationSetProofV1()._solve_master(scope_hash, "policy-cut-test",
        catalog, [cut], time.monotonic() + 10, None)
    assert master.status == "optimal" and master.value == outcome["utility"] + 1
    assert any(dict(master.policy)[observation] != action
               for observation, action in outcome["targetActions"])
    with pytest.raises(InformationSetCompilationStopped) as cancelled:
        query(policy, cancel=lambda: True)
    assert cancelled.value.reason == "cancelled"
    root_observation = observer._path_observation_id(state, start,
        oracle._project(witness.at_prefix(oracle.base_length), oracle.base_length), (), {})
    root_action = witness.plays[oracle.base_length]
    root_code = root_action.card.code + ("+ekka" if root_action.ekka else "")
    different = next(code for code in domains[root_observation] if code != root_code)
    with pytest.raises(InformationSetCompilationStopped) as exhausted:
        query({**policy, root_observation: different}, queries=1)
    assert exhausted.value.reason == "maximum_queries"
    with pytest.raises(InformationSetCompilationStopped) as exhausted:
        query(policy, prefixes=1)
    assert exhausted.value.reason == "maximum_observations"
    with pytest.raises(InformationSetCompilationStopped) as exhausted:
        query(policy, seconds=0)
    assert exhausted.value.reason == "maximum_seconds"
    solved = observer._solve_observed_worlds([(state, oracle, {})], target,
        maximum_iterations=100, maximum_queries=500, maximum_prefixes=1000, maximum_seconds=30)
    assert solved["status"] == "converged" and solved["worldCount"] == 1
    from baloot_platform.research.scenario_contracts import NativeObservedProofLimitsV1
    native_limits = NativeObservedProofLimitsV1.build({"schemaVersion": 1,
        "maximumIterations": 100, "maximumQueries": 500, "maximumObservations": 1000,
        "maximumBoundPrefixes": 1000, "maximumCachedPrefixesPerBranch": 1})
    native_solved = observer._solve_observed_worlds([(state, oracle, {})], target,
        native_limits=native_limits, maximum_seconds=30)
    assert native_solved["value"] == solved["value"]
    assert native_solved["proofMethod"] == "exhaustive_native_paths_v1"
    assert native_solved["resources"]["usage"]["counterexampleQueries"] == len(
        native_solved["queryHashes"])
    assert all(row["kind"] == "native_exhaustive_path_enumeration_v1"
               and row["solverStatus"] == "optimal"
               and row["sha256"] == canonical_sha256(
                   {key: value for key, value in row.items() if key != "sha256"})
               for row in native_solved["queries"])
    from baloot_platform.research.ortools_play_tail import PlayTailSolveStopped
    for phase in ("model_clone", "solver"):
        failed = {"queryPhase": phase, **({"queryModelHash": "a" * 64,
            "baseModelHash": oracle.model_hash, "objective": "feasibility"} if phase == "solver" else {})}
        def stop_native_query(*args, **kwargs):
            raise PlayTailSolveStopped("cancelled", evidence=failed)
        with monkeypatch.context() as patch:
            patch.setattr(oracle, "find_path_below", stop_native_query)
            with pytest.raises(InformationSetCompilationStopped) as stopped_query:
                observer._solve_observed_worlds([(state, oracle, {})], target,
                    maximum_iterations=100, maximum_queries=500, maximum_prefixes=1000, maximum_seconds=30)
        partial_query = stopped_query.value.partial_receipt["evidence"]
        cause = partial_query["causeReceipt"]["evidence"]
        expected_hashes = ["a" * 64] if phase == "solver" else []
        assert cause["queryHashes"] == expected_hashes
        assert partial_query["resources"]["usage"]["counterexampleQueries"] == len(expected_hashes)
        assert cause["causeReceipt"]["evidence"] == failed and cause["pathPhase"] == phase
        assert [row["queryModelHash"] for row in cause["queries"]] == expected_hashes
        assert all(row["status"] == "incomplete" and row["sha256"] == canonical_sha256(
            {key: value for key, value in row.items() if key != "sha256"}) for row in cause["queries"])
    for reason in ("cancelled", "maximum_seconds"):
        original_bindings = observer._observed_policy_bindings
        projection_calls = 0
        def stop_second_projection(*args, **kwargs):
            nonlocal projection_calls
            projection_calls += 1
            if projection_calls == 2:
                raise PlayTailSolveStopped(reason)
            return original_bindings(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(observer, "_observed_policy_bindings", stop_second_projection)
            with pytest.raises(InformationSetCompilationStopped) as stopped_projection:
                observer._solve_observed_worlds([(state, oracle, {})], target,
                    native_limits=native_limits, maximum_seconds=30)
        assert projection_calls == 2 and stopped_projection.value.reason == reason
        partial = stopped_projection.value.partial_receipt["evidence"]
        assert partial["causeReceipt"]["stage"] == "native_exhaustive_policy_projection"
        assert partial["cuts"] and partial["cutWitnesses"] and partial["domains"]
        assert partial["resources"]["usage"]["retainedCuts"] == len(partial["cuts"])
    golden_roots = observer._solve_observed_root_actions([(state, oracle, {})], target,
        native_limits=replace(native_limits, maximum_cached_prefixes_per_branch=1000), maximum_seconds=30)
    unique = observer._certify_observed_root_actions(golden_roots, success_threshold=0,
        claim="selected_action_is_unique_winner", maximum_seconds=10, action_quality=True)
    assert unique["status"] == "satisfied" and unique["winningRootActionCount"] == 1
    winner = next(row["action"] for row in unique["roots"] if row["wins"])
    assert winner.endswith("A")
    assert unique["actionQuality"]["actualAction"] == winner
    assert unique["actionQuality"]["regret"] == 0
    assert unique["actionQuality"]["actualActionOptimal"] is True
    other = next(action for action in golden_roots["legalActions"] if action != winner)
    wrong_selection = observer._certify_observed_root_actions(golden_roots, success_threshold=0,
        claim="selected_action_is_unique_winner", selected_action=other, maximum_seconds=10, action_quality=True)
    assert wrong_selection["status"] == "proof_failed" and wrong_selection["winningRootActionCount"] == 1
    assert wrong_selection["actionQuality"]["regret"] > 0
    assert wrong_selection["actionQuality"]["actualActionOptimal"] is False
    from ortools.sat.python import cp_model
    with monkeypatch.context() as patch:
        patch.setattr("baloot_platform.research.ortools_information_set_compiler._solve_cp_sat_with_cancel_v1",
                      lambda *args, **kwargs: cp_model.UNKNOWN)
        with pytest.raises(InformationSetCompilationStopped) as quality_stopped:
            observer._certify_observed_root_actions(golden_roots, success_threshold=0,
                claim="winning_root_action_count", maximum_seconds=10, action_quality=True)
    partial_quality = quality_stopped.value.partial_receipt["evidence"]
    assert partial_quality["sourceArtifactHash"] == golden_roots["sha256"]
    assert partial_quality["qualificationModelProto"] and partial_quality["bindings"]
    with monkeypatch.context() as patch:
        patch.setattr("baloot_platform.research.ortools_information_set_compiler._solve_cp_sat_with_cancel_v1",
                      lambda *args, **kwargs: cp_model.INFEASIBLE)
        quarantined = observer._certify_observed_root_actions(golden_roots, success_threshold=0,
            claim="winning_root_action_count", maximum_seconds=10, action_quality=True)
    assert quarantined["status"] == "model_error" and quarantined["claimSatisfied"] is None
    assert quarantined["actionQuality"]["modelStatus"] == "quarantined"
    assert native_solved["resources"]["limits"]["cachedPrefixesPerBranch"] == 1
    assert all(count <= 1 for count in native_solved["resources"]["usage"]["cachedPrefixesByBranch"].values())
    with pytest.raises(ValueError, match="cannot mix"):
        observer._solve_observed_worlds([(state, oracle, {})], target,
            native_limits=native_limits, maximum_queries=500, maximum_seconds=30)
    resource = solved["resources"]
    assert resource["kind"] == "native_observed_proof_resources_v1"
    assert resource["usage"]["counterexampleQueries"] == len(solved["queryHashes"])
    assert resource["usage"]["masterIterations"] == len(solved["masterHashes"])
    assert resource["usage"]["observations"] == len(solved["domains"])
    assert resource["usage"]["retainedCuts"] == len(solved["cutHashes"])
    assert all(count <= resource["limits"]["cachedPrefixesPerBranch"]
               for count in resource["usage"]["cachedPrefixesByBranch"].values())
    assert solved["sha256"] == canonical_sha256({key: value for key, value in solved.items() if key != "sha256"})
    assert solved["scopeHash"] == canonical_sha256(solved["scope"])
    assert solved["scope"]["coverage"] == "supplied_worlds_only"
    assert solved["scope"]["utilityScope"] == {"utility": oracle.utility, "boundaryEvaluator": None,
        "playStepCount": len(oracle.build.selected_play_indexes), "basePrefixLength": oracle.base_length,
        "cardCount": len(start.cards)}
    assert tuple(row["receipt"]["queryModelHash"] for row in solved["queries"]) == solved["queryHashes"]
    assert all(row["worldId"] in solved["scope"]["worlds"] for row in solved["queries"])
    assert tuple(canonical_sha256(cut) for cut in solved["cuts"]) == solved["cutHashes"]
    assert set(solved["cutWitnesses"]) == set(solved["cutHashes"])
    for cut_hash, witness_document in solved["cutWitnesses"].items():
        assert witness_document["cutHash"] == cut_hash
        assert witness_document["queryModelHash"] in solved["queryHashes"]
        assert witness_document["worldId"] in solved["scope"]["worlds"]
        assert witness_document["sourceStateHash"] == exact.state_hash
        assert witness_document["sha256"] == canonical_sha256(
            {key: value for key, value in witness_document.items() if key != "sha256"})
        plays = tuple(TailPrefixActionV1.build(row["actor"], row["card"], ekka=row["ekka"])
                      for row in witness_document["plays"])
        terminal, _ = oracle.solve(plays[oracle.base_length:], maximum_seconds=10, cancelled=None)
        assert terminal.terminal_utility == witness_document["utility"]
    tampered = {key: value for key, value in solved.items() if key != "sha256"}
    tampered["value"] += 1
    assert canonical_sha256(tampered) != solved["sha256"]
    assert oracle.enumerate_paths(maximum_nodes=2000, maximum_solutions=1,
                                  maximum_seconds=10, cancelled=None) == "maximum_solutions"
    assert oracle._enumerated_witnesses is None
    with pytest.raises(ValueError, match="positive integer"):
        oracle.enumerate_paths(maximum_nodes=2000, maximum_solutions=0,
                               maximum_seconds=10, cancelled=None)
    assert oracle.enumerate_paths(maximum_nodes=2000, maximum_seconds=10, cancelled=None) is None
    def reference(path):
        node = oracle.results[path]
        if node.terminal_utility is not None:
            return node.terminal_utility
        values = [reference((*path, action)) for action in node.legal_play_actions]
        return (max if node.next_actor.team == target else min)(values)
    assert solved["value"] == reference(())
    with pytest.raises(ValueError, match="visible root"):
        observer._solve_observed_worlds([(state, oracle, {}),
            (state, oracle, {exact.next_actor: ("0" * 64,)})], target,
            maximum_iterations=100, maximum_queries=500, maximum_prefixes=1000, maximum_seconds=30)
    other_actor = next(seat.name.lower() for seat in Seat if seat.name.lower() != exact.next_actor)
    worlds = [(state, oracle, {}), (state, oracle, {other_actor: ("0" * 64,)})]
    two_worlds = observer._solve_observed_worlds(worlds, target,
        maximum_iterations=100, maximum_queries=500, maximum_prefixes=1000, maximum_seconds=30)
    reordered = observer._solve_observed_worlds(list(reversed(worlds)), target,
        maximum_iterations=100, maximum_queries=500, maximum_prefixes=1000, maximum_seconds=30)
    assert two_worlds == reordered
    assert two_worlds["value"] == solved["value"] and two_worlds["worldCount"] == 2
    with pytest.raises(InformationSetCompilationStopped) as exhausted:
        observer._solve_observed_worlds([(state, oracle, {})], target,
            maximum_iterations=100, maximum_queries=1, maximum_prefixes=1000, maximum_seconds=30)
    assert exhausted.value.reason == "maximum_queries"
    assert exhausted.value.partial_receipt["evidence"]["queryHashes"]
    assert exhausted.value.partial_receipt["evidence"]["queries"]
    original_master = OrToolsActorInformationSetProofV1._solve_master
    def stop_after_cut(service, scope, action, catalog, cuts, *args, **kwargs):
        result = original_master(service, scope, action, catalog, cuts, *args, **kwargs)
        return replace(result, status="unknown") if cuts else result
    with monkeypatch.context() as patch:
        patch.setattr(OrToolsActorInformationSetProofV1, "_solve_master", stop_after_cut)
        with pytest.raises(InformationSetCompilationStopped) as interrupted:
            observer._solve_observed_worlds([(state, oracle, {})], target,
                maximum_iterations=100, maximum_queries=500, maximum_prefixes=1000, maximum_seconds=30)
    assert interrupted.value.reason == "master_not_optimal"
    partial = interrupted.value.partial_receipt["evidence"]
    assert partial["causeReceipt"]["solverStatus"] == "unknown"
    assert partial["causeReceipt"]["modelHash"] == partial["masterHashes"][-1]
    assert partial["resources"]["usage"]["counterexampleQueries"] == len(partial["queryHashes"])
    for reason in ("cancelled", "maximum_seconds"):
        stopped_master = [False]
        clock = time.monotonic
        def stop_master(*args, **kwargs):
            result = stop_after_cut(*args, **kwargs)
            stopped_master[0] = result.status == "unknown"
            return result
        with monkeypatch.context() as patch:
            patch.setattr(OrToolsActorInformationSetProofV1, "_solve_master", stop_master)
            patch.setattr(time, "monotonic", lambda: clock() +
                          (60 if stopped_master[0] and reason == "maximum_seconds" else 0))
            with pytest.raises(InformationSetCompilationStopped) as stopped:
                observer._solve_observed_worlds([(state, oracle, {})], target,
                    maximum_iterations=100, maximum_queries=500, maximum_prefixes=1000, maximum_seconds=30,
                    cancel_check=lambda: stopped_master[0] and reason == "cancelled")
        assert stopped.value.reason == reason
        assert stopped.value.partial_receipt["evidence"]["causeReceipt"]["solverStatus"] == "unknown"
    assert partial["cuts"] and partial["domains"]
    assert canonical_sha256(partial["scope"]) == partial["scopeHash"]
    assert sorted(canonical_sha256(cut) for cut in partial["cuts"]) == partial["cutHashes"]
    assert set(partial["cutWitnesses"]) == set(partial["cutHashes"])
    with monkeypatch.context() as patch:
        patch.setattr(oracle, "find_path_below", lambda *args, **kwargs: pytest.fail("mismatched source reached solver"))
        bounded_oracle = compiler._path_oracle(start, target=target, utility="awarded_point_margin",
            play_step_count=4, boundary_evaluator="current_raw_point_margin_v1")
        patch.setattr(bounded_oracle, "find_path_below", lambda *args, **kwargs: pytest.fail("mixed scope reached solver"))
        with pytest.raises(ValueError, match="utility or horizon scope"):
            observer._solve_observed_worlds([(state, oracle, {}), (state, bounded_oracle, {})], target,
                maximum_iterations=100, maximum_queries=500, maximum_prefixes=1000, maximum_seconds=30)
        for field in ("hands", "matchScore", "biddingHistory"):
            substituted = copy.deepcopy(state)
            substituted[field] = []
            with pytest.raises(ValueError, match="normalized model source"):
                observer._solve_observed_worlds([(substituted, oracle, {})], target,
                    maximum_iterations=100, maximum_queries=500, maximum_prefixes=1000, maximum_seconds=30)


@pytest.mark.parametrize("planner", ["legacy", "exhaustive", "mixed", "batched"])
def test_bounded_selection_replays_its_recorded_native_path_method(root, rules, monkeypatch, planner):
    from baloot_platform.research import ortools_information_set_compiler as module
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1

    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    raw["playActions"] = raw["playActions"][:16]
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(raw), decision_point={"kind": "play"}, maximum_seconds=30)
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "claim": "outcome_reachable",
        "strategyClass": "pure_deterministic", "targetTeam": Seat.parse(exact.next_actor).team.name.lower(),
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "plies", "value": 4}, "boundaryEvaluator": "current_raw_point_margin_v1"})
    budget = ScenarioProofBudgetV1.build("test", {
        "maximumNodes": 2000, "maximumDepth": 12, "maximumHorizonActions": 12,
        "maximumHiddenWorlds": 5, "maximumWallSeconds": 30, "maximumCpuSeconds": 30,
        "maximumCacheEntries": 2000, "maximumCacheBytes": 30_000_000,
        "cancellationCheckIntervalNodes": 1, "maximumWorkers": 1,
        "maximumLineCountNodes": 2000, "maximumStateSpace": 2000,
        "nativeObserved": {"schemaVersion": 3 if planner == "batched" else 1, "maximumIterations": 200,
            "maximumQueries": 1000, "maximumObservations": 2000, "maximumBoundPrefixes": 2000,
            "maximumCachedPrefixesPerBranch": 2000,
            **({"counterexampleObjective": "minimum_utility",
                "playAllocationMethod": "deterministic_cp_sat_batched_prefix_v1"}
               if planner == "batched" else {})}})
    observer = module.OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    # An overestimated forecast must not force a cheap bounded selection into CEGIS.
    monkeypatch.setattr(module, "_remaining_native_path_bound_v1", lambda oracle, action, limit: limit + 1)
    with monkeypatch.context() as patch:
        if planner == "batched":
            patch.setattr(ortools_complete_state_graph._CpSatPathOracle, "enumerate_paths",
                lambda *args, **kwargs: pytest.fail("batched best-value proof must retain its measured CEGIS route"))
        first = observer._compile_observed_play_selection(exact, request, budget=budget, maximum_seconds=30)
    best = first["rootArtifact"]["proofs"][first["rootArtifact"]["bestProofHash"]]
    assert best["worldCount"] == 5 and best["scope"]["utilityScope"]["cardCount"] > 12
    assert best.get("proofMethod") == (None if planner == "batched" else "exhaustive_native_paths_v1")
    selected = next(action for action in first["rootArtifact"]["legalActions"]
                    if action != first["rootArtifact"]["selectedAction"])
    request = ScenarioProofRequestV1.build({
        **{key: value for key, value in request.to_dict().items() if key != "sha256"},
        "selectedAction": selected})
    routes = {"legacy": frozenset(), "exhaustive": None, "mixed": frozenset({None}), "batched": None}
    receipt = observer._compile_observed_play_selection(exact, request, budget=budget, maximum_seconds=30,
        _exhaustive_bounded_trick_actions=routes[planner])
    assert receipt["actionQuality"]["bestValue"] == first["actionQuality"]["bestValue"]
    proofs = receipt["rootArtifact"]["proofs"]
    expected_routes = {proof["scope"].get("selectedRootAction"):
        proof.get("proofMethod") == "exhaustive_native_paths_v1" for proof in proofs.values()}
    assert set(expected_routes.values()) == (
        {False, True} if planner in {"mixed", "batched"} else {planner == "exhaustive"})
    calls = {}
    solve = observer._solve_observed_worlds

    def capture(*args, **kwargs):
        calls[kwargs["selected_root_action"]] = kwargs["_prefer_bounded_trick_enumeration"]
        return solve(*args, **kwargs)

    monkeypatch.setattr(observer, "_solve_observed_worlds", capture)
    assert observer._verify_observed_play_selection(receipt, exact, request,
        budget=budget, maximum_seconds=30) == receipt
    assert calls == expected_routes
    worlds, _ = observer._prepare_observed_play_worlds(exact, request,
        maximum_worlds=5, maximum_attempts=2000, maximum_model_bytes=30_000_000, maximum_seconds=30,
        play_allocation_method=budget.native_observed.play_allocation_method)
    assert observer._verify_observed_selected_action(receipt["rootArtifact"], worlds,
        Seat.parse(exact.next_actor).team, native_limits=budget.native_observed,
        maximum_seconds=30, selected_action=selected) == receipt["rootArtifact"]
    if planner == "mixed":
        for malformed in (None, {}, {"proofs": []}, {"proofs": {"bad": 1}},
                {"proofs": {"bad": {"scope": []}}},
                {"proofs": {"bad": {"scope": {"selectedRootAction": []},
                                   "proofMethod": "exhaustive_native_paths_v1"}}}):
            forged = {**receipt, "rootArtifact": malformed}
            forged["sha256"] = canonical_sha256({key: value for key, value in forged.items() if key != "sha256"})
            with pytest.raises(ValueError, match="recorded native path methods are malformed"):
                observer._verify_observed_play_selection(forged, exact, request,
                    budget=budget, maximum_seconds=30)


@pytest.mark.parametrize("opponent_root", [False, True])
@pytest.mark.parametrize("minimum_objective", [False, True])
def test_lazy_native_proof_matches_finite_proof_with_different_hidden_cards(root, rules, record_property, opponent_root, minimum_objective, monkeypatch):
    measured_seconds = 900.0  # Sulaiman p99 542.366s; capped at the 900s file lane.
    import math
    allocation_method = ("deterministic_cp_sat_batched_prefix_v1" if minimum_objective
                         else "deterministic_cp_sat_prefix_v1")
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.ortools_information_set_proof import (
        OrToolsActorInformationSetProofV1, InformationSetProofBudgetV1)
    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    raw["playActions"] = raw["playActions"][:20]
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(raw), decision_point={"kind": "play"}, maximum_seconds=measured_seconds)
    target = Seat.parse(exact.next_actor).team
    if opponent_root:
        target = target.other
    observer = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "claim": "outcome_reachable",
        "strategyClass": "pure_deterministic",
        "targetTeam": target.name.lower(), "utility": "awarded_point_margin",
        "successThreshold": 0, "budgetProfile": "test", "horizon": {"unit": "plies", "value": 4},
        "boundaryEvaluator": "current_raw_point_margin_v1"})
    worlds, preparation = observer._prepare_observed_play_worlds(exact, request,
        maximum_worlds=2, maximum_attempts=2000, maximum_model_bytes=30_000_000, maximum_seconds=measured_seconds,
        play_allocation_method=allocation_method)
    assert preparation['allocationModel']['worldScope']['method'] == allocation_method
    with monkeypatch.context() as patch:
        patch.setattr(observer, "_enumerate_worlds", lambda *args, **kwargs: pytest.fail("unowned exact state reached solver"))
        for field in ("state_hash", "prefix_hash", "event_chain_hash", "sha256"):
            with pytest.raises(ValueError, match="identity mismatch"):
                observer._prepare_observed_play_worlds(replace(exact, **{field: "0" * 64}), request,
                    maximum_worlds=2, maximum_attempts=2000, maximum_model_bytes=30_000_000, maximum_seconds=measured_seconds)
    states = tuple(state for state, _, _ in worlds)
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    for limit, expected in ((1, "maximum_model_bytes"), (30_000_000, "cancelled")):
        with pytest.raises(InformationSetCompilationStopped) as preparation_stopped:
            observer._prepare_observed_play_worlds(exact, request, maximum_worlds=2,
                maximum_attempts=2000, maximum_model_bytes=limit, maximum_seconds=measured_seconds,
                cancel_check=(lambda: True) if expected == "cancelled" else None)
        assert preparation_stopped.value.reason == expected
        assert preparation_stopped.value.partial_receipt["evidence"]["sourceStateHash"] == exact.state_hash
    assert len(states) == 2 and states[0]["hands"] != states[1]["hands"]
    assert preparation["sourceStateHash"] == exact.state_hash and preparation["requestHash"] == request.sha256
    recall = {exact.next_actor: ("0" * 64,)}
    recalled_worlds, recalled_preparation = observer._prepare_observed_play_worlds(exact, request,
        maximum_worlds=2, maximum_attempts=2000, maximum_model_bytes=30_000_000,
        maximum_seconds=measured_seconds, private_recall=recall, play_allocation_method=allocation_method)
    assert all(remembered == recall for _, _, remembered in recalled_worlds)
    assert recalled_preparation["privateRecallHash"] == canonical_sha256(recall)
    assert recalled_preparation["privateHistoryCoverage"] == "declared_history_only"
    assert recalled_preparation["sha256"] != preparation["sha256"]
    assert [item[1].model_hash for item in recalled_worlds] == [item[1].model_hash for item in worlds]
    from baloot_platform.research.ortools_nonplay_graphs import decode_native_model_receipt_v1
    decoded_models = [decode_native_model_receipt_v1(row["modelReceipt"],
        maximum_model_bytes=preparation["maximumModelBytes"]) for row in preparation["models"]]
    assert preparation["modelBytes"] == sum(len(row["modelIdentity"]["model"].encode("utf-8")) for row in decoded_models)
    for (state, oracle, _), row in zip(worlds, preparation["models"], strict=True):
        assert row["worldStateHash"] == canonical_sha256(state)
        restored = decode_native_model_receipt_v1(row["modelReceipt"],
            maximum_model_bytes=preparation["maximumModelBytes"])
        assert row["modelHash"] == restored["sourceModelHash"] == oracle.model_hash
        assert restored["modelIdentity"]["model"] == str(oracle.build.model.proto)
        assert len(canonical_json(row["modelReceipt"])) < len(restored["modelIdentity"]["model"])
        with pytest.raises(ValueError, match="read budget"):
            decode_native_model_receipt_v1(row["modelReceipt"], maximum_model_bytes=1)
        tampered = {**row["modelReceipt"], "sourceModelHash": "0" * 64}
        tampered["sha256"] = canonical_sha256({key: value for key, value in tampered.items() if key != "sha256"})
        with pytest.raises(ValueError, match="restored model identity mismatch"):
            decode_native_model_receipt_v1(tampered, maximum_model_bytes=preparation["maximumModelBytes"])
    lazy = observer._solve_observed_worlds(worlds, target, maximum_iterations=200,
        maximum_queries=1000, maximum_prefixes=2000, maximum_seconds=measured_seconds)
    assert lazy["status"] == "converged"
    recalled = observer._solve_observed_worlds(recalled_worlds, target, maximum_iterations=200,
        maximum_queries=1000, maximum_prefixes=2000, maximum_seconds=measured_seconds)
    assert recalled["value"] == lazy["value"]
    assert recalled["scope"]["rootActor"]["observationHash"] != lazy["scope"]["rootActor"]["observationHash"]
    finite = observer.compile(exact, request, maximum_nodes=2000, maximum_depth=12,
        maximum_worlds=2, maximum_seconds=measured_seconds)
    service = OrToolsActorInformationSetProofV1()
    catalog = service._catalog(finite.graph.problem, 2000)
    # Explicit oracle-test budget; not a published proof tier or performance gate.
    policy_count = math.prod(len(actions) for actions in catalog.domains.values())
    reference = service.prove(finite.graph.problem, budget=InformationSetProofBudgetV1(
        maximum_nodes=2000, maximum_worlds=2, maximum_policies=policy_count, maximum_wall_seconds=30))
    assert reference.completion_status == "exact_complete"
    reference_value = (min if opponent_root else max)(item.robust_value for item in reference.roots)
    assert lazy["value"] == reference_value
    from baloot_platform.research.scenario_contracts import NativeObservedProofLimitsV1
    limits = NativeObservedProofLimitsV1.build({"schemaVersion": 1, "maximumIterations": 200,
        "maximumQueries": 1000, "maximumObservations": 2000, "maximumBoundPrefixes": 2000,
        "maximumCachedPrefixesPerBranch": 2000})
    if minimum_objective:
        limits = replace(limits, counterexample_objective="minimum_utility",
                         play_allocation_method=allocation_method)
    root_proofs = observer._solve_observed_root_actions(worlds, target,
        native_limits=limits, maximum_seconds=measured_seconds)
    assert root_proofs["status"] == "complete" and root_proofs["coverage"] == "supplied_worlds_only"
    state, oracle, remembered = worlds[0]
    witness = next(iter(oracle._enumerated_witnesses.values()))
    prefix_cache = {}
    first_bindings = observer._observed_policy_bindings(
        state, oracle, witness, target, {}, remembered,
        known_domains={}, _prefix_observations=prefix_cache)
    with monkeypatch.context() as patch:
        patch.setattr(type(witness), "at_prefix", lambda *args: pytest.fail(
            "cached policy bindings must not rebuild solved prefix states"))
        cached_bindings = observer._observed_policy_bindings(
            state, oracle, witness, target, {}, remembered,
            known_domains=first_bindings[1], _known_prefix_observations=prefix_cache)
    assert cached_bindings == first_bindings
    if not opponent_root:
        root_observation = lazy["scope"]["rootActor"]["observationHash"]
        chosen_root = dict(lazy["policy"])[root_observation]
        assert root_proofs["roots"][chosen_root]["value"] == lazy["value"]
        selection = observer._solve_observed_selected_action(worlds, target,
            native_limits=limits, maximum_seconds=measured_seconds)
        selected = selection["selectedAction"]
        assert selection["proofs"][selection["bestProofHash"]]["value"] == reference_value
        assert root_proofs["roots"][selected]["value"] == reference_value
        assert selection["bestProofHash"] == selection["selectedProofHash"]
        assert len(selection["proofs"]) == 1
        assert selection["optimalActionsComplete"] is False
        assert selection["winningRootActionCount"] is None
        assert selection["sha256"] == canonical_sha256({k: v for k, v in selection.items() if k != "sha256"})
        alternative = next(action for action in root_proofs["roots"] if action != selected)
        observer_caches = []
        solve_worlds = observer._solve_observed_worlds
        def capture_observer_cache(*args, **kwargs):
            observer_caches.append(kwargs.get("_project_observer_cache"))
            return solve_worlds(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(observer, "_solve_observed_worlds", capture_observer_cache)
            audit = observer._solve_observed_selected_action(worlds, target,
                native_limits=limits, maximum_seconds=measured_seconds, selected_action=alternative)
        assert len(observer_caches) == 2
        assert isinstance(observer_caches[0], dict)
        assert observer_caches[0] is observer_caches[1]
        assert audit["proofs"][audit["selectedProofHash"]]["value"] == root_proofs["roots"][alternative]["value"]
        assert audit["usage"]["counterexampleQueries"] == sum(
            proof["resources"]["usage"]["counterexampleQueries"] for proof in audit["proofs"].values())
        assert observer._verify_observed_selected_action(selection, worlds, target,
            native_limits=limits, maximum_seconds=measured_seconds) == selection
        assert observer._verify_observed_selected_action(audit, worlds, target,
            native_limits=limits, maximum_seconds=measured_seconds, selected_action=alternative) == audit
        forged = copy.deepcopy(selection)
        forged["winningRootActionCount"] = 1
        forged["sha256"] = canonical_sha256({k: v for k, v in forged.items() if k != "sha256"})
        with pytest.raises(ValueError, match="differs from reconstructed proof"):
            observer._verify_observed_selected_action(forged, worlds, target,
                native_limits=limits, maximum_seconds=measured_seconds)
        with pytest.raises(ValueError, match="identity or budget mismatch"):
            observer._verify_observed_selected_action(selection, worlds, target,
                native_limits=replace(limits, maximum_queries=limits.maximum_queries + 1), maximum_seconds=measured_seconds)
        with pytest.raises(InformationSetCompilationStopped) as stopped_selection:
            observer._solve_observed_selected_action(worlds, target,
                native_limits=limits, maximum_seconds=measured_seconds, cancel_check=lambda: True)
        assert stopped_selection.value.reason == "cancelled"
    else:
        with pytest.raises(ValueError, match="acting team's worlds"):
            observer._solve_observed_selected_action(worlds, target,
                native_limits=limits, maximum_seconds=measured_seconds)
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1
    execution_budget = ScenarioProofBudgetV1.build("test", {
        "maximumNodes": 2000, "maximumDepth": 12, "maximumHorizonActions": 12,
        "maximumHiddenWorlds": 2, "maximumWallSeconds": measured_seconds, "maximumCpuSeconds": measured_seconds,
        "maximumCacheEntries": 2000, "maximumCacheBytes": 30_000_000,
        "cancellationCheckIntervalNodes": 1, "maximumWorkers": 1,
        "maximumLineCountNodes": 2000, "maximumStateSpace": 2000, "nativeObserved": limits.to_dict()})
    executed, execution_receipt = observer._prove_observed_play(exact, request,
        budget=execution_budget, maximum_seconds=measured_seconds)
    assert execution_receipt["budgetHash"] == execution_budget.sha256
    assert execution_receipt["rootArtifact"]["sha256"] == executed.graph_hash
    assert execution_receipt["outcome"] == executed.to_dict()
    assert execution_receipt["sha256"] == canonical_sha256({key: value for key, value in execution_receipt.items() if key != "sha256"})
    if not opponent_root:
        selection_execution = observer._compile_observed_play_selection(exact, request,
            budget=execution_budget, maximum_seconds=measured_seconds)
        assert selection_execution["kind"] == "native_observed_play_selection_execution_v1"
        assert "outcome" not in selection_execution
        for field in ("sourceStateHash", "requestHash", "budgetHash", "preparation"):
            assert selection_execution[field] == execution_receipt[field]
        assert selection_execution["rootArtifact"] == selection
        quality = selection_execution["actionQuality"]
        assert quality["actualActionOptimal"] is True and quality["regret"] == 0
        assert quality["actualValue"] == quality["bestValue"] == reference_value
        assert quality["optimalActionsComplete"] is False and quality["optimalActionCount"] is None
        assert quality["analysisScope"]["mode"] == "bounded"
        assert quality["analysisScope"]["coverage"] == "supplied_worlds_only"
        assert quality["qualificationModelHash"] == canonical_sha256(selection_execution["qualityModelProto"])
        assert selection_execution["sha256"] == canonical_sha256({
            key: value for key, value in selection_execution.items() if key != "sha256"})
        assert observer._verify_observed_play_selection(selection_execution, exact, request,
            budget=execution_budget, maximum_seconds=measured_seconds) == selection_execution
        request_core = {key: value for key, value in request.to_dict().items() if key != "sha256"}
        with pytest.raises(ValueError, match="execution identity or budget mismatch"):
            observer._verify_observed_play_selection(selection_execution, exact,
                ScenarioProofRequestV1.build({**request_core, "horizon": {"unit": "plies", "value": 3}}),
                budget=execution_budget, maximum_seconds=measured_seconds)
        with pytest.raises(ValueError, match="scenario proof checksum mismatch"):
            observer._verify_observed_play_selection(selection_execution, exact,
                replace(request, horizon={"unit": "plies", "value": 3}),
                budget=execution_budget, maximum_seconds=measured_seconds)
        with pytest.raises(ValueError, match="differs from owned reconstruction"):
            observer._verify_observed_play_selection(selection_execution, exact, request,
                budget=execution_budget, maximum_seconds=measured_seconds, private_recall=recall)
        with pytest.raises(ValueError, match="cannot certify root-count or scenario claims"):
            observer._compile_observed_play_selection(exact,
                ScenarioProofRequestV1.build({**request_core, "claim": "selected_action_is_unique_winner",
                    "selectedAction": selected}),
                budget=execution_budget, maximum_seconds=measured_seconds)
        with pytest.raises(InformationSetCompilationStopped) as selection_cancelled:
            observer._compile_observed_play_selection(exact, request,
                budget=execution_budget, maximum_seconds=measured_seconds, cancel_check=lambda: True)
        assert selection_cancelled.value.reason == "cancelled"
        assert selection_cancelled.value.partial_receipt["evidence"]["budgetHash"] == execution_budget.sha256
    with pytest.raises(InformationSetCompilationStopped) as execution_cancelled:
        observer._prove_observed_play(exact, request, budget=execution_budget,
            maximum_seconds=measured_seconds, cancel_check=lambda: True)
    assert execution_cancelled.value.reason == "cancelled"
    assert execution_cancelled.value.partial_receipt["evidence"]["budgetHash"] == execution_budget.sha256
    assert root_proofs["sha256"] == canonical_sha256({key: value for key, value in root_proofs.items() if key != "sha256"})
    assert set(root_proofs["legalActions"]) == set(root_proofs["roots"]) == {row.root_action for row in reference.roots}
    adapted = observer._observed_root_outcome(root_proofs, request, maximum_seconds=10)
    assert adapted.completion_status == ("bounded_complete" if any(row.robust_value >= 0 for row in reference.roots)
                                         else "proof_failed")
    assert {row.action_code: row.value for row in adapted.root_actions} == {
        row.root_action: row.robust_value for row in reference.roots}
    assert adapted.graph_hash == root_proofs["sha256"]
    assert adapted.model_document()["nativeRootArtifactHash"] == root_proofs["sha256"]
    assert adapted.principal_variation == ()
    assert adapted.hidden_worlds["complete"] is False and adapted.hidden_worlds["count"] == len(worlds)
    cancelled_outcome = observer._observed_root_outcome(root_proofs, request, maximum_seconds=10,
        cancel_check=lambda: True)
    assert cancelled_outcome.completion_status == "proof_incomplete"
    assert cancelled_outcome.winning_root_action_count is None and not cancelled_outcome.claim_satisfied
    with pytest.raises(ValueError, match="bounded actor scope"):
        observer._observed_root_outcome(root_proofs,
            ScenarioProofRequestV1.build({**{key: value for key, value in request.to_dict().items()
                if key not in {"sha256", "horizon", "boundaryEvaluator"}}, "mode": "exact"}), maximum_seconds=10)
    with pytest.raises(ValueError, match="achieved horizon"):
        observer._observed_root_outcome(root_proofs,
            ScenarioProofRequestV1.build({**{key: value for key, value in request.to_dict().items() if key != "sha256"},
                                         "horizon": {"unit": "plies", "value": 2}}),
            maximum_seconds=10)
    if shared := root_proofs.get("sharedEnumeration"):
        assert shared["proofMethod"] == "exhaustive_native_paths_v1"
        assert root_proofs["usage"]["counterexampleQueries"] == shared["resources"]["usage"]["counterexampleQueries"]
        assert all(row["sharedSourceHash"] == shared["sha256"] for row in root_proofs["roots"].values())
        original_solve = observer._solve_observed_worlds
        def nonexhaustive_first(*args, **kwargs):
            if kwargs.get("selected_root_action") is None:
                result = copy.deepcopy(shared)
                result["proofMethod"] = "counterexample_guided_test"
                result["sha256"] = canonical_sha256({key: value for key, value in result.items() if key != "sha256"})
                return result
            return original_solve(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(observer, "_solve_observed_worlds", nonexhaustive_first)
            fallback = observer._solve_observed_root_actions(worlds, target,
                native_limits=limits, maximum_seconds=measured_seconds)
        assert fallback["status"] == "complete"
        assert "sharedEnumeration" not in fallback
        assert {key: row["value"] for key, row in fallback["roots"].items()} == {
            key: row["value"] for key, row in root_proofs["roots"].items()}
        for counter in ("counterexampleQueries", "masterIterations"):
            assert fallback["usage"][counter] == shared["resources"]["usage"][counter] + sum(
                row["resources"]["usage"][counter] for row in fallback["roots"].values())
        assert fallback["preliminaryProof"]["proofMethod"] == "counterexample_guided_test"
        certified_fallback = observer._certify_observed_root_actions(fallback, success_threshold=0,
            claim="winning_root_action_count", maximum_seconds=10)
        assert certified_fallback["winningRootActionCount"] == sum(row.robust_value >= 0 for row in reference.roots)
        for mutation in ("hash", "scope", "counterexampleQueries", "masterIterations"):
            altered = copy.deepcopy(fallback)
            if mutation == "hash":
                altered["preliminaryProof"]["sha256"] = "0" * 64
            elif mutation == "scope":
                altered["preliminaryProof"]["scope"]["rulesHash"] = "0" * 64
                altered["preliminaryProof"]["scopeHash"] = canonical_sha256(altered["preliminaryProof"]["scope"])
                altered["preliminaryProof"]["sha256"] = canonical_sha256({
                    key: value for key, value in altered["preliminaryProof"].items() if key != "sha256"})
            else:
                altered["usage"][mutation] -= 1
            altered["sha256"] = canonical_sha256({key: value for key, value in altered.items() if key != "sha256"})
            with pytest.raises(ValueError, match="native preliminary proof"):
                observer._certify_observed_root_actions(altered, success_threshold=0,
                    claim="winning_root_action_count", maximum_seconds=10)
        from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
        with monkeypatch.context() as patch:
            patch.setattr(observer, "_solve_observed_worlds", nonexhaustive_first)
            with pytest.raises(InformationSetCompilationStopped) as fallback_exhausted:
                observer._solve_observed_root_actions(worlds, target,
                    native_limits=replace(limits, maximum_queries=shared["resources"]["usage"]["counterexampleQueries"]),
                    maximum_seconds=measured_seconds)
        assert fallback_exhausted.value.reason == "maximum_queries"
        assert fallback_exhausted.value.partial_receipt["evidence"]["roots"] == {}
    else:
        assert root_proofs["usage"]["counterexampleQueries"] == sum(
            row["resources"]["usage"]["counterexampleQueries"] for row in root_proofs["roots"].values())
    root_results = {}
    for root_receipt in reference.roots:
        result = root_proofs["roots"][root_receipt.root_action]
        assert result["status"] == "converged"
        assert result["scope"]["selectedRootAction"] == root_receipt.root_action
        assert result["value"] == root_receipt.robust_value
        root_results[root_receipt.root_action] = result["scopeHash"]
        quality_claim = observer._certify_observed_root_actions(root_proofs, success_threshold=0,
            claim="winning_root_action_count", selected_action=root_receipt.root_action,
            maximum_seconds=10, action_quality=True)
        quality = quality_claim["actionQuality"]
        assert quality["bestValue"] == reference_value
        assert quality["actualValue"] == root_receipt.robust_value
        assert quality["regret"] == (root_receipt.robust_value - reference_value if opponent_root
                                     else reference_value - root_receipt.robust_value)
        assert quality["analysisScope"]["coverage"] == "supplied_worlds_only"
        assert quality["sourceQualificationModelHash"] == quality_claim["modelHash"]
    assert len(set(root_results.values())) == len(reference.roots)
    for threshold in (min(row.robust_value for row in reference.roots), 0,
                      max(row.robust_value for row in reference.roots) + 1):
        expected_count = sum(row.robust_value >= threshold for row in reference.roots)
        for claim, expected in (("winning_root_action_count", True),
                                ("outcome_reachable", expected_count > 0),
                                ("outcome_forced", expected_count == len(reference.roots)),
                                ("selected_action_is_unique_winner", expected_count == 1),
                                ("target_team_has_no_salvage", expected_count == 0)):
            certified = observer._certify_observed_root_actions(root_proofs, success_threshold=threshold,
                claim=claim, maximum_seconds=10)
            assert certified["winningRootActionCount"] == expected_count
            assert certified["claimSatisfied"] == expected
            assert certified["status"] == ("satisfied" if expected else "proof_failed")
            assert certified["coverage"] == "supplied_worlds_only"
            assert certified["sourceArtifactHash"] == root_proofs["sha256"]
            assert certified["modelProto"] and len(certified["bindings"]) == len(reference.roots)
    incomplete_claim = observer._certify_observed_root_actions(root_proofs, success_threshold=0,
        claim="outcome_reachable", maximum_seconds=10, cancel_check=lambda: True)
    assert incomplete_claim["status"] == "incomplete"
    assert incomplete_claim["claimSatisfied"] is None and incomplete_claim["winningRootActionCount"] is None
    for mutation in ("missing", "value", "incomplete"):
        altered = copy.deepcopy(root_proofs)
        if mutation == "missing":
            altered["roots"].pop(altered["legalActions"][0])
        elif mutation == "value":
            altered["roots"][altered["legalActions"][0]]["value"] += 1
        else:
            altered["status"] = "incomplete"
        altered["sha256"] = canonical_sha256({key: value for key, value in altered.items() if key != "sha256"})
        with pytest.raises(ValueError):
            observer._certify_observed_root_actions(altered, success_threshold=0,
                claim="outcome_reachable", maximum_seconds=10)
    if shared:
        altered = copy.deepcopy(root_proofs)
        first_action = altered["legalActions"][0]
        altered["roots"][first_action]["sharedSourceHash"] = "0" * 64
        altered["roots"][first_action]["sha256"] = canonical_sha256({key: value for key, value in
            altered["roots"][first_action].items() if key != "sha256"})
        altered["sha256"] = canonical_sha256({key: value for key, value in altered.items() if key != "sha256"})
        with pytest.raises(ValueError, match="identity or scope disagrees"):
            observer._certify_observed_root_actions(altered, success_threshold=0,
                claim="outcome_reachable", maximum_seconds=10)
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    first_action = root_proofs["legalActions"][0]
    first_queries = (1 if shared else
        root_proofs["roots"][first_action]["resources"]["usage"]["counterexampleQueries"])
    with pytest.raises(InformationSetCompilationStopped) as exhausted:
        observer._solve_observed_root_actions(worlds, target,
            native_limits=replace(limits, maximum_queries=first_queries), maximum_seconds=measured_seconds)
    partial_roots = exhausted.value.partial_receipt["evidence"]
    assert exhausted.value.reason == "maximum_queries"
    assert list(partial_roots["roots"]) == ([] if shared else [first_action])
    assert partial_roots["usage"]["counterexampleQueries"] == first_queries
    if shared:
        cause_usage = partial_roots["causeReceipt"]["evidence"]["resources"]["usage"]
        assert partial_roots["usage"]["masterIterations"] == cause_usage["masterIterations"]
    assert partial_roots["status"] == "incomplete"
    cancelled = [False]
    original_solve = observer._solve_observed_worlds
    def cancel_after_root(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        cancelled[0] = True
        return result
    with monkeypatch.context() as patch:
        patch.setattr(observer, "_solve_observed_worlds", cancel_after_root)
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            observer._solve_observed_root_actions(worlds, target, native_limits=limits,
                maximum_seconds=measured_seconds, cancel_check=lambda: cancelled[0])
    assert stopped.value.reason == "cancelled"
    assert list(stopped.value.partial_receipt["evidence"]["roots"]) == ([] if shared else [first_action])
    with pytest.raises(ValueError, match="root action is not legal"):
        observer._solve_observed_worlds(worlds, target, maximum_iterations=200,
            maximum_queries=1000, maximum_prefixes=2000, maximum_seconds=measured_seconds,
            selected_root_action="invalid")
    reordered = observer._solve_observed_worlds(list(reversed(worlds)), target, maximum_iterations=200,
        maximum_queries=1000, maximum_prefixes=2000, maximum_seconds=measured_seconds)
    assert reordered == lazy
    record_property("nativeWorldComparison", json.dumps({"worldCount": len(worlds),
        "value": lazy["value"], "referenceRootAggregate": reference_value,
        "queryCount": len(lazy["queryHashes"]), "masterCount": len(lazy["masterHashes"]),
        "cutCount": len(lazy["cutHashes"]), "scopeHash": lazy["scopeHash"]}, sort_keys=True))


def test_adverse_witness_extracts_once_and_preserves_base_identity(rules, monkeypatch):
    compiler = _compiler(rules, native_conversion=True)
    start = compiler._parse_start(_seed(rules), expected_rules_hash=rules.sha256)
    oracle = compiler._path_oracle(start, target=Team.NORTH_SOUTH, utility="awarded_point_margin")
    extractor_type = type(oracle.build._extract)
    original = extractor_type.__call__
    calls = []

    def counted(extractor, solver, status, model_hash):
        calls.append(model_hash)
        return original(extractor, solver, status, model_hash)

    monkeypatch.setattr(extractor_type, "__call__", counted)
    _, upper = oracle.query_utility_bounds()
    witness, _, query_hash = oracle.find_path_below(upper + 1,
        conditional_actions={}, maximum_seconds=10, cancelled=None)
    assert calls == [query_hash]
    cached, solver = oracle._solved_witnesses[0]
    assert cached == original(oracle.build._extract, solver,
        ortools_complete_state_graph.cp_model.OPTIMAL, oracle.model_hash)
    assert witness.evidence_hash == cached.evidence_hash
    assert witness.model_hash == query_hash != cached.model_hash
    assert witness.sha256 != cached.sha256


def test_path_oracle_reuses_only_matching_native_witness(rules, monkeypatch):
    compiler = _compiler(rules, native_conversion=True)
    start = compiler._parse_start(_seed(rules), expected_rules_hash=rules.sha256)
    oracle = compiler._path_oracle(start, target=Team.NORTH_SOUTH, utility="awarded_point_margin")
    original = ortools_complete_state_graph._solve_cp_sat_with_cancel_v1
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ortools_complete_state_graph, "_solve_cp_sat_with_cancel_v1", counted)
    root, _ = oracle.solve((), maximum_seconds=10, cancelled=None)
    full, _ = oracle._solved_witnesses[0]
    path = tuple(full.plays[oracle.base_length:])
    cached = [oracle.solve(path[:depth], maximum_seconds=10, cancelled=None)[0]
              for depth in range(1, len(path) + 1)]
    assert len(calls) == 1
    sibling = next(action for action in root.legal_play_actions if action != path[0])
    oracle.solve((sibling,), maximum_seconds=10, cancelled=None)
    assert len(calls) == 2
    restored, _ = oracle.solve(path, maximum_seconds=10, cancelled=None)
    assert restored == cached[-1]
    assert len(calls) == 2  # Ancestor witness survives the divergent sibling.
    assert len(oracle._solved_witnesses) <= len(start.cards) - oracle.base_length + 1
    for depth, expected in enumerate(cached, 1):
        oracle._solved_witnesses.clear()
        actual, _ = oracle.solve(path[:depth], maximum_seconds=10, cancelled=None)
        assert actual == expected
    before = len(calls)
    alternative = next(action for action in root.legal_play_actions if action != path[0])
    oracle.solve((alternative,), maximum_seconds=10, cancelled=None)
    assert len(calls) == before + 1
    for seconds, cancel, reason in ((0, None, "maximum_seconds"), (10, lambda: True, "cancelled")):
        with pytest.raises(ortools_complete_state_graph.PlayTailSolveStopped) as caught:
            oracle.solve((alternative,), maximum_seconds=seconds, cancelled=cancel)
        assert caught.value.solver_status == reason
    assert len(calls) == before + 1


@pytest.mark.parametrize("baloot_value,expected_margin", [(2, -24), (4, 0)])
@pytest.mark.parametrize("prefix_length", [24, 28, 29])
@pytest.mark.parametrize("baloot_enabled", [True, False])
@pytest.mark.parametrize("mode", ["exact", "bounded"])
def test_native_exact_proof_counts_future_baloot(root, baloot_value, expected_margin, prefix_length, baloot_enabled, mode):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    raw_rules = copy.deepcopy(rules.data)
    raw_rules["projects"]["baloot"]["value_game_points"] = baloot_value
    raw_rules["projects"]["baloot"]["enabled"] = baloot_enabled
    rules = RuleBook(raw_rules)
    plays = "north:S7 west:S10 south:SK east:D9 east:H8 north:HJ west:HK south:H7 west:SJ south:SA east:D10 north:S8 east:H9 north:HQ west:HA south:D7 south:D8 east:H10 north:DJ west:DA north:S9 west:SQ south:C9 east:CA"
    plays += " west:C7 south:C10 east:CQ north:DK north:DQ west:C8 south:CJ east:CK"
    prefix = ExactTracePrefixV1.build({
        "schemaVersion": 1, "deckOrder": list(range(32)), "dealer": "east",
        "auctionActions": [{"actor": "north", "action": "hokum", "trumpSuit": "diamonds"},
                           *({"actor": seat, "action": "pass"} for seat in ("west", "south", "east"))],
        "escalationActions": [], "projectActions": [
            {"actor": "north", "action": "declare", "projectId": "PROJECT-B21095B6C944B38C"},
            {"actor": "west", "action": "pass"}, {"actor": "south", "action": "pass"},
            {"actor": "east", "action": "declare", "projectId": "PROJECT-3A7204010DAE03EE"}],
        "playActions": [dict(zip(("actor", "card"), action.split(":")))
                        for action in plays.split()[:prefix_length]],
    })
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "play"}, maximum_seconds=30,
    )
    assert exact.state["balootTeams"] == (["north_south"] if prefix_length == 29 and baloot_enabled else [])
    if prefix_length == 24 and mode == "exact" and baloot_value == 2:
        from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
        observer = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
        start = observer.graph_compiler._parse_start(exact, expected_rules_hash=rules.sha256)
        oracle = observer.graph_compiler._path_oracle(
            start, target=Seat.NORTH.team, utility="awarded_point_margin")
        assert oracle.public_baloot_binding is not None
        witness, _, _ = oracle.find_path_below(1000, conditional_actions={}, maximum_seconds=10, cancelled=None)
        bindings, domains, _ = observer._observed_policy_bindings(
            exact.state, oracle, witness, Seat.NORTH.team, {}, {}, known_domains={}, maximum_seconds=10)
        assert bindings == {} and domains
        cache = {}
        policy = {key: actions[0] for key, actions in domains.items()}
        bindings, _, _ = observer._observed_policy_bindings(exact.state, oracle, witness,
            Seat.NORTH.team, policy, {}, known_domains=domains, maximum_seconds=10, _prefix_observations=cache)
        assert bindings and cache
        from baloot_platform.research.ortools_play_tail import PlayTailSolveStopped
        for seconds, cancel, reason in ((0, None, "maximum_seconds"), (10, lambda: True, "cancelled")):
            with pytest.raises(PlayTailSolveStopped) as stopped:
                observer._observed_policy_bindings(exact.state, oracle, witness,
                    Seat.NORTH.team, policy, {}, known_domains=domains, maximum_seconds=seconds,
                    cancel_check=cancel, _known_prefix_observations=cache)
            assert stopped.value.solver_status == reason
        for seconds in (None, float("nan"), float("inf"), True):
            with pytest.raises(ValueError, match="finite time budget"):
                observer._observed_policy_bindings(exact.state, oracle, witness,
                    Seat.NORTH.team, {}, {}, known_domains={}, maximum_seconds=seconds)
        original = copy.deepcopy(exact.state)
        for length in (4, 5):
            path = tuple(TailPrefixActionV1.build(*action.split(":"))
                         for action in plays.split()[24:24 + length])
            result, _ = oracle.solve(path, maximum_seconds=10, cancelled=None)
            view = observer._dynamic_state(exact.state, start, result, path)
            assert view["balootTeams"] == (
                ["north_south"] if length == 5 and baloot_enabled else [])
            if length == 5:
                assert view["currentTrick"]["plays"][-1]["baloot"] is baloot_enabled
            assert exact.state == original
    request = _request()
    if mode == "bounded":
        request = ScenarioProofRequestV1.build({
            **{key: value for key, value in request.to_dict().items() if key != "sha256"},
            "mode": "bounded", "horizon": {"unit": "plies", "value": 8},
            "boundaryEvaluator": "current_raw_point_margin_v1",
        })
    compiled = _compiler(rules, native_conversion=True).compile(
        exact, request, expected_rules_hash=rules.sha256,
        maximum_nodes=2000, maximum_depth=8, maximum_seconds=30,
    )
    assert compiled.complete
    proof = OrToolsScenarioGraphProofV1().prove(compiled.graph, request)
    # Two points leave the buyer failing (11:13); four produce a 13:13 tie.
    assert {action.value for action in proof.root_actions} == {expected_margin if baloot_enabled else -22}


def test_exact_play_seed_binds_resolved_baloot_points_from_project_domain(rules) -> None:
    exact = _normalized_state(rules)
    state = dict(exact.state)
    state.update(
        {
            "contract": "hokum",
            "trumpSuit": "spades",
            "balootTeams": ["north_south"],
        }
    )
    exact = replace(exact, state=state, state_hash=canonical_sha256(state))

    start = _compiler(rules)._parse_start(
        exact,
        expected_rules_hash=rules.sha256,
    )

    assert start.baloot_points == (rules.project_domain.baloot_value, 0)


@pytest.mark.parametrize("native_conversion", [False, True])
def test_exact_graph_drives_zero_one_and_multiple_root_counts(rules, native_conversion) -> None:
    compilation = _compile(rules, _request(), native_conversion=native_conversion)

    assert compilation.complete is True
    assert compilation.graph is not None
    base_proof = OrToolsScenarioGraphProofV1().prove(
        compilation.graph,
        _request(),
    )
    values = sorted({item.value for item in base_proof.root_actions})
    assert len(values) == 2

    zero = OrToolsScenarioGraphProofV1().prove(
        compilation.graph,
        _request(threshold=max(values) + 1),
    )
    one = OrToolsScenarioGraphProofV1().prove(
        compilation.graph,
        _request(threshold=max(values)),
    )
    multiple = OrToolsScenarioGraphProofV1().prove(
        compilation.graph,
        _request(threshold=min(values)),
    )

    assert zero.winning_root_action_count == 0
    assert one.winning_root_action_count == 1
    assert multiple.winning_root_action_count == 2
    assert one.principal_variation


@pytest.mark.parametrize("native_conversion", [False, True])
@pytest.mark.parametrize("target", list(Team))
def test_contract_encoding_reaches_compiled_graph_and_model_identity(rules, native_conversion, target):
    raw = {**_request().to_dict(), "sha256": "", "utility": "contract_success",
        "targetTeam": target.name.lower()}
    legacy = _compile(rules, ScenarioProofRequestV1.build(raw), native_conversion=native_conversion)
    signed = _compile(rules, ScenarioProofRequestV1.build({**raw,
        "utilityEncoding": "signed_win_loss_v1"}), native_conversion=native_conversion)
    assert legacy.complete and signed.complete
    assert legacy.receipt.model_hash != signed.receipt.model_hash
    old_nodes = {node.node_id: node for node in legacy.graph.nodes}
    new_nodes = {node.node_id: node for node in signed.graph.nodes}
    pending = [(legacy.graph.root_node_id, signed.graph.root_node_id)]
    seen = set()
    while pending:
        pair = pending.pop()
        if pair in seen:
            continue
        seen.add(pair)
        old, new = old_nodes[pair[0]], new_nodes[pair[1]]
        assert old.is_leaf == new.is_leaf
        if old.is_leaf:
            assert new.utility == 2 * old.utility - 1
        else:
            old_edges = {edge.action_id: edge.child_node_id for edge in old.edges}
            new_edges = {edge.action_id: edge.child_node_id for edge in new.edges}
            assert old_edges.keys() == new_edges.keys()
            pending.extend((child, new_edges[action]) for action, child in old_edges.items())


def test_compiled_graph_preserves_action_ids_transpositions_and_line_paths(
    rules,
) -> None:
    compilation = _compile(rules, _request())
    assert compilation.graph is not None
    root = next(
        node
        for node in compilation.graph.nodes
        if node.node_id == compilation.graph.root_node_id
    )
    assert [edge.action_id for edge in root.edges] == [
        Card.parse("S7").id,
        Card.parse("SA").id,
    ]
    assert compilation.receipt.transposition_edge_count > 0

    selected = root.edges[0].action_code
    proof = OrToolsScenarioGraphProofV1().prove(
        compilation.graph,
        _request(
            claim="winning_continuation_line_count",
            selected_action=selected,
            line_universe="all_legal",
            threshold=-1_000,
        ),
    )
    assert proof.continuation_lines is not None
    assert proof.continuation_lines.complete is True
    assert int(proof.continuation_lines.exact_count or "0") > 1


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("normalized_round", [False, True])
def test_on_demand_exact_graph_matches_exhaustive_compilation(
    root, rules, monkeypatch, compact, normalized_round, record_property
):
    import time

    compiler = _compiler(rules, native_conversion=True)
    request = _request()
    source = _seed(rules)
    if normalized_round:
        raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json")
                         .read_text(encoding="utf-8"))["exactTracePrefix"]
        source = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
            ExactTracePrefixV1.build(raw), decision_point={"kind": "play"}, maximum_seconds=30)
    arguments = dict(expected_rules_hash=rules.sha256, maximum_nodes=2000,
        maximum_depth=9 if normalized_round else 8, maximum_seconds=10, _compact_forced=compact)
    started = time.perf_counter()
    baseline = compiler.compile(source, request, **arguments)
    record_property("eagerSeconds", time.perf_counter() - started)
    assert baseline.complete, baseline.receipt.exhaustion_reason
    monkeypatch.setattr(ortools_complete_state_graph._CpSatPathOracle,
        "enumerate_paths", lambda *args, **kwargs: None)
    started = time.perf_counter()
    direct = compiler.compile(source, request, **arguments)
    record_property("onDemandSeconds", time.perf_counter() - started)
    assert direct.complete
    assert direct.graph.to_dict() == baseline.graph.to_dict()
    service = OrToolsScenarioGraphProofV1()
    for action in next(node for node in baseline.graph.nodes
                       if node.node_id == baseline.graph.root_node_id).edges:
        claim = _request(claim="winning_continuation_line_count",
            selected_action=action.action_code, line_universe="all_legal", threshold=-1000)
        expected = service.prove(baseline.graph, claim)
        actual = service.prove(direct.graph, claim)
        assert actual.to_dict() == expected.to_dict()


@pytest.mark.parametrize("native_conversion", [False, True])
def test_bounded_frontier_uses_governed_raw_margin(rules, native_conversion) -> None:
    request = _request(mode="bounded", threshold=0)
    compilation = _compile(rules, request, native_conversion=native_conversion)

    assert compilation.complete is True
    assert compilation.graph is not None
    assert compilation.graph.scope == "bounded"
    assert compilation.receipt.boundary_node_count > 0
    proof = OrToolsScenarioGraphProofV1().prove(compilation.graph, request)
    assert proof.completion_status == "bounded_complete"
    assert {item.value for item in proof.root_actions} == {2}


def test_final_boundary_sharing_precedes_node_budget(rules):
    request = _request(mode="bounded")
    compiler = _compiler(rules)
    kwargs = dict(expected_rules_hash=rules.sha256, maximum_nodes=2,
                  maximum_depth=8, maximum_seconds=30)
    reference = compiler.compile(_normalized_state(rules), request, **kwargs)
    assert not reference.complete
    assert reference.receipt.exhaustion_reason == "maximum_nodes"
    shared = compiler.compile(_normalized_state(rules), request,
                              _merge_final_boundary=True, **kwargs)
    assert shared.complete
    assert len(shared.graph.nodes) == 2
    root = next(node for node in shared.graph.nodes if not node.is_leaf)
    assert len(root.edges) == 2
    assert len({edge.child_node_id for edge in root.edges}) == 1
    proof = OrToolsScenarioGraphProofV1().prove(shared.graph, request)
    assert {item.value for item in proof.root_actions} == {2}


@pytest.mark.parametrize("observed", [False, True])
def test_native_compact_construction_preserves_graph_and_proof_under_node_cap(rules, observed):
    from baloot_platform.research.ortools_proof import ProofGraphSequencePackV1

    compiler = _compiler(rules, native_conversion=True)
    request = _request(claim="winning_continuation_line_count", selected_action="S7",
                       line_universe="all_legal", threshold=-1000)
    source = _seed(rules)
    limits = dict(expected_rules_hash=rules.sha256, maximum_depth=8, maximum_seconds=30)
    if observed:
        limits["decision_observer"] = lambda value: canonical_sha256({
            "actor": value.actor.name, "path": [action.to_dict() for action in value.path]})
    reference = compiler.compile(source, request, maximum_nodes=1000, **limits)
    assert reference.complete
    assert reference.sequence_pack is None
    assert "representation" not in reference.receipt.to_dict()
    expected = ProofGraphSequencePackV1.build(reference.graph,
        observation_ids=reference.decision_observation_ids)
    cap = len(expected.graph.nodes)
    assert cap < len(reference.graph.nodes)
    ordinary = compiler.compile(source, request, maximum_nodes=cap, **limits)
    assert not ordinary.complete and ordinary.receipt.exhaustion_reason == "maximum_nodes"
    compact = compiler.compile(source, request, maximum_nodes=cap, _compact_forced=True, **limits)
    assert compact.complete, compact.receipt.to_dict()
    assert compact.graph == reference.graph
    assert compact.sequence_pack == expected
    assert compact.decision_observation_ids == reference.decision_observation_ids
    evidence = compact.receipt.to_dict()
    assert evidence["representation"]["retainedNodes"] == cap
    assert evidence["representation"]["forcedRecords"] == len(reference.graph.nodes) - cap
    assert evidence["representation"]["packHash"] == expected.sha256
    assert evidence["exploration"]["nodes"] == len(reference.graph.nodes)
    assert canonical_sha256({k: v for k, v in evidence.items() if k != "sha256"}) == evidence["sha256"]
    prover = OrToolsScenarioGraphProofV1()
    assert prover.prove(compact.sequence_pack, request) == prover.prove(reference.graph, request)
    refused = compiler.compile(source, request, maximum_nodes=cap - 1, _compact_forced=True, **limits)
    assert not refused.complete and refused.sequence_pack is None
    assert refused.receipt.exhaustion_reason == "maximum_nodes"
    assert refused.receipt.representation["retainedNodes"] <= cap - 1
    shallow = compiler.compile(source, request, maximum_nodes=cap, _compact_forced=True,
                               **{**limits, "maximum_depth": 7})
    assert not shallow.complete and shallow.receipt.exhaustion_reason == "maximum_depth"


@pytest.mark.parametrize("size", [4, 5, 6, 7])
def test_bounded_proof_accepts_every_midround_hand_size(rules, size):
    hands = tuple(tuple(Card(int(seat) + 4 * index) for index in range(size)) for seat in Seat)
    seed = CompleteStateGraphSeedV1.build(
        exact_state_id=f"TEST-MIDROUND-{size}", rules_hash=rules.sha256,
        remaining_hands=hands, current_trick=(), first_leader=Seat.NORTH,
        next_actor=Seat.NORTH,
        legal_actions=tuple(TailPrefixActionV1(Seat.NORTH, card) for card in hands[int(Seat.NORTH)]),
        contract=Contract.SUN, trump=None, buyer=Seat.WEST, locked=False,
        base_raw=(0, 0), base_tricks=(8 - size, 0), awarded_project_points=(0, 0),
        baloot_points=(0, 0), escalation="normal", last_escalating_team=None,
        starting_match_score=(0, 0),
    )
    request = _request(mode="bounded")
    compilation = _compiler(rules).compile(
        seed, request, expected_rules_hash=rules.sha256,
        maximum_nodes=100, maximum_depth=1, maximum_seconds=15,
    )
    assert compilation.complete is True
    assert compilation.graph is not None
    proof = OrToolsScenarioGraphProofV1().prove(compilation.graph, request)
    assert proof.completion_status == "bounded_complete"
    assert len(proof.root_actions) == size
    assert {item.value for item in proof.root_actions} == {0}


@pytest.mark.parametrize("contract", ("sun", "hokum"))
def test_unresolved_projects_are_only_admitted_for_one_raw_play_frontier(root, contract):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = ExactTracePrefixV1.build({
        "schemaVersion": 1, "deckOrder": list(range(32)), "dealer": "east",
        "roundNumber": 1, "matchScore": [0, 0],
        "auctionActions": [{"actor": "north", "action": contract}] + (
            [{"actor": seat, "action": "pass"} for seat in ("west", "south", "east")]
            if contract == "hokum" else []
        ),
        "escalationActions": [], "projectActions": [{"actor": "north", "action": "pass"}],
        "projectDeclarations": [], "projectAssertions": [], "playActions": [], "stateAssertions": {},
    })
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    assert exact.state["projects"]
    assert exact.state["projectWinningTeam"] is None
    if contract == "hokum":
        assert any(
            {"king", "queen"}.issubset({card["rank"] for card in hand
                                 if card["suit"] == exact.state["trumpSuit"]})
            for hand in exact.state["hands"].values()
        )
    compiler = _compiler(rules)
    compiler._parse_start(exact, expected_rules_hash=rules.sha256, request=_request(mode="bounded"))
    longer = _request(mode="bounded").to_dict()
    longer.pop("sha256")
    longer["horizon"]["value"] = 2
    for request in (None, _request(), ScenarioProofRequestV1.build(longer)):
        with pytest.raises(ValueError, match="requires resolved project comparison"):
            compiler._parse_start(exact, expected_rules_hash=rules.sha256, request=request)
    state = {**exact.state, "projects": [{**exact.state["projects"][0], "scoring": True}]}
    corrupted = replace(exact, state=state, state_hash=canonical_sha256(state))
    with pytest.raises(ValueError, match="awarded project requires a comparison winner"):
        compiler._parse_start(corrupted, expected_rules_hash=rules.sha256, request=_request(mode="bounded"))

    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    for _ in range(len(Seat) - 1):
        prefix = normalizer.append_normalized_action(prefix, exact, exact.legal_actions[0]["code"])
        exact = normalizer.normalize_play_or_project(prefix)
        prefix = normalizer.append_normalized_action(prefix, exact, "pass")
        exact = normalizer.normalize_play_or_project(prefix)
    assert len(exact.state["currentTrick"]["plays"]) == len(Seat) - 1
    assert exact.state["projectWinningTeam"] is None
    one_trick = _request(mode="bounded").to_dict()
    one_trick.pop("sha256")
    one_trick["horizon"]["unit"] = "tricks"
    assert compiler._parse_start(exact, expected_rules_hash=rules.sha256,
        request=ScenarioProofRequestV1.build(one_trick)) == compiler._parse_start(
            exact, expected_rules_hash=rules.sha256, request=_request(mode="bounded"))
    one_trick["horizon"]["value"] = 2
    for request in (None, _request(), ScenarioProofRequestV1.build(one_trick)):
        with pytest.raises(ValueError, match="requires resolved project comparison"):
            compiler._parse_start(exact, expected_rules_hash=rules.sha256, request=request)


def test_no_project_holdings_do_not_skip_pending_declarations(root):
    import random

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    deck = list(range(32))
    random.Random(0).shuffle(deck)
    prefix = ExactTracePrefixV1.build({
        "schemaVersion": 1, "deckOrder": deck, "dealer": "east",
        "roundNumber": 1, "matchScore": [0, 0],
        "auctionActions": [{"actor": "north", "action": "sun"}],
        "escalationActions": [], "projectActions": [{"actor": "north", "action": "pass"}],
        "projectDeclarations": [], "projectAssertions": [], "playActions": [], "stateAssertions": {},
    })
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    assert not exact.state["projects"]
    compiler = _compiler(rules)
    compiler._parse_start(exact, expected_rules_hash=rules.sha256, request=_request(mode="bounded"))
    longer = _request(mode="bounded").to_dict()
    longer.pop("sha256")
    longer["horizon"]["value"] = 2
    for request in (None, _request(), ScenarioProofRequestV1.build(longer)):
        with pytest.raises(ValueError, match="interleaved declaration"):
            compiler._parse_start(exact, expected_rules_hash=rules.sha256, request=request)


def test_graph_solves_each_action_path_once(rules, monkeypatch) -> None:
    paths = []
    original = OrToolsPlayTailStageV1.solve

    def counted(stage, prefix, **kwargs):
        paths.append(
            tuple(
                (int(action.actor), action.card.id, action.ekka)
                for action in prefix
            )
        )
        return original(stage, prefix, **kwargs)

    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", counted)

    compilation = _compile(rules, _request(mode="bounded"))

    assert compilation.complete is True
    assert len(paths) == len(set(paths))


def test_exact_kabout_utility_is_derived_from_native_terminal_scoring(rules) -> None:
    base = _seed(rules)
    seed = CompleteStateGraphSeedV1.build(
        exact_state_id=base.exact_state_id + "-KABOUT",
        rules_hash=base.rules_hash,
        remaining_hands=base.remaining_hands,
        current_trick=base.current_trick,
        first_leader=base.first_leader,
        next_actor=base.next_actor,
        legal_actions=base.legal_actions,
        contract=base.contract,
        trump=base.trump,
        buyer=base.buyer,
        locked=base.locked,
        base_raw=base.base_raw,
        base_tricks=(6, 0),
        awarded_project_points=base.awarded_project_points,
        baloot_points=base.baloot_points,
        escalation=base.escalation,
        last_escalating_team=base.last_escalating_team,
        starting_match_score=base.starting_match_score,
    )
    request_value = _request(threshold=1).to_dict()
    request_value.pop("sha256")
    request_value["utility"] = "kabout"
    request = ScenarioProofRequestV1.build(request_value)
    compilation = _compiler(rules).compile(
        seed,
        request,
        expected_rules_hash=rules.sha256,
        maximum_nodes=2_000,
        maximum_depth=8,
        maximum_seconds=30,
    )

    assert compilation.complete is True
    assert compilation.graph is not None
    assert {
        node.utility for node in compilation.graph.nodes if node.is_leaf
    } <= {-1, 1}
    proof = OrToolsScenarioGraphProofV1().prove(compilation.graph, request)
    assert proof.solver_status == "optimal"
    assert proof.winning_root_action_count is not None


@pytest.mark.parametrize(
    ("profile", "expected_award"),
    (
        ("saudi_common_v1.yaml", 288),
        ("saudi_federation_2023.yaml", 244),
    ),
)
@pytest.mark.parametrize("native_conversion", [False, True])
def test_terminal_graph_scoring_matches_profile_mutation_and_reference_oracle(
    root: Path,
    profile: str,
    expected_award: int,
    monkeypatch, native_conversion,
) -> None:
    rules = RuleBook.load(root / "configs" / "rules" / profile)
    scoring = ScoringDomainIRV1.from_rules(rules.data)
    hands = (
        (Card.parse("SK"), Card.parse("HK")),
        (Card.parse("S7"), Card.parse("H7")),
        (Card.parse("SA"), Card.parse("HA")),
        (Card.parse("S8"), Card.parse("H8")),
    )
    seed = CompleteStateGraphSeedV1.build(
        exact_state_id=f"STATE-FORCED-KABOUT-{profile}",
        rules_hash=rules.sha256,
        remaining_hands=hands,
        current_trick=(),
        first_leader=Seat.SOUTH,
        next_actor=Seat.SOUTH,
        legal_actions=tuple(
            TailPrefixActionV1.build(Seat.SOUTH, card)
            for card in hands[int(Seat.SOUTH)]
        ),
        contract=Contract.SUN,
        trump=None,
        buyer=Seat.SOUTH,
        locked=False,
        base_raw=(90, 0),
        base_tricks=(6, 0),
        awarded_project_points=(100, 0),
        baloot_points=(0, 0),
        escalation="double",
        last_escalating_team=Team.WEST_EAST,
        starting_match_score=(0, 0),
    )
    request = _request()
    if native_conversion:
        def forbidden(*args, **kwargs):
            pytest.fail("native graph scoring used interpreted conversion")
        monkeypatch.setattr(ScoringDomainIRV1, "convert_raw", forbidden)
    compilation = _compiler(rules, native_conversion=native_conversion).compile(
        seed,
        request,
        expected_rules_hash=rules.sha256,
        maximum_nodes=2_000,
        maximum_depth=8,
        maximum_seconds=30,
    )

    reference_award = scoring.round_award(
        contract=Contract.SUN,
        winner=Team.NORTH_SOUTH,
        declarer_team=Team.NORTH_SOUTH,
        escalation="double",
        total_before_award=(126, 0),
        project_points=(100, 0),
        baloot_points=(0, 0),
        kabout=True,
    )
    assert reference_award == (expected_award, 0)
    assert compilation.complete is True
    assert compilation.graph is not None
    assert {
        node.utility for node in compilation.graph.nodes if node.is_leaf
    } == {expected_award}


@pytest.mark.parametrize(
    ("budgets", "reason"),
    [
        ({"maximum_nodes": 1}, "maximum_nodes"),
        ({"maximum_depth": 1}, "maximum_depth"),
        ({"cancelled": lambda: True}, "cancelled"),
    ],
)
def test_incomplete_graph_receipts_preserve_exact_exhaustion_reason(
    rules,
    budgets,
    reason: str,
) -> None:
    compilation = _compile(rules, _request(), **budgets)

    assert compilation.complete is False
    assert compilation.graph is None
    assert compilation.receipt.status == "incomplete"
    assert compilation.receipt.exhaustion_reason == reason
    assert len(compilation.receipt.model_hash) == 64
    assert compilation.receipt.graph_hash is None


def test_graph_and_model_hashes_are_deterministic(rules) -> None:
    first = _compile(rules, _request())
    second = _compile(rules, _request())

    assert first.graph is not None and second.graph is not None
    assert first.graph.sha256 == second.graph.sha256
    assert first.receipt.model_hash == second.receipt.model_hash
    first_wire = first.receipt.to_dict()
    second_wire = second.receipt.to_dict()
    first_wire.pop("elapsedSeconds")
    second_wire.pop("elapsedSeconds")
    first_wire.pop("sha256")
    second_wire.pop("sha256")
    assert first_wire == second_wire


def test_decision_observer_is_opt_in_and_preserves_default_hashes(
    rules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ortools_complete_state_graph,
        "time",
        SimpleNamespace(monotonic=lambda: 0.0),
    )
    observer_ids: set[str] = set()

    def observe(observation) -> str:
        observation_id = canonical_sha256(
            {
                "actor": observation.actor.name.lower(),
                "orderedPublicPath": [
                    action.to_dict() for action in observation.prefix
                ],
            }
        )
        observer_ids.add(observation_id)
        return observation_id

    before = _compile(rules, _request())
    observed = _compile(
        rules,
        _request(),
        decision_observer=observe,
    )
    after = _compile(rules, _request())

    assert before.graph is not None and observed.graph is not None
    assert after.graph is not None
    assert before.decision_observation_ids == after.decision_observation_ids == {}
    observed_decisions = {
        node.node_id for node in observed.graph.nodes if not node.is_leaf
    }
    assert set(observed.decision_observation_ids) == observed_decisions
    assert set(observed.decision_observation_ids.values()) == observer_ids
    assert observed.graph.sha256 != before.graph.sha256
    assert (
        before.graph.sha256,
        before.receipt.sha256,
        before.receipt.graph_hash,
        before.receipt.model_hash,
        before.receipt.proof_state_projection_hash,
    ) == (
        after.graph.sha256,
        after.receipt.sha256,
        after.receipt.graph_hash,
        after.receipt.model_hash,
        after.receipt.proof_state_projection_hash,
    )


def test_ordered_public_path_observation_ids_split_transposition_identity() -> None:
    first_path = tuple(
        TailPrefixActionV1.build(actor, card)
        for actor, card in (
            (Seat.SOUTH, "HA"),
            (Seat.EAST, "HK"),
            (Seat.NORTH, "HJ"),
            (Seat.WEST, "H8"),
            (Seat.SOUTH, "H10"),
            (Seat.EAST, "HQ"),
            (Seat.NORTH, "H9"),
            (Seat.WEST, "H7"),
        )
    )
    second_path = (*first_path[4:], *first_path[:4])
    state_core = {
        "kind": "decision",
        "remainingHands": [[0], [1], [2], [3]],
        "currentTrick": [],
        "nextActor": "south",
        "raw": [100, 50],
        "tricks": [5, 2],
    }

    def observation_id(path: tuple[TailPrefixActionV1, ...]) -> str:
        return canonical_sha256(
            {
                "actor": "south",
                "orderedPublicPath": [action.to_dict() for action in path],
            }
        )

    first_id = observation_id(first_path)
    second_id = observation_id(second_path)
    first_default_core = _transposition_core_v1(state_core, None)
    second_default_core = _transposition_core_v1(dict(state_core), None)

    assert first_id != second_id
    assert canonical_sha256(first_default_core) == canonical_sha256(
        second_default_core
    )
    assert canonical_sha256(_transposition_core_v1(state_core, first_id)) != (
        canonical_sha256(_transposition_core_v1(state_core, second_id))
    )


def test_cp_seed_and_normalized_exact_state_have_identical_proof_inputs(rules) -> None:
    compiler = _compiler(rules)
    request = _request()
    budgets = {
        "expected_rules_hash": rules.sha256,
        "maximum_nodes": 2_000,
        "maximum_depth": 8,
        "maximum_seconds": 30.0,
    }

    exact_compilation = compiler.compile(_normalized_state(rules), request, **budgets)
    seed_compilation = compiler.compile(_seed(rules), request, **budgets)

    assert exact_compilation.graph is not None
    assert seed_compilation.graph is not None
    assert exact_compilation.graph.to_dict() == seed_compilation.graph.to_dict()
    assert exact_compilation.receipt.proof_state_projection_hash == (
        seed_compilation.receipt.proof_state_projection_hash
    )
    assert exact_compilation.receipt.model_hash == seed_compilation.receipt.model_hash
    exact_proof = OrToolsScenarioGraphProofV1().prove(
        exact_compilation.graph,
        request,
    )
    seed_proof = OrToolsScenarioGraphProofV1().prove(
        seed_compilation.graph,
        request,
    )
    assert exact_proof.to_dict() == seed_proof.to_dict()


def test_projection_validation_does_not_enumerate_the_graph(
    rules, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler = _compiler(rules)
    exact = _normalized_state(rules)

    monkeypatch.setattr(
        compiler,
        "_path_oracle",
        lambda *_args, **_kwargs: pytest.fail("projection validation enumerated paths"),
    )

    assert compiler.proof_state_projection_hash(
        exact,
        expected_rules_hash=rules.sha256,
    ) == _seed(rules).proof_state_projection_hash


def test_cp_seed_projection_checksum_is_mandatory(rules) -> None:
    seed = replace(_seed(rules), proof_state_projection_hash="0" * 64)

    with pytest.raises(ValueError, match="projection checksum mismatch"):
        _compiler(rules).compile(
            seed,
            _request(),
            expected_rules_hash=rules.sha256,
            maximum_nodes=2_000,
            maximum_depth=8,
            maximum_seconds=30.0,
        )


def test_graph_transitions_use_native_cp_sat_play_stage(
    rules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = OrToolsPlayTailStageV1.build_model

    def counted(self, *_args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *_args, **kwargs)

    monkeypatch.setattr(OrToolsPlayTailStageV1, "build_model", counted)

    compilation = _compile(rules, _request())

    assert compilation.complete is True
    assert compilation.receipt.explored_node_count > 1
    assert calls == 1


def test_graph_compiler_has_no_engine_bot_service_or_runtime_imports() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "baloot_platform"
        / "research"
        / "ortools_complete_state_graph.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = (
        "core.engine",
        "core.bots",
        "service",
        "runtime",
        "proof_service",
        "reference_play",
    )
    assert not any(any(item in name for item in forbidden) for name in imports)
