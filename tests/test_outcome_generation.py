import time
from types import SimpleNamespace

import pytest

from baloot_platform.analysis.models import BidChoice, InformationState
from baloot_platform.analysis.ismcts_v8 import analyze_information_set_mcts
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.outcome_generation import (
    OutcomePolicyDriver, OutcomePolicyIncomplete, _bid_observation, _informed_bid,
    construct_outcome, load_outcome_policy, validate_outcome_policy,
)
from baloot_platform.research.outcome_spec import compile_outcome_spec
from baloot_platform.research.scenario_contracts import ExactTracePrefixV1


@pytest.fixture
def setup(root):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v7.yaml")
    policy = load_outcome_policy(root / "configs/research/outcome_policy_v1.yaml")
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data["profile"]["id"], mode="sun"), rules)
    return rules, policy, compiled


@pytest.mark.parametrize("section,key,value", [
    ("informed", "deliberateMistakes", True), ("optimal", "onIncomplete", "heuristic"),
    ("optimal", "maximumConflicts", 250001),
])
def test_policy_profile_fails_closed(setup, section, key, value):
    _, policy, _ = setup
    policy[section][key] = value
    with pytest.raises(ValueError):
        validate_outcome_policy(policy)


@pytest.mark.parametrize("key,value", [("maximumWallSeconds", 6), ("maximumCpuSeconds", 6),
    ("maximumDepth", 1000), ("maximumWorkers", 2), ("maximumHiddenWorlds", 2)])
def test_native_budget_keeps_the_frozen_caps(setup, key, value):
    _, policy, _ = setup
    policy["optimal"]["proofBudget"][key] = value
    with pytest.raises(ValueError):
        validate_outcome_policy(policy)


def test_informed_bid_analyzer_only_receives_own_hand(setup, monkeypatch):
    import baloot_platform.research.outcome_generation as module
    rules, policy, _ = setup
    engine = BalootEngine(rules)
    state = engine.create_round(seed=92, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    machine = BiddingMachine(state.dealer, state.purchase_card.suit, AuctionDomainIRV1.from_rules(rules.data), state.purchase_card.rank)
    actor = machine.actor
    observation = _bid_observation(state, machine, actor)
    assert "hands" not in observation and "deckOrder" not in observation
    def analyze(shell, _machine, acting, _rules, **kwargs):
        assert acting == actor
        assert shell.hands[int(actor)] == state.hands[int(actor)]
        assert all(shell.hands[int(seat)] == 0 for seat in Seat if seat != actor)
        assert not shell.undealt and shell.shuffle is None
        return SimpleNamespace(choices=[SimpleNamespace(choice=BidChoice("pass"), expected_match_margin=1,
            team_win_probability=1, best_share=1)], to_dict=lambda: {"test": "sealed shell"})
    monkeypatch.setattr(module, "analyze_bid", analyze)
    choice, _ = _informed_bid(observation, machine, rules, policy["informed"], 1, lambda: False)
    assert choice == BidChoice("pass")


@pytest.mark.real_clock
def test_informed_escalation_uses_one_observer_safe_cohort(setup):
    from baloot_platform.core.enums import Contract
    rules, policy, compiled = setup
    policy["informed"]["particles"] = 2
    driver = OutcomePolicyDriver(rules, compiled, policy, seed=92, deadline=time.monotonic()+10)
    engine = driver.engine
    state = engine.create_round(seed=92, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    assert engine.bid_and_complete_deal(state, forced_contract=Contract.HOKUM, forced_buyer=Seat.NORTH)
    action, scope, evidence = driver._informed_escalation(state, Seat.WEST, ["decline", "double"])
    assert action == min(evidence["means"], key=lambda code: (-evidence["means"][code], code))
    assert len(evidence["values"]["decline"]) == len(evidence["values"]["double"]) == 2
    assert "initialHands" not in scope and "deckOrder" not in scope
    assert scope["projects"] == []
    driver.seed += 1000
    driver.receipts.append({"unrelated": True})
    repeated = driver._informed_escalation(state, Seat.WEST, ["decline", "double"])
    assert repeated == (action, scope, evidence)


@pytest.mark.real_clock
def test_incomplete_exact_proof_is_never_relabelled(setup, monkeypatch):
    import baloot_platform.research.outcome_generation as module
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, ScenarioQualityExecutionBudgetV1
    rules, policy, compiled = setup
    driver = OutcomePolicyDriver(rules, compiled, policy, seed=1, deadline=time.monotonic()+10)
    driver.normalizer = SimpleNamespace(normalize=lambda *args, **kwargs: SimpleNamespace(next_actor="north"))
    def incomplete(*args, **kwargs):
        assert isinstance(kwargs["budget"], ScenarioProofBudgetV1)
        assert isinstance(kwargs["execution_budget"], ScenarioQualityExecutionBudgetV1)
        assert kwargs["maximum_conflicts"] == policy["optimal"]["maximumConflicts"]
        return {"proof": {"completionStatus": "incomplete", "solverStatus": "unknown"}}
    monkeypatch.setattr(module, "_complete_state_quality_artifact_v1", incomplete)
    prefix = ExactTracePrefixV1.build(dict(schemaVersion=1, deckOrder=list(range(32)), dealer="east",
        roundNumber=1, matchScore=[0, 0], auctionActions=[], escalationActions=[],
        projectDeclarations=[], playActions=[], stateAssertions={}))
    with pytest.raises(OutcomePolicyIncomplete, match="optimal_proof_incomplete"):
        driver.exact(prefix, "bid", Seat.NORTH)
    assert not driver.receipts


@pytest.mark.real_clock
def test_complete_proof_returned_after_decision_deadline_is_rejected(setup, monkeypatch):
    import baloot_platform.research.outcome_generation as module
    rules, policy, compiled = setup
    clock = [0.0]
    driver = OutcomePolicyDriver(rules, compiled, policy, seed=1, deadline=100)
    driver.normalizer = SimpleNamespace(normalize=lambda *args, **kwargs: SimpleNamespace(next_actor="north"))
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    def late(*args, **kwargs):
        clock[0] = 6
        return {"proof": {"completionStatus": "exact_complete", "solverStatus": "optimal", "selectedAction": "C7"}}
    monkeypatch.setattr(module, "_complete_state_quality_artifact_v1", late)
    with pytest.raises(OutcomePolicyIncomplete, match="optimal_decision_budget"):
        driver.exact(None, "bid", Seat.NORTH)
    assert driver.receipts == []


@pytest.mark.real_clock
def test_constructor_binds_the_callback_and_reports_every_rejection(setup, monkeypatch):
    import baloot_platform.research.outcome_generation as module
    rules, policy, compiled = setup
    seen = []
    class Compiler:
        def __init__(self, **kwargs):
            assert kwargs["native_cp_sat_scoring"] is True
            seen.append(kwargs["play_model_configurator"])
        def __call__(self, *args, **kwargs):
            raise OutcomePolicyIncomplete("auction_policy_incompatible")
    monkeypatch.setattr(module, "GenericCompleteTraceCompilerV1", Compiler)
    result = construct_outcome(compiled, rules, policy, seed=1, maximum_seconds=2, maximum_attempts=2)
    assert len(seen) == 2 and all(callable(callback) for callback in seen)
    assert result["status"] == "incomplete" and result["accepted"] is None
    assert result["rejectReasons"] == {"auction_policy_incompatible": 2}
    assert len(result["attempts"]) == 2
    assert result["acceptedPerSecond"] == 0 and result["cpuSecondsPerAccepted"] is None


def test_changed_compiled_identity_fails_before_construction(setup):
    rules, policy, compiled = setup
    compiled.source["mode"] = "hokum"
    with pytest.raises(ValueError, match="identity mismatch"):
        construct_outcome(compiled, rules, policy, seed=1)


def test_raw_objective_is_not_silently_run_under_the_old_card_policy(setup, monkeypatch):
    import baloot_platform.research.outcome_generation as module
    rules, policy, _ = setup
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data["profile"]["id"],
        mode="sun", objective="raw_point_margin"), rules)
    def forbidden(**kwargs):
        raise AssertionError("unadmitted objective launched a construction")
    monkeypatch.setattr(module, "GenericCompleteTraceCompilerV1", forbidden)
    with pytest.raises(ValueError, match="not admitted yet"):
        construct_outcome(compiled, rules, policy, seed=1)


def test_mcts_action_profile_supports_ekka_and_cancellation(setup):
    from baloot_platform.core.cards import FULL_DECK
    from baloot_platform.core.models import LegalPlayActionV1
    from baloot_platform.core.serialization import replay_trusted_context
    rules, _, _ = setup
    hands = {seat.name.lower(): [card.code for card in FULL_DECK if int(card.suit) == int(seat)] for seat in Seat}
    state = replay_trusted_context(dict(dealer="east", contract="hokum", buyer="north", trumpSuit="hearts",
        firstLeader="north", initialHands=hands, completedTricks=[], currentTrick=None), rules)
    engine = BalootEngine(rules)
    engine._announce_project_for_turn(state, Seat.NORTH)
    info = InformationState.from_complete(state, Seat.NORTH)
    assert any(project.owner == Seat.NORTH and project.declared for project in info.projects)
    with pytest.raises(InterruptedError):
        analyze_information_set_mcts(info, engine, cancel_check=lambda: True)
    legal = engine.legal_play_actions(state)
    assert any(action.ekka for action in legal)
    result = analyze_information_set_mcts(info, engine, particles=2, exact_allocation_limit=1,
        maximum_sampling_attempts=100, iterations=16, maximum_tree_depth=2, include_ekka=True, seed=9)
    assert {(row.card.code, row.ekka) for row in result.actions} == {(row.card.code, row.ekka) for row in legal}
    assert LegalPlayActionV1(result.recommended_card, result.recommended_ekka) in legal
    assert result.method == "information_set_mcts_actions_v2"


def test_real_native_constructor_calls_outcome_constraints(setup, monkeypatch):
    from baloot_platform.research.ortools_research_executor import GenericCompleteTraceCompilerV1
    from baloot_platform.research.scenario_predicate_manifest_v2 import shared_scenario_predicate_catalog_lowering_manifest_v2
    rules, _, _ = setup
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data["profile"]["id"],
        mode="sun", category="kaboot", declarations=[dict(category="sira", state="eligible")]), rules)
    manifest = shared_scenario_predicate_catalog_lowering_manifest_v2()
    original, kinds = type(manifest).lower_predicate, set()
    def lower(self, predicate, *args, **kwargs):
        kinds.add(predicate.kind)
        return original(self, predicate, *args, **kwargs)
    monkeypatch.setattr(type(manifest), "lower_predicate", lower)
    seen = []
    def reject_model(build, context):
        compiled.constrain_play_model(build)
        assert set(context) == {"deckOrder", "dealer", "matchScore", "auction", "escalation"}
        assert len(context["deckOrder"]) == 32
        seen.append(context["deckOrder"])
        build.model.add_bool_or([])
    result = GenericCompleteTraceCompilerV1(rules=rules, native_cp_sat_scoring=True,
        play_model_configurator=reject_model)(compiled.scenario, (), seed=20261025,
                                            maximum_seconds=12, maximum_conflicts=250000)
    assert seen
    assert result.candidate is None
    assert result.reason == "fixed_deal_play_infeasible"
    assert result.partial_receipt["exhaustionReason"] == "infeasible"
    assert {"auction.contract", "project.exists", "outcome.kabout", "outcome.team_round"} <= kinds


@pytest.fixture(scope="module")
def informed_round(root):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v7.yaml")
    policy = load_outcome_policy(root / "configs/research/outcome_policy_v1.yaml")
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data["profile"]["id"],
        mode="sun", category="big_swing", policies={"default": "informed"}), rules)
    result = construct_outcome(compiled, rules, policy, seed=20261026, maximum_seconds=30, maximum_attempts=1)
    assert result["status"] == "complete", result["rejectReasons"]
    return result


def test_real_informed_full_round_is_native_admitted(informed_round):
    result = informed_round
    accepted = result["accepted"]
    assert len(accepted["exactTracePrefix"]["playActions"]) == 32
    receipts = accepted["decisionReceipts"]
    assert sum(row["decisionKind"] == "play" for row in receipts) == 32
    assert sum(row["decisionKind"] == "project" for row in receipts) == 4
    assert all(row["policy"] == ("spec_fixed" if row["decisionKind"] in {"bid", "escalation"} else "informed")
               for row in receipts)
    assert all(row["guarantee"] in {"forced_by_rules", "sampled_policy_not_exact", "spec_fixed_legal"} for row in receipts)
    assert all(len(row["inputScopeHash"]) == 64 for row in receipts)
    assert accepted["trace"]["score"]["gamePoints"] == accepted["score"]["awardedPoints"]
    assert [row["action"] for row in receipts if row["decisionKind"] == "project"] == accepted["exactTracePrefix"]["projectActions"]
    assert result["attempts"][0]["reason"] == "accepted"


@pytest.mark.real_clock
@pytest.mark.parametrize("fixed", [False, True])
def test_card_policy_never_launches_an_optimal_bid_proof(setup, informed_round, monkeypatch, fixed):
    import baloot_platform.research.outcome_generation as module
    rules, policy, _ = setup
    raw = dict(schemaVersion=1, rulesProfile=rules.data["profile"]["id"], mode="sun", policies={"default": "optimal"})
    if fixed:
        raw["auction"] = [{"actor": "north", "action": "sun"}]
    compiled = compile_outcome_spec(raw, rules)
    driver = OutcomePolicyDriver(rules, compiled, policy, seed=1, deadline=time.monotonic()+10)
    def informed(observation, machine, *args):
        raise AssertionError("a spec-fixed contract must not consult the informed bidder")
    def card_only(prefix, kind, actor):
        assert kind == "play"
        raise OutcomePolicyIncomplete("stop_at_card")
    monkeypatch.setattr(module, "_informed_bid", informed)
    monkeypatch.setattr(driver, "exact", card_only)
    with pytest.raises(OutcomePolicyIncomplete, match="stop_at_card"):
        driver.construct_line(informed_round["accepted"]["trace"])
    bid = next(row for row in driver.receipts if row["decisionKind"] == "bid")
    assert bid["bid_policy"] == bid["policy"] == "spec_fixed"
    assert bid['evidence']['legalityOnly'] is True


def test_native_prefix_keeps_the_current_mandatory_project_before_its_card(setup, informed_round):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    rules, _, _ = setup
    raw = informed_round["accepted"]["exactTracePrefix"].copy()
    raw.pop("sha256")
    raw["playActions"] = raw["playActions"][:3]
    # All four mandatory declarations, but only three cards: the last actor's
    # project must not disappear or be mistaken for a completed first trick.
    assert len(raw["projectActions"]) == 4
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(raw), decision_point={"kind": "play"}, maximum_seconds=5)
    assert normalized.next_actor == raw["projectActions"][3]["actor"]
    assert len(normalized.state["currentTrick"]["plays"]) == 3
    assert sum(event["type"] in {"project_declared", "project_passed"}
               for event in normalized.state["events"]) == 4


@pytest.mark.real_clock
def test_exact_fourth_card_crosses_comparison_without_losing_project_evidence(setup, informed_round):
    from baloot_platform.research.ortools_exact_state import ExactNormalizationStopped
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    rules, policy, compiled = setup
    raw = informed_round["accepted"]["exactTracePrefix"].copy()
    raw.pop("sha256")
    actor = Seat.parse(raw["playActions"][3]["actor"])
    raw["playActions"] = raw["playActions"][:3]
    driver = OutcomePolicyDriver(rules, compiled, policy, seed=1, deadline=time.monotonic()+10)
    try:
        _, artifact = driver.exact(ExactTracePrefixV1.build(raw), "play", actor)
    except (ExactNormalizationStopped, InformationSetCompilationStopped):
        # A bounded native attempt may be incomplete; a project-boundary
        # ValueError, or a silently weakened proof, is never an allowed result.
        assert driver.receipts == []
    else:
        assert artifact["proof"]["completionStatus"] == "exact_complete"
        assert artifact["proof"]["solverStatus"] == "optimal"


@pytest.mark.real_clock
def test_real_late_exact_policy_uses_completed_native_proof(setup, informed_round):
    rules, policy, compiled = setup
    raw = informed_round["accepted"]["exactTracePrefix"].copy()
    raw.pop("sha256")
    actor = Seat.parse(raw["playActions"][28]["actor"])
    raw["playActions"] = raw["playActions"][:28]
    driver = OutcomePolicyDriver(rules, compiled, policy, seed=1, deadline=time.monotonic()+10)
    action, artifact = driver.exact(ExactTracePrefixV1.build(raw), "play", actor)
    assert artifact["proof"]["completionStatus"] == "exact_complete"
    assert artifact["proof"]["solverStatus"] == "optimal"
    roots = artifact["proof"]["rootActions"]
    assert next(row["value"] for row in roots if row["action"] == action) == max(row["value"] for row in roots)
