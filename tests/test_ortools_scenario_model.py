from __future__ import annotations

import pytest
import copy
from dataclasses import replace
from ortools.sat.python import cp_model

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Rank, Seat, Suit, Team
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.auction_compiler import OrToolsCpSatAuctionCompilerV1
from baloot_platform.research.ortools_scenario_model import (
    OrToolsScenarioModelV1,
    OrToolsNativeAuctionFragmentV1,
    attach_auction_legal_actions_v1,
    attach_auction_path_v1,
    attach_auction_hand_values_v1,
    attach_auction_action_values_v1,
    reify_scenario_expression,
)
from baloot_platform.research.scenario_contracts import ScenarioExpressionV1


@pytest.mark.parametrize('cancelled,reused', [(True, False), (True, True), (False, False)])
def test_deal_solve_cancelled_before_start_has_no_response_or_stale_statistics(rules, cancelled, reused):
    from types import SimpleNamespace

    class Solver:
        def __init__(self):
            self.parameters = SimpleNamespace()
            self.calls = int(reused)

        def solve(self, _model):
            self.calls += 1
            return cp_model.UNKNOWN

        def stop_search(self):
            pass

        def status_name(self, status):
            assert status == cp_model.UNKNOWN
            return 'UNKNOWN'

        @property
        def num_branches(self):
            assert self.calls, 'no solver response exists before solve'
            return 7

        @property
        def num_conflicts(self):
            assert self.calls, 'no solver response exists before solve'
            return 3

    solver = Solver()
    result = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST).solve(
        solver=solver, maximum_seconds=1, cancel_check=lambda: cancelled)
    assert result.status == 'not_found_within_budget' and result.solver_status == 'unknown'
    assert result.deck_order is None
    assert solver.calls == int(reused) + int(not cancelled)
    assert (result.branches, result.conflicts) == ((0, 0) if cancelled else (7, 3))


def _native_suit_monopoly_bid(root):
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    bundle = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml")
    rules = bundle.rules
    dealer = Seat.EAST
    buyer = rules.structural_domain.dealer_right(dealer)
    layout = DealLayoutV1.build(rules.structural_domain, dealer=dealer, buyer=buyer)
    hands = {seat: set(range(int(seat) * 8, int(seat) * 8 + 8)) for seat in Seat}
    deck = [None] * 32
    for seat in Seat:
        for position, card in zip(layout.positions_for(seat, final=True), sorted(hands[seat]), strict=True):
            deck[position] = card
    prefix = ExactTracePrefixV1.build(dict(schemaVersion=1, deckOrder=deck, dealer="east",
        auctionActions=[], escalationActions=[], projectDeclarations=[], playActions=[]))
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    assert Seat.parse(exact.next_actor) == buyer
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    return compiler, exact, prefix, hands


@pytest.mark.real_clock
@pytest.mark.parametrize("world_ranking_id,selected_action", ((None, "sun"),
    ("opponent_auction_value_v1", "sun"), ("opponent_auction_value_v1", "hokum:spades")))
def test_native_bid_world_search_is_deterministic_observer_compatible_and_nonqualifying(root, monkeypatch, world_ranking_id, selected_action):
    from baloot_platform.core.exact_state_view import canonical_observer_information_hash_v1
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research import ortools_information_set_compiler as module
    from baloot_platform.research.scenario_contracts import canonical_sha256

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    limits = dict(maximum_seconds=15, maximum_conflicts=250000, maximum_model_bytes=100000000,
        maximum_worlds=3)
    args = dict(selected_action=selected_action, world_ranking_id=world_ranking_id, **limits)
    first, receipt = compiler._find_native_bid_compatible_world(exact, prefix, **args)
    repeated, repeated_receipt = compiler._find_native_bid_compatible_world(exact, prefix, **args)
    assert repeated == first and repeated_receipt == receipt
    assert receipt["status"] == "candidate_found" and receipt["queriesUsed"] == 1
    assert receipt["strategicCoverage"] is receipt["impossibilityProven"] is False
    assert receipt["sha256"] == canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})
    if world_ranking_id is None:
        assert "ranking" not in receipt and "rankingValue" not in receipt
    else:
        assert receipt["ranking"]["scope"] == "construction_preference_only"
        assert receipt["ranking"]["stoppingRule"] == (
            "ranking_optimum" if selected_action.startswith("hokum:") else "first_feasible_solution")
        assert receipt["ranking"]["searchOrder"] == (
            "automatic" if selected_action.startswith("hokum:") else "native_sun_rank_then_deck_v1")
        assert receipt["ranking"]["id"] == world_ranking_id
        assert type(receipt["rankingValue"]) is int
    second, second_receipt = compiler._find_native_bid_compatible_world(exact, prefix,
        excluded_decks=[first.deck_order], **args)
    assert second.deck_order != first.deck_order
    reordered, reordered_receipt = compiler._find_native_bid_compatible_world(exact, prefix,
        excluded_decks=[first.deck_order, first.deck_order], **args)
    assert reordered == second and reordered_receipt == second_receipt
    normalizer = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True)
    observed = canonical_observer_information_hash_v1(complete_state=exact.state, acting_seat=exact.next_actor)
    for world in (first, second):
        assert world.auction_actions == prefix.auction_actions
        assert not world.state_assertions and not world.project_assertions
        state = normalizer.normalize(world, decision_point={"kind": "bid"}, maximum_seconds=15)
        assert canonical_observer_information_hash_v1(
            complete_state=state.state, acting_seat=state.next_actor) == observed
        assert state.legal_actions == exact.legal_actions
    # Changing only hidden cards cannot guide the actor-visible selector.
    equivalent = normalizer.normalize(first, decision_point={"kind": "bid"}, maximum_seconds=15)
    equivalent_world, _ = compiler._find_native_bid_compatible_world(equivalent, first, **args)
    assert equivalent_world.deck_order == first.deck_order
    for changes in ({"selected_action": "invalid"}, {"maximum_seconds": True},
                    {"excluded_decks": [[0] * 32]}, {"excluded_decks": [[True, *range(1, 32)]]},
                    {"world_ranking_id": "unknown"}):
        with pytest.raises(ValueError):
            compiler._find_native_bid_compatible_world(exact, prefix, **{**args, **changes})
    for changes, reason in (({"cancel_check": lambda: True}, "cancelled"),
                            ({"maximum_model_bytes": 1}, "maximum_model_bytes"),
                            ({"maximum_worlds": 1, "excluded_decks": [first.deck_order]}, "maximum_worlds")):
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            compiler._find_native_bid_compatible_world(exact, prefix, **{**args, **changes})
        assert stopped.value.reason == reason
        assert stopped.value.partial_receipt["evidence"]["queriesUsed"] == 0
    with monkeypatch.context() as patch:
        ticks = iter((0, 16))
        patch.setattr(module.time, "monotonic", lambda: next(ticks))
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            compiler._find_native_bid_compatible_world(exact, prefix, **args)
    assert stopped.value.reason == "maximum_seconds"
    for status, expected in ((cp_model.UNKNOWN, "not_found_within_budget"),
                             (cp_model.INFEASIBLE, "candidate_space_exhausted")):
        with monkeypatch.context() as patch:
            patch.setattr(module, "_solve_cp_sat_with_cancel_v1", lambda *a, **kw: status)
            world, stopped = compiler._find_native_bid_compatible_world(exact, prefix, **args)
        assert world is None and stopped["status"] == expected
        assert stopped["queriesUsed"] == 1 and stopped["impossibilityProven"] is False


@pytest.mark.parametrize("stop_first,guarded,minimax", (
    (False, False, False), (True, False, False), ("refuted", False, False),
    pytest.param(False, True, False, id="guarded"),
    pytest.param(False, True, True, id="guarded_minimax")))
def test_observed_bid_world_search_coordinates_budgets_and_fresh_reader(root, monkeypatch, stop_first, guarded, minimax):
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    policy_id = "observer_safe_partner_feed_closed_trick_v1" if guarded else None
    budget = None
    if minimax:
        import yaml
        from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1
        settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
        budget = ScenarioProofBudgetV1.build("actor_information_set_exact_v1",
            settings["proofBudgets"]["profiles"]["actor_information_set_exact_v1"])
    calls = []
    normalization_calls = []
    native_solve = OrToolsScenarioModelV1.solve
    def counted_solve(*args, **kwargs):
        normalization_calls.append(1)
        return native_solve(*args, **kwargs)
    monkeypatch.setattr(OrToolsScenarioModelV1, "solve", counted_solve)
    def prepare(*args, **kwargs):
        assert kwargs["visibility"] == "actor_information_set" and kwargs["fixed_observed_world"] is True
        assert kwargs["compact_purchase_invariant_ownership"] is True
        calls.append("prepare")
        return (object(),), {"queriesUsed": 1}
    def synthesize(*args, **kwargs):
        assert kwargs["required_threshold"] == 0
        assert kwargs["shorten_counterexamples"] is (not guarded or minimax)
        assert kwargs.get("_play_fallback_id") == policy_id
        assert kwargs.get("minimax_budget") is budget
        calls.append("strategy")
        if stop_first is True and calls.count("strategy") == 1:
            raise InformationSetCompilationStopped("maximum_seconds", evidence={"queriesUsed": 1})
        if stop_first == "refuted" and calls.count("strategy") == 1:
            return {"queriesUsed": 1, "status": "scoped_strategy_threshold_refuted",
                    "requiredThreshold": 0, "upperBound": -1}
        return {"queriesUsed": 1, "status": "scoped_strategy_threshold_complete"}
    def verify(*args, **kwargs):
        assert kwargs.get("strategy_play_fallback_id") == policy_id
        assert kwargs.get("minimax_budget") is budget
        calls.append("fresh")
        kwargs["_on_queries_used"](1)
        return 0
    monkeypatch.setattr(compiler, "_prepare_native_bid_branches", prepare)
    monkeypatch.setattr(compiler, "_solve_symbolic_round_policy_bounds", synthesize)
    monkeypatch.setattr(authority, "validate_native_observed_bid_opponent_upper_v1", verify)
    # Real selection and normalization; strategic seams are tested separately below.
    limits = dict(maximum_worlds=2, maximum_seconds=30, maximum_queries=12,
        maximum_world_seconds=15, maximum_world_queries=6, maximum_iterations=2,
        maximum_observations=100, maximum_branches=13, maximum_model_bytes=100000000,
        maximum_evidence_bytes=100000000,
        maximum_conflicts=250000, maximum_depth=64)
    if guarded:
        limits["strategy_play_fallback_id"] = policy_id
    if minimax:
        limits["minimax_budget"] = budget
    result, search = compiler._search_observed_bid_alternative_upper(exact, prefix,
        selected_action="sun", upper_bound=0, **limits)
    assert result["receipt"]["kind"] == "native_observed_bid_opponent_policy_upper_v1"
    assert search["status"] == "upper_bound_proven"
    assert search["strategicCoverage"] is search["impossibilityProven"] is False
    assert search["queriesCharged"] == (11 if stop_first == "refuted" else 12 if stop_first else 6)
    assert len(normalization_calls) == 2 * len(search["attempts"])
    assert calls == (["prepare", "strategy"] if stop_first else []) + ["prepare", "strategy", "fresh"]
    from baloot_platform.research.scenario_contracts import canonical_json
    if not stop_first:
        # The search log fits, but the returned normalized world and proof do not.
        combined_cap = len(canonical_json(search).encode("utf-8")) + 1024
        oversized, bounded = compiler._search_observed_bid_alternative_upper(exact, prefix,
            selected_action="sun", upper_bound=0, **{**limits, "maximum_evidence_bytes": combined_cap})
        assert oversized is None and bounded["exhaustionReason"] == "maximum_evidence_bytes"
        assert bounded["attempts"][-1]["stage"] == "fresh_verification"
        assert len(bounded["attempts"]) == 1
        for mutation in ({"requiredThreshold": False}, {"requiredThreshold": 1},
                         {"upperBound": True}, {"upperBound": 0}):
            with monkeypatch.context() as patch:
                patch.setattr(compiler, "_solve_symbolic_round_policy_bounds", lambda *a, **kw:
                    {"queriesUsed": 1, "status": "scoped_strategy_threshold_refuted",
                     "requiredThreshold": 0, "upperBound": -1, **mutation})
                with pytest.raises(ValueError, match="refutation contradicts"):
                    compiler._search_observed_bid_alternative_upper(exact, prefix,
                        selected_action="sun", upper_bound=0, **limits)
    if stop_first:
        first, second = search["attempts"]
        assert first["status"] == ("opponent_threshold_refuted" if stop_first == "refuted" else "incomplete")
        assert first["queriesCharged"] == (5 if stop_first == "refuted" else 6)
        assert first["worldPrefix"]["deckOrder"] != second["worldPrefix"]["deckOrder"]
        if stop_first is True:
            assert first["partial"]["exhaustionReason"] == "maximum_seconds"
        else:
            assert "partial" not in first
    calls.clear()
    result, cancelled = compiler._search_observed_bid_alternative_upper(exact, prefix,
        selected_action="sun", upper_bound=0, cancel_check=lambda: True, **limits)
    assert result is None and cancelled["status"] == "cancelled" and not calls
    result, exhausted = compiler._search_observed_bid_alternative_upper(exact, prefix,
        selected_action="sun", upper_bound=0, **{**limits, "maximum_evidence_bytes": 1})
    assert result is None and len(exhausted["attempts"]) == 1 and not calls
    assert exhausted["exhaustionReason"] == "maximum_evidence_bytes"
    from baloot_platform.research.scenario_contracts import canonical_json
    partial_cap = len(canonical_json(search["attempts"][0]).encode("utf-8")) // 2
    result, exhausted = compiler._search_observed_bid_alternative_upper(exact, prefix,
        selected_action="sun", upper_bound=0, **{**limits, "maximum_evidence_bytes": partial_cap})
    assert result is None and exhausted["exhaustionReason"] == "maximum_evidence_bytes"
    assert len(exhausted["attempts"]) == 1 and "worldPrefix" in exhausted["attempts"][0]
    assert len(canonical_json(exhausted).encode("utf-8")) > partial_cap  # Preserve the offending diagnostic.
    calls.clear()
    result, exhausted = compiler._search_observed_bid_alternative_upper(exact, prefix,
        selected_action="sun", upper_bound=0, **{**limits, "maximum_model_bytes": 1})
    assert result is None and len(exhausted["attempts"]) == 1 and not calls
    assert exhausted["attempts"][0]["partial"]["exhaustionReason"] == "maximum_model_bytes"
    result, exhausted = compiler._search_observed_bid_alternative_upper(exact, prefix,
        selected_action="sun", upper_bound=0, **{**limits, "maximum_queries": 1})
    assert result is None and exhausted["queriesCharged"] == 1 and not calls
    assert exhausted["impossibilityProven"] is False
    with monkeypatch.context() as patch:
        patch.setattr(compiler, "_solve_symbolic_round_policy_bounds", lambda *a, **kw:
            {"queriesUsed": 1, "status": "scoped_strategy_threshold_complete"})
        def failed_readback(*args, **kwargs):
            raise InformationSetCompilationStopped("maximum_queries", evidence={"queriesUsed": 1})
        patch.setattr(authority, "validate_native_observed_bid_opponent_upper_v1", failed_readback)
        result, exhausted = compiler._search_observed_bid_alternative_upper(exact, prefix,
            selected_action="sun", upper_bound=0, **{**limits, "maximum_worlds": 1})
    assert result is None and exhausted["status"] == "not_found_within_budget"
    assert exhausted["attempts"][0]["stage"] == "fresh_verification"
    assert exhausted["queriesCharged"] == limits["maximum_world_queries"]
    with pytest.raises(ValueError):
        compiler._search_observed_bid_alternative_upper(exact, prefix,
            selected_action="sun", upper_bound=True, **limits)


def test_observed_penultimate_pass_refutes_floor_without_fixed_play_policy(root):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    compiler, _, initial, _ = _native_suit_monopoly_bid(root)
    raw = initial.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[{"action": "pass"}] * 6, projectActions=[])
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    stopped = compiler._prove_observed_bid_dominance(exact, prefix,
        selected_action="pass", opponent_policy={"auction_action": "pass", "escalation_action": "decline"},
        threshold_refutation_probe=dict(schemaVersion=1, maximumSeconds=60, maximumQueries=200,
            worldRankingId="opponent_auction_value_v1", opponentAuctionAction="pass"),
        alternative_proof=dict(schemaVersion=1, upperBound=1, maximumWorlds=1,
            maximumSeconds=10, maximumQueries=50, maximumWorldSeconds=10, maximumWorldQueries=50),
        maximum_evidence_bytes=100000000, maximum_iterations=2,
        maximum_queries=1000, maximum_observations=100, maximum_branches=13,
        maximum_model_bytes=100000000, maximum_conflicts=250000, maximum_depth=64, maximum_seconds=30)
    assert stopped["status"] == "incomplete" and stopped["stage"] == "visible_strategy"
    assert stopped["thresholdRefutationProbe"]["status"] == "threshold_refuted"
    counterworld = ExactTracePrefixV1.build(stopped["thresholdRefutationProbe"]["worldPrefix"])
    assert counterworld.sha256 == stopped["thresholdRefutationProbe"]["worldPrefixHash"]
    assert counterworld.auction_actions == prefix.auction_actions
    upper = stopped["thresholdRefutationProbe"]["upperProof"]
    assert upper["upperBound"] == 0
    assert upper["scope"]["nextDealAnalyzed"] is False
    assert stopped["selectedProof"]["status"] == "scoped_strategy_threshold_refuted"
    assert stopped["selectedProof"]["strategy"]["upperBound"] == 0
    assert stopped["alternatives"] == [] and stopped["certificate"] is None
    assert "incumbentAttempt" not in stopped


@pytest.mark.parametrize("version", (1, 2))
def test_observed_bid_world_search_produces_real_fresh_all_pass_cap(root, version):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    compiler, _, initial, _ = _native_suit_monopoly_bid(root)
    raw = initial.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[{"action": "pass"}] * 7, projectActions=[])
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    searcher = (compiler._search_observed_bid_alternative_upper if version == 1
        else compiler._search_observed_bid_information_set_upper)
    search_limits = dict(maximum_seconds=30, maximum_queries=100,
        maximum_iterations=2, maximum_observations=100, maximum_branches=13,
        maximum_model_bytes=100000000, maximum_evidence_bytes=100000000,
        maximum_conflicts=250000, maximum_depth=64)
    if version == 1:
        search_limits.update(maximum_worlds=1, maximum_world_seconds=30, maximum_world_queries=100)
    result, search = searcher(exact, prefix, selected_action="pass", upper_bound=0, **search_limits)
    assert result is not None and search["status"] == "upper_bound_proven"
    assert result["receipt"]["upperBound"] == 0
    if version == 1:
        assert result["receipt"]["preparation"]["scope"]["fixedObservedWorld"]["coverage"] == "single_compatible_deal"
        assert search["queriesCharged"] <= 100 and len(search["attempts"]) == 1
    else:
        from baloot_platform.research.result_authority import _validate_native_observed_bid_refutation_search_v1
        from baloot_platform.research.scenario_contracts import canonical_sha256
        assert "fixedObservedWorld" not in result["receipt"]["preparation"]["scope"]
        assert result["receipt"]["strategy"]["kind"] == "native_symbolic_round_strategy_threshold_v1"
        assert "playFallback" not in result["receipt"]["strategy"]["scope"]
        configuration = dict(schemaVersion=2, method="information_set_refutation_v1",
            upperBound=0, maximumSeconds=30, maximumQueries=100)
        context = dict(configuration=configuration, source_prefix=prefix, rules=compiler.rules,
            action="pass", target_team=Seat.parse(exact.next_actor).team.name.lower(),
            limits={"maximumIterations": 2, "maximumObservations": 100, "maximumBranches": 13,
                "maximumModelBytes": 100000000, "maximumConflicts": 250000, "maximumDepth": 64},
            maximum_evidence_bytes=100000000)
        assert _validate_native_observed_bid_refutation_search_v1(search, result["receipt"], **context) == search["queriesCharged"]
        for key, value in (("status", "incomplete"), ("queriesCharged", True), ("sourcePrefixHash", "0" * 64),
                           ("upperBound", 1), ("queryCountComplete", False)):
            forged = {**search, key: value}
            forged["sha256"] = canonical_sha256({k: v for k, v in forged.items() if k != "sha256"})
            with pytest.raises(ValueError):
                _validate_native_observed_bid_refutation_search_v1(forged, result["receipt"], **context)
        absent, incomplete = searcher(exact, prefix, selected_action="pass", upper_bound=0,
            **{**search_limits, "maximum_evidence_bytes": 1})
        assert absent is None and incomplete["status"] == "incomplete"
        assert incomplete["queriesCharged"] == 100 and incomplete["impossibilityProven"] is False
        absent, not_refuted = searcher(exact, prefix, selected_action="pass", upper_bound=-1, **search_limits)
        assert absent is None and not_refuted["status"] == "threshold_not_refuted"
        absent, cancelled = searcher(exact, prefix, selected_action="pass", upper_bound=0,
            **search_limits, cancel_check=lambda: True)
        assert absent is None and cancelled["status"] == "incomplete" and not cancelled["queryCountComplete"]
    assert result["receipt"]["preparation"]["scope"]["nextDealAnalyzed"] is False
    # Passing at this terminal auction root yields zero, so it cannot meet
    # a positive proposed floor. Native synthesis must establish that before
    # spending any budget searching for upper premises of the other bids.
    stopped = compiler._prove_observed_bid_dominance(exact, prefix,
        selected_action="pass", opponent_policy={"auction_action": "pass", "escalation_action": "decline"},
        alternative_proof=(dict(schemaVersion=1, upperBound=1, maximumWorlds=1,
            maximumSeconds=15, maximumQueries=50, maximumWorldSeconds=15, maximumWorldQueries=50)
            if version == 1 else dict(schemaVersion=2, method="information_set_refutation_v1",
                upperBound=1, maximumSeconds=15, maximumQueries=50)),
        maximum_evidence_bytes=100000000, maximum_iterations=2,
        maximum_queries=100, maximum_observations=100, maximum_branches=13,
        maximum_model_bytes=100000000, maximum_conflicts=250000, maximum_depth=64, maximum_seconds=30)
    assert stopped["status"] == "incomplete" and stopped["stage"] == "visible_strategy"
    assert stopped["selectedProof"]["status"] == "scoped_strategy_threshold_refuted"
    assert stopped["selectedProof"]["strategy"]["upperBound"] == 0
    assert stopped["cause"] == "sufficient_threshold_not_proved"
    assert stopped["selectedProof"]["strategy"]["scope"]["utilityBounds"] == [0, 0]
    assert 0 < stopped["queriesUsed"] == stopped["selectedProof"]["queriesUsed"] <= 100
    assert stopped["alternativeSearches"] == [] and stopped["alternatives"] == []
    assert stopped["certificate"] is None


@pytest.mark.real_clock
def test_native_bid_substage_ceiling_does_not_refuse_remaining_deadline(root, monkeypatch):
    import yaml
    from baloot_platform.research import ortools_research_executor as executor
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.scenario_contracts import (
        ScenarioProofBudgetV1, ScenarioProofRequestV1, ScenarioQualityExecutionBudgetV1)
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    tier = "offline_heavyweight_v1"
    budget = ScenarioProofBudgetV1.build(tier, settings["proofBudgets"]["profiles"][tier])
    execution = ScenarioQualityExecutionBudgetV1.build(settings["qualityExecution"])
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="complete_state",
        claim="winning_root_action_count", targetTeam="north_south", utility="awarded_point_margin",
        budgetProfile=tier, selectedAction="sun"))
    prove = type(compiler)._prove_native_bid_selected_action
    calls = []
    def attempt(self, *args, **kwargs):
        calls.append(kwargs)
        assert 0 < kwargs["maximum_seconds"] <= 10
        if len(calls) == 1:
            assert kwargs["incumbent_maximum_seconds"] == execution.bid_incumbent_proof["maximumSeconds"] > 10
            receipt = prove(self, *args, **kwargs)
            assert receipt["status"] == "incomplete" and receipt["queriesUsed"] == 0
            assert receipt["cause"]["exhaustionReason"] == "alternative_budget_exhausted"
            return receipt
        assert len(calls) == 2 and "incumbent_maximum_seconds" not in kwargs
        return {"status": "incomplete", "limits": {"maximumSeconds": kwargs["maximum_seconds"]}}
    monkeypatch.setattr(type(compiler), "_prove_native_bid_selected_action", attempt)
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        executor._complete_state_quality_artifact_v1(compiler.rules, exact, prefix, {}, request,
            budget=budget, execution_budget=execution, maximum_nodes=budget.maximum_nodes,
            maximum_conflicts=100, maximum_depth=64, maximum_seconds=10, cancel_check=None,
            opportunity_order=execution.escalation_opportunity_order)
    assert stopped.value.reason == "native_bid_quality_incomplete" and len(calls) == 2
    assert stopped.value.partial_receipt["evidence"]["nativeBidDominance"]["incumbentAttempt"]["queriesUsed"] == 0


def test_native_bid_refuted_sufficient_threshold_skips_synthesis_and_fallback(root, monkeypatch):
    import yaml
    from baloot_platform.research import ortools_research_executor as executor, result_authority as authority
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.scenario_contracts import (
        ScenarioProofBudgetV1, ScenarioProofRequestV1, ScenarioQualityExecutionBudgetV1)
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
    tier = "offline_heavyweight_v1"
    budget = ScenarioProofBudgetV1.build(tier, settings["proofBudgets"]["profiles"][tier])
    execution = ScenarioQualityExecutionBudgetV1.build(settings["qualityExecution"])
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="complete_state",
        claim="winning_root_action_count", targetTeam="north_south", utility="awarded_point_margin",
        budgetProfile=tier, selectedAction="sun"))
    alternatives = sorted(row["code"] for row in exact.legal_actions if row["code"] != "sun")
    receipts = tuple({"scope": {"selectedRootAction": action}, "upperBound": upper,
        "queriesUsed": 1, "branches": [], "limits": {"branches": 13, "modelBytes": 100000000,
            "conflictsPerSolve": 100, "queries": 1, "seconds": 1, "maximumDepth": 64}}
        for action, upper in ((alternatives[0], 22), ("sun", -38)))
    validated = []
    def validate(receipt, **kwargs):
        assert kwargs["exact"] is exact and kwargs["source_prefix"] is prefix
        assert kwargs["action"] == receipt["scope"]["selectedRootAction"]
        validated.append(kwargs["action"])
        return receipt["upperBound"]
    monkeypatch.setattr(authority, "validate_native_bid_world_upper_v1", validate)
    def unused(*args, **kwargs):
        pytest.fail("a refuted sufficient threshold must not launch another model")
    monkeypatch.setattr(type(compiler), "_prepare_native_bid_branches", unused)
    monkeypatch.setattr(type(compiler), "_solve_native_bid_policy_bounds", unused)
    result = compiler._prove_native_bid_selected_action(exact, prefix, Seat.NORTH.team,
        selected_root_action="sun", opponent_strategy={"auction_action": "pass", "escalation_action": "decline"},
        maximum_iterations=10, maximum_queries=100, maximum_observations=100,
        maximum_branches=13, maximum_model_bytes=100000000, maximum_conflicts=100,
        maximum_depth=64, maximum_seconds=10, incumbent_maximum_seconds=2,
        incumbent_maximum_queries=10, incumbent_native_play_policy_id="observer_safe_partner_feed_compact_v1",
        alternative_first=True, _prior_alternatives=receipts)
    assert validated == [alternatives[0], "sun"]
    assert result["status"] == "incomplete" and result["actualActionOptimal"] is None
    assert result["cause"] == {"reason": "selected_threshold_exceeds_certified_upper",
        "requiredThreshold": 22, "selectedUpperBound": -38, "selectedUpperPreparation": receipts[1]}
    assert result["selectedValueProof"] is None and result["queriesUsed"] == 2
    calls = []
    def stopped(*args, **kwargs):
        calls.append(kwargs)
        assert len(calls) == 1, "fallback cannot retry the same refuted threshold"
        return result
    monkeypatch.setattr(type(compiler), "_prove_native_bid_selected_action", stopped)
    with pytest.raises(InformationSetCompilationStopped) as incomplete:
        executor._complete_state_quality_artifact_v1(compiler.rules, exact, prefix, {}, request,
            budget=budget, execution_budget=execution, maximum_nodes=budget.maximum_nodes,
            maximum_conflicts=100, maximum_depth=64, maximum_seconds=90, cancel_check=None,
            opportunity_order=execution.escalation_opportunity_order)
    assert incomplete.value.reason == "native_bid_quality_incomplete" and len(calls) == 1
    assert incomplete.value.partial_receipt["evidence"]["nativeBidDominance"] == result


@pytest.mark.parametrize("opponent_play_policy_id,resume_after_caps,alternative_first", [
    pytest.param(None, False, None, id="None-False"),
    pytest.param("observer_safe_trick_control_v1", False, None, id="observer_safe_trick_control_v1-False"),
    pytest.param(None, True, None, id="None-True"),
    pytest.param("observer_safe_trick_control_v1", False, False, id="configured-incumbent-first")])
def test_native_bid_threshold_certificate_roundtrips_without_exact_score(root, record_property, opponent_play_policy_id, resume_after_caps, alternative_first, monkeypatch, deterministic_work):
    import json
    import time
    import yaml
    from baloot_platform.research.result_authority import (
        build_native_bid_dominance_quality_artifact_v1, reverify_native_bid_dominance_quality_v1,
        validate_native_bid_dominance_budget_v1, validate_native_bid_dominance_claim_v1,
        validate_native_bid_dominance_quality_artifact_v1)
    from baloot_platform.research.scenario_contracts import (
        ExactTracePrefixV1, ScenarioProofBudgetV1, ScenarioProofRequestV1,
        ScenarioQualityExecutionBudgetV1, canonical_sha256)
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    # Expose the Ace without changing final ownership. This smaller legal bid
    # universe isolates threshold admission rather than broad opening search.
    submitted = prefix.to_dict()
    submitted.pop("sha256")
    deck = submitted["deckOrder"]
    layout = DealLayoutV1.build(compiler.rules.structural_domain, dealer=Seat.EAST, buyer=Seat.NORTH)
    ace = deck.index(Card.parse("SA").id)
    deck[ace], deck[layout.purchase_position] = deck[layout.purchase_position], deck[ace]
    prefix = ExactTracePrefixV1.build(submitted)
    exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    profile = "offline_heavyweight_v1"
    budget = ScenarioProofBudgetV1.build(profile, settings["proofBudgets"]["profiles"][profile])
    raw = copy.deepcopy(settings["qualityExecution"])
    raw["bidIncumbentProof"] = dict(schemaVersion=2, maximumSeconds=20.0,
        maximumQueries=64, nativePlayPolicyId="observer_safe_partner_feed_compact_v1")
    if alternative_first is not None:
        raw["bidIncumbentProof"]["completeStateAlternativeFirst"] = alternative_first
    if opponent_play_policy_id is None:
        raw.pop("bidOpponentPlayPolicyId", None)
    else:
        raw["bidOpponentPlayPolicyId"] = opponent_play_policy_id
    execution = ScenarioQualityExecutionBudgetV1.build(raw)
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="complete_state",
        claim="winning_root_action_count", targetTeam="north_south", utility="awarded_point_margin",
        budgetProfile=profile, selectedAction="sun"))
    from baloot_platform.research.ortools_research_executor import _complete_state_quality_artifact_v1
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1)
    owned, calls = [], []
    prove = OrToolsActorInformationSetCompilerV1._prove_native_bid_selected_action

    def production_prove(self, *args, **kwargs):
        calls.append(kwargs)
        result = prove(self, *args, **kwargs)
        owned.extend(kwargs.get("_owned_incumbent_models", ()))
        return result

    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_prove_native_bid_selected_action", production_prove)
    if resume_after_caps:
        solve = OrToolsActorInformationSetCompilerV1._solve_native_bid_policy_bounds
        def refuse_incumbent(self, *args, **kwargs):
            if kwargs.get("incumbent_only"):
                # Fault injection after real alternative proofs. The resumed
                # strategy, qualification and fresh readback remain real CP-SAT.
                raise InformationSetCompilationStopped("incumbent_threshold_refuted",
                    stage="native_bid_policy_bounds", evidence={"queriesUsed": 0})
            assert kwargs["required_threshold"] is not None
            return solve(self, *args, **kwargs)
        monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_solve_native_bid_policy_bounds", refuse_incumbent)
    progress = []
    conflicts = settings["searchBudgets"]["profiles"][profile]["maximumConflicts"]
    depth = min(budget.maximum_depth, budget.maximum_horizon_actions)
    started = time.perf_counter()
    artifact = _complete_state_quality_artifact_v1(compiler.rules, exact, prefix, {}, request,
        budget=budget, execution_budget=execution, maximum_nodes=budget.maximum_nodes,
        maximum_conflicts=conflicts, maximum_depth=depth, maximum_seconds=90, cancel_check=None,
        opportunity_order=execution.escalation_opportunity_order, progress_callback=progress.append,
        _search_generated=True)
    certificate = artifact["graph"]
    assert len(calls) == (2 if resume_after_caps else 1)
    assert calls[0]["alternative_first"] is (alternative_first is not False)
    if opponent_play_policy_id is None and not resume_after_caps:
        prepare = OrToolsActorInformationSetCompilerV1._prepare_native_bid_branches
        def reject_redundant_upper(self, *args, **kwargs):
            assert not kwargs.get("optimize_alternative_upper"), "completed alternative was solved twice"
            return prepare(self, *args, **kwargs)
        monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_prepare_native_bid_branches", reject_redundant_upper)
        reused_models = []
        reused = prove(compiler, exact, prefix, Seat.NORTH.team, **{
            **calls[0], "maximum_seconds": 80, "progress_callback": None,
            "_owned_incumbent_models": reused_models,
            "_prior_alternatives": tuple(row["preparation"] for row in artifact["graph"]["alternatives"])})
        assert reused["status"] == "selected_action_optimal"
        assert reused["alternatives"] == artifact["graph"]["alternatives"]
        assert reused["queriesUsed"] == (reused["selectedValueProof"]["queriesUsed"]
            + sum(row["preparation"]["queriesUsed"] for row in reused["alternatives"]) + 1)
        monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_prepare_native_bid_branches", prepare)
        # The ordinary artifact and fresh solver readback below must also admit
        # reused premises; reuse never bypasses either independent check.
        artifact = build_native_bid_dominance_quality_artifact_v1(reused, rules=compiler.rules,
            source_prefix=prefix, normalized=exact, request=request, native_incumbent_models=tuple(reused_models))
        certificate = artifact["graph"]
        owned = reused_models
    record_property("threshold_production_seconds", time.perf_counter() - started)
    assert certificate["status"] == "selected_action_optimal", certificate.get("cause")
    for phase in ("selected", "fallback") if resume_after_caps else ("selected",):
        counters = [row["queryCount"] for row in progress if row.get("proofPhase") == phase]
        assert counters and counters == sorted(counters)
        assert max(counters) <= (certificate["incumbentAttempt"] if phase == "selected" and resume_after_caps else certificate)["queriesUsed"]
    assert certificate["kind"] == ("native_bid_selected_action_dominance_v4" if resume_after_caps else
        "native_bid_selected_action_dominance_v2" if alternative_first is False else "native_bid_selected_action_dominance_v3")
    strategy = certificate["selectedValueProof"]["strategy"]
    assert strategy["kind"] == ("native_symbolic_round_strategy_threshold_v1" if resume_after_caps else
        "native_symbolic_round_policy_bounds_v2" if alternative_first is False else "native_incumbent_threshold_floor_v1")
    assert strategy["lowerBound"] == max(row["upperBound"] for row in certificate["alternatives"])
    # The two policies are witnesses on opposite sides of minimax, not a
    # joint rollout. Never constrain the adversary in the selected floor.
    selected_scope = certificate["selectedValueProof"]["preparation"]["scope"]
    assert selected_scope.get("nonplayPolicy") is None
    assert selected_scope.get("opponentPlayPolicy") is None
    if not resume_after_caps:
        for row in strategy["queries"]:
            floor = row["receipt"]
            assert floor["policy"]["team"] == request.target_team
            assert floor["policy"]["nativePlayPolicy"]["team"] == request.target_team
            if alternative_first is False:
                assert floor["query"]["solverStatus"] == "optimal"
                assert floor["query"]["status"] == "witness_found"
                assert floor["query"]["candidate"]["utility"] >= strategy["lowerBound"]
            else:
                assert floor["query"]["solverStatus"] == "infeasible"
                assert floor["query"]["status"] == "no_matching_outcome"
    for alternative in certificate["alternatives"]:
        scope = alternative["preparation"]["scope"]
        assert scope["nonplayPolicy"]["team"] == "west_east"
        if opponent_play_policy_id is not None:
            assert scope["opponentPlayPolicy"]["team"] == "west_east"
        assert alternative["upperBound"] <= strategy["lowerBound"]
    assert certificate["actionQuality"]["actualValue"] is certificate["actionQuality"]["bestValue"] is None
    assert certificate["actionQuality"]["actualActionOptimal"] is True
    assert certificate["actionQuality"]["regret"] == 0 and certificate["optimalActionCount"] is None
    validate_native_bid_dominance_budget_v1(certificate, budget, maximum_conflicts=conflicts,
        execution_budget=execution, opportunity_order=execution.escalation_opportunity_order)
    artifact = build_native_bid_dominance_quality_artifact_v1(certificate, rules=compiler.rules,
        source_prefix=prefix, normalized=exact, request=request, native_incumbent_models=tuple(owned) if owned else None)
    validate_native_bid_dominance_quality_artifact_v1(json.loads(json.dumps(artifact)), rules=compiler.rules,
        request=request, source_prefix_hash=prefix.sha256, exact_state_id=exact.exact_state_id,
        native_incumbent_models=tuple(owned) if owned else None)
    started = time.perf_counter()
    reverify_native_bid_dominance_quality_v1(artifact, rules=compiler.rules, request=request, budget=budget,
        execution_budget=execution, maximum_conflicts=conflicts,
        opportunity_order=execution.escalation_opportunity_order, maximum_seconds=90)
    record_property("threshold_fresh_read_seconds", time.perf_counter() - started)
    if resume_after_caps:
        basis = certificate["incumbentAttempt"]
        assert calls[1]["_alternative_basis"] == basis
        assert certificate["alternatives"] == basis["alternatives"]
        assert certificate["queriesUsed"] == certificate["selectedValueProof"]["queriesUsed"] + 1
        from baloot_platform.research.result_authority import validate_native_bid_dominance_resume_basis_v1
        for change in ("state", "action", "missing_alternative", "query_budget", "upper"):
            forged = copy.deepcopy(basis)
            if change == "state":
                forged["stateHash"] = "0" * 64
            elif change == "action":
                forged["selectedAction"] = "pass"
            elif change == "missing_alternative":
                forged["alternatives"].pop()
            elif change == "query_budget":
                forged["queriesUsed"] = forged["limits"]["maximumQueries"]
            else:
                forged["alternatives"][0]["upperBound"] += 1
            forged["sha256"] = canonical_sha256({k: v for k, v in forged.items() if k != "sha256"})
            with pytest.raises(ValueError):
                validate_native_bid_dominance_resume_basis_v1(forged, rules=compiler.rules,
                    source_prefix=prefix, normalized=exact, selected_action="sun",
                    opponent_policy=basis["opponentStrategy"], opponent_play_policy_id=opponent_play_policy_id,
                    limits=certificate["limits"])
        forged = copy.deepcopy(certificate)
        forged["kind"] = "native_bid_selected_action_dominance_v1"
        with pytest.raises(ValueError, match="strategy evidence kind"):
            validate_native_bid_dominance_claim_v1(forged, legal_actions=tuple(row["code"] for row in exact.legal_actions))
        return
    # Rehashed UNKNOWN is not absence evidence; a changed floor is not the
    # threshold owned by the caps. A floor cannot become an exact-score proof.
    for change in ("unknown", "threshold", "kind"):
        forged = copy.deepcopy(certificate)
        proof = forged["selectedValueProof"]["strategy"]
        if change == "unknown":
            query = proof["queries"][0]["receipt"]["query"]
            query["solverStatus"] = "unknown"
            query["sha256"] = canonical_sha256({k: v for k, v in query.items() if k != "sha256"})
            parent = proof["queries"][0]["receipt"]
            parent["sha256"] = canonical_sha256({k: v for k, v in parent.items() if k != "sha256"})
        elif change == "threshold":
            if alternative_first is False:
                proof["lowerBound"] += 1
            else:
                proof["scope"]["incumbentThreshold"] += 1
                proof["scopeHash"] = canonical_sha256(proof["scope"])
        else:
            forged["kind"] = "native_bid_selected_action_dominance_v1"
        proof["sha256"] = canonical_sha256({k: v for k, v in proof.items() if k != "sha256"})
        with pytest.raises(ValueError):
            validate_native_bid_dominance_claim_v1(forged,
                legal_actions=tuple(row["code"] for row in exact.legal_actions), native_incumbent_models=tuple(owned))

    # The remaining-seconds check below measures real elapsed time between mocked calls.
    deterministic_work.release()

    def incomplete_prove(self, *args, **kwargs):
        calls.append(kwargs)
        return {**certificate, "status": "incomplete"}

    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_prove_native_bid_selected_action", incomplete_prove)
    with pytest.raises(InformationSetCompilationStopped, match="native_bid_quality_incomplete"):
        _complete_state_quality_artifact_v1(compiler.rules, exact, prefix, {}, request,
            budget=budget, execution_budget=execution, maximum_nodes=budget.maximum_nodes,
            maximum_conflicts=conflicts, maximum_depth=depth, maximum_seconds=90, cancel_check=None,
            opportunity_order=execution.escalation_opportunity_order, _search_generated=True)
    assert len(calls) == 3 and calls[-2]["alternative_first"] is (alternative_first is not False)
    assert "alternative_first" not in calls[-1]
    assert calls[-1]["maximum_queries"] == calls[-2]["maximum_queries"] - certificate["queriesUsed"]
    assert 0 < calls[-1]["maximum_seconds"] < calls[-2]["maximum_seconds"]


@pytest.fixture(scope="module")
def native_bid_readback_case(root, request):
    import yaml
    from baloot_platform.research.ortools_research_executor import resolve_quality_auction_continuation_v1
    from baloot_platform.research.scenario_contracts import (
        ExactTracePrefixV1, ScenarioProofRequestV1, ScenarioProofBudgetV1, ScenarioQualityExecutionBudgetV1,
    )
    mode = getattr(request, "param", "incumbent")
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    profile = "offline_heavyweight_v1"
    budget = ScenarioProofBudgetV1.build(profile, settings["proofBudgets"]["profiles"][profile])
    execution_raw = copy.deepcopy(settings["qualityExecution"])
    if mode == "refined":
        execution_raw.pop("bidIncumbentProof")
    elif mode in {"native", "paired"}:
        execution_raw["bidIncumbentProof"] = dict(schemaVersion=2, maximumSeconds=20.0,
            maximumQueries=64, nativePlayPolicyId="observer_safe_partner_feed_v1")
        if mode == "paired":
            execution_raw["bidOpponentPlayPolicyId"] = "observer_safe_trick_control_v1"
    execution = ScenarioQualityExecutionBudgetV1.build(execution_raw)
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="complete_state",
        claim="winning_root_action_count", targetTeam="north_south", utility="awarded_point_margin",
        budgetProfile=profile))
    submitted = prefix.to_dict()
    submitted.pop("sha256")
    submitted["auctionActions"] = [{"actor": exact.next_actor, "action": "sun"}]
    progress = []
    # These readback/forgery cases own historical exact-value/interval receipts.
    # Production now tries threshold-first receipts, whose candidate may be
    # absent. Their distinct semantics are covered by the threshold roundtrip
    # test above; do not accidentally mutate them as optimized-value receipts.
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    prove = OrToolsActorInformationSetCompilerV1._prove_native_bid_selected_action
    def interval_first(self, *args, **kwargs):
        kwargs.pop("alternative_first", None)
        return prove(self, *args, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        if mode != "refined":
            patch.setattr(OrToolsActorInformationSetCompilerV1, "_prove_native_bid_selected_action", interval_first)
        result = resolve_quality_auction_continuation_v1(compiler.rules, prefix, request,
            recorded_trace=ExactTracePrefixV1.build(submitted), continue_recorded_prefix=True,
            maximum_nodes=budget.maximum_nodes, maximum_depth=min(budget.maximum_depth, budget.maximum_horizon_actions),
            maximum_worlds=1, maximum_seconds=execution.maximum_wall_seconds,
            maximum_decision_seconds=min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds),
            maximum_evidence_bytes=execution.maximum_evidence_bytes, opportunity_order=execution.escalation_opportunity_order,
            proof_budget_profiles=settings["proofBudgets"]["profiles"],
            search_budget_profiles=settings["searchBudgets"]["profiles"], execution_budget=execution,
            checkpoint_callback=progress.append)
    assert result["status"] == "complete", result.get("partialReceipt")
    certificate = next(iter(result["artifacts"].values()))["graph"]
    assert certificate["kind"] == ("native_bid_selected_action_dominance_v1" if mode == "refined"
                                    else "native_bid_selected_action_dominance_v2")
    assert max(row["queryCount"] for row in progress if row["kind"] == "quality_solver_progress_v1") > certificate["selectedValueProof"]["queriesUsed"]
    return compiler.rules, prefix, request, execution, settings, result


@pytest.mark.parametrize("native_bid_readback_case,mutation", [
    ("incumbent", None), ("incumbent", "floor"), ("incumbent", "floor_and_score"), ("refined", None),
    ("native", None), ("native", "floor"), ("paired", None),
], indirect=["native_bid_readback_case"], scope="module")
def test_native_bid_admission_reproves_consistently_rehashed_bounds(native_bid_readback_case, mutation):
    from baloot_platform.research.ortools_exact_state import (
        OrToolsExactNormalizationReceiptV1, OrToolsNormalizedExactStateV1,
    )
    from baloot_platform.research.ortools_information_set_proof import build_native_bid_dominance_model_v1
    if mutation is None and native_bid_readback_case[5]["artifacts"]:
        graph = next(iter(native_bid_readback_case[5]["artifacts"].values()))["graph"]
        if graph.get("opponentPlayPolicyId") is not None or graph["kind"] == "native_bid_selected_action_dominance_v1":
            assert all(row["preparation"]["kind"] == "native_bid_streamed_alternative_preparation_v1"
                       for row in graph["alternatives"])
    from baloot_platform.research.result_authority import (
        build_native_bid_dominance_quality_artifact_v1, validate_native_decision_quality_evidence_v1,
    )
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofRequestV1, canonical_sha256
    rules, prefix, request, execution, settings, original = native_bid_readback_case
    value = copy.deepcopy(original)
    if mutation is not None:
        artifact = next(iter(value["artifacts"].values()))
        graph = artifact["graph"]
        selected, quality = graph["selectedValueProof"], graph["actionQuality"]
        strategy = selected["strategy"]
        floor = strategy["lowerBound"] + 1
        strategy["lowerBound"] = floor
        def rehash(document):
            document["sha256"] = canonical_sha256({k: v for k, v in document.items() if k != "sha256"})
        for wrapper in strategy["queries"]:
            receipt, query = wrapper["receipt"], wrapper["receipt"]["query"]
            query["objectiveLowerBound"] = query["candidate"]["utility"] = floor
            if mutation == "floor_and_score":
                query["candidate"]["awardedPoints"][0] += 1
            rehash(query)
            rehash(receipt)
        rehash(strategy)
        rehash(selected)
        model, selected_value, claim, best, regret, bindings = build_native_bid_dominance_model_v1(floor, graph["alternatives"])
        qualification = graph["qualificationModel"]
        qualification.update(selectedProofHash=canonical_sha256(selected), modelProto=str(model.proto),
            selectedValueVariable=selected_value.name, claimVariable=claim.name, claimLiteralIndex=int(claim.index),
            bestValueVariable=best.name, regretVariable=regret.name, alternatives=bindings)
        rehash(qualification)
        quality["selectedValueBounds"]["lower"] = floor
        quality["qualificationModelHash"] = qualification["sha256"]
        quality["sha256"] = canonical_sha256({**{k: v for k, v in quality.items() if k != "sha256"},
            "modelProto": str(model.proto)})
        rehash(graph)
        allocation = artifact["allocationModel"]
        raw = allocation["normalizedState"]
        normalized = OrToolsNormalizedExactStateV1(exact_state_id=raw["exactStateId"],
            prefix_hash=raw["prefixHash"], rules_hash=raw["rulesHash"], state_hash=raw["stateHash"],
            phase=raw["phase"], decision_kind=raw["decisionKind"], next_actor=raw["nextActor"],
            legal_actions=tuple(raw["legalActions"]), state=raw["state"],
            project_declarations_verified=raw["projectDeclarationsVerified"],
            project_assertions_verified=raw["projectAssertionsVerified"], event_chain_hash=raw["eventChainHash"],
            sha256=raw["sha256"], normalization_receipt=OrToolsExactNormalizationReceiptV1(allocation["normalizationReceipt"]))
        native_models = None
        if "incumbentNativePlayPolicyId" in graph:
            from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
            # Only reconstruct source models for envelope creation. The admission
            # below must freshly refute the forged floor, not re-prove the valid
            # original and every unrelated alternative before testing the forgery.
            limits = graph["limits"]
            scope = selected["preparation"]["scope"]
            native_models, _ = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._prepare_native_bid_branches(
                normalized, ExactTracePrefixV1.build(allocation["sourcePrefix"]),
                visibility="complete_state", selected_root_action=graph["selectedAction"],
                compact_known_initial_ownership=scope.get("compactKnownInitialOwnership", False),
                compact_purchase_invariant_ownership=scope.get("ownershipCompaction") == "purchase_recipient_invariant_v1",
                maximum_branches=limits["maximumBranchesPerPreparation"],
                maximum_model_bytes=limits["maximumModelBytesPerPreparation"],
                maximum_queries=limits["maximumQueries"], maximum_depth=limits["maximumDepth"],
                maximum_conflicts=limits["maximumConflictsPerSolve"],
                maximum_seconds=execution.maximum_wall_seconds, opportunity_order=execution.escalation_opportunity_order)
        rebuilt = build_native_bid_dominance_quality_artifact_v1(graph, rules=rules,
            source_prefix=ExactTracePrefixV1.build(allocation["sourcePrefix"]), normalized=normalized,
            request=ScenarioProofRequestV1.build(artifact["request"]),
            **({"native_incumbent_models": native_models} if native_models is not None else {}))
        digest = canonical_sha256(rebuilt)
        value["artifacts"] = {digest: rebuilt}
        value["decisions"][0].update(proofArtifactHash=digest, actionQualityHash=quality["sha256"])
        rehash(value["decisions"][0])
        value["sha256"] = canonical_sha256({k: v for k, v in value.items() if k not in {"sha256", "executionReceipt"}})
    kwargs = dict(source_prefix=prefix, rules_hash=rules.sha256, rules=rules, request=request,
        execution_budget=execution, search_budget_profiles=settings["searchBudgets"]["profiles"],
        opportunity_order=execution.escalation_opportunity_order)
    if mutation is None:
        validate_native_decision_quality_evidence_v1(value, **kwargs)
        if execution.bid_opponent_play_policy_id is not None:
            from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1
            raw_execution = execution.to_dict()
            raw_execution.pop("sha256")
            raw_execution["bidOpponentPlayPolicyId"] = "observer_safe_partner_feed_v1"
            with pytest.raises(ValueError, match="frozen opponent play policy"):
                validate_native_decision_quality_evidence_v1(value, **{**kwargs,
                    "execution_budget": ScenarioQualityExecutionBudgetV1.build(raw_execution)})
    else:
        with pytest.raises(ValueError, match="native bound refuted"):
            validate_native_decision_quality_evidence_v1(value, **kwargs)
    assert original == native_bid_readback_case[-1]


@pytest.mark.real_clock
@pytest.mark.parametrize("native_bid_readback_case", ["paired"], indirect=True, scope="module")
def test_native_bid_immediate_wrapper_reuses_only_owned_artifact(native_bid_readback_case, monkeypatch):
    """Immediate validation reuses solver truth; distinct readers never do."""
    from baloot_platform.research import ortools_research_executor as executor
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.scenario_contracts import canonical_sha256

    rules, prefix, request, execution, settings, original = native_bid_readback_case
    digest = original["decisions"][0]["proofArtifactHash"]
    assert "incumbentNativePlayPolicyId" in original["artifacts"][digest]["graph"]

    def generated(*args, **kwargs):
        kwargs["_verified_artifacts"].add(digest)
        return copy.deepcopy(original)

    def distinct_read(*args, **kwargs):
        raise RuntimeError("fresh native bid read")

    monkeypatch.setattr(executor, "_resolve_quality_continuation_v1", generated)
    monkeypatch.setattr(authority, "reverify_native_bid_dominance_quality_v1", distinct_read)
    context = dict(source_prefix=prefix, rules_hash=rules.sha256, rules=rules, request=request,
        execution_budget=execution, search_budget_profiles=settings["searchBudgets"]["profiles"],
        opportunity_order=execution.escalation_opportunity_order)
    immediate = executor.resolve_quality_auction_continuation_v1(rules, prefix, request,
        maximum_seconds=execution.maximum_wall_seconds,
        execution_budget=execution, search_budget_profiles=settings["searchBudgets"]["profiles"])
    assert {k: v for k, v in immediate.items() if k != "executionReceipt"} == {
        k: v for k, v in original.items() if k != "executionReceipt"}
    assert immediate["executionReceipt"]["elapsedSeconds"] >= 0
    with pytest.raises(RuntimeError, match="fresh native bid read"):
        authority.validate_native_decision_quality_evidence_v1(original, **context)
    with pytest.raises(TypeError, match="verified_artifacts"):
        authority.validate_native_decision_quality_evidence_v1(original, **context,
            verified_artifacts=frozenset({digest}))

    changed = copy.deepcopy(original)
    changed["decisions"][0]["regret"] = 1
    changed["decisions"][0]["sha256"] = canonical_sha256({
        k: v for k, v in changed["decisions"][0].items() if k != "sha256"})
    changed["sha256"] = canonical_sha256({
        k: v for k, v in changed.items() if k not in {"sha256", "executionReceipt"}})
    with pytest.raises(ValueError, match="decision binding mismatch"):
        authority._validate_native_decision_quality_evidence_v1(changed, **context,
            verified_artifacts=frozenset({digest}))

    changed = copy.deepcopy(original)
    artifact = changed["artifacts"].pop(digest)
    artifact["proof"]["winningRootActionCount"] = 1
    replacement = canonical_sha256(artifact)
    changed["artifacts"][replacement] = artifact
    changed["decisions"][0]["proofArtifactHash"] = replacement
    changed["decisions"][0]["sha256"] = canonical_sha256({
        k: v for k, v in changed["decisions"][0].items() if k != "sha256"})
    changed["sha256"] = canonical_sha256({
        k: v for k, v in changed.items() if k not in {"sha256", "executionReceipt"}})
    with pytest.raises(RuntimeError, match="fresh native bid read"):
        authority._validate_native_decision_quality_evidence_v1(changed, **context,
            verified_artifacts=frozenset({digest}))


@pytest.mark.real_clock
@pytest.mark.parametrize("stop", ["cancelled", "maximum_seconds", "invalid_deadline"])
@pytest.mark.parametrize("native_bid_readback_case", ["incumbent"], indirect=True, scope="module")
def test_native_bid_readback_obeys_limits_before_starting_solver(native_bid_readback_case, monkeypatch, stop):
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.result_authority import reverify_native_bid_dominance_quality_v1
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, ScenarioProofRequestV1
    rules, _prefix, _request, execution, settings, result = native_bid_readback_case
    artifact = next(iter(result["artifacts"].values()))
    request = ScenarioProofRequestV1.build(artifact["request"])
    budget = ScenarioProofBudgetV1.build(request.budget_profile,
        settings["proofBudgets"]["profiles"][request.budget_profile])
    def forbidden_solver():
        pytest.fail("stopped native readback started a solver")
    monkeypatch.setattr("ortools.sat.python.cp_model.CpSolver", forbidden_solver)
    with pytest.raises(ValueError if stop == "invalid_deadline" else InformationSetCompilationStopped) as raised:
        reverify_native_bid_dominance_quality_v1(artifact, rules=rules, request=request, budget=budget,
            maximum_conflicts=settings["searchBudgets"]["profiles"][request.budget_profile]["maximumConflicts"],
            execution_budget=execution, opportunity_order=execution.escalation_opportunity_order,
            maximum_seconds=True if stop == "invalid_deadline" else 1e-12 if stop == "maximum_seconds" else 1,
            cancel_check=lambda: stop == "cancelled")
    if stop != "invalid_deadline":
        assert raised.value.reason == stop


@pytest.mark.parametrize("mixed_hands", [False, True], ids=["suit_monopoly", "cross_suit_sevens"])
def test_complete_state_round_quality_certifies_every_action(root, tmp_path, record_property, mixed_hands):
    import json
    import time
    import yaml
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_research_executor import (
        OrToolsUnifiedResearchExecutorV1, resolve_quality_round_continuation_v1,
    )
    from baloot_platform.research.scenario_contracts import (
        ExactTracePrefixV1, ScenarioPlayingQualityPolicyV1, ScenarioProofBudgetV1,
        ScenarioQualityExecutionBudgetV1,
    )
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    profile = "offline_heavyweight_v1"
    budget = ScenarioProofBudgetV1.build(profile, settings["proofBudgets"]["profiles"][profile])
    execution = ScenarioQualityExecutionBudgetV1.build(settings["qualityExecution"])
    order = tuple(settings["qualityExecution"]["escalationOpportunityOrder"])
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["projectActions"] = []
    if mixed_hands:
        # Change the initial deal, never repair a completed continuation.
        # Exchanging opposing sevens creates a choice between retaining the
        # lead and surrendering it; equal-value suit-monopoly play is insufficient.
        raw["deckOrder"] = [{0: 8, 8: 0}.get(card, card) for card in raw["deckOrder"]]
    source = ExactTracePrefixV1.build(raw)
    submitted = ExactTracePrefixV1.build({**raw,
        "auctionActions": [{"actor": exact.next_actor, "action": "sun"}]})
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact",
        "visibility": "complete_state", "decisions": {kind: {
            "utility": "awarded_point_margin", "budgetProfile": profile}
            for kind in ("bid", "escalation", "project", "play")}})
    decision = {"kind": "escalation", "selector": {}}
    started = time.monotonic()
    result = resolve_quality_round_continuation_v1(compiler.rules, source, decision, policy,
        recorded_trace=None if mixed_hands else submitted,
        recorded_events=None if mixed_hands else [],
        continue_recorded_prefix=not mixed_hands, search_generated=mixed_hands,
        maximum_nodes=budget.maximum_nodes,
        maximum_depth=min(budget.maximum_depth, budget.maximum_horizon_actions),
        maximum_worlds=execution.maximum_hidden_worlds_per_decision,
        maximum_seconds=execution.maximum_wall_seconds,
        maximum_decision_seconds=min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds),
        maximum_escalation_decisions=execution.maximum_escalation_decisions,
        maximum_evidence_bytes=execution.maximum_evidence_bytes, opportunity_order=order,
        proof_budget_profiles=settings["proofBudgets"]["profiles"],
        search_budget_profiles=settings["searchBudgets"]["profiles"], execution_budget=execution)
    record_property("wholeRoundProofSeconds", time.monotonic() - started)
    receipt = tmp_path / "complete-state-round-evidence.json"
    receipt.write_text(json.dumps({"sourcePrefix": source.to_dict(), "rules": compiler.rules.data,
        "settings": settings, "decisionPoint": decision, "result": result},
        ensure_ascii=False, sort_keys=True), encoding="utf-8")
    record_property("wholeRoundEvidencePath", str(receipt))
    assert result["status"] == "complete", {"status": result["status"], "evidencePath": str(receipt)}
    bundle = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml")
    executor = OrToolsUnifiedResearchExecutorV1(compiler.rules,
        bundle_revision_id="CBR-FULL-ROUND-REGRESSION", bundle_hash=bundle.sha256,
        backend_id="ortools_cp_sat_v3", quality_policy=policy, quality_execution=execution,
        quality_proof_profiles=settings["proofBudgets"]["profiles"],
        quality_search_profiles=settings["searchBudgets"]["profiles"])
    candidate = executor.candidate_from_quality_trace({
        "completeExactTracePrefix": result["finalPrefix"], "playingQualityEvidence": {
            "sourcePrefix": source.to_dict(), "decisionPoint": decision,
            "receipt": {key: value for key, value in result.items() if key != "executionReceipt"}}},
        maximum_seconds=execution.maximum_wall_seconds, origin="exact_trace_prefix")
    assert candidate.trace["completeExactTracePrefix"] == result["finalPrefix"]
    auction = result["auction"]
    assert len(auction["decisions"]) == 1
    artifact = auction["artifacts"][auction["decisions"][0]["proofArtifactHash"]]
    record_property("openingBidQueries", artifact["graph"]["queriesUsed"])
    assert artifact["proof"]["completionStatus"] == "exact_complete"
    assert artifact["proof"]["completionScope"] == "selected_bid_current_deal"
    assert artifact["proof"]["claimSatisfied"] is None
    assert artifact["proof"]["winningRootActionCount"] is None
    quality = artifact["proofModel"]["actionQuality"]
    assert quality["actualAction"] == "sun" and quality["actualActionOptimal"] is True
    assert quality["regret"] == 0 and quality["optimalActionCount"] is None
    assert artifact["graph"]["selectedValueProof"]["preparation"]["scope"]["ownershipCompaction"] == "purchase_recipient_invariant_v1"
    assert {row["action"] for row in artifact["graph"]["alternatives"]} == {
        row["code"] for row in exact.legal_actions if row["code"] != "sun"}
    play = result["postAuction"]["projectPlay"]
    assert len(result["finalPrefix"]["playActions"]) == 32
    assert sum(row["decisionKind"] == "project" for row in play["decisions"]) == 4
    assert sum(row["decisionKind"] == "play" for row in play["decisions"]) == 32
    if mixed_hands:
        pair = next(iter(play["pairedPolicyEvidence"].values()))
        assert pair["synthesis"]["proofs"]["north_south"]["value"] == 58
        assert pair["synthesis"]["proofs"]["west_east"]["value"] == -58
        assert all("nativePlayPolicy" in proof for proof in pair["synthesis"]["proofs"].values())
        assert any(row["decisionKind"] == "play" and len(
            play["artifacts"][row["proofArtifactHash"]]["proof"]["rootActions"]) > 1
            for row in play["decisions"])
        # Independent deviation query: the opposing certified policy can
        # refute a heart-seven opening even against every buyer response.
        from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
        from baloot_platform.research.ortools_play_tail import TailPrefixActionV1
        paired_prefix = ExactTracePrefixV1.build(pair["sourcePrefix"])
        paired_exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize_play_or_project(
            paired_prefix, maximum_seconds=1)
        paired_request = policy.request_for("play", paired_exact.next_actor)
        worlds, _, _ = compiler._prepare_complete_state_play(paired_exact, paired_prefix, paired_request,
            budget=budget, maximum_seconds=10, require_single_project_branch=True)
        source_oracle = worlds[0][1]
        opponent_oracle = compiler.graph_compiler._path_oracle(source_oracle.start, target=Team.WEST_EAST,
            utility=paired_request.utility, cancel_check=None, play_step_count=len(source_oracle.start.cards),
            project_prefix=paired_prefix, project_continuation_actions=source_oracle.project_continuation_actions)
        opponent_bound = opponent_oracle.with_native_play_policy(team=Team.WEST_EAST,
            policy_id=execution.bid_opponent_play_policy_id)
        better_response, _, query_hash = opponent_bound.find_path_below(-57,
            path=(TailPrefixActionV1(Seat.NORTH, Card.parse("H7")),), conditional_actions={},
            maximum_seconds=30, cancelled=None, feasibility_only=True)
        assert better_response is None, "The bad opening must not match the certified +58 value"
        record_property("badOpeningRefutationQueryHash", query_hash)
    for stage in (auction, play):
        for row in stage["decisions"]:
            proof = stage["artifacts"][row["proofArtifactHash"]]
            assert proof["request"]["visibility"] == "complete_state"
            assert proof["proof"]["completionStatus"] == "exact_complete"
            assert row["actualActionOptimal"] is True and row["regret"] == 0

    # Downstream replay verifies usability without strengthening the native proof.
    from importlib.metadata import version
    from baloot_platform.research.runtime_compatibility import check_runtime_compatibility_v1
    final_prefix = ExactTracePrefixV1.build(result["finalPrefix"])
    compatibility = check_runtime_compatibility_v1(exact_state=candidate.trace["completeExactNormalization"],
        semantic_result_hash=result["sha256"], bundle_hash=bundle.sha256,
        trace_hash=final_prefix.sha256, rules=compiler.rules, engine_version=version("baloot-platform"),
        match_score_before=final_prefix.match_score, project_actions=final_prefix.project_actions)
    assert compatibility.receipt.to_dict()["status"] == "passed", compatibility.to_dict()


def test_flagship_opening_has_positive_native_buyer_policy_floor(root, tmp_path, record_property):
    """A losing tactical suffix must not certify optimal play from the opening."""
    import json
    import time
    import yaml
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.result_authority import validate_native_incumbent_lower_bound_v2
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofBudgetV1

    bundle = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml")
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    tier = "offline_heavyweight_v1"
    budget = ScenarioProofBudgetV1.build(tier, settings["proofBudgets"]["profiles"][tier])
    native = budget.native_observed
    incumbent = settings["qualityExecution"]["bidIncumbentProof"]
    fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))
    raw = copy.deepcopy(fixture["exactTracePrefix"])
    raw.update(auctionActions=[], escalationActions=[], projectActions=[], projectDeclarations=[],
        projectAssertions=[], playActions=[], stateAssertions={})
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(bundle.rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=budget.maximum_wall_seconds)
    target = Seat.parse(exact.next_actor).team
    policies = ({seat: value for seat, value in incumbent["playPolicies"].items()
        if Seat.parse(seat).team == target} if incumbent["schemaVersion"] == 1 else None)
    owned_models = []
    incumbent_args = ({"incumbent_play_policies": policies} if policies is not None else {
        "incumbent_native_play_policy_id": incumbent["nativePlayPolicyId"],
        "_owned_incumbent_models": owned_models})
    started = time.monotonic()
    result = OrToolsActorInformationSetCompilerV1(bundle.rules, native_cp_sat=True)._solve_native_bid_policy_bounds(
        exact, prefix, target, visibility="complete_state", selected_root_action="sun",
        maximum_iterations=native.maximum_iterations, maximum_queries=incumbent["maximumQueries"],
        maximum_observations=native.maximum_observations, maximum_branches=budget.maximum_state_space,
        maximum_model_bytes=budget.maximum_cache_bytes,
        maximum_conflicts=settings["searchBudgets"]["profiles"][tier]["maximumConflicts"],
        maximum_depth=min(budget.maximum_depth, budget.maximum_horizon_actions),
        maximum_seconds=incumbent["maximumSeconds"],
        opportunity_order=tuple(settings["qualityExecution"]["escalationOpportunityOrder"]),
        **incumbent_args, incumbent_only=True)
    record_property("openingFloorSeconds", time.monotonic() - started)
    receipt = tmp_path / "flagship-opening-floor.json"
    receipt.write_text(json.dumps({"sourcePrefix": prefix.to_dict(), "rules": bundle.rules.data,
        "settings": settings, "result": result}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    record_property("openingFloorEvidencePath", str(receipt))
    assert result["status"] == "scoped_incumbent_bound_complete"
    preparation = result["preparation"]
    assert preparation["constructionComplete"] is True
    assert preparation["scope"]["visibility"] == "complete_state"
    assert preparation["scope"]["selectedRootAction"] == "sun"
    assert preparation["scope"].get("nonplayPolicy") is None
    floor = validate_native_incumbent_lower_bound_v2(result["strategy"],
        target_team=target.name.lower(), play_policies=policies, rules_hash=bundle.rules.sha256,
        **({"native_play_policy_id": incumbent["nativePlayPolicyId"], "source_models": tuple(owned_models)}
           if policies is None else {}),
        source_model_hashes=tuple(row["sourceModelHash"] for row in preparation["branches"] if row["retained"]))
    record_property("buyerMarginLowerBound", floor)
    assert target == Team.NORTH_SOUTH and floor > 0
    assert fixture["proof"]["targetTeam"] == target.other.name.lower()
    assert fixture["proof"]["utility"] == "round_win"
    terminal_raw = copy.deepcopy(fixture["exactTracePrefix"])
    terminal_raw["playActions"].extend({"card": card} for card in fixture["expected"]["principalVariation"])
    terminal_raw["stateAssertions"] = {}
    terminal = OrToolsExactStateNormalizerV1(bundle.rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(terminal_raw), decision_point={"kind": "play"},
        maximum_seconds=budget.maximum_wall_seconds)
    points = terminal.state["roundScore"]["awardedPoints"]
    margin = points[int(target)] - points[int(target.other)]
    record_property("goldenTerminalBuyerMargin", margin)
    assert terminal.phase == "scored" and margin < 0 < floor


@pytest.mark.real_clock
def test_native_bid_preparation_progress_is_safe_and_cannot_change_receipt(root):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
    from baloot_platform.research.service import _operator_quality_solver_progress_v1

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    normalizer = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True)
    for _ in range(7):
        raw = prefix.to_dict()
        raw.pop("sha256")
        raw["auctionActions"].append({"actor": exact.next_actor, "action": "pass"})
        prefix = ExactTracePrefixV1.build(raw)
        exact = normalizer.normalize(prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    limits = dict(visibility="complete_state", selected_root_action="pass",
        maximum_branches=13, maximum_model_bytes=100000000, maximum_conflicts=250000,
        maximum_queries=100, maximum_depth=64, maximum_seconds=15)
    progress = []
    _, original = compiler._prepare_native_bid_branches(exact, prefix, **limits)
    _, observed = compiler._prepare_native_bid_branches(exact, prefix,
        progress_callback=progress.append, **limits)
    def semantic(receipt):
        result = copy.deepcopy(receipt)
        result.pop("sha256")
        for branch in result["branches"]:
            query = branch.get("depthQuery")
            if query is not None:
                assert 0 < query.pop("maximumSeconds") <= limits["maximum_seconds"]
                query.pop("sha256")  # Remaining-time allowance is execution evidence.
        return result
    assert semantic(observed) == semantic(original)
    assert progress[0]["queryCount"] == 0
    assert progress[-1]["queryCount"] == observed["queriesUsed"]
    assert [row["queryCount"] for row in progress] == sorted(row["queryCount"] for row in progress)
    assert all(set(row) == {"kind", "phase", "masterCount", "queryCount", "bindingCount"}
               and _operator_quality_solver_progress_v1(row) is not None for row in progress)
    def failed_callback(row):
        raise RuntimeError("operator disconnected")
    _, disconnected = compiler._prepare_native_bid_branches(exact, prefix,
        progress_callback=failed_callback, **limits)
    assert semantic(disconnected) == semantic(original)
    with pytest.raises(TypeError, match="callback"):
        compiler._prepare_native_bid_branches(exact, prefix, progress_callback=False, **limits)


@pytest.mark.real_clock
@pytest.mark.parametrize("case", ("complete", "threshold", "budget", "cancelled", "overcharge"))
@pytest.mark.parametrize("reserve", (False, True))
@pytest.mark.parametrize("play_policy", (None, "observer_safe_trick_control_v1"))
def test_observed_bid_pipeline_preserves_visibility_and_one_budget(root, monkeypatch, case, reserve, play_policy):
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    actions = sorted(row["code"] for row in exact.legal_actions if row["code"] != "sun")
    calls, progress = [], []
    minimax_options = {}
    maximum_queries = 2 * len(actions) + 4 if case == "budget" else 100
    incumbent_queries = [0]

    def prepare(actual, source, **kwargs):
        assert actual is exact and source is prefix
        assert kwargs["compact_purchase_invariant_ownership"] is True
        assert kwargs.get("opponent_play_policy_id") == play_policy
        assert kwargs["visibility"] == "complete_state" and "fixed_observed_world" not in kwargs
        assert kwargs["maximum_queries"] == maximum_queries - 2 * len(calls)
        calls.append(kwargs["selected_root_action"])
        kwargs["progress_callback"]({"kind": "quality_solver_progress_v1", "phase": "preparation",
            "masterCount": 0, "queryCount": 2, "bindingCount": 0})
        return (), {"queriesUsed": 2, "upper": len(calls)}

    def solve(actual, source, target, **kwargs):
        assert kwargs.get("minimax_budget") is minimax_options.get("minimax_budget")
        assert "opponent_play_policy_id" not in kwargs
        assert kwargs["visibility"] == "actor_information_set" and "fixed_observed_world" not in kwargs
        assert kwargs["selected_root_action"] == "sun"
        assert kwargs["required_threshold"] == len(actions)
        assert kwargs["maximum_queries"] == maximum_queries - 2 * len(actions) - incumbent_queries[0]
        calls.append("strategy")
        kwargs["progress_callback"]({"kind": "quality_solver_progress_v1", "phase": "master",
            "masterCount": 1, "queryCount": 3, "bindingCount": 0})
        return {"status": "scoped_strategy_threshold_refuted" if case == "threshold"
            else "scoped_strategy_threshold_complete", "queriesUsed": 101 if case == "overcharge" else 3,
            "strategy": {}, "preparation": {}}

    def compose(**kwargs):
        assert kwargs.get("minimax_budget") is minimax_options.get("minimax_budget")
        assert kwargs.get("opponent_play_policy_id") == play_policy
        assert [row["action"] for row in kwargs["alternatives"]] == actions
        assert all(row["exact"] is exact and row["sourcePrefix"] is prefix for row in kwargs["alternatives"])
        assert kwargs["maximum_queries"] == (maximum_queries - 2 * len(actions) - incumbent_queries[0] - 3) // (2 if reserve else 1)
        calls.append("verification")
        return {"status": "selected_action_optimal", "executionReceipt": {"queriesUsed": 4}}

    monkeypatch.setattr(compiler, "_prepare_native_bid_branches", prepare)
    monkeypatch.setattr(compiler, "_solve_native_bid_policy_bounds", solve)
    monkeypatch.setattr(authority, "validate_native_bid_world_upper_v1", lambda receipt, **kwargs: receipt["upper"])
    monkeypatch.setattr(authority, "compose_native_observed_bid_dominance_v1", compose)
    limits = dict(maximum_iterations=100, maximum_queries=maximum_queries, maximum_observations=100,
        maximum_branches=100, maximum_model_bytes=1_000_000, maximum_conflicts=100,
        maximum_seconds=10, maximum_depth=64)
    def run(**extra):
        return compiler._prove_observed_bid_dominance(exact, prefix, selected_action="sun",
            opponent_policy={"auction_action": "pass", "escalation_action": "decline"},
            cancel_check=lambda: case == "cancelled", reserve_artifact_verification=reserve,
            opponent_play_policy_id=play_policy, progress_callback=progress.append, **limits, **extra)
    if case == "overcharge":
        with pytest.raises(ValueError, match="query accounting"):
            run()
        return
    result = run()
    assert [row["queryCount"] for row in progress] == ([] if case == "cancelled" else
        [*range(2, 2 * len(actions) + 1, 2), 2 * len(actions) + 3])
    if case == "complete":
        assert result["status"] == "selected_action_optimal"
        assert result["queriesUsed"] == 2 * len(actions) + 7 and result["queryCountComplete"] is True
        assert calls == [*actions, "strategy", "verification"]
        from baloot_platform.research.scenario_contracts import canonical_sha256
        def owned(document):
            core = {key: value for key, value in document.items() if key != "sha256"}
            return {**core, "sha256": canonical_sha256(core)}
        retained = [dict(action=row["action"], receipt=owned({**row["receipt"],
            "kind": "native_bid_streamed_world_upper_v1", "constructionComplete": True,
            "limits": dict(branches=100, modelBytes=1_000_000, conflictsPerSolve=100,
                maximumDepth=64, queries=100 - 2 * index, seconds=20)}))
            for index, row in enumerate(result["alternatives"])]
        parent = owned({**result, "status": "incomplete", "maximumSeconds": 20,
            "alternatives": retained})
        prior = [{"receipt": {"evidence": {"nativeObservedBidDominance": parent}}}]
        reuse = authority._prior_bid_alternative_receipts_v1(prior,
            source_prefix_hash=prefix.sha256, rules_hash=compiler.rules.sha256, maximum_seconds=30)
        assert len(reuse) == len(actions)
        assert not authority._prior_bid_alternative_receipts_v1(prior,
            source_prefix_hash="wrong-owner", rules_hash=compiler.rules.sha256, maximum_seconds=30)
        assert not authority._prior_bid_alternative_receipts_v1(prior,
            source_prefix_hash=prefix.sha256, rules_hash=compiler.rules.sha256, maximum_seconds=10)
        calls.clear()
        reused = run(_prior_alternatives=reuse)
        assert reused["status"] == "selected_action_optimal"
        assert reused["queriesUsed"] == result["queriesUsed"]
        assert calls == ["strategy", "verification"]
        from baloot_platform.config import build_bundle
        from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1
        settings = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component("challenge")["scenarioResearchV4"]
        minimax_options["minimax_budget"] = ScenarioProofBudgetV1.build("actor_information_set_exact_v1",
            settings["proofBudgets"]["profiles"]["actor_information_set_exact_v1"])
        calls.clear()
        shortened = run(strategy_play_fallback_id="observer_safe_partner_feed_secure_control_v1", **minimax_options)
        assert shortened["status"] == "selected_action_optimal"
        assert calls == [*actions, "strategy", "verification"]
        minimax_options.clear()
    else:
        assert result["status"] == "incomplete" and result["certificate"] is None
        assert result["queryCountComplete"] is False
        assert calls == ([] if case == "cancelled" else [*actions, "strategy"])
    if case == "complete" and not reserve and play_policy is not None:
        probe = dict(schemaVersion=1, maximumSeconds=2, maximumQueries=20,
            worldRankingId="opponent_auction_value_v1", opponentAuctionAction="sun")
        extra = dict(threshold_refutation_probe=probe, maximum_evidence_bytes=1_000_000)
        incumbent_queries[0] = 4  # Shared assertions check all charged auxiliary queries.
        for outcome in ("threshold_not_refuted", "incomplete"):
            def no_refutation(*args, **kwargs):
                assert kwargs["configuration"]["maximumSeconds"] <= 2
                assert kwargs["configuration"]["maximumQueries"] == 20
                assert kwargs["opponent_play_policy_id"] == play_policy
                calls.append("probe")
                return None, {"status": outcome, "queriesCharged": 4}
            monkeypatch.setattr(compiler, "_probe_observed_bid_threshold_refutation", no_refutation)
            calls.clear()
            continued = run(**extra)
            assert continued["status"] == "selected_action_optimal"
            assert continued["queriesUsed"] == 2 * len(actions) + 11
            assert calls == [*actions, "probe", "strategy", "verification"]
        for reason in ("cancelled", "maximum_evidence_bytes"):
            def interrupted(*args, **kwargs):
                calls.append("probe")
                raise InformationSetCompilationStopped(reason, stage="native_bid_threshold_probe",
                    evidence={"queriesCharged": 4})
            monkeypatch.setattr(compiler, "_probe_observed_bid_threshold_refutation", interrupted)
            calls.clear()
            stopped = run(**extra)
            assert stopped["status"] == "incomplete" and stopped["certificate"] is None
            assert stopped["queriesUsed"] == 2 * len(actions) + 4
            assert calls == [*actions, "probe"]
        monkeypatch.setattr(compiler, "_probe_observed_bid_threshold_refutation",
            lambda *a, **k: (None, {"queriesCharged": maximum_queries}))
        calls.clear()
        with pytest.raises(ValueError, match="query accounting"):
            run(**extra)
    if case == "complete" and not reserve and play_policy is None:
        incumbent = dict(schemaVersion=2, nativePlayPolicyId="observer_safe_partner_feed_v1",
                         maximumSeconds=2.0, maximumQueries=20)
        floor = {"kind": "native_observed_bid_visible_policy_floor_v1",
                 "branches": [{"receipt": {"bindings": []}}]}
        def prove_floor(actual, source, **kwargs):
            assert actual is exact and source is prefix
            assert kwargs["nonplay_policy"] == {"auction_action": None, "escalation_action": None}
            assert kwargs["required_threshold"] == len(actions)
            assert kwargs["maximum_seconds"] <= 2 and kwargs["maximum_queries"] == 20
            assert kwargs.get("observed_project_policy_id") == incumbent.get("observedProjectPolicyId")
            assert kwargs.get("observed_auction_policy_id") == incumbent.get("observedAuctionPolicyId")
            calls.append("native_floor")
            return floor, {"queriesUsed": 2}
        monkeypatch.setattr(compiler, "_prove_observed_bid_visible_policy_floor", prove_floor)
        calls.clear()
        native = run(incumbent_proof=incumbent)
        assert native["status"] == "selected_action_optimal" and native["incumbentProof"] == incumbent
        assert calls == [*actions, "native_floor", "verification"]
        calls.clear()
        incumbent["observedProjectPolicyId"] = "known_project_max_value_v1"
        assert run(incumbent_proof=incumbent)["status"] == "selected_action_optimal"
        calls.clear()
        incumbent["observedAuctionPolicyId"] = "prefer_legal_sun_then_pass_v1"
        assert run(incumbent_proof=incumbent)["status"] == "selected_action_optimal"
        incumbent.pop("observedAuctionPolicyId")
        incumbent.pop("observedProjectPolicyId")
        floor.update(status="refuted", queriesUsed=3)
        incumbent_queries[0] = 3
        calls.clear()
        refuted = run(incumbent_proof=incumbent)
        assert refuted["status"] == "selected_action_optimal"
        assert refuted["incumbentAttempt"]["strategy"]["status"] == "refuted"
        assert refuted["queriesUsed"] == 2 * len(actions) + 10
        assert calls == [*actions, "native_floor", "strategy", "verification"]
        calls.clear()
        deferred = run(incumbent_proof=incumbent, _defer_bid_fallback=True)
        assert deferred["status"] == "incomplete" and deferred["certificate"] is None
        assert deferred["queriesUsed"] == 2 * len(actions) + 3
        assert deferred["incumbentAttempt"]["strategy"]["status"] == "refuted"
        assert calls == [*actions, "native_floor"]
        floor["queriesUsed"] = incumbent["maximumQueries"] + 1
        calls.clear()
        with pytest.raises(ValueError, match="incumbent query accounting"):
            run(incumbent_proof=incumbent)
        floor["queriesUsed"] = 3
        for stopped_stage, method in (("native_bid_preparation", "_prepare_native_bid_branches"),
                                      ("native_observed_bid_floor", "_prove_observed_bid_visible_policy_floor")):
            partial = {"queriesUsed": 3, "constructionComplete": False}
            def stopped(*args, **kwargs):
                raise InformationSetCompilationStopped("maximum_seconds", stage=stopped_stage, evidence=partial)
            with monkeypatch.context() as patch:
                patch.setattr(compiler, method, stopped)
                calls.clear()
                interrupted = run(incumbent_proof=incumbent)
                if stopped_stage == "native_bid_preparation":
                    assert interrupted["status"] == "incomplete" and interrupted["queryCountComplete"] is False
                    assert interrupted["queriesUsed"] == 3 and interrupted["cause"]["evidence"] == partial
                    assert interrupted["activePreparation"] == partial and not calls
                else:
                    assert interrupted["status"] == "selected_action_optimal"
                    assert interrupted["queriesUsed"] == 2 * len(actions) + 10
                    assert interrupted["incumbentAttempt"]["partialReceipt"]["evidence"] == partial
                    assert calls == [*actions, "strategy", "verification"]
        # Exhausting the parent budget or cancellation must never start fallback.
        import time
        for reason in ("maximum_seconds", "cancelled", "maximum_model_bytes"):
            clock = [0.0]
            def parent_stopped(*args, **kwargs):
                if reason == "maximum_seconds":
                    clock[0] = limits["maximum_seconds"]
                raise InformationSetCompilationStopped(reason, stage="native_observed_bid_floor",
                    evidence={"queriesUsed": 3})
            with monkeypatch.context() as guard:
                guard.setattr(time, "monotonic", lambda: clock[0])
                guard.setattr(compiler, "_prove_observed_bid_visible_policy_floor", parent_stopped)
                calls.clear()
                exhausted = run(incumbent_proof=incumbent)
                assert exhausted["status"] == "incomplete" and calls == actions
                assert exhausted["queriesUsed"] == 2 * len(actions) + 3


@pytest.mark.real_clock
@pytest.mark.parametrize("case", ("complete", "incomplete", "deadline", "overspent"))
@pytest.mark.parametrize("play_policy", (None, "observer_safe_trick_control_v1"))
def test_observed_bid_quality_dispatch_reserves_verification_and_time(root, monkeypatch, case, play_policy):
    import time
    import yaml
    from baloot_platform.research import ortools_research_executor as executor, result_authority as authority
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, ScenarioQualityExecutionBudgetV1, ScenarioProofRequestV1
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    tier = "offline_heavyweight_v1"
    budget = ScenarioProofBudgetV1.build(tier, settings["proofBudgets"]["profiles"][tier])
    execution = ScenarioQualityExecutionBudgetV1.build({**{key: value for key, value in settings["qualityExecution"].items()
        if key not in {"bidIncumbentProof", "bidOpponentPlayPolicyId"}},
        **({"bidOpponentPlayPolicyId": play_policy,
            "observedBidStrategyPlayFallbackId": "observer_safe_partner_feed_secure_control_v1"}
           if play_policy is not None else {})})
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="actor_information_set",
        strategyClass="pure_deterministic", claim="winning_root_action_count", targetTeam="north_south",
        utility="awarded_point_margin", successThreshold=1, budgetProfile=tier, selectedAction="sun"))
    clock, calls, progress = [time.monotonic()], [], []
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    def prove(self, state, source, **kwargs):
        assert kwargs.get("opponent_play_policy_id") == play_policy
        assert kwargs.get("strategy_play_fallback_id") == execution.observed_bid_strategy_play_fallback_id
        assert kwargs.get("minimax_budget") is (budget if execution.observed_bid_strategy_play_fallback_id is not None else None)
        assert kwargs.get("alternative_proof") == execution.bid_alternative_proof
        if execution.bid_alternative_proof is not None:
            assert kwargs["maximum_evidence_bytes"] == execution.maximum_evidence_bytes
        calls.append("prove")
        assert state is exact and source is prefix
        assert kwargs["reserve_artifact_verification"] is True
        assert kwargs["maximum_queries"] == budget.native_observed.maximum_queries
        assert kwargs["maximum_seconds"] == 10
        assert kwargs["opportunity_order"] == execution.escalation_opportunity_order
        kwargs["progress_callback"]({"kind": "quality_solver_progress_v1", "phase": "master",
            "masterCount": 1, "queryCount": 0, "bindingCount": 0})
        clock[0] += 11 if case == "deadline" else 4
        return {"status": "incomplete" if case == "incomplete" else "selected_action_optimal",
            "queriesUsed": budget.native_observed.maximum_queries if case == "overspent" else 1,
            "certificate": {"sha256": "owned-certificate", "limits": {"maximumQueries": 1}}}
    def build(attempt, **kwargs):
        assert kwargs.get("opponent_play_policy_id") == play_policy
        assert kwargs["execution_budget"] is execution
        assert kwargs["verified_certificates"] == frozenset({"owned-certificate"})
        calls.append("artifact")
        assert kwargs["maximum_seconds"] == 6 and kwargs["budget"] is budget
        return {"test": "dispatch-only"}
    monkeypatch.setattr(executor.OrToolsActorInformationSetCompilerV1, "_prove_observed_bid_dominance", prove)
    monkeypatch.setattr(authority, "_build_native_observed_bid_quality_artifact_v1", build)
    def run():
        return executor._observed_bid_quality_artifact_v1(compiler.rules, exact, prefix, request,
            budget=budget, execution_budget=execution, maximum_conflicts=100,
            maximum_depth=budget.maximum_depth, maximum_seconds=10, progress_callback=progress.append)
    if case == "complete":
        assert run() == {"test": "dispatch-only"}
        assert calls == ["prove", "artifact"]

    else:
        with pytest.raises(ValueError if case == "overspent" else InformationSetCompilationStopped):
            run()
        assert calls == ["prove"]
    assert progress == [{"kind": "quality_solver_progress_v1", "phase": "master",
        "masterCount": 1, "queryCount": 0, "bindingCount": 0,
        "decisionKind": "bid", "proofPhase": "selected"}]
    if case == "complete" and play_policy is None:
        raw = execution.to_dict()
        raw.pop("sha256")
        raw["bidAlternativeProof"] = dict(schemaVersion=1, upperBound=33, maximumWorlds=2,
            maximumSeconds=4, maximumQueries=100, maximumWorldSeconds=2, maximumWorldQueries=50)
        execution = ScenarioQualityExecutionBudgetV1.build(raw)
        clock[0] -= 4
        calls.clear()
        assert run() == {"test": "dispatch-only"}
        assert calls == ["prove", "artifact"]

        # Frozen sub-stage ceilings can exceed the remaining 10-second slice.
        # They are not minimum launch allowances or permission to reset time.
        raw = execution.to_dict()
        raw.pop("sha256")
        raw["bidAlternativeProof"]["maximumSeconds"] = 20
        raw["bidThresholdRefutation"] = dict(schemaVersion=1, maximumSeconds=20,
            maximumQueries=100, worldRankingId="opponent_auction_value_v1", opponentAuctionAction="pass")
        raw["bidIncumbentProof"] = dict(schemaVersion=2, maximumSeconds=20,
            maximumQueries=64, nativePlayPolicyId="observer_safe_partner_feed_v1")
        execution = ScenarioQualityExecutionBudgetV1.build(raw)
        clock[0] -= 4
        calls.clear()
        assert run() == {"test": "dispatch-only"}
        assert calls == ["prove", "artifact"]


@pytest.mark.real_clock
@pytest.mark.parametrize("visibility", ["actor_information_set", "complete_state"])
def test_generated_bid_selection_shares_budget_and_distinguishes_local_timeout(root, monkeypatch, visibility):
    import time
    import yaml
    from baloot_platform.research import ortools_research_executor as executor
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, ScenarioProofRequestV1, ScenarioQualityExecutionBudgetV1
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    tier = "offline_heavyweight_v1"
    budget = ScenarioProofBudgetV1.build(tier, settings["proofBudgets"]["profiles"][tier])
    execution = ScenarioQualityExecutionBudgetV1.build(settings["qualityExecution"])
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility=visibility,
        strategyClass="pure_deterministic", claim="winning_root_action_count", targetTeam="north_south",
        utility="awarded_point_margin", successThreshold=1, budgetProfile=tier))
    clock, calls, cancelled = [0.0], [], [False]
    complete_helper = executor._complete_state_quality_artifact_v1
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    def prove(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            clock[0] = 4.0
            raise InformationSetCompilationStopped("cancelled", stage="local_native_deadline")
        assert kwargs["maximum_seconds"] == 56
        assert kwargs["_prior_selection_attempts"][0]["queriesCharged"] == calls[0]["_maximum_queries"]
        assert sum(row["_maximum_queries"] for row in calls) <= budget.native_observed.maximum_queries
        clock[0] += 40  # Fits the remainder, not the former 28-second slice.
        return {"test": "dispatch-only"}
    helper = "_complete_state_quality_artifact_v1" if visibility == "complete_state" else "_observed_bid_quality_artifact_v1"
    monkeypatch.setattr(executor, helper, prove)
    def run(search_generated=False, maximum_seconds=60):
        return executor._select_observed_bid_quality_artifact_v1(compiler.rules, exact, prefix, request,
            budget=budget, execution_budget=execution, maximum_conflicts=100,
            maximum_depth=budget.maximum_depth, maximum_seconds=maximum_seconds, cancel_check=lambda: cancelled[0],
            maximum_nodes=budget.maximum_nodes, decision={"kind": "bid"},
            opportunity_order=execution.escalation_opportunity_order,
            _search_generated=search_generated)
    assert run() == {"test": "dispatch-only"}
    assert calls[0]["maximum_seconds"] == 60
    assert [row["_selected_action"] for row in calls] == [row["code"] for row in exact.legal_actions[:2]]
    if execution.bid_incumbent_proof["schemaVersion"] == 2:
        calls.clear()
        clock[0] = 0
        def staged_prove(*args, **kwargs):
            calls.append(kwargs)
            clock[0] += 1
            if kwargs["_defer_bid_fallback"]:
                raise InformationSetCompilationStopped("native_bid_quality_incomplete", stage="complete_state_quality")
            return {"test": "deep-dispatch"}
        monkeypatch.setattr(executor, helper, staged_prove)
        assert run(search_generated=True, maximum_seconds=600) == {"test": "deep-dispatch"}
        actions = [row["code"] for row in exact.legal_actions]
        assert calls[0]["maximum_seconds"] == 600
        assert [row["_selected_action"] for row in calls] == actions + actions[:1]
        assert all(row["_staged_bid_selection"] for row in calls)
        assert sum(row["_maximum_queries"] for row in calls) <= budget.native_observed.maximum_queries
    # A real cancellation cannot launch another action.
    calls.clear()
    clock[0] = 0
    def stop(*args, **kwargs):
        calls.append(kwargs)
        cancelled[0] = True
        raise InformationSetCompilationStopped("cancelled", stage="native_bid_preparation")
    monkeypatch.setattr(executor, helper, stop)
    with pytest.raises(InformationSetCompilationStopped) as interrupted:
        run()
    assert interrupted.value.reason == "cancelled" and len(calls) == 1
    assert len(interrupted.value.partial_receipt["evidence"]["attempts"]) == 1
    if visibility == "complete_state":
        # The production dispatcher must use this selector, not enumerate the full graph.
        monkeypatch.setattr(executor, "_select_observed_bid_quality_artifact_v1", lambda *a, **kw: kw)
        dispatched = complete_helper(compiler.rules, exact, prefix, {"kind": "bid"}, request,
            budget=budget, execution_budget=execution, maximum_conflicts=100, maximum_nodes=budget.maximum_nodes,
            maximum_depth=budget.maximum_depth, maximum_seconds=60, cancel_check=None,
            opportunity_order=execution.escalation_opportunity_order)
        assert dispatched["maximum_nodes"] == budget.maximum_nodes
        assert dispatched["maximum_conflicts"] == 100


@pytest.mark.parametrize("play_policy,fallback,native_search,generated", ((None, False, False, False),
    ("observer_safe_trick_control_v1", False, False, False), (None, True, False, False),
    (None, False, True, False), (None, False, False, True)))
@pytest.mark.real_clock
def test_observed_last_bid_pass_has_real_native_dominance_certificate(root, tmp_path, record_property, monkeypatch, play_policy, fallback, native_search, generated):
    import json
    import time
    import yaml
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_research_executor import resolve_quality_auction_continuation_v1
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.scenario_contracts import (
        ExactTracePrefixV1, ScenarioProofBudgetV1, ScenarioProofRequestV1, ScenarioQualityExecutionBudgetV1,
    )

    bundle = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml")
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    if play_policy is not None:
        settings["qualityExecution"]["bidOpponentPlayPolicyId"] = play_policy
    else:
        settings["qualityExecution"].pop("bidOpponentPlayPolicyId", None)
        settings["qualityExecution"]["bidIncumbentProof"] = dict(schemaVersion=2,
            nativePlayPolicyId="observer_safe_partner_feed_v1", maximumSeconds=20.0, maximumQueries=64)
    if native_search:
        settings["qualityExecution"]["bidAlternativeProof"] = dict(schemaVersion=1, upperBound=-1,
            # All alternatives share this existing frozen hidden-world ceiling.
            maximumWorlds=settings["qualityExecution"]["maximumHiddenWorldsPerDecision"],
            maximumSeconds=100, maximumQueries=2000, maximumWorldSeconds=100, maximumWorldQueries=1000,
            worldRankingId="opponent_auction_value_v1")
    if generated:
        settings["qualityExecution"]["bidIncumbentProof"]["completeStateNonplayPolicy"] = {
            "schemaVersion": 1, "auctionAction": "pass", "escalationAction": "decline",
            "projectPolicyId": "full_hand_project_max_value_v1"}
    tier = "offline_heavyweight_v1"
    budget = ScenarioProofBudgetV1.build(tier, settings["proofBudgets"]["profiles"][tier])
    native = budget.native_observed
    hands = {Seat.NORTH: {Card.parse(suit + rank).id for suit in "SHDC" for rank in ("A", "10")},
        Seat.SOUTH: {Card.parse(suit + rank).id for suit in "SHDC" for rank in ("J", "9")},
        Seat.EAST: {Card.parse(card).id for card in ("S7", "SQ", "H8", "HK", "D7", "DQ", "C8", "CK")}}
    hands[Seat.WEST] = set(range(32)) - set.union(*hands.values())
    layout = DealLayoutV1.build(bundle.rules.structural_domain, dealer=Seat.EAST, buyer=Seat.EAST)
    deck = [None] * 32
    for seat in Seat:
        for position, card in zip(layout.positions_for(seat, final=True), sorted(hands[seat]), strict=True):
            deck[position] = card
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "dealer": "east", "deckOrder": deck,
        "auctionActions": [{"action": "pass"}] * 7, "escalationActions": [],
        "projectActions": [], "projectDeclarations": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(bundle.rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=budget.maximum_wall_seconds)
    base_request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="actor_information_set",
        strategyClass="pure_deterministic", claim="winning_root_action_count", targetTeam="west_east",
        utility="awarded_point_margin", successThreshold=1, budgetProfile=tier))
    submitted = prefix.to_dict()
    submitted.pop("sha256")
    submitted["auctionActions"].append({"action": "pass"})
    execution = ScenarioQualityExecutionBudgetV1.build(settings["qualityExecution"])
    if fallback:
        raw = execution.to_dict()
        raw.pop("sha256")
        raw["observedBidStrategyPlayFallbackId"] = "observer_safe_partner_feed_secure_control_v1"
        execution = ScenarioQualityExecutionBudgetV1.build(raw)
        settings["qualityExecution"] = raw
        from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
        original_floor = OrToolsActorInformationSetCompilerV1._prove_observed_bid_visible_policy_floor
        def bounded_floor(self, *args, **kwargs):
            return original_floor(self, *args, **{**kwargs, "maximum_seconds": 0.001})
        monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_prove_observed_bid_visible_policy_floor", bounded_floor)
    def distinct_read(*args, **kwargs):
        raise RuntimeError("fresh observed bid read")
    # Production composes freshly verified premises, then reuses its owned
    # certificate locally. External readers must still reconstruct the proof.
    with monkeypatch.context() as patch:
        patch.setattr(authority, "validate_native_observed_bid_quality_artifact_v1", distinct_read)
        started = time.monotonic()
        production = resolve_quality_auction_continuation_v1(bundle.rules, prefix, base_request,
            recorded_trace=None if generated else ExactTracePrefixV1.build(submitted), maximum_nodes=budget.maximum_nodes,
            maximum_depth=min(budget.maximum_depth, budget.maximum_horizon_actions),
            maximum_worlds=execution.maximum_hidden_worlds_per_decision,
            maximum_seconds=min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds),
            maximum_decision_seconds=min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds),
            maximum_evidence_bytes=execution.maximum_evidence_bytes,
            opportunity_order=execution.escalation_opportunity_order, execution_budget=execution,
            proof_budget_profiles=settings["proofBudgets"]["profiles"],
            search_budget_profiles=settings["searchBudgets"]["profiles"], search_generated=generated)
        production_seconds = time.monotonic() - started
    record_property("observedBidProductionSeconds", production_seconds)
    assert abs(production_seconds - production["executionReceipt"]["elapsedSeconds"]) < 1
    (tmp_path / "observed-last-bid-production.json").write_text(json.dumps(production, sort_keys=True), encoding="utf-8")
    assert production["status"] == "complete", production.get("partialReceipt")
    assert production["operation"] == ("generate" if generated else "audit_recorded")
    assert len(production["decisions"]) == 1 and production["auctionOutcome"]["allPassed"] is True
    artifact = next(iter(production["artifacts"].values()))
    result = artifact["allocationModel"]["attempt"]
    if generated:
        assert result["selectionSchedule"] == "incumbent_then_deep_v1"
    assert artifact["allocationModel"]["schemaVersion"] == 2
    assert artifact["allocationModel"]["generationQueryCountUnit"] == "conservative_budget_charges"
    assert result.get("strategyPlayFallbackId") == execution.observed_bid_strategy_play_fallback_id
    if fallback:
        assert result["selectedProof"]["strategy"]["kind"] == "native_guarded_round_strategy_threshold_v2"
    assert result["artifactVerificationReserved"] is True
    assert artifact["allocationModel"]["generationQueriesUsed"] <= native.maximum_queries
    receipt = tmp_path / "observed-last-bid-certificate.json"
    receipt.write_text(json.dumps({"sourcePrefix": prefix.to_dict(), "rules": bundle.rules.data,
        "settings": settings, "result": result}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    record_property("observedBidEvidencePath", str(receipt))
    assert result["status"] == "selected_action_optimal", {
        "status": result["status"], "stage": result["stage"], "receipt": str(receipt)}
    certificate = result["certificate"]
    assert certificate["visibility"] == "actor_information_set"
    assert certificate["actualActionOptimal"] is True and certificate["wholeRoundCertified"] is False
    assert certificate["selectedAction"] == "pass"
    assert all(row["upperBound"] < 0 for row in certificate["alternatives"])
    assert result["queryCountComplete"] is True
    assert result["queriesUsed"] <= native.maximum_queries
    if native_search:
        assert result["alternativeProof"] == execution.bid_alternative_proof
        assert len(result["alternativeSearches"]) == len(certificate["alternatives"])
        assert all(row["status"] == "upper_bound_proven" for row in result["alternativeSearches"])

    from copy import deepcopy
    from baloot_platform.research.result_authority import build_native_observed_bid_quality_artifact_v1
    from baloot_platform.research.scenario_contracts import ScenarioProofRequestV1, canonical_sha256
    request = ScenarioProofRequestV1.build(artifact["request"])
    assert request.selected_action == (None if generated else "pass")
    if generated:
        assert result["priorSelectionAttempts"] == []
        # Rehashing the generated selection history cannot recycle its budget.
        changed = deepcopy(result)
        changed["priorSelectionAttempts"] = [{"action": "sun"}]
        changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
        with pytest.raises(ValueError, match="selection attempts"):
            authority.build_native_observed_bid_quality_artifact_v1(changed,
                rules=bundle.rules, source_prefix=prefix, normalized=exact, request=request,
                budget=budget, maximum_conflicts=settings["searchBudgets"]["profiles"][tier]["maximumConflicts"],
                opportunity_order=execution.escalation_opportunity_order,
                maximum_seconds=budget.maximum_wall_seconds, execution_budget=execution)
    digest = canonical_sha256(artifact)
    read_context = dict(source_prefix=prefix, rules_hash=bundle.rules.sha256,
        rules=bundle.rules, request=base_request, execution_budget=execution,
        search_budget_profiles=settings["searchBudgets"]["profiles"])
    token = canonical_sha256({"artifactHash": digest, "proofBudgetHash": budget.sha256,
        "executionBudgetHash": execution.sha256,
        "maximumConflicts": settings["searchBudgets"]["profiles"][tier]["maximumConflicts"]})
    with monkeypatch.context() as patch:
        patch.setattr(authority, "validate_native_observed_bid_quality_artifact_v1", distinct_read)
        authority._validate_native_decision_quality_evidence_v1(production, **read_context,
            verified_artifacts=frozenset({token}))
        with pytest.raises(RuntimeError, match="fresh observed bid read"):
            authority.validate_native_decision_quality_evidence_v1(production, **read_context)
        with pytest.raises(TypeError, match="verified_artifacts"):
            authority.validate_native_decision_quality_evidence_v1(production, **read_context,
                verified_artifacts=frozenset({token}))
        # A changed frozen execution context or conflict limit cannot reuse the
        # token, even when the semantic artifact itself is unchanged.
        execution_raw = execution.to_dict()
        execution_raw.pop("sha256")
        execution_raw["maximumEvidenceBytes"] -= 1
        changed_profiles = copy.deepcopy(settings["searchBudgets"]["profiles"])
        changed_profiles[tier]["maximumConflicts"] -= 1
        for change in ({"execution_budget": ScenarioQualityExecutionBudgetV1.build(execution_raw)},
                       {"search_budget_profiles": changed_profiles}):
            with pytest.raises(RuntimeError, match="fresh observed bid read"):
                authority._validate_native_decision_quality_evidence_v1(production,
                    **{**read_context, **change}, verified_artifacts=frozenset({token}))
        stale_execution = copy.deepcopy(execution)
        stale_execution.bid_incumbent_proof["maximumQueries"] -= 1
        with pytest.raises(ValueError, match="checksum mismatch"):
            authority._validate_native_decision_quality_evidence_v1(production,
                **{**read_context, "execution_budget": stale_execution},
                verified_artifacts=frozenset({token}))
        changed = copy.deepcopy(production)
        changed["decisions"][0]["regret"] = 1
        changed["decisions"][0]["sha256"] = canonical_sha256({
            k: v for k, v in changed["decisions"][0].items() if k != "sha256"})
        changed["sha256"] = canonical_sha256({
            k: v for k, v in changed.items() if k not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="decision binding mismatch"):
            authority._validate_native_decision_quality_evidence_v1(changed, **read_context,
                verified_artifacts=frozenset({token}))
        changed = copy.deepcopy(production)
        changed["decisions"][0]["exactStateId"] = "wrong"
        changed["decisions"][0]["sha256"] = canonical_sha256({
            k: v for k, v in changed["decisions"][0].items() if k != "sha256"})
        changed["sha256"] = canonical_sha256({
            k: v for k, v in changed.items() if k not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="decision ownership mismatch"):
            authority._validate_native_decision_quality_evidence_v1(changed, **read_context,
                verified_artifacts=frozenset({token}))
    if fallback:
        started = time.monotonic()
        authority.validate_native_decision_quality_evidence_v1(production, **read_context)
        record_property("observedBidExternalReadSeconds", time.monotonic() - started)
    context = dict(rules=bundle.rules, source_prefix=prefix, normalized=exact, request=request,
        budget=budget, maximum_conflicts=settings["searchBudgets"]["profiles"][tier]["maximumConflicts"],
        opportunity_order=tuple(settings["qualityExecution"]["escalationOpportunityOrder"]),
        execution_budget=execution,
        maximum_seconds=budget.maximum_wall_seconds,
        **({"opponent_play_policy_id": play_policy} if play_policy is not None else {}))
    quality = artifact["proofModel"]["actionQuality"]
    if fallback:
        for replacement in (None, "observer_safe_trick_control_v1"):
            changed_execution = execution.to_dict()
            changed_execution.pop("sha256")
            changed_execution.pop("observedBidStrategyPlayFallbackId")
            if replacement is not None:
                changed_execution["observedBidStrategyPlayFallbackId"] = replacement
            with pytest.raises(ValueError, match="strategy fallback differs from frozen configuration"):
                build_native_observed_bid_quality_artifact_v1(result, **{**context,
                    "execution_budget": ScenarioQualityExecutionBudgetV1.build(changed_execution)})
    assert quality["actualActionOptimal"] is True and quality["regret"] == 0
    assert quality["actualValue"] is None and quality["optimalActionCount"] is None
    assert artifact["proof"]["claimSatisfied"] is None
    assert artifact["proof"]["winningRootActionCount"] is None
    if play_policy is not None:
        assert result["opponentPlayPolicy"] == certificate["opponentPlayPolicy"]
        assert certificate["opponentPlayPolicy"]["policyId"] == play_policy
        with pytest.raises(ValueError, match="differs from frozen configuration"):
            build_native_observed_bid_quality_artifact_v1(result,
                **{**context, "opponent_play_policy_id": None})
    else:
        assert result["incumbentProof"] == execution.observed_bid_incumbent_proof
        if fallback:
            assert result["selectedProof"]["strategy"]["kind"] != "native_observed_bid_visible_policy_floor_v1"
            assert result["incumbentAttempt"]["partialReceipt"]["exhaustionReason"] == "maximum_seconds"
            assert result["incumbentAttempt"]["visibility"] == "actor_information_set"
            for key, value in (("queriesUsed", -1), ("status", "scoped_strategy_threshold_complete"),
                               ("visibility", "complete_state"), ("maximumSeconds", 21)):
                changed = deepcopy(result)
                changed["incumbentAttempt"][key] = value
                changed["incumbentAttempt"]["sha256"] = canonical_sha256({
                    k: v for k, v in changed["incumbentAttempt"].items() if k != "sha256"})
                changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
                with pytest.raises(ValueError, match="incumbent fallback"):
                    build_native_observed_bid_quality_artifact_v1(changed, **context)
            changed = deepcopy(result)
            interrupted = changed["incumbentAttempt"]["partialReceipt"]
            interrupted["evidence"] = None
            interrupted["sha256"] = canonical_sha256({k: v for k, v in interrupted.items() if k != "sha256"})
            changed["incumbentAttempt"]["sha256"] = canonical_sha256({
                k: v for k, v in changed["incumbentAttempt"].items() if k != "sha256"})
            changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
            with pytest.raises(ValueError, match="substitutes the interrupted incumbent"):
                build_native_observed_bid_quality_artifact_v1(changed, **context)
        else:
            assert result["selectedProof"]["strategy"]["kind"] == "native_observed_bid_visible_policy_floor_v1"
        with pytest.raises(ValueError, match="strategy fallback differs from frozen" if fallback else "incumbent.*differs from frozen"):
            build_native_observed_bid_quality_artifact_v1(result, **{**context, "execution_budget": None})
    (tmp_path / "observed-last-bid-quality.json").write_text(
        json.dumps(artifact, sort_keys=True), encoding="utf-8")
    for change in ({"queriesUsed": result["queriesUsed"] - 1}, {"queryCountComplete": False},
                   {"status": "incomplete"}, {"selectedAction": "sun"},
                   {"limits": {**result["limits"], "maximumDepth": budget.maximum_depth + 1}}):
        changed = {**deepcopy(result), **change}
        changed["sha256"] = canonical_sha256({key: value for key, value in changed.items() if key != "sha256"})
        with pytest.raises(ValueError):
            build_native_observed_bid_quality_artifact_v1(changed, **context)


@pytest.mark.real_clock
@pytest.mark.parametrize("case", ("valid", "owner", "graph", "request", "state", "cancelled", "deadline"))
def test_observed_bid_readback_owns_normalization_and_one_deadline(root, monkeypatch, case):
    from copy import deepcopy
    from math import nextafter
    import yaml
    import time
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.scenario_contracts import ScenarioProofRequestV1, ScenarioProofBudgetV1, canonical_sha256
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    tier = "offline_heavyweight_v1"
    budget = ScenarioProofBudgetV1.build(tier, settings["proofBudgets"]["profiles"][tier])
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="actor_information_set",
        strategyClass="pure_deterministic", claim="outcome_forced", targetTeam="north_south",
        utility="awarded_point_margin", successThreshold=1, budgetProfile=tier, selectedAction="sun"))
    allocation = dict(schemaVersion=1, sourcePrefixHash=prefix.sha256, exactStateId=exact.exact_state_id,
        rulesHash=compiler.rules.sha256, sourcePrefix=prefix.to_dict(), normalizedState=exact.to_dict(),
        normalizationReceipt=exact.normalization_receipt.to_dict(), attempt={"certificate": {"test": "flow-only"}})
    if case == "owner":
        allocation["exactStateId"] = "wrong"
    if case == "state":
        allocation["normalizedState"]["nextActor"] = "east"
    allocation["sha256"] = canonical_sha256(allocation)
    artifact = dict(request=request.to_dict(), allocationModel=allocation,
        graph=deepcopy(allocation["attempt"]["certificate"]))
    if case == "graph":
        artifact["graph"] = {}
    if case == "request":
        artifact["request"] = {}
    # Adding ten seconds to this clock value rounds the deadline upward.
    clock, calls = [nextafter(247.0, 0.0)], []
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    def normalize(self, value, **kwargs):
        assert value == prefix and 0 < kwargs["maximum_seconds"] <= 10
        calls.append("normalize")
        clock[0] += 11 if case == "deadline" else 2
        return exact
    def build(*args, **kwargs):
        calls.append("reprove")
        assert 0 < kwargs["maximum_seconds"] <= 8
        return artifact
    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize", normalize)
    monkeypatch.setattr(authority, "build_native_observed_bid_quality_artifact_v1", build)
    context = dict(rules=compiler.rules, request=request, source_prefix_hash=prefix.sha256,
        exact_state_id=exact.exact_state_id, budget=budget, maximum_conflicts=1,
        opportunity_order=tuple(settings["qualityExecution"]["escalationOpportunityOrder"]),
        maximum_seconds=10, cancel_check=lambda: case == "cancelled")
    if case == "valid":
        authority.validate_native_observed_bid_quality_artifact_v1(artifact, **context)
        assert calls == ["normalize", "reprove"]
    else:
        with pytest.raises(InformationSetCompilationStopped if case in {"cancelled", "deadline"} else ValueError):
            authority.validate_native_observed_bid_quality_artifact_v1(artifact, **context)
        assert calls == (["normalize"] if case in {"state", "deadline"} else [])


@pytest.mark.parametrize("native_floor", (False, True))
@pytest.mark.parametrize("observed_upper", (False, True, "mixed", "refuted"))
@pytest.mark.parametrize("play_policy", (None, "observer_safe_trick_control_v1"))
def test_observed_bid_dominance_composes_verified_bounds_without_exact_value_claim(root, monkeypatch, native_floor, observed_upper, play_policy):
    import json
    from baloot_platform.config import build_bundle
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, canonical_sha256
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    settings = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component("challenge")["scenarioResearchV4"]
    minimax_budget = ScenarioProofBudgetV1.build("actor_information_set_exact_v1",
        settings["proofBudgets"]["profiles"]["actor_information_set_exact_v1"])
    expected_budget = None
    actions = sorted(row["code"] for row in exact.legal_actions)
    selected = actions[0]
    alternatives = [{"action": action, "exact": exact, "sourcePrefix": prefix, "receipt": {"action": action}}
        for action in actions[1:]]
    if observed_upper:
        for row in (alternatives[:1] if observed_upper == "mixed" else alternatives):
            receipt = {"schemaVersion": 1, "kind": ("native_observed_bid_refuted_upper_v1"
                if observed_upper == "refuted" else "native_observed_bid_opponent_policy_upper_v1"),
                "strategy": {}, "preparation": {}, "upperBound": 5}
            row["receipt"] = {**receipt, "sha256": canonical_sha256(receipt)}
    calls = []
    def lower(*args, **kwargs):
        assert "opponent_play_policy_id" not in kwargs
        assert kwargs.get("strategy_play_fallback_id") == (None if native_floor else "observer_safe_partner_feed_secure_control_v1")
        assert kwargs.get("minimax_budget") is (None if native_floor else expected_budget)
        calls.append(("lower", kwargs))
        kwargs["_on_queries_used"](2)
        return kwargs["required_threshold"]
    def upper(*args, **kwargs):
        assert kwargs.get("opponent_play_policy_id") == play_policy
        assert "minimax_budget" not in kwargs
        calls.append(("upper", kwargs))
        kwargs["_on_queries_used"](3)
        return 5
    # Composition test only. Fresh native premise readers have separate real fixtures.
    lower_reader = "validate_native_observed_bid_visible_floor_v1" if native_floor else "validate_native_observed_bid_threshold_v1"
    monkeypatch.setattr(authority, lower_reader, lower)
    monkeypatch.setattr(authority, "reverify_native_bid_world_upper_v1", upper)
    monkeypatch.setattr(authority, "validate_native_observed_bid_opponent_upper_v1", upper)
    monkeypatch.setattr(authority, "validate_native_observed_bid_refuted_upper_v1", upper)
    arguments = dict(rules=compiler.rules, exact=exact, source_prefix=prefix, selected_action=selected,
        strategy={"kind": "native_observed_bid_visible_policy_floor_v1"} if native_floor else {},
        preparation={}, alternatives=alternatives,
        opponent_policy={"auction_action": "pass", "escalation_action": "decline"},
        maximum_branches=13, maximum_model_bytes=100000000, maximum_conflicts=250000,
        maximum_seconds=30, maximum_queries=100, maximum_depth=64, maximum_observations=100,
        strategy_play_fallback_id="observer_safe_partner_feed_secure_control_v1",
        opponent_play_policy_id=play_policy)
    if play_policy is not None and observed_upper:
        with pytest.raises(ValueError, match="cannot relabel synthesized"):
            authority.compose_native_observed_bid_dominance_v1(**arguments, required_threshold=5)
        assert not calls
        return
    for floor, expected in ((5, True), (6, True), (4, None)):
        calls.clear()
        expected_budget = minimax_budget if floor == 6 else None
        arguments["minimax_budget"] = expected_budget
        result = authority.compose_native_observed_bid_dominance_v1(**arguments, required_threshold=floor)
        assert result["actualActionOptimal"] is expected
        assert result["status"] == ("selected_action_optimal" if expected else "bounds_do_not_establish_dominance")
        assert result["actualValue"] is result["bestValue"] is result["optimalActionCount"] is None
        assert result["wholeRoundCertified"] is False
        if play_policy is None:
            assert "opponentPlayPolicy" not in result
        else:
            assert result["opponentPlayPolicy"]["policyId"] == play_policy
            assert result["opponentPlayPolicy"]["team"] == Seat.parse(exact.next_actor).team.other.name.lower()
        assert result["opponentPolicy"] == (None if observed_upper else arguments["opponent_policy"])
        for bound, alternative in zip(result["alternatives"], alternatives, strict=True):
            if alternative["receipt"].get("kind") in {
                    "native_observed_bid_opponent_policy_upper_v1", "native_observed_bid_refuted_upper_v1"}:
                assert bound["sourceProofKind"] == alternative["receipt"]["kind"]
                assert "opponentPolicy" not in bound
            elif observed_upper:
                assert bound["opponentPolicy"] == arguments["opponent_policy"]
        assert result["limits"]["maximumQueries"] == 100
        assert result["limits"]["queriesPerRecheck"] * len(actions) + result["limits"]["comparisonQueriesReserved"] <= 100
        assert result["escalationOpportunityOrder"] is None
        assert result["executionReceipt"]["queriesUsed"] == 3 + 3 * (len(actions) - 1)
        assert result["executionReceipt"]["comparisonQueriesUsed"] == 1
        assert len(calls) == len(actions)
        assert sum(row[1]["maximum_queries"] for row in calls) + 1 <= 100
        assert all(0 < row[1]["maximum_seconds"] <= 30 for row in calls)


        stored = json.loads(json.dumps(result))
        assert stored == result
        context = {k: v for k, v in arguments.items() if k not in {"strategy", "preparation", "alternatives"}}
        context["alternative_worlds"] = {row["action"]: (row["exact"], row["sourcePrefix"]) for row in alternatives}
        assert authority.reverify_native_observed_bid_dominance_v1(stored,
            **context, required_threshold=floor) == result
        # Production readback receives the policy recorded in the certificate,
        # not the unused synthesis-time template retained by this test caller.
        assert authority.reverify_native_observed_bid_dominance_v1(stored,
            **{**context, "opponent_policy": stored["opponentPolicy"]},
            required_threshold=floor) == result
        if observed_upper == "mixed":
            malformed = copy.deepcopy(stored)
            fixed = next(row for row in malformed["alternatives"] if "opponentPolicy" in row)
            fixed["opponentPolicy"] = {"auction_action": True, "escalation_action": "decline"}
            malformed["sha256"] = canonical_sha256({k: v for k, v in malformed.items() if k != "sha256"})
            calls.clear()
            with pytest.raises(ValueError):
                authority.reverify_native_observed_bid_dominance_v1(malformed,
                    **{**context, "opponent_policy": None}, required_threshold=floor)
            assert not calls
        assert stored["premises"]["sourcePrefix"] == prefix.to_dict()
        assert canonical_sha256(stored["premises"]["selectedStrategy"]) == stored["selectedProofHash"]
        assert canonical_sha256(stored["premises"]["selectedPreparation"]) == stored["preparationHash"]
        for premise, bound in zip(stored["premises"]["alternativeWorlds"], stored["alternatives"], strict=True):
            assert premise["action"] == bound["action"]
            assert canonical_sha256(premise["receipt"]) == bound["sourceProofHash"]
        assert stored["sha256"] == canonical_sha256({k: v for k, v in stored.items() if k != "sha256"})
        altered = copy.deepcopy(stored)
        altered["premises"]["alternativeWorlds"][0]["receipt"]["action"] = "wrong"
        assert altered["sha256"] != canonical_sha256({k: v for k, v in altered.items() if k != "sha256"})
        # The certificate owns copies; caller mutation must not change its content.
        alternatives[0]["receipt"]["changedAfterComposition"] = True
        assert "changedAfterComposition" not in result["premises"]["alternativeWorlds"][0]["receipt"]
        alternatives[0]["receipt"].pop("changedAfterComposition")
        for key, forged_value in (("selectedLowerBound", 999), ("modelProto", "forged"),
                                  ("observerRootHash", "0" * 64)):
            forged = copy.deepcopy(stored)
            forged[key] = forged_value
            forged["sha256"] = canonical_sha256({k: v for k, v in forged.items() if k != "sha256"})
            with pytest.raises(ValueError, match="differs from fresh"):
                authority.reverify_native_observed_bid_dominance_v1(forged, **context, required_threshold=floor)
    for invalid_receipt in (None, [], "invalid", True):
        calls.clear()
        invalid = [{**row, "receipt": invalid_receipt} for row in alternatives]
        with pytest.raises(ValueError, match="alternative coverage mismatch"):
            authority.compose_native_observed_bid_dominance_v1(
                **{**arguments, "alternatives": invalid}, required_threshold=5)
        assert not calls  # Reject malformed premises before spending any proof budget.
    for change in ({"alternatives": alternatives[:-1]}, {"alternatives": alternatives + alternatives[:1]},
                   {"maximum_queries": 1}, {"maximum_conflicts": True}, {"required_threshold": True},
                   {"opponent_policy": False}, {"opponent_policy": []}, {"minimax_budget": {}},
                   {"minimax_budget": minimax_budget, "strategy_play_fallback_id": None}) + (
                       ({"opponent_policy": None},) if observed_upper not in (True, "refuted") else ()):
        calls.clear()
        with pytest.raises(ValueError):
            authority.compose_native_observed_bid_dominance_v1(**{**arguments, "required_threshold": 5, **change})
        assert not calls
    calls.clear()
    with pytest.raises(InformationSetCompilationStopped):
        authority.compose_native_observed_bid_dominance_v1(**arguments, required_threshold=5, cancel_check=lambda: True)
    assert not calls
    for count in (None, True, -1, 101):
        def broken_accounting(*args, **kwargs):
            if count is not None:
                kwargs["_on_queries_used"](count)
            return 5
        with monkeypatch.context() as patch:
            patch.setattr(authority, lower_reader, broken_accounting)
            with pytest.raises(ValueError, match="query accounting"):
                authority.compose_native_observed_bid_dominance_v1(**arguments, required_threshold=5)
    if observed_upper:
        for change in ({"schemaVersion": True}, {"upperBound": True}, {"unexpected": 1}):
            forged = copy.deepcopy(alternatives)
            forged[0]["receipt"].update(change)
            forged[0]["receipt"]["sha256"] = canonical_sha256({
                k: v for k, v in forged[0]["receipt"].items() if k != "sha256"})
            with pytest.raises(ValueError, match="receipt integrity"):
                authority.compose_native_observed_bid_dominance_v1(
                    **{**arguments, "alternatives": forged}, required_threshold=5)


@pytest.mark.parametrize("upper_kind", ("fixed", "synthesized", "mixed", "searched", "searched_refuted", "probe", "probe_unrestricted"))
def test_observed_bid_alternative_worlds_survive_owned_quality_readback(root, monkeypatch, upper_kind):
    """Native world legality; stub strategic readers only to isolate admission/accounting."""
    refuted_search = upper_kind == "searched_refuted"
    if refuted_search:
        upper_kind = "searched"
    unrestricted_probe = upper_kind == "probe_unrestricted"
    if unrestricted_probe:
        upper_kind = "probe"
    from copy import deepcopy
    from baloot_platform.config import build_bundle
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import (
        ScenarioProofBudgetV1, ScenarioProofRequestV1, ScenarioQualityExecutionBudgetV1, canonical_sha256)

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    world_prefix, _ = compiler._find_native_bid_compatible_world(exact, prefix,
        selected_action="sun", maximum_seconds=15, maximum_conflicts=250000,
        maximum_model_bytes=100000000, maximum_worlds=1)
    assert world_prefix != prefix
    normalizer = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True)
    world = normalizer.normalize(world_prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    settings = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component("challenge")["scenarioResearchV4"]
    tier = "offline_heavyweight_v1"
    budget = ScenarioProofBudgetV1.build(tier, settings["proofBudgets"]["profiles"][tier])
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="actor_information_set",
        strategyClass="pure_deterministic", claim="outcome_forced", targetTeam="north_south",
        utility="awarded_point_margin", successThreshold=1, budgetProfile=tier, selectedAction="sun"))
    actions = sorted(row["code"] for row in exact.legal_actions if row["code"] != "sun")
    order = tuple(settings["qualityExecution"]["escalationOpportunityOrder"])
    limits = dict(maximum_iterations=10, maximum_queries=100, maximum_observations=100,
        maximum_branches=13, maximum_model_bytes=1000000, maximum_conflicts=100,
        maximum_seconds=10, maximum_depth=64, opportunity_order=order)

    def hashed(core):
        return {**core, "sha256": canonical_sha256(core)}

    stage_limits = dict(branches=13, modelBytes=1000000, conflictsPerSolve=100,
        maximumDepth=64, queries=100, seconds=10)
    proof_order = []
    def prepare(actual, source, **kwargs):
        assert actual is world and source is world_prefix
        proof_order.append("alternative")
        return (), hashed(dict(kind="native_bid_streamed_world_upper_v1", queriesUsed=2, limits=stage_limits))
    def selected(actual, source, target, **kwargs):
        assert actual is exact and source is prefix and kwargs["visibility"] == "actor_information_set"
        assert kwargs["required_threshold"] == 5
        proof_order.append("selected")
        if upper_kind == "searched":
            assert kwargs["maximum_queries"] == limits["maximum_queries"]
        return hashed(dict(status="scoped_strategy_threshold_complete", queriesUsed=3,
            preparation=hashed(dict(queriesUsed=2, limits=stage_limits)),
            strategy=hashed(dict(queriesUsed=1, limits={**stage_limits, "masterIterations": 10, "observations": 100}))))
    def lower(*args, **kwargs):
        assert kwargs["exact"].to_dict() == exact.to_dict() and kwargs["source_prefix"] == prefix
        kwargs["_on_queries_used"](2)
        return 5
    def upper(*args, **kwargs):
        assert kwargs["exact"].to_dict() == world.to_dict() and kwargs["source_prefix"] == world_prefix
        kwargs["_on_queries_used"](1)
        return 5
    monkeypatch.setattr(compiler, "_prepare_native_bid_branches", prepare)
    monkeypatch.setattr(compiler, "_solve_native_bid_policy_bounds", selected)
    monkeypatch.setattr(authority, "validate_native_bid_world_upper_v1", lambda *a, **k: 5)
    monkeypatch.setattr(authority, "validate_native_observed_bid_threshold_v1", lower)
    monkeypatch.setattr(authority, "reverify_native_bid_world_upper_v1", upper)
    monkeypatch.setattr(authority, "validate_native_observed_bid_opponent_upper_v1", upper)
    monkeypatch.setattr(authority, "validate_native_observed_bid_refuted_upper_v1", upper)
    search_configuration = dict(schemaVersion=1, upperBound=5, maximumWorlds=len(actions), maximumSeconds=4,
        maximumQueries=40, maximumWorldSeconds=2, maximumWorldQueries=10)
    if refuted_search:
        search_configuration = dict(schemaVersion=2, method="information_set_refutation_v1",
            upperBound=5, maximumSeconds=4, maximumQueries=40)
    def search(actual, source, **kwargs):
        assert actual is exact and source is prefix
        assert proof_order[0] == "selected"
        proof_order.append("alternative")
        if refuted_search:
            assert "strategy_play_fallback_id" not in kwargs and "minimax_budget" not in kwargs
            owned_limits = {**stage_limits, "seconds": kwargs["maximum_seconds"], "queries": kwargs["maximum_queries"]}
            preparation = hashed(dict(queriesUsed=2, limits=owned_limits))
            strategy = hashed(dict(queriesUsed=1, limits={**owned_limits, "masterIterations": 10, "observations": 100}))
            receipt = hashed(dict(schemaVersion=1, kind="native_observed_bid_refuted_upper_v1",
                preparation=preparation, strategy=strategy, upperBound=5))
            generation = hashed(dict(schemaVersion=1, kind="native_bid_policy_bounds_v1",
                status="scoped_strategy_threshold_refuted", visibility="actor_information_set",
                targetTeam="north_south", strategicCoverage=False, selectedRootAction=kwargs["selected_action"],
                queriesUsed=3, maximumSeconds=kwargs["maximum_seconds"], preparation=preparation, strategy=strategy))
            log = hashed(dict(schemaVersion=1, kind="native_observed_bid_information_set_search_v1",
                status="upper_bound_proven", stage="fresh_verification", strategicCoverage=False,
                impossibilityProven=False, queryCountComplete=True, action=kwargs["selected_action"],
                sourcePrefixHash=prefix.sha256, rulesHash=compiler.rules.sha256, upperBound=5,
                queriesCharged=7, verificationQueriesUsed=4, generation=generation,
                limits=dict(seconds=kwargs["maximum_seconds"], queries=kwargs["maximum_queries"],
                    masterIterations=10, observations=100, branches=13, modelBytes=1000000,
                    evidenceBytes=100000000, conflicts=100, depth=64)))
            return dict(action=kwargs["selected_action"], exact=world, sourcePrefix=world_prefix, receipt=receipt), log
        preparation = hashed(dict(queriesUsed=2, limits=stage_limits))
        strategy = hashed(dict(queriesUsed=1, limits={**stage_limits, "masterIterations": 10, "observations": 100}))
        receipt = hashed(dict(schemaVersion=1, kind="native_observed_bid_opponent_policy_upper_v1",
            preparation=preparation, strategy=strategy, upperBound=5))
        search_receipt = hashed(dict(schemaVersion=1, kind="native_observed_bid_world_search_v1",
            status="upper_bound_proven", strategicCoverage=False, impossibilityProven=False,
            sourcePrefixHash=prefix.sha256, rulesHash=compiler.rules.sha256,
            action=kwargs["selected_action"], upperBound=5, queriesCharged=7,
            attempts=[dict(index=0, queryAllowance=10, stage="fresh_verification", status="upper_bound_proven",
                queriesCharged=7, selection={"queriesUsed": 1}, preparation=preparation, strategy=strategy,
                verificationQueriesUsed=1, worldPrefix=world_prefix.to_dict())],
            limits=dict(worlds=kwargs["maximum_worlds"], seconds=kwargs["maximum_seconds"], queries=kwargs["maximum_queries"],
                worldSeconds=2, worldQueries=10, iterationsPerWorld=10, observationsPerWorld=100,
                branches=13, modelBytes=1000000, evidenceBytes=100000000, conflicts=100, depth=64)))
        return dict(action=kwargs["selected_action"], exact=world, sourcePrefix=world_prefix, receipt=receipt), search_receipt
    searcher_name = ("_search_observed_bid_information_set_upper" if refuted_search else "_search_observed_bid_alternative_upper")
    monkeypatch.setattr(compiler, searcher_name, search)
    probe_configuration = dict(schemaVersion=1, maximumSeconds=2, maximumQueries=4,
        worldRankingId="opponent_auction_value_v1", opponentAuctionAction="sun")
    probe_policy = None if unrestricted_probe else "observer_safe_trick_control_v1"
    def inconclusive_probe(actual, source, **kwargs):
        assert actual is exact and source is prefix and kwargs["threshold"] == 5
        return None, hashed(dict(schemaVersion=1, kind="native_bid_threshold_probe_v1",
            status="incomplete", stage="selection", configuration=probe_configuration,
            selectedAction="sun", requiredThreshold=5, rulesHash=compiler.rules.sha256,
            sourcePrefixHash=prefix.sha256, strategicCoverage=False, queriesCharged=4,
            worldPrefix=world_prefix.to_dict(), worldPrefixHash=world_prefix.sha256,
            queryCountComplete=False, cause={"exhaustionReason": "maximum_seconds"}))
    monkeypatch.setattr(compiler, "_probe_observed_bid_threshold_refutation", inconclusive_probe)
    attempt = compiler._prove_observed_bid_dominance(exact, prefix, selected_action="sun",
        opponent_policy={"auction_action": "pass", "escalation_action": "decline"},
        reserve_artifact_verification=True,
        **({"threshold_refutation_probe": probe_configuration, "opponent_play_policy_id": probe_policy,
            "maximum_evidence_bytes": 100000000} if upper_kind == "probe" else {}),
        **({"alternative_proof": search_configuration, "maximum_evidence_bytes": 100000000}
           if upper_kind == "searched" else {"alternative_worlds": {action: (world, world_prefix) for action in actions}}), **limits)
    assert attempt["status"] == "selected_action_optimal"
    assert proof_order == (["selected", *(["alternative"] * len(actions))] if upper_kind == "searched"
                           else [*(["alternative"] * len(actions)), "selected"])
    assert attempt["certificate"]["limits"]["maximumQueries"] == (100 - (7 if upper_kind == "searched" else 2) * len(actions) - 3 - 2 - (4 if upper_kind == "probe" else 0)) // 2
    if upper_kind == "searched":
        # A failed or stopped selected proof cannot use any alternative cap.
        # Preserve its partial evidence without launching that expensive search.
        from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
        for status in ("scoped_strategy_threshold_refuted", "incomplete", "cancelled", "exhausted_native"):
            with monkeypatch.context() as patch:
                def unfinished(*args, **kwargs):
                    assert kwargs["required_threshold"] == search_configuration["upperBound"]
                    if status == "cancelled":
                        raise InformationSetCompilationStopped("cancelled", stage="native_bid_preparation",
                            evidence={"queriesUsed": 2})
                    if status == "exhausted_native":
                        raise InformationSetCompilationStopped("maximum_queries", stage="native_bid_policy_bounds",
                            evidence={"queriesUsed": 2})
                    return {"status": status, "queriesUsed": 3, "strategy": {}, "preparation": {}}
                def unused(*args, **kwargs):
                    pytest.fail("alternative search started without a selected floor")
                patch.setattr(compiler, "_solve_native_bid_policy_bounds", unfinished)
                patch.setattr(compiler, searcher_name, unused)
                stopped = compiler._prove_observed_bid_dominance(exact, prefix, selected_action="sun",
                    opponent_policy={"auction_action": "pass", "escalation_action": "decline"},
                    alternative_proof=search_configuration, maximum_evidence_bytes=100000000, **limits)
                assert stopped["status"] == "incomplete" and stopped["stage"] == "visible_strategy"
                assert stopped["certificate"] is None and stopped["alternativeSearches"] == []
                assert stopped["alternatives"] == [] and stopped["queriesUsed"] == (2 if status in {"cancelled", "exhausted_native"} else 3)
    if upper_kind in {"synthesized", "mixed"}:
        # Admission/accounting fixture, not a synthesized strategic proof.
        for row in attempt["alternatives"][:1] if upper_kind == "mixed" else attempt["alternatives"]:
            row["receipt"] = hashed(dict(schemaVersion=1, kind="native_observed_bid_opponent_policy_upper_v1",
                preparation=row["receipt"], upperBound=5,
                strategy=hashed(dict(queriesUsed=1, limits={**stage_limits,
                    "masterIterations": 10, "observations": 100}))))
        stage_queries = 3 + sum(row["receipt"]["preparation"]["queriesUsed"] + row["receipt"]["strategy"]["queriesUsed"]
            if row["receipt"].get("kind") == "native_observed_bid_opponent_policy_upper_v1"
            else row["receipt"]["queriesUsed"] for row in attempt["alternatives"])
        certificate = authority.compose_native_observed_bid_dominance_v1(rules=compiler.rules,
            exact=exact, source_prefix=prefix, selected_action="sun", strategy=attempt["selectedProof"]["strategy"],
            preparation=attempt["selectedProof"]["preparation"], required_threshold=5,
            alternatives=[{**row, "exact": world, "sourcePrefix": world_prefix} for row in attempt["alternatives"]],
            opponent_policy={"auction_action": "pass", "escalation_action": "decline"},
            maximum_queries=(100 - stage_queries - 2) // 2,
            **{key: value for key, value in limits.items() if key not in {"maximum_queries", "maximum_iterations"}})
        attempt = hashed({**{key: value for key, value in attempt.items() if key != "sha256"},
            "certificate": certificate, "queriesUsed": stage_queries + certificate["executionReceipt"]["queriesUsed"]})
    normalized_prefixes = []
    original_normalize = OrToolsExactStateNormalizerV1.normalize
    def normalize(self, value, **kwargs):
        normalized_prefixes.append(value.sha256)
        return original_normalize(self, value, **kwargs)
    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize", normalize)
    context = dict(rules=compiler.rules, source_prefix=prefix, normalized=exact, request=request,
        budget=budget, maximum_conflicts=100, opportunity_order=order, maximum_seconds=10)
    if upper_kind == "searched":
        context["execution_budget"] = ScenarioQualityExecutionBudgetV1.build({
            **{key: value for key, value in settings["qualityExecution"].items()
               if key not in {"bidOpponentPlayPolicyId", "bidIncumbentProof"}},
            "bidAlternativeProof": search_configuration})
    if upper_kind == "probe":
        context.update(opponent_play_policy_id=probe_policy,
            execution_budget=ScenarioQualityExecutionBudgetV1.build({
                **{key: value for key, value in settings["qualityExecution"].items()
                   if key not in {"bidIncumbentProof", "bidOpponentPlayPolicyId"}},
                **({"bidOpponentPlayPolicyId": probe_policy} if probe_policy is not None else {}),
                "bidThresholdRefutation": probe_configuration}))
    artifact = authority.build_native_observed_bid_quality_artifact_v1(attempt, **context)
    if upper_kind == "fixed":
        changed = deepcopy(attempt)
        changed["selectedProof"]["strategy"]["scope"] = {"minimaxProofBudgetHash": "0" * 64}
        changed["selectedProof"]["strategy"] = hashed({key: value for key, value in changed["selectedProof"]["strategy"].items() if key != "sha256"})
        changed["selectedProof"] = hashed({key: value for key, value in changed["selectedProof"].items() if key != "sha256"})
        changed["certificate"]["premises"]["selectedStrategy"] = deepcopy(changed["selectedProof"]["strategy"])
        changed["certificate"]["selectedProofHash"] = canonical_sha256(changed["selectedProof"]["strategy"])
        changed["certificate"] = hashed({key: value for key, value in changed["certificate"].items() if key != "sha256"})
        changed = hashed({key: value for key, value in changed.items() if key != "sha256"})
        with pytest.raises(ValueError, match="minimax budget differs"):
            authority.build_native_observed_bid_quality_artifact_v1(changed, **context)
        with pytest.raises(ValueError, match="minimax budget differs"):
            authority._build_native_observed_bid_quality_artifact_v1(
                changed, **context, verified_certificates=frozenset({changed["certificate"]["sha256"]}))
    for schedule in ([], {}, "unknown"):
        changed = hashed({**{key: value for key, value in attempt.items() if key != "sha256"},
            "selectionSchedule": schedule})
        with pytest.raises(ValueError, match="selection schedule"):
            authority.build_native_observed_bid_quality_artifact_v1(changed, **context)
    if upper_kind == "probe":
        assert artifact["allocationModel"]["generationQueryCountUnit"] == "conservative_budget_charges"
        for mutation in ({"queriesCharged": True}, {"requiredThreshold": 6}, {"status": "threshold_refuted"},
                         {"worldPrefixHash": "wrong"},
                         {"sourcePrefixHash": world_prefix.sha256}, {"configuration": {**probe_configuration, "maximumQueries": 5}}):
            changed = deepcopy(attempt)
            changed["thresholdRefutationProbe"].update(mutation)
            changed["thresholdRefutationProbe"] = hashed({
                key: value for key, value in changed["thresholdRefutationProbe"].items() if key != "sha256"})
            with pytest.raises(ValueError, match="probe"):
                authority.build_native_observed_bid_quality_artifact_v1(
                    hashed({key: value for key, value in changed.items() if key != "sha256"}), **context)
        with pytest.raises(ValueError, match="probe"):
            authority.build_native_observed_bid_quality_artifact_v1(attempt,
                **{key: value for key, value in context.items() if key != "execution_budget"})
    if upper_kind == "searched":
        assert artifact["allocationModel"]["generationQueryCountUnit"] == "conservative_budget_charges"
        for mutation in ({"queriesCharged": True}, {"upperBound": 6}, {"status": "not_found_within_budget"}):
            changed = deepcopy(attempt)
            changed["alternativeSearches"][0].update(mutation)
            changed["alternativeSearches"][0] = hashed({
                k: v for k, v in changed["alternativeSearches"][0].items() if k != "sha256"})
            with pytest.raises(ValueError, match="search"):
                authority.build_native_observed_bid_quality_artifact_v1(
                    hashed({k: v for k, v in changed.items() if k != "sha256"}), **context)
    assert normalized_prefixes == [world_prefix.sha256]  # One witness/uniqueness pair per distinct world.
    checked = attempt["certificate"]["executionReceipt"]["queriesUsed"]
    assert artifact["allocationModel"]["generationQueriesUsed"] == attempt["queriesUsed"] + checked + 2
    if upper_kind == "fixed":
        token = frozenset({attempt["certificate"]["sha256"]})
        def fresh_replay(*args, **kwargs):
            raise RuntimeError("fresh certificate replay")
        with monkeypatch.context() as patch:
            patch.setattr(authority, "reverify_native_observed_bid_dominance_v1", fresh_replay)
            normalized_prefixes.clear()
            owned = authority._build_native_observed_bid_quality_artifact_v1(
                attempt, **context, verified_certificates=token)
            assert not normalized_prefixes
            assert owned["graph"] == artifact["graph"] and owned["graph"] is not attempt["certificate"]
            assert owned["allocationModel"]["schemaVersion"] == 2
            assert owned["allocationModel"]["generationQueryCountUnit"] == "conservative_budget_charges"
            assert owned["allocationModel"]["generationQueriesUsed"] == artifact["allocationModel"]["generationQueriesUsed"]
            for tokens in (frozenset(), frozenset({"wrong"})):
                with pytest.raises(RuntimeError, match="fresh certificate replay"):
                    authority._build_native_observed_bid_quality_artifact_v1(
                        attempt, **context, verified_certificates=tokens)
            with pytest.raises(RuntimeError, match="fresh certificate replay"):
                authority.build_native_observed_bid_quality_artifact_v1(attempt, **context, receipt_version=2)
            with pytest.raises(TypeError, match="verified_certificates"):
                authority.build_native_observed_bid_quality_artifact_v1(attempt, **context, verified_certificates=token)
            changed = deepcopy(attempt)
            changed["certificate"]["exactStateId"] = "wrong"
            changed["certificate"] = hashed({k: v for k, v in changed["certificate"].items() if k != "sha256"})
            changed = hashed({k: v for k, v in changed.items() if k != "sha256"})
            with pytest.raises(ValueError, match="certificate context mismatch"):
                authority._build_native_observed_bid_quality_artifact_v1(changed, **context, verified_certificates=token)
        # Stored v2 evidence must still take the public normalization/reproof path.
        authority.validate_native_observed_bid_quality_artifact_v1(owned,
            rules=compiler.rules, request=request, source_prefix_hash=prefix.sha256,
            exact_state_id=exact.exact_state_id, budget=budget, maximum_conflicts=100,
            opportunity_order=order, maximum_seconds=10)
        for field, replacement in (("generationQueryCountUnit", None), ("generationQueriesUsed", 0), ("schemaVersion", 1)):
            changed = deepcopy(owned)
            changed["allocationModel"][field] = replacement
            changed["allocationModel"] = hashed({k: v for k, v in changed["allocationModel"].items() if k != "sha256"})
            with pytest.raises(ValueError, match="differs from fresh native evidence"):
                authority.validate_native_observed_bid_quality_artifact_v1(changed,
                    rules=compiler.rules, request=request, source_prefix_hash=prefix.sha256,
                    exact_state_id=exact.exact_state_id, budget=budget, maximum_conflicts=100,
                    opportunity_order=order, maximum_seconds=10)
    assert artifact["graph"]["wholeRoundCertified"] is False
    assert artifact["proofModel"]["actionQuality"]["actualValue"] is None
    normalized_prefixes.clear()
    authority.validate_native_observed_bid_quality_artifact_v1(artifact,
        rules=compiler.rules, request=request, source_prefix_hash=prefix.sha256,
        exact_state_id=exact.exact_state_id, budget=budget, maximum_conflicts=100,
        opportunity_order=order, maximum_seconds=10, execution_budget=context.get("execution_budget"),
        **({"opponent_play_policy_id": probe_policy} if upper_kind == "probe" else {}))
    assert normalized_prefixes == [prefix.sha256, world_prefix.sha256]
    normalized_prefixes.clear()
    if upper_kind in {"synthesized", "mixed", "searched"}:
        # Rehash every owner so these failures must come from budget/schema
        # validation, not a stale outer checksum. Fail before normalization.
        for key, value in (("upperBound", True), ("unexpected", 1),
                           ("preparationQueries", True), ("strategyQueries", -1),
                           ("modelBytes", stage_limits["modelBytes"] + 1)):
            changed = deepcopy(attempt)
            row = next(row for row in changed["alternatives"]
                if row["receipt"].get("kind") in {
                    "native_observed_bid_opponent_policy_upper_v1", "native_observed_bid_refuted_upper_v1"})
            receipt = row["receipt"]
            if key in {"preparationQueries", "strategyQueries", "modelBytes"}:
                child = receipt["preparation" if key != "strategyQueries" else "strategy"]
                if key == "modelBytes":
                    child["limits"]["modelBytes"] = value
                else:
                    child["queriesUsed"] = value
                child["sha256"] = canonical_sha256({k: v for k, v in child.items() if k != "sha256"})
            else:
                receipt[key] = value
            receipt["sha256"] = canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})
            premise = next(p for p in changed["certificate"]["premises"]["alternativeWorlds"] if p["action"] == row["action"])
            premise["receipt"] = deepcopy(receipt)
            bound = next(p for p in changed["certificate"]["alternatives"] if p["action"] == row["action"])
            bound["sourceProofHash"] = canonical_sha256(receipt)
            changed["certificate"] = hashed({k: v for k, v in changed["certificate"].items() if k != "sha256"})
            with pytest.raises(ValueError):
                authority.build_native_observed_bid_quality_artifact_v1(
                    hashed({k: v for k, v in changed.items() if k != "sha256"}), **context)
            assert normalized_prefixes == []

    def changed_attempt(prefix_data):
        changed = deepcopy(attempt)
        for row in changed["certificate"]["premises"]["alternativeWorlds"]:
            row["sourcePrefix"] = prefix_data
        for search in changed.get("alternativeSearches", []):
            if "attempts" in search:
                search["attempts"][-1]["worldPrefix"] = prefix_data
            search["sha256"] = canonical_sha256({k: v for k, v in search.items() if k != "sha256"})
        changed["certificate"] = hashed({k: v for k, v in changed["certificate"].items() if k != "sha256"})
        return hashed({k: v for k, v in changed.items() if k != "sha256"})
    raw = world_prefix.to_dict()
    raw.pop("sha256")
    raw["matchScore"] = [1, 0]
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
    with pytest.raises(ValueError, match="changes the public prefix"):
        authority.build_native_observed_bid_quality_artifact_v1(changed_attempt(ExactTracePrefixV1.build(raw).to_dict()), **context)
    assert normalized_prefixes == []
    # Swapping a visible card keeps the public timeline but must fail native observer equivalence.
    layout = DealLayoutV1.build(compiler.rules.structural_domain, dealer=Seat.parse(prefix.dealer))
    raw = world_prefix.to_dict()
    raw.pop("sha256")
    visible = layout.positions_for(Seat.parse(exact.next_actor), final=False)[0]
    hidden = next(p for p in range(32) if p not in layout.positions_for(Seat.parse(exact.next_actor), final=False)
        and p != layout.purchase_position)
    raw["deckOrder"][visible], raw["deckOrder"][hidden] = raw["deckOrder"][hidden], raw["deckOrder"][visible]
    with pytest.raises(ValueError, match="observer mismatch"):
        authority.build_native_observed_bid_quality_artifact_v1(changed_attempt(ExactTracePrefixV1.build(raw).to_dict()), **context)
    for invalid in ({"sun": (world, world_prefix)}, {actions[0]: (exact, world_prefix)}, {actions[0]: None}):
        with pytest.raises(ValueError):
            compiler._prove_observed_bid_dominance(exact, prefix, selected_action="sun",
                opponent_policy={"auction_action": "pass", "escalation_action": "decline"},
                alternative_worlds=invalid, **limits)


@pytest.mark.parametrize("project_policy_id,auction_policy_id,selected_action", (
    ("known_project_max_value_v1", None, "sun"),
    ("full_hand_project_max_value_v1", None, "sun"),
    ("full_hand_project_max_value_v1", "prefer_legal_sun_then_pass_v1", "pass"),
    ("full_hand_project_max_value_v1", "prefer_root_actor_sun_partner_pass_v1", "pass")))
def test_observed_known_project_floor_native_query_and_readback(root, monkeypatch, project_policy_id,
        auction_policy_id, selected_action):
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, canonical_sha256

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    actor = Seat.parse(exact.next_actor)
    layout = DealLayoutV1.build(compiler.rules.structural_domain, dealer=Seat.parse(prefix.dealer))
    raw = prefix.to_dict()
    raw.pop("sha256")
    deck = raw["deckOrder"]
    known = [Card.of(suit, Rank.ACE).id for suit in Suit] + [Card.of(Suit.CLUBS, Rank.SEVEN).id]
    for position, card in zip(layout.positions_for(actor, final=False), known, strict=True):
        other = deck.index(card)
        deck[position], deck[other] = deck[other], deck[position]
    other = deck.index(Card.of(Suit.HEARTS, Rank.SEVEN).id)
    deck[layout.purchase_position], deck[other] = deck[other], deck[layout.purchase_position]
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    nonplay = {"auction_action": None, "escalation_action": None}
    limits = dict(maximum_branches=13, maximum_model_bytes=100000000,
        maximum_conflicts=250000, maximum_seconds=60, maximum_queries=100, maximum_depth=64)
    models, preparation = compiler._prepare_native_bid_branches(exact, prefix,
        visibility="actor_information_set", selected_root_action=selected_action,
        **({"target_native_auction_policy_id": auction_policy_id} if auction_policy_id else
           {"policy_team": actor.team, "nonplay_policy": nonplay}), **limits)
    # A positive all-hidden-world guarantee, not a vacuous domain lower bound.
    threshold = 34
    args = dict(selected_action=selected_action, native_play_policy_id="observer_safe_partner_feed_compact_v1",
        observed_project_policy_id=project_policy_id, nonplay_policy=nonplay,
        required_threshold=threshold, **limits)
    if auction_policy_id:
        args["observed_auction_policy_id"] = auction_policy_id
    strategy, preparation = compiler._prove_observed_bid_visible_policy_floor(exact, prefix, **args)
    assert strategy["observedProjectPolicyId"] == args["observed_project_policy_id"]
    for row in strategy["branches"]:
        if project_policy_id == "known_project_max_value_v1":
            assert set(row["projectActions"]) == {actor.name.lower()}
            assert row["projectActions"][actor.name.lower()].startswith("declare:PROJECT-")
            assert row["receipt"]["policy"]["projectActions"] == row["projectActions"]
            assert len(row["receipt"]["bindings"]) == 1
            assert row["receipt"]["bindings"][0]["status"] == "bound"
        else:
            assert "projectActions" not in row
            assert row["receipt"]["policy"]["nativeProjectPolicyId"] == project_policy_id
            assert row["receipt"]["policy"]["nativeProjectPolicyBinding"]["kind"] == "native_visible_project_policy_binding_v1"
            assert len(row["receipt"]["bindings"]) == int(auction_policy_id is not None)
    if auction_policy_id:
        assert strategy["observedAuctionPolicyId"] == auction_policy_id
        assert preparation["scope"]["targetNativeAuctionPolicyId"] == auction_policy_id
        assert [row["partitionIndex"] for row in preparation["branches"] if row["retained"]] == [1]
        assert all(row["reachability"]["policyBindings"][0]["status"] == "bound" for row in preparation["branches"])
    owner = dict(rules=compiler.rules, exact=exact, source_prefix=prefix,
        selected_action=selected_action, required_threshold=threshold, maximum_observations=100, **limits)
    used = []
    assert authority.validate_native_observed_bid_visible_floor_v1(strategy, preparation,
        **owner, _on_queries_used=used.append) == threshold
    query_cost = (2 if project_policy_id == "known_project_max_value_v1" else 1) + int(auction_policy_id is not None)
    assert used == [preparation["queriesUsed"] + query_cost * len(models)]
    if auction_policy_id:
        altered = copy.deepcopy(strategy)
        altered.pop("observedAuctionPolicyId")
        altered["sha256"] = canonical_sha256({k: v for k, v in altered.items() if k != "sha256"})
        with pytest.raises(ValueError, match="differs from preparation"):
            authority.validate_native_observed_bid_visible_floor_v1(altered, preparation, **owner)
        with pytest.raises(ValueError, match="cannot mix auction authorities"):
            compiler._prove_observed_bid_visible_policy_floor(exact, prefix,
                **{**args, "nonplay_policy": {"auction_action": "pass", "escalation_action": None}})
    # Admission mutations reuse native reconstructed models; they cannot select a new declaration.
    monkeypatch.setattr(authority, "validate_native_observed_bid_preparation_v1", lambda *a, **kw: models)
    changed = copy.deepcopy(strategy)
    if project_policy_id == "known_project_max_value_v1":
        changed["branches"][0]["projectActions"][actor.name.lower()] = "pass"
    else:
        changed["branches"][0]["projectActions"] = {actor.name.lower(): "pass"}
    changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
    with pytest.raises(ValueError, match="substitutes actor-visible declarations|branch coverage mismatch"):
        authority.validate_native_observed_bid_visible_floor_v1(changed, preparation, **owner)
    with pytest.raises(ValueError, match="unsupported native observed project"):
        compiler._prove_observed_bid_visible_policy_floor(exact, prefix,
            **{**args, "observed_project_policy_id": "unknown"})


def test_native_target_auction_preparation_preserves_interrupted_binding(root, monkeypatch):
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    stopped = {"status": "incomplete", "solverStatus": "UNKNOWN", "action": "prefer_legal_sun_then_pass_v1"}
    monkeypatch.setattr(OrToolsScenarioModelV1, "_bind_native_policy_choice", lambda *a, **kw: stopped)
    with pytest.raises(InformationSetCompilationStopped) as failure:
        compiler._prepare_native_bid_branches(exact, prefix, visibility="actor_information_set",
            selected_root_action="pass", target_native_auction_policy_id=stopped["action"],
            maximum_branches=13, maximum_model_bytes=100000000, maximum_conflicts=250000,
            maximum_seconds=30, maximum_queries=100, maximum_depth=64)
    evidence = failure.value.partial_receipt["evidence"]
    assert evidence["constructionComplete"] is False and evidence["queriesUsed"] == 1
    assert evidence["branches"][0]["policyBindings"] == [stopped]
    assert evidence["branches"][0]["reachability"] is None


def test_observed_native_policy_floor_requires_every_prepared_branch(root, monkeypatch):
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.scenario_contracts import canonical_sha256
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    target = Seat.parse(exact.next_actor).team
    nonplay = {"auction_action": None, "escalation_action": "decline"}
    limits = dict(maximum_branches=13, maximum_model_bytes=100000000,
        maximum_conflicts=250000, maximum_seconds=60, maximum_queries=100, maximum_depth=64)
    models, legacy_preparation = compiler._prepare_native_bid_branches(exact, prefix,
        visibility="actor_information_set", selected_root_action="sun",
        policy_team=target, nonplay_policy=nonplay, **limits)
    threshold = -max(max(map(int, variable.proto.domain)) for model in models
        for variable in model._native_round_query_context()[1])
    policy_id = "observer_safe_trick_control_v1"
    strategy, preparation = compiler._prove_observed_bid_visible_policy_floor(exact, prefix,
        selected_action="sun", native_play_policy_id=policy_id, nonplay_policy=nonplay,
        required_threshold=threshold, **limits)
    assert "nonplayPolicy" not in preparation["scope"]
    assert "nonplayPolicy" in legacy_preparation["scope"]
    assert preparation["queriesUsed"] < legacy_preparation["queriesUsed"]
    rows = strategy["branches"]
    owner = dict(rules=compiler.rules, exact=exact, source_prefix=prefix, selected_action="sun",
        required_threshold=threshold, maximum_observations=100, **limits)
    used = []
    assert authority.validate_native_observed_bid_visible_floor_v1(strategy, preparation,
        **owner, _on_queries_used=used.append) == threshold
    assert used == [preparation["queriesUsed"] + 2 * len(models)]
    legacy = {**strategy, "preparationHash": canonical_sha256(legacy_preparation),
        "branches": [{"sourceModelHash": model._model_hash(search_branching="automatic"),
        "receipt": model.find_native_round_with_constant_policy_below(target=target, policy_team=target,
            **nonplay, native_play_policy_id=policy_id, threshold=threshold, optimize_margin=False,
            maximum_seconds=30, maximum_conflicts=250000)} for model in models]}
    legacy["sha256"] = canonical_sha256({key: value for key, value in legacy.items() if key != "sha256"})
    assert authority.validate_native_observed_bid_visible_floor_v1(legacy, legacy_preparation, **owner) == threshold
    # Rehashing a counterfeit policy scope must not bypass fresh reconstruction.
    for policy in (None, {"auction_action": "pass", "escalation_action": "decline"}):
        altered_preparation = copy.deepcopy(preparation)
        altered_preparation["scope"]["nonplayPolicy"] = policy
        altered_preparation["scopeHash"] = canonical_sha256(altered_preparation["scope"])
        altered_preparation["sha256"] = canonical_sha256(
            {k: v for k, v in altered_preparation.items() if k != "sha256"})
        altered = {**strategy, "preparationHash": canonical_sha256(altered_preparation)}
        altered["sha256"] = canonical_sha256({k: v for k, v in altered.items() if k != "sha256"})
        with pytest.raises(ValueError, match="preparation"):
            authority.validate_native_observed_bid_visible_floor_v1(altered, altered_preparation, **owner)
    with pytest.raises(InformationSetCompilationStopped, match="cancelled") as cancelled:
        compiler._prove_observed_bid_visible_policy_floor(exact, prefix,
            selected_action="sun", native_play_policy_id=policy_id, nonplay_policy=nonplay,
            required_threshold=threshold, cancel_check=lambda: True, **limits)
    assert cancelled.value.partial_receipt["evidence"]["queriesUsed"] == 0
    with monkeypatch.context() as guard:
        guard.setattr(compiler, "_prepare_native_bid_branches", lambda *a, **kw: (models, preparation))
        with pytest.raises(InformationSetCompilationStopped, match="maximum_queries") as exhausted:
            compiler._prove_observed_bid_visible_policy_floor(exact, prefix,
                selected_action="sun", native_play_policy_id=policy_id, nonplay_policy=nonplay,
                required_threshold=threshold,
                **{**limits, "maximum_queries": preparation["queriesUsed"] + 1})
        assert exhausted.value.partial_receipt["evidence"]["queriesUsed"] == preparation["queriesUsed"]
        assert exhausted.value.partial_receipt["evidence"]["branches"] == []
        witness = copy.deepcopy(rows[0]["receipt"])
        witness["query"].update(status="witness_found", solverStatus="feasible",
                                candidate={"utility": threshold - 1})
        guard.setattr(type(models[0]), "find_native_round_with_constant_policy_below", lambda *a, **kw: witness)
        refuted, refuted_preparation = compiler._prove_observed_bid_visible_policy_floor(exact, prefix,
            selected_action="sun", native_play_policy_id=policy_id, nonplay_policy=nonplay,
            required_threshold=threshold, **limits)
        assert refuted["status"] == "refuted"
        assert refuted["kind"] == "native_observed_bid_policy_floor_refutation_v1"
        assert refuted["branches"][0]["receipt"]["query"]["candidate"]["utility"] == threshold - 1
        assert refuted["queriesUsed"] == preparation["queriesUsed"] + 2
        with pytest.raises(InformationSetCompilationStopped, match="maximum_model_bytes") as oversized:
            compiler._prove_observed_bid_visible_policy_floor(exact, prefix,
                selected_action="sun", native_play_policy_id=policy_id, nonplay_policy=nonplay,
                required_threshold=threshold,
                **{**limits, "maximum_model_bytes": preparation["modelBytes"]})
        assert oversized.value.partial_receipt["evidence"]["branches"][0]["receipt"]["modelBytes"]
        with pytest.raises(ValueError):
            authority.validate_native_observed_bid_visible_floor_v1(refuted, refuted_preparation, **owner)
        witness["bindings"].append(witness["bindings"][0])
        with pytest.raises(ValueError, match="reserved allowance"):
            compiler._prove_observed_bid_visible_policy_floor(exact, prefix,
                selected_action="sun", native_play_policy_id=policy_id, nonplay_policy=nonplay,
                required_threshold=threshold, **limits)
    with pytest.raises(InformationSetCompilationStopped):
        authority.validate_native_observed_bid_visible_floor_v1(strategy, preparation,
            **owner, cancel_check=lambda: True)
    # Fresh reconstruction above; reuse its models only for admission mutations.
    monkeypatch.setattr(authority, "validate_native_observed_bid_preparation_v1", lambda *a, **kw: models)
    with pytest.raises(InformationSetCompilationStopped, match="maximum_model_bytes"):
        authority.validate_native_observed_bid_visible_floor_v1(strategy, preparation,
            **{**owner, "maximum_model_bytes": preparation["modelBytes"]})
    for invalid in (True, -1, "100", None):
        altered = copy.deepcopy(strategy)
        altered["branches"][0]["receipt"]["modelBytes"]["retainedUpperBound"] = invalid
        altered["sha256"] = canonical_sha256({k: v for k, v in altered.items() if k != "sha256"})
        with pytest.raises(ValueError, match="model-byte"):
            authority.validate_native_observed_bid_visible_floor_v1(altered, preparation, **owner)
    for changes in ({"branches": []}, {"branches": rows + rows},
                    {"branches": [{**rows[0], "sourceModelHash": "0" * 64}]},
                    {"branches": [{**rows[0], "receipt": None}, *rows[1:]]},
                    {"requiredThreshold": threshold + 1}, {"nativePlayPolicyId": "unknown"},
                    {"preparationHash": "0" * 64}, {"schemaVersion": True}, {"extra": True}):
        altered = {**strategy, **changes}
        altered["sha256"] = canonical_sha256({k: v for k, v in altered.items() if k != "sha256"})
        with pytest.raises(ValueError):
            authority.validate_native_observed_bid_visible_floor_v1(altered, preparation, **owner)
    with pytest.raises(InformationSetCompilationStopped):
        authority.validate_native_observed_bid_visible_floor_v1(strategy, preparation,
            **{**owner, "maximum_queries": preparation["queriesUsed"] + 1})


@pytest.mark.parametrize("policy_id", ["observer_safe_trick_control_v1", "observer_safe_tactical_category_v1"])
def test_observed_all_pass_floor_is_native_and_refutes_positive_margin(root, policy_id):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.result_authority import validate_native_observed_bid_visible_floor_v1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    compiler, _, initial, _ = _native_suit_monopoly_bid(root)
    raw = initial.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[{"action": "pass"}] * 7, projectActions=[])
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    limits = dict(maximum_branches=13, maximum_model_bytes=100000000,
        maximum_conflicts=250000, maximum_seconds=20, maximum_queries=100, maximum_depth=64)
    args = dict(selected_action="pass", native_play_policy_id=policy_id,
        nonplay_policy={"auction_action": None, "escalation_action": "decline"}, **limits)
    floor, preparation = compiler._prove_observed_bid_visible_policy_floor(
        exact, prefix, required_threshold=0, **args)
    assert len(floor["branches"]) == 1
    query = floor["branches"][0]["receipt"]
    assert query["query"]["solverStatus"] == "infeasible"
    assert query["scope"]["allPassedCovered"] is True
    assert "nativePlayPolicyBinding" not in query["policy"]  # No card play exists.
    assert validate_native_observed_bid_visible_floor_v1(floor, preparation,
        rules=compiler.rules, exact=exact, source_prefix=prefix, selected_action="pass",
        required_threshold=0, maximum_observations=100, **limits) == 0
    refuted, _ = compiler._prove_observed_bid_visible_policy_floor(
        exact, prefix, required_threshold=1, **args)
    assert refuted["status"] == "refuted"
    witness = refuted["branches"][0]["receipt"]["query"]["candidate"]
    assert witness["utility"] == 0 and witness["allPassed"] is True
    assert len(witness["auctionActions"]) == 8
    assert witness["playObservations"] == []
    assert refuted["queriesUsed"] <= limits["maximum_queries"]


def test_all_pass_alternative_cap_distinguishes_refutation_from_exhaustion(root):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
    from baloot_platform.research.result_authority import reverify_native_bid_world_upper_v1

    compiler, _, initial, _ = _native_suit_monopoly_bid(root)
    raw = initial.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[{"action": "pass"}] * 7, projectActions=[])
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    policy = {"auction_action": "pass", "escalation_action": "decline"}
    limits = dict(maximum_branches=13, maximum_model_bytes=100000000,
        maximum_conflicts=250000, maximum_seconds=20, maximum_queries=100, maximum_depth=64)
    args = dict(visibility="complete_state", selected_root_action="pass",
        policy_team=Seat.parse(exact.next_actor).team.other, nonplay_policy=policy, **limits)
    models, receipt = compiler._prepare_native_bid_branches(exact, prefix, alternative_upper_bound=0, **args)
    assert not models and receipt["constructionComplete"]
    assert reverify_native_bid_world_upper_v1(receipt, rules=compiler.rules, exact=exact,
        source_prefix=prefix, action="pass", opponent_policy=policy, **limits) == 0
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._prepare_native_bid_branches(exact, prefix, alternative_upper_bound=-1, **args)
    assert stopped.value.reason == "alternative_upper_bound_refuted"
    partial = stopped.value.partial_receipt["evidence"]
    assert partial["constructionComplete"] is False  # Not a strategic proof-failed claim.
    query = next(row["alternativeQuery"]["query"] for row in partial["branches"] if row["retained"])
    assert query["status"] == "witness_found" and query["solverStatus"] in {"optimal", "feasible"}
    assert query["candidate"]["utility"] == 0 and query["candidate"]["allPassed"] is True
    assert partial["queriesUsed"] < limits["maximum_queries"]


@pytest.mark.parametrize("threshold,fallback,minimax_shortening", ((14, None, False), (18, None, False),
    (18, "observer_safe_partner_feed_secure_control_v1", False),
    (18, "observer_safe_partner_feed_secure_control_v1", True)))
def test_observed_first_bid_project_floor_survives_fresh_verification(root, threshold, fallback, minimax_shortening):
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.result_authority import validate_native_observed_bid_threshold_v1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofBudgetV1

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    actor = Seat.parse(exact.next_actor)
    layout = DealLayoutV1.build(compiler.rules.structural_domain, dealer=Seat.parse(prefix.dealer))
    raw = prefix.to_dict()
    raw.pop("sha256")
    deck = raw["deckOrder"]
    known = [Card.of(suit, Rank.ACE).id for suit in Suit] + [Card.of(Suit.CLUBS, Rank.SEVEN).id]
    for position, card in zip(layout.positions_for(actor, final=False), known, strict=True):
        other = deck.index(card)
        deck[position], deck[other] = deck[other], deck[position]
    other = deck.index(Card.of(Suit.HEARTS, Rank.SEVEN).id)
    deck[layout.purchase_position], deck[other] = deck[other], deck[layout.purchase_position]
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    settings = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component("challenge")["scenarioResearchV4"]
    limits = dict(maximum_branches=13, maximum_model_bytes=100000000,
        maximum_conflicts=250000, maximum_seconds=60, maximum_queries=100, maximum_depth=64,
        maximum_observations=100,
        opportunity_order=settings["qualityExecution"]["escalationOpportunityOrder"])
    progress = []
    reader_options = {}
    if minimax_shortening:
        budget = ScenarioProofBudgetV1.build("actor_information_set_exact_v1",
            settings["proofBudgets"]["profiles"]["actor_information_set_exact_v1"])
        reader_options["minimax_budget"] = budget
    result = compiler._solve_native_bid_policy_bounds(exact, prefix, actor.team,
        visibility="actor_information_set", selected_root_action="sun", required_threshold=threshold,
        maximum_iterations=10, progress_callback=progress.append,
        strategy_play_fallback_id=fallback, **reader_options, **limits)
    assert progress and all(row["kind"] == "quality_solver_progress_v1" for row in progress)
    assert all(set(row) <= {"kind", "phase", "masterCount", "queryCount", "bindingCount", "solverStatus"}
               for row in progress)
    assert not prefix.auction_actions
    assert result["status"] == "scoped_strategy_threshold_complete"
    assert result["strategy"]["lowerBound"] == threshold
    assert result["strategy"]["upperBound"] > threshold  # No exact-value or best-bid claim.
    assert result["strategicCoverage"] is False
    assert any(row["retained"] and row["partitionIndex"] != 0 for row in result["preparation"]["branches"])
    assert validate_native_observed_bid_threshold_v1(result["strategy"], result["preparation"],
        rules=compiler.rules, exact=exact, source_prefix=prefix, selected_action="sun",
        required_threshold=threshold, strategy_play_fallback_id=fallback, **reader_options, **limits) == threshold
    if minimax_shortening:
        for wrong in (None, settings["proofBudgets"]["profiles"]["actor_information_set_exact_v1"], ScenarioProofBudgetV1.build("offline_heavyweight_v1",
                settings["proofBudgets"]["profiles"]["offline_heavyweight_v1"])):
            with pytest.raises(ValueError, match="minimax"):
                validate_native_observed_bid_threshold_v1(result["strategy"], result["preparation"],
                    rules=compiler.rules, exact=exact, source_prefix=prefix, selected_action="sun",
                    required_threshold=threshold, strategy_play_fallback_id=fallback,
                    minimax_budget=wrong, **limits)
    if fallback is not None:
        assert result["strategy"]["kind"] == "native_guarded_round_strategy_threshold_v2"
        assert (result["strategy"]["scope"].get("counterexampleShortening") is True) == minimax_shortening
        for wrong in (None, "observer_safe_trick_control_v1"):
            with pytest.raises(ValueError, match="caller authority"):
                validate_native_observed_bid_threshold_v1(result["strategy"], result["preparation"],
                    rules=compiler.rules, exact=exact, source_prefix=prefix, selected_action="sun",
                    required_threshold=threshold, strategy_play_fallback_id=wrong, **reader_options, **limits)


@pytest.mark.real_clock
def test_observed_last_bid_pass_has_native_current_deal_threshold(root, monkeypatch):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.result_authority import (
        validate_native_observed_bid_preparation_v1, validate_native_strategy_threshold_v1,
        validate_native_observed_bid_threshold_v1)
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, canonical_sha256

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    normalizer = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True)
    for _ in range(7):
        raw = prefix.to_dict()
        raw.pop("sha256")
        raw["auctionActions"].append({"actor": exact.next_actor, "action": "pass"})
        prefix = ExactTracePrefixV1.build(raw)
        exact = normalizer.normalize(prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    target = Seat.parse(exact.next_actor).team
    limits = dict(maximum_branches=13, maximum_model_bytes=100000000,
        maximum_conflicts=250000, maximum_seconds=30, maximum_queries=100, maximum_depth=64)
    result = compiler._solve_native_bid_policy_bounds(exact, prefix, target,
        visibility="actor_information_set", selected_root_action="pass", required_threshold=0,
        maximum_iterations=10, maximum_observations=100, **limits)
    assert result["status"] == "scoped_strategy_threshold_complete"
    assert result["strategicCoverage"] is False
    models = validate_native_observed_bid_preparation_v1(result["preparation"], rules=compiler.rules,
        exact=exact, source_prefix=prefix, selected_action="pass", **limits)
    strategy = result["strategy"]
    model = compiler._build_native_bid_branch(exact, prefix,
        visibility="actor_information_set", branch=None)
    model.model.add(model.auction_fragment.choices[len(prefix.auction_actions)]["pass"] == 1)
    actor = Seat.parse(exact.next_actor)
    scope = {"kind": "native_symbolic_round_policy_scope_v1", "rulesHash": compiler.rules.sha256,
        "targetTeam": target.name.lower(), "coverage": "supplied_deal_branches_only",
        "utility": "awarded_point_margin", "utilityBounds": [0, 0],
        "counterexampleShortening": True,
        "requiredThreshold": 0, "objective": "strategy_threshold_satisfaction",
        "branches": [{"modelHash": model._model_hash(search_branching="automatic"),
            "scope": dict(model._native_round_query_context()[0])}],
        "fixedRoot": {"observationHash": model._auction_observation_hash(actor,
            [compiler._card(card) for card in exact.state["hands"][actor.name.lower()]],
            compiler._card(exact.state["purchaseCard"]), ["pass"] * 7),
            "legalActions": sorted(row["code"] for row in exact.legal_actions), "selectedAction": "pass"}}
    assert strategy["scope"] == scope
    assert scope["fixedRoot"]["selectedAction"] == "pass"
    assert scope["fixedRoot"]["legalActions"] == sorted(row["code"] for row in exact.legal_actions)
    assert {row["modelHash"] for row in scope["branches"]} == {
        row["sourceModelHash"] for row in result["preparation"]["branches"] if row["retained"]}
    assert validate_native_strategy_threshold_v1(strategy, expected_scope=scope) == 0
    reader_args = dict(rules=compiler.rules, exact=exact, source_prefix=prefix, selected_action="pass",
        required_threshold=0, maximum_observations=100, **limits)
    # Fresh verification has its own deadline, not the original proof allowance.
    reader_args["maximum_seconds"] = 15
    measured = []
    assert validate_native_observed_bid_threshold_v1(strategy, result["preparation"],
        **reader_args, _on_queries_used=measured.append) == 0
    assert len(measured) == 1 and result["preparation"]["queriesUsed"] < measured[0] <= reader_args["maximum_queries"]
    legacy = copy.deepcopy(strategy)
    legacy["scope"].pop("counterexampleShortening")
    legacy["scopeHash"] = canonical_sha256(legacy["scope"])
    legacy["sha256"] = canonical_sha256({key: value for key, value in legacy.items() if key != "sha256"})
    assert validate_native_observed_bid_threshold_v1(legacy, result["preparation"], **reader_args) == 0
    with pytest.raises(ValueError):
        validate_native_observed_bid_threshold_v1({**strategy, "scope": None}, result["preparation"], **reader_args)
    import baloot_platform.research.result_authority as authority
    from baloot_platform.research.ortools_information_set_compiler import (
        OrToolsActorInformationSetCompilerV1, InformationSetCompilationStopped)
    with pytest.raises(InformationSetCompilationStopped):
        validate_native_observed_bid_threshold_v1(strategy, result["preparation"],
            **reader_args, cancel_check=lambda: True)
    # The real reconstruction above owns this model; substitute it only to
    # exercise the reader's post-reconstruction rejection paths cheaply.
    monkeypatch.setattr(authority, "validate_native_observed_bid_preparation_v1", lambda *a, **kw: models)
    altered = copy.deepcopy(strategy)
    altered["scope"]["fixedRoot"]["observationHash"] = "0" * 64
    with pytest.raises(ValueError, match="ownership or coverage"):
        validate_native_observed_bid_threshold_v1(altered, result["preparation"], **reader_args)
    for status in ("policy_extension_required", "policy_counterexample_in_branch", "incomplete"):
        monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_refine_symbolic_round_policy",
            lambda *a, **kw: {"status": status, "queriesUsed": 1})
        with pytest.raises(ValueError, match="did not survive"):
            validate_native_observed_bid_threshold_v1(strategy, result["preparation"], **reader_args)
    # This is a zero-award redeal boundary, not a played-round or optimal-bid proof.
    assert result["preparation"]["scope"]["nextDealAnalyzed"] is False


@pytest.mark.parametrize("with_opponent_policy,fixed_world", ((False, False), (True, False), (False, True)))
def test_observed_bid_preparation_reader_reconstructs_hidden_universe(root, monkeypatch, with_opponent_policy, fixed_world):
    from baloot_platform.research.result_authority import validate_native_observed_bid_preparation_v1
    from baloot_platform.research.scenario_contracts import canonical_sha256
    from baloot_platform.research.ortools_information_set_compiler import (
        OrToolsActorInformationSetCompilerV1, InformationSetCompilationStopped)

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    limits = dict(maximum_branches=13, maximum_model_bytes=100000000,
        maximum_conflicts=250000, maximum_seconds=60, maximum_queries=100, maximum_depth=64)
    selected = next(row["code"] for row in exact.legal_actions if row["code"].startswith("hokum:"))
    policy = {"auction_action": "pass", "escalation_action": "decline"} if with_opponent_policy else None
    models, receipt = compiler._prepare_native_bid_branches(exact, prefix,
        visibility="actor_information_set", selected_root_action=selected,
        policy_team=Seat.parse(exact.next_actor).team.other if policy else None,
        nonplay_policy=policy, fixed_observed_world=fixed_world,
        **({"compact_purchase_invariant_ownership": True} if fixed_world else {}), **limits)
    assert len(models) >= (1 if policy else 2)
    assert all(model._native_auction_policy_root == (
        len(prefix.auction_actions), selected, Seat.parse(exact.next_actor).team) for model in models)
    for model in models:
        deck_hints = dict(zip(model.model.proto.solution_hint.vars,
            model.model.proto.solution_hint.values, strict=True))
        assert deck_hints == ({variable.index: card for variable, card in
            zip(model.card_at_position, prefix.deck_order, strict=True)} if fixed_world else {})
    record = next(row for row in receipt["branches"] if row["retained"])
    compiler._validate_native_bid_information_bindings(models[0], record,
        proto_hash=record["cpModelProtoHash"])
    changed_model = copy.copy(models[0])
    changed_model.model = models[0].model.clone()
    changed_model.model.add(False)
    with pytest.raises(ValueError, match="binding identity mismatch"):
        compiler._validate_native_bid_information_bindings(changed_model, record)
    owner = dict(rules=compiler.rules, exact=exact, source_prefix=prefix,
        selected_action=selected, opponent_policy=policy, fixed_observed_world=fixed_world, **limits)
    if fixed_world:
        assert receipt["scope"]["fixedObservedWorld"] == {
            "coverage": "single_compatible_deal", "stateHash": exact.state_hash,
            "prefixHash": exact.prefix_hash, "exactStateId": exact.exact_state_id}
        for model in models:
            assert getattr(model, "_complete_information_deck", None) is None
        probe = compiler._native_bid_root_model(exact, prefix,
            visibility="actor_information_set", fixed_observed_world=True)
        probe.model.add(probe.card_at_position[0] != prefix.deck_order[0])
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = 5
        assert solver.solve(probe.model) == cp_model.INFEASIBLE
        for invalid in (1, "true", None):
            with pytest.raises(ValueError, match="fixed observed world"):
                compiler._native_bid_root_model(exact, prefix,
                    visibility="actor_information_set", fixed_observed_world=invalid)
        with pytest.raises(ValueError, match="fixed observed world"):
            compiler._native_bid_root_model(exact, prefix,
                visibility="complete_state", fixed_observed_world=True)
    # Real producer and fresh reader reconstruction; no patched acceptance path.
    validate_native_observed_bid_preparation_v1(receipt, **owner)
    if fixed_world:
        with monkeypatch.context() as guard:
            def must_not_reconstruct(*args, **kwargs):
                pytest.fail("scope mismatch must fail before reconstruction")
            guard.setattr(OrToolsActorInformationSetCompilerV1, "_prepare_native_bid_branches",
                must_not_reconstruct)
            with pytest.raises(ValueError, match="world scope mismatch"):
                validate_native_observed_bid_preparation_v1(receipt,
                    **{**owner, "fixed_observed_world": False})
    with pytest.raises(InformationSetCompilationStopped):
        validate_native_observed_bid_preparation_v1(receipt, **owner, cancel_check=lambda: True)

    # Reuse that reconstructed universe only for cheap resealed mutation checks.
    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_prepare_native_bid_branches",
        lambda *args, **kwargs: (models, receipt))
    for seconds in (None, True, 0, -1, "30"):
        invalid = copy.deepcopy(receipt)
        invalid["limits"]["seconds"] = seconds
        invalid["sha256"] = canonical_sha256({key: value for key, value in invalid.items() if key != "sha256"})
        with pytest.raises(ValueError, match="time receipt mismatch"):
            validate_native_observed_bid_preparation_v1(invalid, **owner)
    invalid = copy.deepcopy(receipt)
    depth = next(row["depthQuery"] for row in invalid["branches"] if isinstance(row.get("depthQuery"), dict))
    depth["maximumSeconds"] = invalid["limits"]["seconds"] + 1
    depth["sha256"] = canonical_sha256({key: value for key, value in depth.items() if key != "sha256"})
    invalid["sha256"] = canonical_sha256({key: value for key, value in invalid.items() if key != "sha256"})
    with pytest.raises(ValueError, match="depth receipt mismatch"):
        validate_native_observed_bid_preparation_v1(invalid, **owner)
    mutations = []
    if fixed_world:
        assert receipt["scope"]["ownershipCompaction"] == "purchase_recipient_invariant_v1"
        missing_world = copy.deepcopy(receipt)
        missing_world["scope"].pop("fixedObservedWorld")
        missing_world["scopeHash"] = canonical_sha256(missing_world["scope"])
        mutations.append(missing_world)
        missing_compaction = copy.deepcopy(receipt)
        missing_compaction["scope"].pop("ownershipCompaction")
        missing_compaction["scopeHash"] = canonical_sha256(missing_compaction["scope"])
        mutations.append(missing_compaction)
    wrong_root = copy.deepcopy(receipt)
    wrong_root["scope"]["rootHash"] = "0" * 64
    wrong_root["scopeHash"] = canonical_sha256(wrong_root["scope"])
    mutations.append(wrong_root)
    missing = copy.deepcopy(receipt)
    missing["branches"].pop()
    mutations.append(missing)
    reordered = copy.deepcopy(receipt)
    reordered["branches"].reverse()
    mutations.append(reordered)
    binding = copy.deepcopy(receipt)
    retained = next(row for row in binding["branches"] if row["retained"])
    retained["informationBindings"]["purchaseRecipient"]["index"] += 1
    retained["informationBindings"]["sha256"] = canonical_sha256({
        key: value for key, value in retained["informationBindings"].items() if key != "sha256"})
    mutations.append(binding)
    if policy:
        substituted = copy.deepcopy(receipt)
        substituted["scope"]["nonplayPolicy"]["team"] = Seat.parse(exact.next_actor).team.name.lower()
        substituted["scopeHash"] = canonical_sha256(substituted["scope"])
        mutations.append(substituted)
    for changed in mutations:
        changed["sha256"] = canonical_sha256({key: value for key, value in changed.items() if key != "sha256"})
        with pytest.raises(ValueError, match="reconstructed universe|world scope mismatch"):
            validate_native_observed_bid_preparation_v1(changed, **owner)


@pytest.mark.parametrize("shorten_counterexamples,readback_witness_hints,guarded", (
    (False, False, False), (True, False, False), (False, True, False), (True, True, False),
    pytest.param(False, False, True, id="guarded"),
    pytest.param(True, False, True, id="guarded_minimax")))
def test_observed_opponent_upper_freshly_verifies_single_world_only(root, monkeypatch, shorten_counterexamples, readback_witness_hints, guarded):
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofBudgetV1, canonical_sha256

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    layout = DealLayoutV1.build(compiler.rules.structural_domain, dealer=Seat.parse(prefix.dealer))
    normalizer = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True)
    observations = []
    # One different hidden world is compatible; one changed actor hand is not.
    for seat in (Seat.EAST, Seat.parse(exact.next_actor)):
        raw = prefix.to_dict()
        raw.pop("sha256")
        first = layout.positions_for(Seat.WEST, final=False)[0]
        second = layout.positions_for(seat, final=False)[0]
        raw["deckOrder"][first], raw["deckOrder"][second] = raw["deckOrder"][second], raw["deckOrder"][first]
        observations.append(normalizer.normalize(ExactTracePrefixV1.build(raw),
            decision_point={"kind": "bid"}, maximum_seconds=15))
    observer, incompatible = observations
    assert observer.state_hash != exact.state_hash
    limits = dict(maximum_branches=13, maximum_model_bytes=100000000,
        maximum_conflicts=250000, maximum_seconds=60, maximum_queries=100, maximum_depth=64)
    models, preparation = compiler._prepare_native_bid_branches(exact, prefix,
        visibility="actor_information_set", fixed_observed_world=True,
        selected_root_action="sun", **limits)
    maximum = max(max(map(int, variable.proto.domain)) for model in models
        for variable in model._native_round_query_context()[1])
    guarded_args, reader_args = {}, {}
    if guarded:
        policy_id = "observer_safe_partner_feed_closed_trick_v1"
        guarded_args["_play_fallback_id"] = policy_id
        reader_args["strategy_play_fallback_id"] = policy_id
        if shorten_counterexamples:
            import yaml
            settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
            budget = ScenarioProofBudgetV1.build("actor_information_set_exact_v1",
                settings["proofBudgets"]["profiles"]["actor_information_set_exact_v1"])
            guarded_args["minimax_budget"] = reader_args["minimax_budget"] = budget
    strategy = compiler._solve_symbolic_round_policy_bounds(models, Seat.parse(exact.next_actor).team.other,
        required_threshold=-maximum, maximum_iterations=2, maximum_observations=100,
        shorten_counterexamples=shorten_counterexamples,
        readback_witness_hints=readback_witness_hints,
        **guarded_args,
        **{key: value for key, value in limits.items() if key not in {"maximum_model_bytes", "maximum_depth"}})
    assert strategy["status"] == "scoped_strategy_threshold_complete"
    owner = dict(rules=compiler.rules, observer_exact=observer, exact=exact, source_prefix=prefix,
        action="sun", upper_bound=maximum, maximum_observations=100, **limits, **reader_args)
    used = []
    assert authority.validate_native_observed_bid_opponent_upper_v1(strategy, preparation,
        **owner, _on_queries_used=used.append) == maximum
    assert preparation["queriesUsed"] < used[0] <= limits["maximum_queries"]
    # This deliberately loose native cap tests the fresh reader, not optimality.
    assert preparation["strategicCoverage"] is False and strategy["strategicCoverage"] is False
    with monkeypatch.context() as guard:
        def must_not_reconstruct(*args, **kwargs):
            pytest.fail("incompatible observer must fail before reconstruction")
        guard.setattr(authority, "validate_native_observed_bid_preparation_v1", must_not_reconstruct)
        with pytest.raises(ValueError, match="differs from observer root"):
            authority.validate_native_observed_bid_opponent_upper_v1(strategy, preparation,
                **{**owner, "observer_exact": incompatible})
        if guarded:
            for wrong in ({"strategy_play_fallback_id": None},
                          {"strategy_play_fallback_id": "observer_safe_trick_control_v1"},
                          *([{"minimax_budget": None}, {"minimax_budget": {}}] if shorten_counterexamples else [])):
                with pytest.raises(ValueError, match="caller authority|typed caller budget"):
                    authority.validate_native_observed_bid_opponent_upper_v1(strategy, preparation,
                        **{**owner, **wrong})
    with pytest.raises(InformationSetCompilationStopped):
        authority.validate_native_observed_bid_opponent_upper_v1(strategy, preparation,
            **owner, cancel_check=lambda: True)
    # The positive path above rebuilt native models; reuse only that result for
    # scoped tampering checks without repeating expensive construction.
    monkeypatch.setattr(authority, "validate_native_observed_bid_preparation_v1", lambda *a, **kw: models)
    for change in ({"targetTeam": Seat.parse(exact.next_actor).team.name.lower()},
                   {"requiredThreshold": 1 - maximum}, {"branches": []},
                   {"counterexampleShortening": "true"}, {"counterexampleShortening": 1},
                   {"readbackWitnessHints": True}, {"readbackWitnessHints": "unknown"}):
        corrupted = copy.deepcopy(strategy)
        corrupted["scope"].update(change)
        corrupted["scopeHash"] = canonical_sha256(corrupted["scope"])
        corrupted["sha256"] = canonical_sha256({k: v for k, v in corrupted.items() if k != "sha256"})
        with pytest.raises(ValueError):
            authority.validate_native_observed_bid_opponent_upper_v1(corrupted, preparation, **owner)
    monkeypatch.setattr(type(compiler), "_refine_symbolic_round_policy",
        lambda *a, **kw: {"status": "policy_counterexample_in_branch", "queriesUsed": 1})
    with pytest.raises(ValueError, match="did not survive"):
        authority.validate_native_observed_bid_opponent_upper_v1(strategy, preparation, **owner)


@pytest.mark.parametrize("contract", (Contract.SUN, Contract.HOKUM, Contract.ASHKAL))
def test_fixed_world_initial_ownership_compaction_preserves_native_upper(root, monkeypatch, contract):
    from baloot_platform.research import ortools_project_fragment
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    if contract == Contract.ASHKAL:
        from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
        from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
        for _ in range(2):
            raw = prefix.to_dict()
            raw.pop("sha256")
            raw["auctionActions"].append({"actor": exact.next_actor, "action": "pass"})
            prefix = ExactTracePrefixV1.build(raw)
            exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(
                prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    actor = Seat.parse(exact.next_actor)
    purchase = compiler._card(exact.state["purchaseCard"])
    trump = purchase.suit if contract == Contract.HOKUM else None
    action = "hokum:" + trump.name.lower() if contract == Contract.HOKUM else contract.value
    assert action in {row["code"] for row in exact.legal_actions}
    captured = []
    attach = ortools_project_fragment.attach_project_lifecycle_fragment_v1
    def record(*args, **kwargs):
        captured.append(kwargs.get("fixed_card_owners"))
        return attach(*args, **kwargs)
    monkeypatch.setattr(ortools_project_fragment, "attach_project_lifecycle_fragment_v1", record)
    models, values = [], []
    branch = next(row for row in compiler._native_bid_branch_specs(Seat.parse(prefix.dealer))
        if row is not None and row[0] == contract and row[1] == trump and not row[3])
    for compact in (False, True, "purchase_invariant", "observed_fixed"):
        model = compiler._build_native_bid_branch(exact, prefix,
            visibility="actor_information_set" if compact == "observed_fixed" else "complete_state",
            branch=branch, **({"compact_purchase_invariant_ownership": True,
                              "fixed_observed_world": True} if compact == "observed_fixed" else
                             {"compact_purchase_invariant_ownership": True} if compact == "purchase_invariant"
                             else {"compact_known_initial_ownership": compact}))
        model.model.add(model.auction_fragment.choices[len(prefix.auction_actions)][action] == 1)
        if contract == Contract.ASHKAL:
            invalid = model.model.clone()
            invalid.add(model.auction_fragment.states[-1]["recipient"] != int(actor.partner))
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.max_time_in_seconds = 15
            assert solver.solve(invalid) == cp_model.INFEASIBLE
        result = model.find_native_round_margin_below(target=actor.team.other, threshold=100000,
            maximum_seconds=30, maximum_conflicts=250000)
        assert result["solverStatus"] == "optimal" and result["status"] == "witness_found"
        values.append(result["candidate"]["utility"])
        models.append(model)
    assert len(set(values)) == 1
    assert captured[0] is None
    expected = {compiler._card(card).id: seat for seat in Seat for card in exact.state["hands"][seat.name.lower()]}
    assert len(expected) == 20
    assert captured[1] == tuple(expected.get(card) for card in range(32))
    assert captured[1][purchase.id] is None
    assert sum(owner is None for owner in captured[1]) == 12
    invariant = {}
    for position, card in enumerate(prefix.deck_order):
        owners = {row[position] for row in models[2]._owners_by_recipient.values()}
        if len(owners) == 1:
            invariant[card] = Seat(next(iter(owners)))
    assert captured[2] == tuple(invariant.get(card) for card in range(32))
    assert captured[3] == captured[2]
    assert getattr(models[3], "_complete_information_deck", None) is None
    assert sum(owner is None for owner in captured[2]) == 4
    assert len(models[2].model.proto.variables) < len(models[1].model.proto.variables)
    assert models[0]._complete_information_deck == models[1]._complete_information_deck == tuple(prefix.deck_order)
    assert len(models[1].model.proto.variables) < len(models[0].model.proto.variables)
    assert len(str(models[1].model.proto)) < len(str(models[0].model.proto))
    for invalid in (1, "true", None):
        with pytest.raises(ValueError, match="compaction"):
            compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
                branch=branch, compact_known_initial_ownership=invalid)
    with pytest.raises(ValueError, match="compaction"):
        compiler._build_native_bid_branch(exact, prefix, visibility="actor_information_set",
            branch=branch, compact_known_initial_ownership=True)
    for changes in ({"visibility": "actor_information_set"}, {"compact_known_initial_ownership": True},
                    {"compact_purchase_invariant_ownership": 1}):
        with pytest.raises(ValueError, match="compaction"):
            compiler._build_native_bid_branch(exact, prefix, branch=branch,
                **{"visibility": "complete_state", "compact_purchase_invariant_ownership": True, **changes})


def test_retained_opening_bid_owner_projection_parity(root, record_property):
    """Measure native owner projection on LAF-1408636F3A, not a synthetic deal."""
    import time
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    rules = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").rules
    prefix = ExactTracePrefixV1.build(dict(schemaVersion=1, dealer="east",
        deckOrder=[7,24,25,0,4,6,1,3,5,10,11,12,27,31,13,30,9,29,16,18,2,26,28,19,21,23,14,15,17,8,20,22],
        auctionActions=[], escalationActions=[], projectActions=[], projectDeclarations=[], playActions=[]))
    assert prefix.sha256 == "5b864fed26d605b937dda3ca397e67b088517756ffac0f4a984aa5f692672deb"
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    branch = next(row for row in compiler._native_bid_branch_specs(Seat.EAST)
                  if row is not None and row[0] == Contract.SUN and not row[3])
    context = dict(contract=branch[0], trump=branch[1], first_leader=branch[2],
                   locked=branch[3], match_score=prefix.match_score)
    started = time.monotonic()
    probe = compiler._native_bid_root_model(exact, prefix, visibility="complete_state")
    probe.model.add(probe.auction_fragment.choices[0]["sun"] == 1)
    probe._add_native_round_escalation(**context)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 2
    assert solver.solve(probe.model) == cp_model.OPTIMAL
    recipient = solver.value(probe.purchase_recipient)
    owners = tuple(Seat(solver.value(owner)) for owner in probe.owner_of_card)
    different = probe.model.clone()
    different.add(probe.purchase_recipient != recipient)
    assert solver.solve(different) == cp_model.INFEASIBLE
    projection_seconds = time.monotonic() - started
    record_property("nativeOwnerProjectionSeconds", projection_seconds)

    # A reachable auction witness alone cannot justify fixing the owners.
    variable = compiler._native_bid_root_model(exact, prefix, visibility="complete_state")
    variable.model.add(variable.auction_fragment.choices[0]["pass"] == 1)
    variable._add_native_round_escalation(**context)
    assert solver.solve(variable.model) == cp_model.OPTIMAL
    variable.model.add(variable.purchase_recipient != solver.value(variable.purchase_recipient))
    assert solver.solve(variable.model) == cp_model.OPTIMAL

    # The same native deal table already fixes 28 positions for every recipient.
    invariant = {}
    for position, card in enumerate(prefix.deck_order):
        possible = {row[position] for row in probe._owners_by_recipient.values()}
        if len(possible) == 1:
            invariant[card] = Seat(next(iter(possible)))
    assert len(invariant) == 28
    unrestricted = compiler._native_bid_root_model(exact, prefix, visibility="complete_state")
    unrestricted.model.add_forbidden_assignments(
        [unrestricted.owner_of_card[card] for card in sorted(invariant)],
        [tuple(int(invariant[card]) for card in sorted(invariant))])
    assert solver.solve(unrestricted.model) == cp_model.INFEASIBLE

    measurements = []
    for projected in (False, "purchase_invariant", True):
        started = time.monotonic()
        if projected == "purchase_invariant":
            model = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
                branch=branch, compact_purchase_invariant_ownership=True)
            model.model.add(model.auction_fragment.choices[0]["sun"] == 1)
        elif projected:
            model = compiler._native_bid_root_model(exact, prefix, visibility="complete_state")
            model.model.add(model.auction_fragment.choices[0]["sun"] == 1)
            model.add_native_round_continuation(**context, fixed_card_owners=owners)
        else:
            model = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
                branch=branch, compact_known_initial_ownership=True)
            model.model.add(model.auction_fragment.choices[0]["sun"] == 1)
        outcome = model.find_native_round_margin_below(target=Seat.NORTH.team.other,
            threshold=100000, maximum_seconds=12, maximum_conflicts=250000)
        measurement = dict(projected=projected, seconds=time.monotonic()-started,
            variables=len(model.model.proto.variables), constraints=len(model.model.proto.constraints),
            modelBytes=len(str(model.model.proto).encode("utf-8")),
            solverStatus=outcome["solverStatus"], status=outcome["status"],
            value=(outcome.get("candidate") or {}).get("utility"))
        measurements.append(measurement)
        record_property(str(projected) if projected else "baseline", str(measurement))
    assert all(row["solverStatus"] == "optimal" and row["status"] == "witness_found"
               for row in measurements), measurements
    assert len({row["value"] for row in measurements}) == 1, measurements
    assert measurements[1]["modelBytes"] < measurements[0]["modelBytes"], measurements
    assert measurements[1]["variables"] < measurements[0]["variables"], measurements
    record_property("nativeOwnerProjectionFaster",
        projection_seconds + measurements[2]["seconds"] < measurements[0]["seconds"])
    record_property("purchaseInvariantCompactionFaster", measurements[1]["seconds"] < measurements[0]["seconds"])


def test_observed_bid_threshold_probe_stops_unattainable_floor_without_claiming_bad_play(root, monkeypatch):
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import (
        OrToolsActorInformationSetCompilerV1, InformationSetCompilationStopped)
    from baloot_platform.research.ortools_research_executor import (
        _observed_bid_quality_artifact_v1, _quality_budget_snapshots)
    from baloot_platform.research.scenario_contracts import (
        ExactTracePrefixV1, ScenarioProofRequestV1, ScenarioQualityExecutionBudgetV1)

    bundle = build_bundle(root, bundle_path="configs/bundle_ortools_research_native_v3_intelligent.yaml")
    settings = bundle.component("challenge")["scenarioResearchV4"]
    prefix = ExactTracePrefixV1.build(dict(schemaVersion=1, deckOrder=list(range(32)), dealer="east",
        auctionActions=[], escalationActions=[], projectDeclarations=[], playActions=[]))
    exact = OrToolsExactStateNormalizerV1(bundle.rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    before = exact.to_dict()
    tier = "offline_heavyweight_v1"
    budget = _quality_budget_snapshots({tier}, settings["proofBudgets"]["profiles"])[0][tier]
    execution = ScenarioQualityExecutionBudgetV1.build({**settings["qualityExecution"],
        "bidThresholdRefutation": dict(schemaVersion=1, maximumSeconds=60, maximumQueries=256,
            worldRankingId="opponent_auction_value_v1", opponentAuctionAction="sun")})
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="actor_information_set",
        strategyClass="pure_deterministic", claim="outcome_forced", targetTeam="north_south",
        utility="awarded_point_margin", successThreshold=1, budgetProfile=tier, selectedAction="pass"))
    solve = OrToolsActorInformationSetCompilerV1._solve_native_bid_policy_bounds

    def only_refutation(self, *args, **kwargs):
        assert kwargs.get("threshold_refutation") is not None, "do not synthesize an impossible floor"
        return solve(self, *args, **kwargs)

    def no_incumbent(*args, **kwargs):
        pytest.fail("a freshly refuted floor must stop before incumbent construction")

    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_solve_native_bid_policy_bounds", only_refutation)
    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_prove_observed_bid_visible_policy_floor", no_incumbent)
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        _observed_bid_quality_artifact_v1(bundle.rules, exact, prefix, request, budget=budget,
            execution_budget=execution, maximum_depth=64, maximum_seconds=180,
            maximum_conflicts=settings["searchBudgets"]["profiles"][tier]["maximumConflicts"])
    attempt = stopped.value.partial_receipt["evidence"]["nativeObservedBidDominance"]
    assert attempt["status"] == "incomplete" and attempt["cause"] == "sufficient_threshold_refuted"
    selected = attempt["selectedProof"]
    assert selected["status"] == "scoped_strategy_threshold_refuted"
    assert selected["strategy"]["upperBound"] == -42
    assert selected["strategy"]["claimValue"] is True and selected["strategicCoverage"] is False
    probe = attempt["thresholdRefutationProbe"]
    assert probe["status"] == "threshold_refuted" and probe["queriesCharged"] <= 256
    assert attempt["queriesUsed"] <= budget.native_observed.maximum_queries
    assert probe["upperProof"]["observerRootHash"] == selected["strategy"]["observerRootHash"]
    assert "incumbentAttempt" not in attempt and attempt["certificate"] is None
    assert exact.to_dict() == before  # No deal/action repair; no optimal-action artifact.


@pytest.mark.parametrize("policy_id", ["observer_safe_trick_control_v1", "observer_safe_tactical_category_v1"])
@pytest.mark.parametrize("compact", (False, True))
def test_native_world_upper_binds_opponent_visible_policy(root, policy_id, monkeypatch, compact):
    import hashlib
    import sys
    from baloot_platform.research.result_authority import (
        validate_native_bid_world_upper_v1, validate_native_bid_threshold_refutation_v1,
        reverify_native_bid_world_upper_v1)
    from baloot_platform.research.scenario_contracts import canonical_sha256
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    opponent = Seat.parse(exact.next_actor).team.other
    args = dict(visibility="complete_state", selected_root_action="sun", policy_team=opponent,
        nonplay_policy={"auction_action": "pass", "escalation_action": "decline"},
        maximum_branches=13, maximum_model_bytes=100000000, maximum_conflicts=250000,
        maximum_seconds=30, maximum_queries=100, maximum_depth=64, optimize_alternative_upper=True,
        opponent_play_policy_id=policy_id, compact_purchase_invariant_ownership=compact)
    model_hash = OrToolsScenarioModelV1._model_hash
    reused_digests = []
    def checked_hash(model, **kwargs):
        if sys._getframe(1).f_code.co_name == "_prepare_native_bid_branches":
            digest = hashlib.sha256(str(model.model.proto).encode("utf-8")).hexdigest()
            assert kwargs.get("proto_hash") == digest
            reused_digests.append(digest)
        return model_hash(model, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(OrToolsScenarioModelV1, "_model_hash", checked_hash)
        _, receipt = compiler._prepare_native_bid_branches(exact, prefix, **args)
    assert reused_digests
    owner = dict(rules=compiler.rules, exact=exact, source_prefix=prefix, action="sun",
        opponent_policy=args["nonplay_policy"], maximum_model_bytes=100000000,
        opponent_play_policy_id=policy_id, maximum_depth=64)
    assert validate_native_bid_world_upper_v1(receipt, **owner) == 92
    assert receipt["scope"].get("ownershipCompaction") == (
        "purchase_recipient_invariant_v1" if compact else None)
    if compact:
        for marker in (None, "unknown", True):
            changed = copy.deepcopy(receipt)
            if marker is None:
                changed["scope"].pop("ownershipCompaction")
            else:
                changed["scope"]["ownershipCompaction"] = marker
            changed["scopeHash"] = canonical_sha256(changed["scope"])
            changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
            with pytest.raises(ValueError):
                validate_native_bid_world_upper_v1(changed, **owner)
    measured = []
    fresh_limits = dict(maximum_branches=13, maximum_conflicts=250000,
        maximum_seconds=30, maximum_queries=100)
    assert reverify_native_bid_world_upper_v1(receipt, **owner, **fresh_limits,
        _on_queries_used=measured.append) == 92
    assert len(measured) == 1 and measured[0] > 0
    if compact and policy_id == "observer_safe_trick_control_v1":
        _, inconclusive = compiler._prepare_native_bid_branches(
            exact, prefix, _refutation_threshold=92, **args)
        assert inconclusive["constructionComplete"] is False
        assert inconclusive["upperBound"] is None
        assert len(inconclusive["branches"]) < len(receipt["branches"])
        assert inconclusive["queriesUsed"] < receipt["queriesUsed"]
        assert any(row.get("alternativeUpperBound") == 92 for row in inconclusive["branches"])
        with pytest.raises(ValueError):
            validate_native_bid_world_upper_v1(inconclusive, **owner)
        for invalid in (True, 92.0, "92"):
            with pytest.raises(ValueError, match="refutation threshold"):
                compiler._prepare_native_bid_branches(exact, prefix, _refutation_threshold=invalid, **args)
        # Exercise probe coordination with the real partial native receipt.
        # An inconclusive refutation must neither invoke certification nor
        # charge unexecuted cells as though its whole allowance was exhausted.
        with monkeypatch.context() as patch:
            patch.setattr(compiler, "_find_native_bid_compatible_world",
                lambda *a, **kw: (prefix, {"queriesUsed": 1}))
            patch.setattr(compiler, "_prepare_native_bid_branches",
                lambda *a, **kw: ((), inconclusive))
            patch.setattr(compiler, "_solve_native_bid_policy_bounds",
                lambda *a, **kw: pytest.fail("partial coverage cannot refute the threshold"))
            proof, probe = compiler._probe_observed_bid_threshold_refutation(exact, prefix,
                selected_action="sun", threshold=92,
                configuration={"maximumSeconds": 30, "maximumQueries": 100,
                    "worldRankingId": "opponent_auction_value_v1", "opponentAuctionAction": "pass"},
                opponent_play_policy_id=policy_id, opponent_escalation_action="decline",
                maximum_evidence_bytes=100000000, maximum_branches=13,
                maximum_model_bytes=100000000, maximum_conflicts=250000,
                maximum_depth=64, maximum_iterations=100, maximum_observations=1000,
                opportunity_order=None, cancel_check=None)
        assert proof is None and probe["status"] == "threshold_not_refuted"
        assert probe["queryCountComplete"] and probe["queriesCharged"] == 3 + inconclusive["queriesUsed"]
        threshold_args = {k: v for k, v in args.items() if k != "optimize_alternative_upper"}
        _, cap = compiler._prepare_native_bid_branches(exact, prefix, alternative_upper_bound=92, **threshold_args)
        assert cap["kind"] == "native_bid_streamed_alternative_preparation_v1"
        assert cap["scope"]["opponentPlayPolicy"] == receipt["scope"]["opponentPlayPolicy"]
        assert all(row["alternativeQuery"]["query"]["solverStatus"] == "infeasible"
                   for row in cap["branches"] if row["retained"])
        assert reverify_native_bid_world_upper_v1(cap, **owner, **fresh_limits) == 92
        with pytest.raises(ValueError, match="policy substitution"):
            reverify_native_bid_world_upper_v1(cap, **{**owner, "opponent_play_policy_id": None}, **fresh_limits)
    for substituted in (None, "unknown", "observer_safe_tactical_category_v1"
                        if policy_id == "observer_safe_trick_control_v1" else "observer_safe_trick_control_v1"):
        with pytest.raises(ValueError):
            reverify_native_bid_world_upper_v1(receipt,
                **{**owner, "opponent_play_policy_id": substituted}, **fresh_limits)
    assert validate_native_bid_threshold_refutation_v1(receipt, **owner,
        observer_exact=exact, required_threshold=93) == 92
    for threshold in (92, 91, True, 93.0):
        with pytest.raises(ValueError):
            validate_native_bid_threshold_refutation_v1(receipt, **owner,
                observer_exact=exact, required_threshold=threshold)
    bound = {"receipt": receipt, "exact": exact, "prefix": prefix,
        "opponentPolicy": args["nonplay_policy"], "opponentPlayPolicyId": policy_id}
    limits = dict(maximum_iterations=1, maximum_queries=1, maximum_observations=100,
        maximum_branches=13, maximum_model_bytes=100000000, maximum_conflicts=250000,
        maximum_seconds=30, maximum_depth=64, visibility="actor_information_set", selected_root_action="sun",
        required_threshold=93, threshold_refutation=bound)
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    import baloot_platform.research.ortools_information_set_compiler as module
    with monkeypatch.context() as patch:
        def no_search(*args, **kwargs):
            pytest.fail("a certified refutation must bypass construction and strategy search")
        patch.setattr(compiler, "_prepare_native_bid_branches", no_search)
        refuted = compiler._solve_native_bid_policy_bounds(exact, prefix, opponent.other, **limits)
        assert refuted["status"] == "scoped_strategy_threshold_refuted"
        assert refuted["queriesUsed"] == refuted["strategy"]["queriesUsed"] == 1
        assert refuted["preparation"] is None and refuted["strategicCoverage"] is False
        assert refuted["strategy"]["upperBound"] == 92 and refuted["strategy"]["lowerBound"] is None
        assert refuted["strategy"]["claimValue"] is True
        assert refuted["strategy"]["reusedProofQueries"] == receipt["queriesUsed"]
        assert refuted["strategy"]["worldUpperProof"] == receipt
        for changes in ({"maximum_depth": True}, {"maximum_depth": 1},
                        {"threshold_refutation": {**bound, "extra": 1}}, {"visibility": "complete_state"}):
            with pytest.raises(ValueError):
                compiler._solve_native_bid_policy_bounds(exact, prefix, opponent.other, **{**limits, **changes})
        with pytest.raises(InformationSetCompilationStopped) as capped:
            compiler._solve_native_bid_policy_bounds(exact, prefix, opponent.other,
                **{**limits, "maximum_branches": 1})
        assert capped.value.reason == "maximum_branches"
        patch.setattr(compiler, "native_cp_sat", False)
        with pytest.raises(ValueError):
            compiler._solve_native_bid_policy_bounds(exact, prefix, opponent.other, **limits)
        patch.setattr(compiler, "native_cp_sat", True)
        with pytest.raises(InformationSetCompilationStopped) as cancelled:
            compiler._solve_native_bid_policy_bounds(exact, prefix, opponent.other, **limits, cancel_check=lambda: True)
        assert cancelled.value.reason == "cancelled"
        assert cancelled.value.partial_receipt["evidence"]["queriesUsed"] == 0
        patch.setattr(module, "_solve_cp_sat_with_cancel_v1", lambda *args: cp_model.UNKNOWN)
        with pytest.raises(InformationSetCompilationStopped) as unknown:
            compiler._solve_native_bid_policy_bounds(exact, prefix, opponent.other, **limits)
        retained = unknown.value.partial_receipt["evidence"]
        assert retained["status"] == "incomplete" and retained["queriesUsed"] == 1
        assert retained["strategy"]["worldUpperProof"] == receipt
        assert retained["strategy"]["claimValue"] is None
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["deckOrder"][0], raw["deckOrder"][3] = raw["deckOrder"][3], raw["deckOrder"][0]
    changed_observer = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(raw), decision_point={"kind": "bid"})
    with pytest.raises(ValueError, match="observer mismatch"):
        validate_native_bid_threshold_refutation_v1(receipt, **owner,
            observer_exact=changed_observer, required_threshold=93)
    assert receipt["scope"]["opponentPlayPolicy"]["team"] == opponent.name.lower()
    retained = next(row for row in receipt["branches"] if row["retained"])
    changed = copy.deepcopy(receipt)
    next(row for row in changed["branches"] if row["retained"])["depthQuery"] = None
    changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
    with pytest.raises(ValueError):
        validate_native_bid_world_upper_v1(changed, **owner)
    binding = retained["playPolicyBinding"]
    assert binding["sourceCpModelProtoHash"] != binding["boundCpModelProtoHash"]
    assert len(retained["alternativeQuery"]["query"]["candidate"]["play"]["plays"]) == 32
    with pytest.raises(ValueError):
        validate_native_bid_world_upper_v1(receipt, **{**owner, "opponent_play_policy_id": None})
    for field, value in (("policy", {**binding["policy"], "team": opponent.other.name.lower()}),
                         ("sourceCpModelProtoHash", binding["boundCpModelProtoHash"]),
                         ("boundCpModelProtoHash", "0" * 64), ("schemaVersion", True)):
        changed = copy.deepcopy(receipt)
        row = next(row for row in changed["branches"] if row["retained"])
        row["playPolicyBinding"][field] = value
        row["playPolicyBinding"]["sha256"] = canonical_sha256({k: v for k, v in row["playPolicyBinding"].items() if k != "sha256"})
        changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
        with pytest.raises(ValueError):
            validate_native_bid_world_upper_v1(changed, **owner)
    for changes in ({"optimize_alternative_upper": False}, {"opponent_play_policy_id": "unknown"},
                    {"policy_team": opponent.other}):
        with pytest.raises(ValueError):
            compiler._prepare_native_bid_branches(exact, prefix, **{**args, **changes})


@pytest.mark.parametrize("optional_projects", [False, True])
def test_native_observed_project_bounds_cover_complete_continuation(root, optional_projects):
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofBudgetV1, canonical_sha256
    from baloot_platform.research.result_authority import reverify_native_observed_project_bounds_v1
    compiler, _, prefix, _ = _native_suit_monopoly_bid(root)
    if optional_projects:
        from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
        compiler = OrToolsActorInformationSetCompilerV1(RuleBook.load(
            root / "configs/rules/saudi_project_optional_unratified_v1.yaml"), native_cp_sat=True)
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[{"actor": "north", "action": "sun"}], projectActions=[])
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize_play_or_project(prefix)
    assert exact.decision_kind == "project"
    assert len(exact.legal_actions) == (2 if optional_projects else 1)
    bounds = compiler._solve_native_post_auction_policy_bounds(exact, prefix,
        Seat.parse(exact.next_actor).team, selected_root_action=None,
        compact_public_ownership=optional_projects,
        visibility="actor_information_set", opportunity_order=("west", "east", "north", "south"),
        escalation_events=[], maximum_iterations=20, maximum_queries=60, maximum_observations=64,
        maximum_model_bytes=100_000_000, maximum_conflicts=250_000, maximum_depth=36, maximum_seconds=60)
    assert bounds["status"] == "scoped_bound_converged"
    assert bounds["strategy"]["lowerBound"] == bounds["strategy"]["upperBound"]
    assert 68 <= bounds["strategy"]["lowerBound"] <= 92
    assert bounds["decisionKind"] == "project" and bounds["visibility"] == "actor_information_set"
    assert bounds["preparation"].get("ownershipCompaction") == ("public_fixed_v1" if optional_projects else None)
    assert bounds["strategicCoverage"] is False  # Internal bounds are not yet admission evidence.
    queries = bounds["strategy"]["queries"]
    floor = next(row["receipt"] for row in queries if row["purpose"] == (
        "root_policy_path_floor" if optional_projects else "universal_path_floor"))
    upper = next(row for row in queries if row["purpose"] == "adversarial_world_upper_bound")
    assert upper["sourceWorldQueryHash"] == floor["sha256"]
    assert upper["receipt"]["candidate"]["deckOrder"] == floor["candidate"]["deckOrder"]
    assert floor["candidate"]["utility"] == -upper["receipt"]["candidate"]["utility"] == 68
    if optional_projects:
        binding = next(row["receipt"] for row in queries if row["purpose"] == "project_root_policy_binding")
        assert binding["status"] == "bound" and binding["action"].startswith("declare:")
        assert floor["projectPolicyBindings"] == [binding]
        assert upper["receipt"]["projectPolicyBindings"] == []
    profile = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component(
        "challenge")["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    budget = ScenarioProofBudgetV1.build("offline_heavyweight_v1", profile)
    context = dict(rules=compiler.rules, source_prefix=prefix, normalized=exact,
        selected_root_action=None, opportunity_order=("west", "east", "north", "south"),
        escalation_events=[], budget=budget, maximum_conflicts=250_000, maximum_seconds=60)
    reverify_native_observed_project_bounds_v1(bounds, **context)
    import copy
    if optional_projects:
        wrong_compaction = copy.deepcopy(bounds)
        wrong_compaction["preparation"]["ownershipCompaction"] = True
        wrong_compaction["sha256"] = canonical_sha256({k: v for k, v in wrong_compaction.items() if k != "sha256"})
        with pytest.raises(ValueError, match="compaction identity"):
            reverify_native_observed_project_bounds_v1(wrong_compaction, **context)
    altered = copy.deepcopy(bounds)
    altered["strategy"]["queries"][-1]["sourceWorldQueryHash"] = "0" * 64
    altered["strategy"]["sha256"] = canonical_sha256({key: value for key, value in altered["strategy"].items() if key != "sha256"})
    altered["sha256"] = canonical_sha256({key: value for key, value in altered.items() if key != "sha256"})
    with pytest.raises(ValueError, match="fresh native reconstruction"):
        reverify_native_observed_project_bounds_v1(altered, **context)
    from baloot_platform.research.scenario_contracts import ScenarioProofRequestV1
    from baloot_platform.research.result_authority import (
        build_native_observed_play_quality_artifact_v1, validate_native_observed_play_quality_artifact_v1,
        native_featured_play_proof_supported_v1,
    )
    selected = (bounds["strategy"]["policy"][bounds["preparation"]["rootObservationHash"]]
                if optional_projects else exact.legal_actions[0]["code"])
    selected_bounds = bounds if optional_projects else compiler._solve_native_post_auction_policy_bounds(exact, prefix,
        Seat.parse(exact.next_actor).team, selected_root_action=selected,
        visibility="actor_information_set", opportunity_order=context["opportunity_order"], escalation_events=[],
        maximum_iterations=20, maximum_queries=60, maximum_observations=64,
        maximum_model_bytes=100_000_000, maximum_conflicts=250_000, maximum_depth=36, maximum_seconds=60)
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="actor_information_set",
        strategyClass="pure_deterministic", claim="winning_root_action_count",
        targetTeam=Seat.parse(exact.next_actor).team.name.lower(), utility="awarded_point_margin", successThreshold=0,
        budgetProfile=budget.profile_id, selectedAction=selected))
    artifact_context = {key: value for key, value in context.items() if key != "selected_root_action"}
    artifact_context.update(request=request, decision_kind="project",
        maximum_seconds=min(180, budget.maximum_wall_seconds, budget.maximum_cpu_seconds))
    artifact = build_native_observed_play_quality_artifact_v1(selected_bounds, bounds, **artifact_context)
    if optional_projects:
        assert artifact["graph"]["selectionSource"] == "verified_best_strategy"
        assert artifact["graph"]["selectedProofHash"] == artifact["graph"]["bestProofHash"] == bounds["sha256"]
        assert artifact["graph"]["queriesUsed"] == 2 * bounds["queriesUsed"] + 1
    assert artifact["proofModel"]["kind"] == "native_observed_project_quality_model_v1"
    assert artifact["proofModel"]["actionQuality"]["actualActionOptimal"] is True
    assert artifact["proof"]["winningRootActionCount"] is None
    assert not native_featured_play_proof_supported_v1(artifact)
    validate_native_observed_play_quality_artifact_v1(artifact, **artifact_context)
    with pytest.raises(ValueError, match="owned exact selected-action"):
        build_native_observed_play_quality_artifact_v1(selected_bounds, bounds,
            **{**artifact_context, "decision_kind": "play"})


def test_native_sun_branch_strategy_bounds_cover_all_32_plays(root, monkeypatch):
    """Certify one fixed Sun branch, not opening optimality or hidden worlds."""
    compiler, exact, prefix, hands = _native_suit_monopoly_bid(root)
    dealer, buyer = Seat.parse(prefix.dealer), Seat.parse(exact.next_actor)
    # The selected bid must not omit another reachable contract/mode branch.
    for spec in compiler._native_bid_branch_specs(dealer):
        probe = compiler._native_bid_root_model(exact, prefix, visibility="complete_state")
        probe.model.add(probe.auction_fragment.choices[0]["sun"] == 1)
        if spec is None:
            probe.model.add(probe.auction_fragment.states[-1]["allPassed"] == 1)
        else:
            contract, trump, leader, locked = spec
            probe._add_native_round_escalation(contract=contract, trump=trump,
                first_leader=leader, locked=locked, match_score=prefix.match_score)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = 15
        expected = cp_model.OPTIMAL if spec == (Contract.SUN, None, buyer, False) else cp_model.INFEASIBLE
        assert solver.solve(probe.model) == expected
    source = compiler._build_native_bid_branch(
        exact, prefix, visibility="complete_state", branch=(Contract.SUN, None, buyer, False),
        cancel_check=None)
    source.model.add(source.auction_fragment.choices[0]["sun"] == 1)
    fragment = source._native_round_fragment
    # Candidate strategy only: native constraints must prove legality and value.
    chosen = {seat: max((p for p in fragment.projects.projects
        if p.owner == seat and {card.id for card in p.cards} <= hands[seat]),
        key=lambda p: (p.value, p.high_rank, len(p.cards), p.project_id)) for seat in Seat}
    illegal = source.model.clone()
    illegal.add_bool_or([fragment.projects.declaration_actions[int(seat)]["declare:" + p.project_id].Not()
        for seat, p in chosen.items()])
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 15
    assert solver.solve(illegal) == cp_model.INFEASIBLE

    bounds = []
    before = str(source.model.proto)
    for team in (buyer.team, buyer.team.other):
        actions = {seat.name.lower(): "declare:" + project.project_id
            for seat, project in chosen.items() if seat.team == team}
        maximum = max(max(map(int, v.proto.domain)) for v in source._native_round_query_context()[1])
        result = source.find_native_round_with_constant_policy_below(
            target=team, policy_team=team, auction_action=None, escalation_action=None,
            project_actions=actions, threshold=maximum + 1,
            maximum_seconds=30, maximum_conflicts=250_000)
        assert result["status"] == "queried"
        assert result["policy"]["projectActions"] == actions
        assert len(result["bindings"]) == 2
        assert all(binding["status"] == "bound" for binding in result["bindings"])
        receipt = result["query"]
        assert receipt["solverStatus"] == "optimal"
        candidate = receipt["candidate"]
        assert candidate is not None  # Nonempty witnesses rule out vacuous legality.
        assert len(candidate["play"]["plays"]) == 32
        assert len(candidate["play"]["tricks"]) == 8
        bounds.append(candidate["utility"] if team == buyer.team else -candidate["utility"])
        assert str(source.model.proto) == before
        if team == buyer.team:
            repeated = source.find_native_round_with_constant_policy_below(
                target=team, policy_team=team, auction_action=None, escalation_action=None,
                project_actions=dict(reversed(list(actions.items()))), threshold=maximum + 1,
                maximum_seconds=30, maximum_conflicts=250_000)
            assert repeated == result
    assert bounds == [92, 92]
    models, upper = compiler._prepare_native_bid_branches(exact, prefix, visibility="complete_state",
        selected_root_action="sun", policy_team=buyer.team.other,
        nonplay_policy={"auction_action": "pass", "escalation_action": "decline"},
        maximum_branches=13, maximum_model_bytes=100000000, maximum_conflicts=250000,
        maximum_seconds=30, maximum_queries=100, optimize_alternative_upper=True)
    assert models == () and upper["constructionComplete"]
    assert upper["kind"] == "native_bid_streamed_world_upper_v1"
    assert upper["upperBound"] == 92
    assert upper["worldStateHash"] == exact.state_hash and upper["worldPrefixHash"] == prefix.sha256
    assert not upper["strategicCoverage"]
    from baloot_platform.research.result_authority import validate_native_bid_world_upper_v1
    from baloot_platform.research.scenario_contracts import canonical_sha256
    owner = dict(rules=compiler.rules, exact=exact, source_prefix=prefix, action="sun",
        opponent_policy={"auction_action": "pass", "escalation_action": "decline"}, maximum_model_bytes=100000000)
    assert validate_native_bid_world_upper_v1(upper, **owner) == 92
    models, threshold_proof = compiler._prepare_native_bid_branches(exact, prefix, visibility="complete_state",
        selected_root_action="sun", policy_team=buyer.team.other,
        nonplay_policy=owner["opponent_policy"], alternative_upper_bound=92,
        maximum_branches=13, maximum_model_bytes=100000000, maximum_conflicts=250000,
        maximum_seconds=30, maximum_queries=100)
    assert not models and threshold_proof["constructionComplete"]
    retained_thresholds = [row["alternativeQuery"]["query"] for row in threshold_proof["branches"] if row["retained"]]
    assert retained_thresholds
    assert all(row["status"] == "no_matching_outcome" and row["solverStatus"] == "infeasible"
        and row["queryMode"] == "feasibility_witness" for row in retained_thresholds)
    from baloot_platform.research.result_authority import reverify_native_bid_world_upper_v1
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    fresh_limits = dict(maximum_branches=13, maximum_conflicts=250000,
        maximum_seconds=30, maximum_queries=100)
    threshold_measured = []
    assert reverify_native_bid_world_upper_v1(threshold_proof, **owner, **fresh_limits,
        _on_queries_used=threshold_measured.append) == 92
    assert threshold_measured == [threshold_proof["queriesUsed"]]
    measured = []
    assert reverify_native_bid_world_upper_v1(upper, **owner, **fresh_limits,
        _on_queries_used=measured.append) == 92
    assert measured == [upper["queriesUsed"]]
    with pytest.raises(InformationSetCompilationStopped):
        reverify_native_bid_world_upper_v1(upper, **owner, **fresh_limits, cancel_check=lambda: True)
    # Real reconstruction above; isolate substitutions on the fresh-reader seam.
    for change in ("bound", "model", "unexpected_models"):
        reconstructed = copy.deepcopy(upper)
        cell = next(row for row in reconstructed["branches"] if row["retained"])
        if change == "bound":
            reconstructed["upperBound"] = 91
        elif change == "model":
            cell["sourceModelHash"] = "0" * 64
        with monkeypatch.context() as patch:
            patch.setattr(type(compiler), "_prepare_native_bid_branches",
                lambda *a, **kw: ((object(),) if change == "unexpected_models" else (), reconstructed))
            with pytest.raises(ValueError):
                reverify_native_bid_world_upper_v1(upper, **owner, **fresh_limits)
    retained_index = next(i for i, row in enumerate(upper["branches"]) if row["retained"])
    query_path = ("branches", retained_index, "alternativeQuery")
    def reseal(value):
        if isinstance(value, dict):
            for item in value.values():
                reseal(item)
            if "sha256" in value:
                value["sha256"] = canonical_sha256({k: v for k, v in value.items() if k != "sha256"})
        elif isinstance(value, list):
            for item in value:
                reseal(item)
    forged = copy.deepcopy(upper)
    for change in ("scope", "threshold", "query"):
        tampered_threshold = copy.deepcopy(threshold_proof)
        if change == "scope":
            tampered_threshold["scope"]["rootHash"] = "0" * 64
            tampered_threshold["scopeHash"] = canonical_sha256(tampered_threshold["scope"])
        elif change == "threshold":
            tampered_threshold["alternativeUpperBound"] = 91
        else:
            branch = next(row for row in tampered_threshold["branches"] if row["retained"])
            branch["alternativeQuery"]["query"]["solverStatus"] = "unknown"
        reseal(tampered_threshold)
        with monkeypatch.context() as patch:
            # Fresh proof already computed above; isolate retained substitutions.
            patch.setattr(type(compiler), "_prepare_native_bid_branches", lambda *a, **kw: ((), threshold_proof))
            with pytest.raises(ValueError):
                reverify_native_bid_world_upper_v1(tampered_threshold, **owner, **fresh_limits)
    forged["upperBound"] = 91
    for cell in forged["branches"]:
        if cell["retained"]:
            cell["alternativeUpperBound"] = 91
            outcome = cell["alternativeQuery"]["query"]
            outcome["objectiveLowerBound"] = -91
            outcome["candidate"]["utility"] = -91
    reseal(forged)
    # Checksums and internally consistent bounds are not semantic proof.
    assert validate_native_bid_world_upper_v1(forged, **owner) == 91
    with pytest.raises(ValueError, match="failed fresh verification"):
        reverify_native_bid_world_upper_v1(forged, **owner, **fresh_limits)
    for path, value in ((("worldStateHash",), "wrong"), (("worldExactStateId",), "wrong"),
            (("worldEffectivePrefixHash",), "wrong"), (("observerRootHash",), "wrong"),
            (("upperBound",), 91), (("schemaVersion",), True), (("branches",), upper["branches"][:-1]),
            (query_path + ("query", "solverStatus"), "feasible"),
            (query_path + ("query", "targetTeam"), buyer.team.name.lower()),
            (query_path + ("query", "objectiveLowerBound"), 0),
            (query_path + ("query", "exhaustionReason"), "maximum_seconds"),
            (query_path + ("query", "baseModelHash"), "wrong-model"),
            (query_path + ("bindings", 0, "solverStatus"), "OPTIMAL"),
            (query_path + ("bindings", 0, "observationHash"), "wrong-observation"),
            (query_path + ("policy", "auctionAction"), "sun")):
        tampered = copy.deepcopy(upper)
        parent = tampered
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = copy.deepcopy(value)
        reseal(tampered)
        with pytest.raises(ValueError):
            validate_native_bid_world_upper_v1(tampered, **owner)
    swapped = copy.deepcopy(upper)
    owned_query = swapped["branches"][retained_index]["alternativeQuery"]
    owned_query["scope"]["contract"] = "hokum"
    owned_query["query"]["scope"]["contract"] = "hokum"
    reseal(swapped)
    with pytest.raises(ValueError, match="world upper ownership"):
        validate_native_bid_world_upper_v1(swapped, **owner)
    excluded = [i for i, row in enumerate(upper["branches"]) if not row["retained"]]
    swapped = copy.deepcopy(upper)
    swapped["branches"][excluded[0]]["reachability"] = copy.deepcopy(swapped["branches"][excluded[1]]["reachability"])
    reseal(swapped)
    with pytest.raises(ValueError, match="world upper ownership"):
        validate_native_bid_world_upper_v1(swapped, **owner)
    with pytest.raises(ValueError, match="world upper ownership"):
        validate_native_bid_world_upper_v1(upper, **owner, opportunity_order=[Seat.NORTH] * 4)
    assert fragment.scope["strategicCoverage"] is False
    args = dict(target=buyer.team, policy_team=buyer.team, auction_action=None,
        escalation_action=None, threshold=1000, maximum_seconds=30, maximum_conflicts=250_000)
    for actions in ({"NORTH": "pass"}, {"north": True}, {"east": "pass"},
                    {"north": "unknown"}, ["pass"]):
        with pytest.raises(ValueError, match="project"):
            source.find_native_round_with_constant_policy_below(**args, project_actions=actions)
    invalid = source.find_native_round_with_constant_policy_below(**args, project_actions={"north": "pass"})
    assert invalid["status"] == "invalid_policy" and invalid["query"] is None
    actions = {seat.name.lower(): "declare:" + project.project_id
        for seat, project in chosen.items() if seat.team == buyer.team}
    cancelled = source.find_native_round_with_constant_policy_below(
        **args, project_actions=actions, cancel_check=lambda: True)
    assert cancelled["status"] == "incomplete" and cancelled["bindings"] == []
    import baloot_platform.research.ortools_scenario_model as module
    original_solve = module._solve_cp_sat_with_cancel_v1
    calls = 0

    def stop_second(*solve_args):
        nonlocal calls
        calls += 1
        return original_solve(*solve_args) if calls == 1 else cp_model.UNKNOWN

    with monkeypatch.context() as patch:
        patch.setattr(module, "_solve_cp_sat_with_cancel_v1", stop_second)
        partial = source.find_native_round_with_constant_policy_below(**args, project_actions=actions)
    assert partial["status"] == "incomplete" and partial["query"] is None
    assert [binding["status"] for binding in partial["bindings"]] == ["bound", "incomplete"]
    assert str(source.model.proto) == before


@pytest.mark.parametrize("played", [0, 3])
def test_native_early_play_uniform_bounds_keep_forced_project_coverage(root, played):
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.result_authority import build_native_complete_state_play_quality_artifact_v1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofBudgetV1, ScenarioProofRequestV1
    compiler, _, prefix, _ = _native_suit_monopoly_bid(root)
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[{"actor": "north", "action": "sun"}], projectActions=[])
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True)
    project = normalizer.normalize_play_or_project(prefix)
    assert project.decision_kind == "project" and len(project.legal_actions) == 1
    project_prefix = prefix
    prefix = normalizer.append_normalized_action(prefix, project, project.legal_actions[0]["code"])
    exact = normalizer.normalize_play_or_project(prefix)
    while len(prefix.play_actions) < played or exact.decision_kind == "project":
        prefix = normalizer.append_normalized_action(prefix, exact, exact.legal_actions[0]["code"])
        exact = normalizer.normalize_play_or_project(prefix)
    target = Seat.parse(exact.next_actor).team
    profile = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component(
        "challenge")["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    budget = ScenarioProofBudgetV1.build("offline_heavyweight_v1", profile)
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="complete_state",
        claim="outcome_forced", targetTeam=target.name.lower(), utility="awarded_point_margin",
        successThreshold=1, budgetProfile=budget.profile_id))
    result = compiler._prove_complete_state_play(exact, prefix, request, budget=budget, maximum_seconds=60)
    expected = 92 if target == Team.NORTH_SOUTH else -92
    assert result["certificate"]["winningRootActionCount"] == (8 if expected > 0 else 0)
    assert {row["value"] for row in result["certificate"]["roots"]} == {expected}
    preparation = result["preparation"]
    assert preparation["remainingActions"] == 35 - 2 * played
    assert len(preparation["projectCoverage"]["sequences"]) == 1
    for proof in result["roots"]["roots"].values():
        assert proof["proofMethod"] == "uniform_native_path_bounds_v1"
        assert proof["strategyCoverage"] == "all_legal_continuations_value_invariant"
        assert proof["branchCount"] == proof["worldCount"] == 1
        assert proof["scope"]["branches"] == preparation["branches"]
        assert {query["value"] for query in proof["queries"]} == {expected}
        assert all(query["solverStatus"] == "optimal" for query in proof["queries"])
        assert all(len(query["nativePlay"]["plays"]) == 32 for query in proof["queries"])
    artifact = build_native_complete_state_play_quality_artifact_v1(result, rules=compiler.rules,
        source_prefix=prefix, normalized=exact, request=request, budget=budget)
    assert artifact["proof"]["completionStatus"] == ("exact_complete" if expected > 0 else "proof_failed")
    assert artifact["proofModel"]["actionQuality"]["scope"] == "compiled_complete_state"

    if played == 0:
        import copy
        from baloot_platform.research.ortools_research_executor import resolve_quality_play_continuation_v1
        from baloot_platform.research.result_authority import verify_native_complete_state_policy_pair_v1
        from baloot_platform.research.scenario_contracts import (
            canonical_sha256, ScenarioPlayingQualityPolicyV1, ScenarioQualityExecutionBudgetV1)
        policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact",
            "visibility": "complete_state", "decisions": {kind: {
                "utility": "awarded_point_margin", "budgetProfile": budget.profile_id}
                for kind in ("bid", "escalation", "project", "play")}})
        request = policy.request_for("play", exact.next_actor)
        paired = compiler._synthesize_complete_state_policy_pair(exact, prefix, request,
            budget=budget, maximum_seconds=60,
            native_play_policy_ids={team.name.lower(): "observer_safe_partner_feed_v1" for team in Team})
        assert paired["queriesUsed"] == 2
        assert len(paired["preparation"]["projectCoverage"]["sequences"]) == 1
        execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
            "maximumWallSeconds": 120, "maximumEvidenceBytes": 30_000_000,
            "maximumEscalationDecisions": 20, "completeStatePlayPolicyPair": True})
        generated = resolve_quality_play_continuation_v1(compiler.rules, project_prefix, policy,
            execution_budget=execution, maximum_nodes=100_000, maximum_depth=35,
            maximum_worlds=1, maximum_seconds=120, maximum_decision_seconds=60,
            maximum_evidence_bytes=30_000_000, proof_budget_profiles={budget.profile_id: profile})
        partial = generated.get("partialReceipt") or {}
        assert generated["status"] == "complete", {
            key: partial.get(key) for key in ("stage", "exhaustionReason", "reason")}
        assert len(generated["finalPrefix"]["playActions"]) == 32
        assert len(generated["finalPrefix"]["projectActions"]) == 4
        assert len(generated["decisions"]) == 36
        assert generated["decisions"][0]["decisionKind"] == "project"
        assert all(row["actualActionOptimal"] is True and row["regret"] == 0 for row in generated["decisions"])
        # Generation already performs fresh owned readback; do not duplicate it.
        evidence = next(iter(generated["pairedPolicyEvidence"].values()))
        assert evidence["synthesis"]["preparation"]["basePrefixLength"] == 0
        assert evidence["synthesis"]["preparation"]["sourcePrefixHash"] == prefix.sha256
        assert evidence["synthesis"]["queriesUsed"] == 4
        assert all(proof["proofMethod"] == "uniform_native_path_bounds_v1"
            and "selectedRootAction" not in proof["scope"]
            and all(query["selectedRootAction"] is None for query in proof["queries"])
            for proof in evidence["proofs"].values())
        verified = evidence["receipt"]
        assert verified["wholeRoundCertified"] is False
        assert verified["counterexampleAndReachabilityQueries"] == (
            sum(len(row["queryHashes"]) for row in verified["guarantees"].values())
            + len(verified["continuation"]["queryHashes"]))
        assert verified["counterexampleAndReachabilityQueries"] <= verified["queryLimit"]
        assert len(verified["continuation"]["decisions"]) == 32
        assert len(verified["continuation"]["projectActions"]) == 4
        assert verified["continuation"]["projectCoverageHash"] == paired["preparation"]["projectCoverage"]["sha256"]
        assert verified["continuation"]["terminalValue"] == expected
        refused = resolve_quality_play_continuation_v1(compiler.rules, prefix, policy,
            paired_policy_proofs=paired["proofs"], maximum_nodes=100_000, maximum_depth=32,
            maximum_worlds=1, maximum_seconds=120, maximum_decision_seconds=60,
            maximum_evidence_bytes=30_000_000, proof_budget_profiles={budget.profile_id: profile})
        assert refused["status"] == "incomplete" and refused["decisions"] == []
        assert refused["partialReceipt"]["exhaustionReason"] == "maximum_depth"
        substituted = copy.deepcopy(paired["proofs"])
        proof = substituted[target.name.lower()]
        proof["scope"]["branches"][0]["coverageHash"] = "0" * 64
        proof["scopeHash"] = canonical_sha256(proof["scope"])
        proof["sha256"] = canonical_sha256({key: value for key, value in proof.items() if key != "sha256"})
        with pytest.raises(ValueError, match="paired policy receipt ownership"):
            verify_native_complete_state_policy_pair_v1(substituted, rules=compiler.rules,
                normalized=exact, source_prefix=prefix, request=request, budget=budget)


def test_native_fixed_sun_coordinator_proves_full_round_value(root):
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    target = Seat.parse(exact.next_actor).team
    result = compiler._solve_native_bid_policy_bounds(exact, prefix, target,
        visibility="complete_state", selected_root_action="sun", maximum_iterations=10,
        maximum_queries=20, maximum_observations=1000, maximum_branches=13,
        maximum_model_bytes=100_000_000, maximum_conflicts=250_000, maximum_seconds=60)
    assert result["status"] == "scoped_bound_converged"
    assert result["strategicCoverage"] is False
    assert result["preparation"]["constructionComplete"]
    assert sum(row["retained"] for row in result["preparation"]["branches"]) == 1
    strategy = result["strategy"]
    from baloot_platform.research.result_authority import _validate_native_final_branch_proofs_v2
    assert strategy["kind"] == "native_symbolic_round_policy_bounds_v2"
    _validate_native_final_branch_proofs_v2(strategy, target_team=target.name.lower())
    assert sorted(row["sourceModelHash"] for row in result["preparation"]["branches"] if row["retained"]) == sorted(
        row["modelHash"] for row in strategy["scope"]["branches"])
    assert strategy["lowerBound"] == strategy["upperBound"] == 92
    assert len(strategy["queries"]) == 2
    upper, lower = (row["receipt"] for row in strategy["queries"])
    assert upper["solverStatus"] == "optimal"
    assert upper["targetTeam"] == target.other.name.lower()
    assert upper["candidate"]["utility"] == -92
    assert len(upper["candidate"]["play"]["plays"]) == 32
    assert lower["solverStatus"] == "infeasible"
    assert lower["targetTeam"] == target.name.lower() and lower["threshold"] == 92
    assert lower["candidate"] is None
    # This certifies a fixed bid value, not a count of optimal opening moves.
    assert "rootActionValues" not in result and "winningRootActionCount" not in result


@pytest.mark.real_clock
@pytest.mark.parametrize("minimax_shortening", (False, True))
def test_guarded_native_threshold_search_and_fresh_reader(root, minimax_shortening):
    import time
    from baloot_platform.research.result_authority import (
        validate_native_strategy_threshold_v1, _reverify_observed_bid_policy_threshold)
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, canonical_sha256

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    actor = Seat.parse(exact.next_actor)
    def build():
        source = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
            branch=(Contract.SUN, None, actor, False), compact_known_initial_ownership=True)
        source.constrain_native_auction_root_choice("sun", actor=actor)
        return source
    source = build()
    before = str(source.model.proto)
    limits = dict(maximum_iterations=3, maximum_queries=100, maximum_observations=1000,
        maximum_branches=1, maximum_conflicts=250000, maximum_seconds=30)
    if minimax_shortening:
        import yaml
        settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
        limits.update(shorten_counterexamples=True, minimax_budget=ScenarioProofBudgetV1.build(
            "actor_information_set_exact_v1", settings["proofBudgets"]["profiles"]["actor_information_set_exact_v1"]))
    root_key = source._auction_observation_hash(actor, exact.state["hands"][actor.name.lower()],
        exact.state["purchaseCard"], [])
    strategy = compiler._solve_symbolic_round_policy_bounds([source], actor.team, **limits,
        required_threshold=90, _play_fallback_id="observer_safe_partner_feed_secure_control_v1",
        fixed_root=(root_key, tuple(sorted(row["code"] for row in exact.legal_actions)), "sun"))
    assert strategy["kind"] == "native_guarded_round_strategy_threshold_v2"
    assert strategy["scope"]["policyRepresentation"] == "native_fallback_plus_deviations_v1"
    assert strategy["scope"].get("minimaxProofBudgetHash") == (
        limits["minimax_budget"].sha256 if minimax_shortening else None)
    assert strategy["status"] == "scoped_strategy_threshold_complete"
    assert not strategy["strategicCoverage"] and len(strategy["masters"]) <= 3
    assert validate_native_strategy_threshold_v1(strategy, expected_scope=strategy["scope"]) == 90
    assert all(row["policyIndependent"] is False for row in strategy["finalBranchProofs"])
    assert str(source.model.proto) == before

    def read(value):
        fresh = build()
        assert str(fresh.model.proto) == before
        return _reverify_observed_bid_policy_threshold(value, {"queriesUsed": 0}, [fresh],
            compiler=compiler, scope=strategy["scope"], target=actor.team, required_threshold=90,
            deadline=time.monotonic()+30, maximum_queries=100, maximum_observations=1000,
            maximum_conflicts=250000, cancel_check=None, on_queries_used=None)
    assert read(strategy) == 90
    hint_core = {key: copy.deepcopy(strategy[key]) for key in ("scopeHash", "domains", "policy", "playContexts")}
    hint = {**hint_core, "sha256": canonical_sha256(hint_core)}
    original_hint = copy.deepcopy(hint)
    hinted = compiler._solve_symbolic_round_policy_bounds([build()], actor.team, **limits,
        required_threshold=90, _play_fallback_id="observer_safe_partner_feed_secure_control_v1",
        fixed_root=(root_key, tuple(sorted(row["code"] for row in exact.legal_actions)), "sun"),
        _strategy_hint=hint)
    assert hinted["status"] == "scoped_strategy_threshold_complete" and hint == original_hint
    assert hinted["queriesUsed"] > 0 and hinted["scopeHash"] == strategy["scopeHash"]
    assert read(hinted) == 90  # A hint never replaces independent reconstruction.
    for change in ({"sha256": "0"*64}, {"scopeHash": "0"*64}, {"lowerBound": 999},
                   {"domains": []}, {"domains": {root_key: []}},
                   {"policy": {root_key: "pass"}}, {"policy": {"missing": "sun"}},
                   {"playContexts": {"0"*64: {}}}):
        wrong = {**copy.deepcopy(hint), **change}
        if "sha256" not in change:
            wrong["sha256"] = canonical_sha256({k: v for k, v in wrong.items() if k != "sha256"})
        with pytest.raises(ValueError, match="strategy hint"):
            compiler._solve_symbolic_round_policy_bounds([build()], actor.team, **limits,
                required_threshold=90, _play_fallback_id="observer_safe_partner_feed_secure_control_v1",
                fixed_root=(root_key, tuple(sorted(row["code"] for row in exact.legal_actions)), "sun"),
                _strategy_hint=wrong)
    legacy = copy.deepcopy(strategy)
    legacy["kind"] = "native_guarded_round_strategy_threshold_v1"
    legacy["scope"].pop("policyRepresentation")
    legacy["scopeHash"] = canonical_sha256(legacy["scope"])
    legacy["sha256"] = canonical_sha256({key: value for key, value in legacy.items() if key != "sha256"})
    assert validate_native_strategy_threshold_v1(legacy, expected_scope=legacy["scope"]) == 90
    for change in ({"kind": "native_symbolic_round_strategy_threshold_v1"},
                   {"kind": "native_guarded_round_strategy_threshold_v1"},
                   {"playContexts": {"0"*64: {}}}):
        changed = {**copy.deepcopy(strategy), **change}
        changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
        with pytest.raises(ValueError, match="ownership|coverage"):
            read(changed)
    # A resealed but substituted fallback model hash still fails a fresh solve.
    changed = copy.deepcopy(strategy)
    proof = changed["finalBranchProofs"][0]
    retained = changed["refinements"].pop(proof["refinementHash"])
    binding = retained["core"]["fallbackBinding"]
    binding["boundCpModelProtoHash"] = "0" * 64
    binding["sha256"] = canonical_sha256({k: v for k, v in binding.items() if k != "sha256"})
    lookup = {canonical_sha256(row["receipt"]): row["receipt"] for row in changed["queries"]}
    identity = canonical_sha256({**retained["core"], "queries": [lookup[key] for key in retained["queryHashes"]]})
    changed["refinements"][identity] = retained
    proof["refinementHash"] = identity
    changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
    with pytest.raises(ValueError, match="fresh native binding"):
        read(changed)


@pytest.mark.real_clock
@pytest.mark.parametrize("minimax_shortening", (False, True))
def test_guarded_mixed_redeal_and_play_search_keeps_auction_witness_out_of_card_hints(root, minimax_shortening):
    import time
    import yaml
    from baloot_platform.research.result_authority import _reverify_observed_bid_policy_threshold
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    actor = Seat.parse(exact.next_actor)
    def build():
        redeal = compiler._native_bid_root_model(exact, prefix, visibility="complete_state")
        redeal.add_native_redeal_continuation(match_score=prefix.match_score)
        purchased = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
            branch=(Contract.SUN, None, actor, False), compact_known_initial_ownership=True)
        purchased.constrain_native_auction_root_choice("sun", actor=actor)
        return [redeal, purchased]
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
    budget = ScenarioProofBudgetV1.build("actor_information_set_exact_v1",
        settings["proofBudgets"]["profiles"]["actor_information_set_exact_v1"])
    options = dict(shorten_counterexamples=True, minimax_budget=budget) if minimax_shortening else {}
    strategy = compiler._solve_symbolic_round_policy_bounds(build(), actor.team,
        required_threshold=1, _play_fallback_id="observer_safe_partner_feed_secure_control_v1",
        maximum_iterations=4, maximum_queries=100, maximum_observations=1000,
        maximum_branches=2, maximum_conflicts=250000, maximum_seconds=30, **options)
    assert strategy["status"] == "scoped_strategy_threshold_complete" and not strategy["strategicCoverage"]
    assert len(strategy["masters"]) >= 2 and strategy["cuts"]
    witnesses = [row["receipt"] for row in strategy["queries"] if row["receipt"].get("status") == "witness_found"]
    assert witnesses and any(row["candidate"].get("allPassed") for row in witnesses)
    assert all("playCardHintHash" not in row["receipt"] for row in strategy["queries"]
        if row["receipt"]["scope"].get("allPassedCovered"))
    assert _reverify_observed_bid_policy_threshold(strategy, {"queriesUsed": 0}, build(),
        compiler=compiler, scope=strategy["scope"], target=actor.team, required_threshold=1,
        deadline=time.monotonic()+30, maximum_queries=100, maximum_observations=1000,
        maximum_conflicts=250000, cancel_check=None, on_queries_used=None) == 1


@pytest.mark.real_clock
@pytest.mark.parametrize("minimax_shortening", (False, True))
def test_guarded_threshold_refines_the_ace_sacrifice_and_reverifies(root, rules, monkeypatch, minimax_shortening):
    import json
    import time
    import yaml
    from baloot_platform.core.models import Project
    from baloot_platform.core.project_ir import project_id_v1
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.result_authority import (
        _reverify_observed_bid_policy_threshold, validate_native_strategy_threshold_v1)
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofBudgetV1, canonical_sha256
    from baloot_platform.research.ortools_information_set_proof import OrToolsActorInformationSetProofV1

    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    legacy = normalizer.normalize(ExactTracePrefixV1.build(raw), decision_point={"kind": "play"})
    awarded = next(row for row in legacy.state["projects"] if row["scoring"])
    project = Project(owner=Seat.parse(awarded["owner"]), kind=awarded["kind"],
        cards=tuple(Card(card["id"]) for card in awarded["cards"]), value=awarded["value"],
        high_rank=awarded["highRank"], category_priority=rules.project_domain.category_priorities[awarded["kind"]])
    raw.update(projectDeclarations=[], projectAssertions=[], projectActions=[
        {"actor": seat.name.lower(), "action": "declare", "projectId": project_id_v1(project)}
        if seat == project.owner else {"actor": seat.name.lower(), "action": "pass"}
        for seat in rules.project_domain.declaration_seats(Seat.parse(raw["dealer"]))])
    prefix = ExactTracePrefixV1.build(raw)
    exact = normalizer.normalize(prefix, decision_point={"kind": "play"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    def build():
        return compiler._native_post_auction_root_model(exact, prefix,
            opportunity_order=("west", "east", "north", "south"), escalation_events=[])
    source = build()
    preference_args, master_preferences = {}, []
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
    budget = ScenarioProofBudgetV1.build("actor_information_set_exact_v1",
        settings["proofBudgets"]["profiles"]["actor_information_set_exact_v1"])
    # Functional proof/readback coverage uses the governed tier, not a machine-
    # dependent smoke timeout. Dedicated performance gates remain separate.
    proof_seconds = budget.maximum_wall_seconds
    if minimax_shortening:
        preference_args = dict(shorten_counterexamples=True, minimax_budget=budget)
        original_master = OrToolsActorInformationSetProofV1._solve_master
        def observed_master(service, *args, **kwargs):
            master_preferences.append(dict(kwargs.get("preferred_actions", {})))
            return original_master(service, *args, **kwargs)
        monkeypatch.setattr(OrToolsActorInformationSetProofV1, "_solve_master", observed_master)
    strategy = compiler._solve_symbolic_round_policy_bounds([source], Team.WEST_EAST,
        required_threshold=1, _play_fallback_id="observer_safe_trick_control_v1",
        maximum_iterations=3, maximum_queries=100, maximum_observations=1000,
        maximum_branches=1, maximum_conflicts=250000, maximum_seconds=proof_seconds, **preference_args)
    assert strategy["status"] == "scoped_strategy_threshold_complete"
    assert len(strategy["masters"]) >= 2 and strategy["cuts"]
    first_witness = next(row["receipt"] for row in strategy["queries"]
        if row["receipt"].get("status") == "witness_found")
    assert any(row["receipt"].get("deckHintHash") == canonical_sha256(first_witness["candidate"]["deckOrder"])
        and row["receipt"].get("playCardHintHash") == canonical_sha256(
            [Card.parse(play["card"]).id for play in first_witness["candidate"]["play"]["plays"]])
        for row in strategy["queries"])
    root = next(row for wrapper in strategy["queries"]
        for row in (wrapper["receipt"].get("candidate") or {}).get("playObservations", ())
        if row["playIndex"] == len(prefix.play_actions))
    assert root["selectedAction"] != "CA"  # The baseline really failed, not a synthetic cut.
    assert strategy["policy"][root["observationHash"]] == "CA"
    if minimax_shortening:
        assert any(row.get(root["observationHash"]) == "CA" for row in master_preferences)
        assert any(row["receipt"].get("prefixPlayCount") == len(prefix.play_actions)
            and row["receipt"].get("kind") == "native_counterexample_prefix_minimax_v1"
            for row in strategy["queries"])
    play_keys = set(strategy["playContexts"]) & set(strategy["policy"])
    assert play_keys == {root["observationHash"]}
    effective = strategy["refinements"][strategy["finalBranchProofs"][0]["refinementHash"]]["core"]
    assert len(effective["playOverrides"]) == 1
    assert _reverify_observed_bid_policy_threshold(strategy, {"queriesUsed": 0}, [build()],
        compiler=compiler, scope=strategy["scope"], target=Team.WEST_EAST, required_threshold=1,
        deadline=time.monotonic()+proof_seconds, maximum_queries=100, maximum_observations=1000,
        maximum_conflicts=250000, cancel_check=None, on_queries_used=None) == 1
    # Warm policy proposals still need fresh truth. A legal but losing proposal
    # must be corrected, not certified using the previous successful receipt.
    for proposal in ("CA", root["selectedAction"]):
        hint_core = {key: copy.deepcopy(strategy[key]) for key in ("scopeHash", "domains", "policy", "playContexts")}
        hint_core["playContexts"] = {key: info for key, info in hint_core["playContexts"].items()
            if key in hint_core["domains"]}
        hint_core["policy"][root["observationHash"]] = proposal
        hinted = compiler._solve_symbolic_round_policy_bounds([build()], Team.WEST_EAST,
            required_threshold=1, _play_fallback_id="observer_safe_trick_control_v1",
            maximum_iterations=3, maximum_queries=100, maximum_observations=1000,
            maximum_branches=1, maximum_conflicts=250000, maximum_seconds=proof_seconds,
            _strategy_hint={**hint_core, "sha256": canonical_sha256(hint_core)}, **preference_args)
        assert hinted["status"] == "scoped_strategy_threshold_complete" and hinted["queriesUsed"] > 0
        assert hinted["policy"][root["observationHash"]] == "CA"
        if proposal == "CA":
            assert len(hinted["masters"]) == 1
        else:
            assert len(hinted["masters"]) > 1  # A losing hint must be freshly corrected.
        assert _reverify_observed_bid_policy_threshold(hinted, {"queriesUsed": 0}, [build()],
            compiler=compiler, scope=strategy["scope"], target=Team.WEST_EAST, required_threshold=1,
            deadline=time.monotonic()+proof_seconds, maximum_queries=100, maximum_observations=1000,
            maximum_conflicts=250000, cancel_check=None, on_queries_used=None) == 1
    without_deviation = compiler._refine_symbolic_round_policy(build(), Team.WEST_EAST, {},
        threshold=1, _play_fallback_id="observer_safe_trick_control_v1", _play_contexts={},
        maximum_queries=100, maximum_observations=1000, maximum_conflicts=250000,
        maximum_seconds=15)
    assert without_deviation["status"] == "policy_counterexample_in_branch"
    assert without_deviation["candidate"]["utility"] < 1
    wrong_hint = copy.deepcopy(first_witness)
    wrong_hint["targetTeam"] = Team.NORTH_SOUTH.name.lower()
    wrong_hint["sha256"] = canonical_sha256({key: value for key, value in wrong_hint.items() if key != "sha256"})
    with pytest.raises(ValueError, match="guarded search hint ownership"):
        compiler._refine_symbolic_round_policy(build(), Team.WEST_EAST, {},
            threshold=1, _play_fallback_id="observer_safe_trick_control_v1", _play_contexts={},
            _search_hint=wrong_hint, maximum_queries=100, maximum_observations=1000,
            maximum_conflicts=250000, maximum_seconds=15)
    bad_hint = copy.deepcopy(first_witness)
    bad_hint["candidate"]["deckOrder"][1] = bad_hint["candidate"]["deckOrder"][0]
    bad_hint["sha256"] = canonical_sha256({key: value for key, value in bad_hint.items() if key != "sha256"})
    with pytest.raises(ValueError, match="guarded search hint requires complete legal card sequences"):
        compiler._refine_symbolic_round_policy(build(), Team.WEST_EAST, {},
            threshold=1, _play_fallback_id="observer_safe_trick_control_v1", _play_contexts={},
            _search_hint=bad_hint, maximum_queries=100, maximum_observations=1000,
            maximum_conflicts=250000, maximum_seconds=15)
    changed = copy.deepcopy(strategy)
    proof = changed["finalBranchProofs"][0]
    retained = changed["refinements"].pop(proof["refinementHash"])
    key = root["observationHash"]
    retained["core"]["playOverrides"].pop(key)
    retained["core"]["bindings"] = [row for row in retained["core"]["bindings"]
                                     if row["observationHash"] != key]
    binding = retained["core"]["fallbackBinding"]
    binding["overridesHash"] = canonical_sha256(retained["core"]["playOverrides"])
    binding["sha256"] = canonical_sha256({k: v for k, v in binding.items() if k != "sha256"})
    terminal = changed["queries"][-1]["receipt"]
    terminal["playPolicyBindings"] = [row for row in terminal["playPolicyBindings"]
                                       if row["observationHash"] != key]
    terminal["sha256"] = canonical_sha256({k: v for k, v in terminal.items() if k != "sha256"})
    retained["queryHashes"][-1] = canonical_sha256(terminal)
    lookup = {canonical_sha256(row["receipt"]): row["receipt"] for row in changed["queries"]}
    identity = canonical_sha256({**retained["core"], "queries": [lookup[value] for value in retained["queryHashes"]]})
    changed["refinements"][identity] = retained
    proof["refinementHash"] = identity
    changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
    with pytest.raises(ValueError, match="ownership or coverage"):
        validate_native_strategy_threshold_v1(changed, expected_scope=strategy["scope"])


def test_native_strategy_threshold_is_not_exact_value(root, monkeypatch):
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.scenario_contracts import canonical_sha256
    from types import SimpleNamespace
    import baloot_platform.research.ortools_information_set_compiler as module
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    actor = Seat.parse(exact.next_actor)
    source = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
        branch=(Contract.SUN, None, actor, False), compact_known_initial_ownership=True)
    source.model.add(source.auction_fragment.choices[0]["sun"] == 1)
    limits = dict(maximum_iterations=10, maximum_queries=100, maximum_observations=1000,
        maximum_branches=13, maximum_conflicts=250000, maximum_seconds=30)
    receipt = compiler._solve_symbolic_round_policy_bounds([source], actor.team, **limits, required_threshold=90)
    assert receipt["kind"] == "native_symbolic_round_strategy_threshold_v1"
    assert receipt["status"] == "scoped_strategy_threshold_complete"
    assert receipt["lowerBound"] == receipt["requiredThreshold"] == 90
    assert receipt["upperBound"] > 90  # Native value is92, not the capped master90.
    assert receipt["scope"]["objective"] == "strategy_threshold_satisfaction"
    assert receipt["strategicCoverage"] is False
    assert len(receipt["finalBranchProofs"]) == 1
    assert receipt["masters"][-1]["searchValue"] == 90
    assert not any(row.get("purpose") == "cooperative_upper_bound" for row in receipt["queries"])
    from copy import deepcopy
    from baloot_platform.research.result_authority import (
        validate_native_strategy_threshold_v1, _validate_native_final_branch_proofs_v2)
    owner = deepcopy(receipt["scope"])
    assert validate_native_strategy_threshold_v1(receipt, expected_scope=owner) == 90
    with pytest.raises(ValueError):
        _validate_native_final_branch_proofs_v2(receipt, target_team=actor.team.name.lower())
    for change in ({"status": "incomplete"}, {"lowerBound": 92}, {"requiredThreshold": True},
                   {"masterSearchValue": 92}, {"upperBound": 90}, {"finalBranchProofs": []},
                   {"masters": []}, {"strategicCoverage": True}, {"schemaVersion": True},
                   {"policy": {"missing": "pass"}}):
        changed = {**deepcopy(receipt), **change}
        changed["sha256"] = canonical_sha256({key: value for key, value in changed.items() if key != "sha256"})
        with pytest.raises(ValueError):
            validate_native_strategy_threshold_v1(changed, expected_scope=owner)
    with pytest.raises(ValueError):
        validate_native_strategy_threshold_v1(receipt, expected_scope={**owner, "rulesHash": "0" * 64})
    for change in ({"targetTeam": []}, {"requiredThreshold": True}, {"utilityBounds": [0, False]}):
        with pytest.raises(ValueError):
            validate_native_strategy_threshold_v1(receipt, expected_scope={**owner, **change})
    # Even rehashed, an absent-outcome query cannot switch threshold or return UNKNOWN.
    for field, value in (("threshold", 89), ("solverStatus", "unknown")):
        changed = deepcopy(receipt)
        query = changed["queries"][-1]["receipt"]
        query[field] = value
        query["sha256"] = canonical_sha256({key: item for key, item in query.items() if key != "sha256"})
        retained = next(iter(changed["refinements"].values()))
        retained["queryHashes"][-1] = canonical_sha256(query)
        identity = canonical_sha256({**retained["core"], "queries": [query]})
        changed["refinements"] = {identity: retained}
        changed["finalBranchProofs"][0]["refinementHash"] = identity
        changed["sha256"] = canonical_sha256({key: item for key, item in changed.items() if key != "sha256"})
        with pytest.raises(ValueError):
            validate_native_strategy_threshold_v1(changed, expected_scope=owner)
    coordinated = compiler._solve_native_bid_policy_bounds(exact, prefix, actor.team,
        **limits, maximum_model_bytes=100000000, visibility="complete_state",
        selected_root_action="sun", required_threshold=90)
    assert coordinated["status"] == "scoped_strategy_threshold_complete"
    assert coordinated["strategy"]["lowerBound"] == 90
    assert coordinated["strategy"]["upperBound"] > 90
    fixed_owner = deepcopy(coordinated["strategy"]["scope"])
    assert validate_native_strategy_threshold_v1(coordinated["strategy"], expected_scope=fixed_owner) == 90
    changed = deepcopy(coordinated["strategy"])
    changed["policy"][fixed_owner["fixedRoot"]["observationHash"]] = "pass"
    changed["sha256"] = canonical_sha256({key: value for key, value in changed.items() if key != "sha256"})
    with pytest.raises(ValueError):
        validate_native_strategy_threshold_v1(changed, expected_scope=fixed_owner)
    for changes in ({"required_threshold": True}, {"required_threshold": 90, "tighten_utility_ceiling": True},
                    {"required_threshold": 90, "incumbent_play_policies": {}}):
        with pytest.raises(ValueError):
            compiler._solve_symbolic_round_policy_bounds([source], actor.team, **limits, **changes)
    with pytest.raises(InformationSetCompilationStopped) as refused:
        compiler._solve_symbolic_round_policy_bounds([source], actor.team, **limits, required_threshold=10**20)
    assert refused.value.reason == "strategy_threshold_out_of_domain"
    # A sufficient guarantee below the utility minimum is valid, not an exact value.
    below = compiler._solve_symbolic_round_policy_bounds(
        [source], actor.team, **limits, required_threshold=-10000)
    assert below["scope"]["utilityBounds"][0] > -10000
    assert validate_native_strategy_threshold_v1(below, expected_scope=below["scope"]) == -10000
    with pytest.raises(InformationSetCompilationStopped):
        compiler._solve_symbolic_round_policy_bounds(
            [source], actor.team, **limits, required_threshold=-(2**63))
    with monkeypatch.context() as patch:
        patch.setattr(module.OrToolsActorInformationSetProofV1, "_solve_master",
            lambda *args, **kwargs: SimpleNamespace(model_hash="test-master", status="optimal", policy={}, value=89))
        missed = compiler._solve_symbolic_round_policy_bounds([source], actor.team, **limits, required_threshold=90)
    assert missed["status"] == "scoped_strategy_threshold_refuted"
    assert missed["upperBound"] == missed["masterSearchValue"] == 89
    assert missed["lowerBound"] is None and missed["strategicCoverage"] is False
    with pytest.raises(ValueError):
        validate_native_strategy_threshold_v1(missed, expected_scope=missed["scope"])
    with monkeypatch.context() as patch:
        patch.setattr(module.OrToolsActorInformationSetProofV1, "_solve_master",
            lambda *args, **kwargs: SimpleNamespace(model_hash="unknown-master", status="unknown", policy={}, value=None))
        with pytest.raises(InformationSetCompilationStopped) as unfinished:
            compiler._solve_symbolic_round_policy_bounds([source], actor.team, **limits, required_threshold=90)
    assert unfinished.value.reason == "master_not_optimal"
    assert unfinished.value.partial_receipt["evidence"]["masterSearchValue"] is None
    assert unfinished.value.partial_receipt["evidence"]["lowerBound"] is None


@pytest.mark.real_clock
def test_native_strategy_threshold_refutation_matches_exact_tail(root):
    import time
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.result_authority import (
        validate_native_strategy_threshold_v1, _reverify_native_strategy_refuted_upper_v1)
    from baloot_platform.research.scenario_contracts import canonical_sha256

    rules, exact, prefix, request, values, order, events = _native_mixed_tail_quality_context(root, play_count=27)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    source = compiler._native_post_auction_root_model(exact, prefix,
        opportunity_order=order, escalation_events=[
            {"actor": row["actor"], "actionCode": row["selectedAction"]} for row in events])
    before = str(source.model.proto)
    threshold = max(values.values()) + 1
    result = compiler._solve_symbolic_round_policy_bounds([source], Team[request.target_team.upper()],
        required_threshold=threshold, maximum_iterations=64, maximum_queries=1000,
        maximum_observations=1000, maximum_branches=1, maximum_conflicts=250000, maximum_seconds=30)
    assert result["status"] == "scoped_strategy_threshold_refuted"
    assert result["upperBound"] == result["masterSearchValue"] == max(values.values())
    assert result["lowerBound"] is None and result["strategicCoverage"] is False
    assert result["cuts"] and result["masters"][-1]["status"] == "optimal"
    assert not result["finalBranchProofs"]
    assert result["sha256"] == canonical_sha256({k: v for k, v in result.items() if k != "sha256"})
    assert str(source.model.proto) == before
    with pytest.raises(ValueError):
        validate_native_strategy_threshold_v1(result, expected_scope=result["scope"])
    checked = []
    context = dict(compiler=compiler, scope=result["scope"], target=Team[request.target_team.upper()],
        upper_bound=threshold - 1, preparation_queries=0, deadline=time.monotonic() + 30,
        maximum_queries=1000, maximum_observations=1000, maximum_conflicts=250000,
        on_queries_used=checked.append)
    assert _reverify_native_strategy_refuted_upper_v1(result, [source], **context) == threshold - 1
    assert len(checked) == 1 and 0 < checked[0] <= 1000
    for key, value in (("status", "incomplete"), ("kind", "native_guarded_round_strategy_threshold_v2"),
                       ("lowerBound", threshold), ("strategicCoverage", True), ("scopeHash", "0" * 64),
                       ("schemaVersion", True), ("masters", [1])):
        invalid = {**result, key: value}
        invalid["sha256"] = canonical_sha256({k: v for k, v in invalid.items() if k != "sha256"})
        with pytest.raises(ValueError):
            _reverify_native_strategy_refuted_upper_v1(invalid, [source], **context)
    # Rehashing a claimed tighter cap is not proof: the fresh solver must reject it.
    forged = copy.deepcopy(result)
    forged["scope"]["requiredThreshold"] -= 1
    forged["scopeHash"] = canonical_sha256(forged["scope"])
    for key in ("requiredThreshold", "upperBound", "masterSearchValue"):
        forged[key] -= 1
    forged["masters"][-1]["searchValue"] -= 1
    forged["sha256"] = canonical_sha256({k: v for k, v in forged.items() if k != "sha256"})
    with pytest.raises(ValueError, match="fresh strategy solving"):
        _reverify_native_strategy_refuted_upper_v1(forged, [source],
            **{**context, "scope": forged["scope"], "upper_bound": threshold - 2})
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    for stopped in ({"cancel_check": lambda: True}, {"deadline": time.monotonic() - 1},
                    {"maximum_queries": 1}):
        with pytest.raises(InformationSetCompilationStopped):
            _reverify_native_strategy_refuted_upper_v1(result, [source], **{**context, **stopped})
    assert str(source.model.proto) == before


@pytest.mark.parametrize("record_model_bytes", (False, True))
def test_native_constant_policy_bounds_unfixed_full_round(root, record_model_bytes):
    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    actor = Seat.parse(exact.next_actor)
    source = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
        branch=(Contract.SUN, None, actor, False))
    source.model.add(source.auction_fragment.choices[0]["pass"] == 1)
    before = str(source.model.proto)
    result = source.find_native_round_with_constant_policy_below(
        target=actor.team.other, policy_team=actor.team.other, auction_action="pass",
        escalation_action="decline", threshold=1000, maximum_seconds=30, maximum_conflicts=250_000,
        record_model_bytes=record_model_bytes)
    assert result["status"] == "queried" and result["strategicCoverage"] is False
    query = result["query"]
    assert query["solverStatus"] == "optimal" and query["candidate"]["utility"] == -92
    assert len(query["candidate"]["play"]["plays"]) == 32
    assert len(query["candidate"]["play"]["tricks"]) == 8
    assert str(source.model.proto) == before
    repeated = source.find_native_round_with_constant_policy_below(
        target=actor.team.other, policy_team=actor.team.other, auction_action="pass",
        escalation_action="decline", threshold=1000, maximum_seconds=30, maximum_conflicts=250_000,
        record_model_bytes=record_model_bytes)
    assert repeated == result
    if record_model_bytes:
        sizes = result["modelBytes"]
        assert sizes["unit"] == "utf8_cp_model_proto" and sizes["includesSolverWorkingMemory"] is False
        assert sizes["source"] == len(before.encode("utf-8"))
        phases = [sizes["source"] * 2, sizes["source"] + query["modelBytes"]["owner"] + query["modelBytes"]["outcomeQuery"]]
        phases.extend(sizes["source"] + row["modelBytes"]["owner"] + row["modelBytes"]["legalityProbe"]
            for row in result["bindings"])
        assert sizes["retainedUpperBound"] == max(phases)
        assert sizes["retainedUpperBound"] > sizes["source"] * 2
    else:
        assert "modelBytes" not in result and "modelBytes" not in query


def test_native_guarded_play_policy_keeps_fallback_and_separate_authority(root, monkeypatch):
    from baloot_platform.research.result_authority import validate_native_visible_policy_threshold_v1
    from baloot_platform.research.scenario_contracts import canonical_sha256
    import baloot_platform.research.ortools_scenario_model as module

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    actor = Seat.parse(exact.next_actor)
    source = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
        branch=(Contract.SUN, None, actor, False))
    source.model.add(source.auction_fragment.choices[0]["sun"] == 1)
    args = dict(target=actor.team, policy_team=actor.team, auction_action=None,
        escalation_action="decline", native_play_policy_id="observer_safe_partner_feed_secure_control_v1",
        threshold=1000, optimize_margin=False, maximum_seconds=30, maximum_conflicts=250000)
    baseline = source.find_native_round_with_constant_policy_below(**args)
    assert baseline["status"] == "queried"
    observation = baseline["query"]["candidate"]["playObservations"][0]
    alternative = next(code for code in observation["legalActions"] if code != observation["selectedAction"])
    source.model.add(source.native_play_observation_literal(information=observation["information"]) == 1)
    original = str(source.model.proto)
    baseline = source.find_native_round_with_constant_policy_below(**args)
    assert source.find_native_round_with_constant_policy_below(**args, native_play_overrides=[]) == baseline
    override = {"information": observation["information"], "action": alternative}
    result = source.find_native_round_with_constant_policy_below(**args, native_play_overrides=[override])
    assert result["status"] == "queried" and not result["strategicCoverage"]
    assert result["kind"] == "native_guarded_play_policy_round_query_v1"
    assert result["policy"]["scope"] == "future_nonplay_constants_and_guarded_native_visible_play"
    plays = result["query"]["candidate"]["playObservations"]
    assert plays[0]["selectedAction"] == alternative
    # The first move alone is replaced. At the actor's next lead the unchanged
    # fallback still selects the Ace which the override conserved.
    following = next(row for row in plays[1:] if row["actor"] == observation["actor"])
    assert following["selectedAction"] == observation["selectedAction"]
    binding = result["policy"]["nativePlayPolicyBinding"]
    assert binding["kind"] == "native_guarded_play_policy_binding_v1"
    assert binding["overridesHash"] == canonical_sha256(result["policy"]["playOverrides"])
    assert result["query"]["baseModelHash"] == result["policy"]["playPolicyModelHash"]
    owner = {key: value for key, value in args.items() if key not in {"threshold", "optimize_margin"}}
    with pytest.raises(ValueError, match="ownership"):
        validate_native_visible_policy_threshold_v1(result, **owner,
            rules=compiler.rules, scenario=source, required_threshold=1)
    floor = source.find_native_round_with_constant_policy_below(
        **{**args, "threshold": 1}, native_play_overrides=[override])
    reader = dict(owner, rules=compiler.rules, scenario=source, required_threshold=1,
        native_play_overrides=[override])
    assert validate_native_visible_policy_threshold_v1(floor, **reader) == 1
    assert not floor["strategicCoverage"]
    for omitted in (None, []):
        with pytest.raises(ValueError, match="ownership"):
            validate_native_visible_policy_threshold_v1(floor, **{**reader, "native_play_overrides": omitted})
    changed = {**override, "action": observation["selectedAction"]}
    with pytest.raises(ValueError, match="evidence mismatch"):
        validate_native_visible_policy_threshold_v1(floor, **{**reader, "native_play_overrides": [changed]})
    with pytest.raises(ValueError, match="explicit sequence"):
        validate_native_visible_policy_threshold_v1(floor, **{**reader, "native_play_overrides": {}})
    with pytest.raises(ValueError, match="reverification"):
        validate_native_visible_policy_threshold_v1(result, **{**reader, "required_threshold": args["threshold"]})
    with pytest.raises(ValueError, match="unique future"):
        source.find_native_round_with_constant_policy_below(**args, native_play_overrides=[override, override])
    wrong_team = next(row for row in plays if Seat.parse(row["actor"]).team != actor.team)
    with pytest.raises(ValueError, match="policy-team"):
        source.find_native_round_with_constant_policy_below(**args,
            native_play_overrides=[{"information": wrong_team["information"], "action": wrong_team["selectedAction"]}])
    illegal = next(code for code in source._native_round_fragment.play.native_legal_action_literals(0)
                   if code not in observation["legalActions"])
    rejected = source.find_native_round_with_constant_policy_below(**args,
        native_play_overrides=[{**override, "action": illegal}])
    # A witness in the relaxed legality universe cannot prove this observation
    # reachable in the owner. It must stop without binding or claiming invalidity.
    assert rejected["status"] == "incomplete" and rejected["query"] is None
    assert rejected["bindings"][-1]["legalityScope"] == "actor_visible_play_prefix_relaxation_v1"
    assert rejected["bindings"][-1]["status"] == "incomplete"
    with monkeypatch.context() as patch:
        patch.setattr(module, "_solve_cp_sat_with_cancel_v1", lambda *_: cp_model.UNKNOWN)
        stopped = source.find_native_round_with_constant_policy_below(**args, native_play_overrides=[override])
        assert stopped["status"] == "incomplete" and stopped["query"] is None
        from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
        with pytest.raises(InformationSetCompilationStopped):
            validate_native_visible_policy_threshold_v1(floor, **reader)
    assert str(source.model.proto) == original and not source._observed_play_bindings


@pytest.mark.real_clock
@pytest.mark.parametrize("record_bytes", (False, True))
def test_native_visible_policy_query_owns_its_clone_and_identity(root, record_bytes, monkeypatch):
    from baloot_platform.research.ortools_play_tail import native_observer_safe_policy_identity_v1
    from baloot_platform.research.scenario_contracts import canonical_sha256

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    actor = Seat.parse(exact.next_actor)
    source = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
        branch=(Contract.SUN, None, actor, False))
    source.model.add(source.auction_fragment.choices[0]["sun"] == 1)
    before = str(source.model.proto)
    args = dict(target=actor.team, policy_team=actor.team, auction_action=None,
        escalation_action="decline", native_play_policy_id="observer_safe_trick_control_v1",
        threshold=1000, optimize_margin=False, maximum_seconds=30, maximum_conflicts=250000,
        record_model_bytes=record_bytes)
    result = source.find_native_round_with_constant_policy_below(**args)
    assert result["status"] == "queried" and not result["strategicCoverage"]
    assert result["query"]["status"] == "witness_found"
    assert len(result["query"]["candidate"]["play"]["plays"]) == 32
    policy = result["policy"]
    assert policy["scope"] == "future_nonplay_constants_and_native_visible_play"
    assert policy["nativePlayPolicy"] == native_observer_safe_policy_identity_v1(args["native_play_policy_id"], actor.team)
    binding = policy["nativePlayPolicyBinding"]
    assert binding["policy"] == policy["nativePlayPolicy"]
    assert binding["sourceCpModelProtoHash"] != binding["boundCpModelProtoHash"]
    assert binding["sha256"] == canonical_sha256({k: v for k, v in binding.items() if k != "sha256"})
    assert result["query"]["baseModelHash"] == policy["playPolicyModelHash"]
    paired = source.find_native_round_with_constant_policy_below(**args,
        other_native_play_policy_id="observer_safe_partner_feed_v1")
    assert paired["status"] == "queried" and not paired["strategicCoverage"]
    assert len(paired["query"]["candidate"]["play"]["plays"]) == 32
    pair_policy = paired["policy"]
    assert pair_policy["scope"] == "future_nonplay_constants_and_paired_native_visible_play"
    assert pair_policy["otherNativePlayPolicy"] == native_observer_safe_policy_identity_v1(
        "observer_safe_partner_feed_v1", actor.team.other)
    other_binding = pair_policy["otherNativePlayPolicyBinding"]
    assert other_binding["sourceCpModelProtoHash"] == pair_policy["nativePlayPolicyBinding"]["boundCpModelProtoHash"]
    assert other_binding["sha256"] == canonical_sha256({k: v for k, v in other_binding.items() if k != "sha256"})
    assert pair_policy["playPolicyModelHash"] == paired["query"]["baseModelHash"]
    if record_bytes:
        assert result["modelBytes"]["retainedUpperBound"] >= len(before.encode()) + policy["nativePlayPolicyModelBytes"]
    else:
        from baloot_platform.research.result_authority import validate_native_visible_policy_threshold_v1
        from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
        floor = source.find_native_round_with_constant_policy_below(**{**args, "threshold": 1})
        owner = {key: value for key, value in args.items() if key not in {"threshold", "optimize_margin"}}
        owner.update(rules=compiler.rules, scenario=source, required_threshold=1)
        with pytest.raises(ValueError, match="ownership"):
            validate_native_visible_policy_threshold_v1(paired, **owner)
        assert validate_native_visible_policy_threshold_v1(floor, **owner) == 1
        resealed_pair = source.find_native_round_with_constant_policy_below(
            **{**args, "threshold": 1}, other_native_play_policy_id="observer_safe_partner_feed_v1")
        resealed_pair["policy"]["scope"] = "future_nonplay_constants_and_native_visible_play"
        resealed_pair["policy"].pop("otherNativePlayPolicy")
        resealed_pair["policy"].pop("otherNativePlayPolicyBinding")
        resealed_pair["sha256"] = canonical_sha256({k: v for k, v in resealed_pair.items() if k != "sha256"})
        with pytest.raises(ValueError, match="evidence mismatch"):
            validate_native_visible_policy_threshold_v1(resealed_pair, **owner)
        with pytest.raises(ValueError, match="ownership"):
            validate_native_visible_policy_threshold_v1(floor, **{**owner, "policy_team": actor.team.other})
        with pytest.raises(InformationSetCompilationStopped):
            validate_native_visible_policy_threshold_v1(floor, **{**owner, "cancel_check": lambda: True})
        # Real producer and fresh solve above; isolate resealed substitution checks.
        with monkeypatch.context() as patch:
            patch.setattr(OrToolsScenarioModelV1, "find_native_round_with_constant_policy_below", lambda *a, **kw: floor)
            for field, value in (("team", "west_east"), ("escalationAction", None),
                                 ("nativePlayPolicy", {}), ("playPolicyModelHash", "0" * 64),
                                 ("nativePlayPolicyBinding", {})):
                altered = copy.deepcopy(floor)
                altered["policy"][field] = value
                altered["sha256"] = canonical_sha256({k: v for k, v in altered.items() if k != "sha256"})
                with pytest.raises(ValueError, match="evidence mismatch"):
                    validate_native_visible_policy_threshold_v1(altered, **owner)
            for field, value in (("solverStatus", "unknown"), ("status", "witness_found")):
                incomplete = copy.deepcopy(floor)
                incomplete["query"][field] = value
                patch.setattr(OrToolsScenarioModelV1, "find_native_round_with_constant_policy_below", lambda *a, **kw: incomplete)
                with pytest.raises(ValueError, match="reverification"):
                    validate_native_visible_policy_threshold_v1(floor, **owner)
    for changes in ({"cancel_check": lambda: True}, {"maximum_seconds": 1e-12}):
        stopped = source.find_native_round_with_constant_policy_below(**{**args, **changes})
        assert stopped["status"] == "incomplete" and stopped["query"] is None
    for changes in ({"native_play_policy_id": True}, {"native_play_policy_id": "unknown"},
                    {"play_policies": {}}, {"escalation_observations": []},
                    {"other_native_play_policy_id": "unknown"},
                    {"native_play_policy_id": None, "other_native_play_policy_id": "observer_safe_partner_feed_v1"}):
        with pytest.raises(ValueError):
            source.find_native_round_with_constant_policy_below(**{**args, **changes})
    assert str(source.model.proto) == before
    assert source._native_round_fragment.play.model is source.model


def _native_ranked_play_policies(team):
    from baloot_platform.research.scenario_contracts import canonical_sha256

    policies = {}
    for seat in Seat:
        if seat.team == team:
            core = dict(schemaVersion=1, seat=seat.name.lower(), policyId="fixed_action_map_v1",
                version="1.0.0", seed=0, choices=[], materialization="seeded_legal_action_order_v1")
            policies[seat.name.lower()] = {key: value for key, value in core.items() if key != "seat"}
            policies[seat.name.lower()]["profileHash"] = canonical_sha256(core)
    return policies


@pytest.mark.real_clock
def test_native_ranked_target_policy_leaves_opponents_unrestricted(root, monkeypatch):
    from baloot_platform.core.models import LegalPlayActionV1
    from baloot_platform.research.scenario_contracts import canonical_sha256
    from baloot_platform.research.ortools_proof import (
        DeclarativeActionPolicyV1, ProofGraphEdgeV1, select_declarative_policy_action_v1)

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    actor = Seat.parse(exact.next_actor)
    source = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
        branch=(Contract.SUN, None, actor, False))
    source.model.add(source.auction_fragment.choices[0]["sun"] == 1)
    policies = _native_ranked_play_policies(actor.team)
    args = dict(target=actor.team, policy_team=actor.team, auction_action=None,
        escalation_action=None, play_policies=policies, threshold=1000,
        maximum_seconds=30, maximum_conflicts=250000)
    before = str(source.model.proto)
    result = source.find_native_round_with_constant_policy_below(**args)
    assert result["status"] == "queried" and not result["strategicCoverage"]
    query = result["query"]
    assert query["solverStatus"] == "optimal"
    refinement = dict(threshold=1000, maximum_queries=3, maximum_observations=100,
        maximum_conflicts=250000, maximum_seconds=30)
    hinted = compiler._refine_symbolic_round_policy(source, actor.team, {},
        **refinement, observation_hint=result)
    assert hinted["status"] == "policy_extension_required"
    assert hinted["observationHintHash"] == result["sha256"]
    assert len(hinted["queries"]) == 1 and not hinted["bindings"]
    assert hinted["queries"][0]["baseModelHash"] == result["sourceModelHash"]
    assert hinted["queries"][0]["baseModelHash"] != query["baseModelHash"]
    assert not hinted["strategicCoverage"]
    for field, value in (("sourceModelHash", "0" * 64), ("targetTeam", "west_east")):
        tampered_hint = {**result, field: value}
        tampered_hint["sha256"] = canonical_sha256({k: v for k, v in tampered_hint.items() if k != "sha256"})
        with pytest.raises(ValueError, match="hint ownership or checksum"):
            compiler._refine_symbolic_round_policy(source, actor.team, {},
                **refinement, observation_hint=tampered_hint)
    tampered_hint = copy.deepcopy(result)
    tampered_hint["query"]["candidate"]["utility"] += 1
    tampered_hint["sha256"] = canonical_sha256({k: v for k, v in tampered_hint.items() if k != "sha256"})
    with pytest.raises(ValueError, match="hint ownership or checksum"):
        compiler._refine_symbolic_round_policy(source, actor.team, {},
            **refinement, observation_hint=tampered_hint)
    rows = query["candidate"]["playObservations"]
    assert len(rows) == 32
    for row in rows:
        if row["actor"] not in policies:
            continue
        actions = [LegalPlayActionV1.from_code(code) for code in row["legalActions"]]
        edges = [ProofGraphEdgeV1.build(action.card.id + 32 * int(action.ekka),
            action.code, "reference") for action in actions]
        selected = select_declarative_policy_action_v1(
            DeclarativeActionPolicyV1.build(row["actor"], policies[row["actor"]]), edges)
        assert row["selectedAction"] == selected.action_code
    assert str(source.model.proto) == before and not source._observed_play_bindings
    other = next(row for row in rows if row["actor"] not in policies and len(row["legalActions"]) > 1)
    alternate = LegalPlayActionV1.from_code(next(code for code in other["legalActions"]
        if code != other["selectedAction"]))
    owned = copy.copy(source)
    owned.model = source.model.clone()
    slot = owned._native_round_fragment.play.play_state_context.play_slots[other["playIndex"]]
    owned.model.add(owned._native_round_fragment.play._extract.actors[other["playIndex"]]
                    == int(Seat.parse(other["actor"])))
    owned.model.add(slot.card == alternate.card.id)
    owned.model.add(slot.ekka == int(alternate.ekka))
    assert owned.find_native_round_with_constant_policy_below(**args)["status"] == "queried"
    assert str(source.model.proto) == before
    with pytest.raises(ValueError, match="exactly the policy team"):
        source.find_native_round_with_constant_policy_below(**{**args, "play_policies": {}})
    tampered = copy.deepcopy(policies)
    tampered[actor.name.lower()]["seed"] += 1
    with pytest.raises(ValueError, match="profileHash"):
        source.find_native_round_with_constant_policy_below(**{**args, "play_policies": tampered})
    stopped = source.find_native_round_with_constant_policy_below(**args, cancel_check=lambda: True)
    assert stopped["status"] == "incomplete" and stopped["query"] is None
    import baloot_platform.research.ortools_scenario_model as module
    with monkeypatch.context() as patch:
        patch.setattr(module, "_solve_cp_sat_with_cancel_v1", lambda *_: cp_model.UNKNOWN)
        unknown = source.find_native_round_with_constant_policy_below(**args)
    assert unknown["status"] == "incomplete" and unknown["query"]["candidate"] is None
    assert str(source.model.proto) == before and not source._observed_play_bindings


def test_native_ranked_policy_hidden_worlds_and_authored_play(root):
    from baloot_platform.core.models import LegalPlayActionV1

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    actor = Seat.parse(exact.next_actor)
    source = compiler._build_native_bid_branch(exact, prefix, visibility="actor_information_set",
        branch=(Contract.SUN, None, actor, False))
    source.model.add(source.auction_fragment.choices[0]["sun"] == 1)
    assert source._complete_information_deck is None
    original = str(source.model.proto)
    layout = DealLayoutV1.build(source.rules.structural_domain, dealer=Seat.EAST, buyer=actor)
    left, right = (layout.positions_for(seat, final=True)[0] for seat in (Seat.WEST, Seat.EAST))
    first_actions = []
    args = dict(target=actor.team, policy_team=actor.team, auction_action=None,
        escalation_action=None, play_policies=_native_ranked_play_policies(actor.team),
        threshold=1000, maximum_seconds=30, maximum_conflicts=250000)
    for swap in (False, True):
        owned = copy.copy(source)
        owned.model = source.model.clone()
        deck = list(prefix.deck_order)
        if swap:
            deck[left], deck[right] = deck[right], deck[left]
            for seat in (Seat.WEST, Seat.EAST):
                positions = layout.positions_for(seat, final=False)
                for position, card in zip(positions, sorted(deck[index] for index in positions), strict=True):
                    deck[position] = card
        for variable, card in zip(owned.card_at_position, deck, strict=True):
            owned.model.add(variable == card)
        result = owned.find_native_round_with_constant_policy_below(**args)
        assert result["status"] == "queried" and result["query"]["solverStatus"] == "optimal", swap
        first_actions.append(result["query"]["candidate"]["playObservations"][0])
    assert first_actions[0]["observationHash"] == first_actions[1]["observationHash"]
    assert first_actions[0]["selectedAction"] == first_actions[1]["selectedAction"]
    # An authored action stays fixed even when the current policy ranks another first.
    first = first_actions[-1]
    alternative = LegalPlayActionV1.from_code(next(code for code in first["legalActions"]
        if code != first["selectedAction"]))
    owned._recorded_play_count = 1
    slot = owned._native_round_fragment.play.play_state_context.play_slots[0]
    owned.model.add(slot.card == alternative.card.id)
    owned.model.add(slot.ekka == int(alternative.ekka))
    prefixed = owned.find_native_round_with_constant_policy_below(**args)
    assert prefixed["status"] == "queried"
    assert prefixed["query"]["candidate"]["playObservations"][0]["selectedAction"] == alternative.code
    assert prefixed["policy"]["playPrefixLength"] == 1
    assert str(source.model.proto) == original and not source._observed_play_bindings


def test_native_opponent_ceiling_uses_owned_policy_without_restricting_adversary(root, monkeypatch):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    compiler, _, prefix, _ = _native_suit_monopoly_bid(root)
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["deckOrder"] = [{0: 8, 8: 0}.get(card, card) for card in raw["deckOrder"]]
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    source = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
        branch=(Contract.SUN, None, Seat.NORTH, False), compact_known_initial_ownership=True)
    source.model.add(source.auction_fragment.choices[0]["sun"] == 1)
    before = str(source.model.proto)
    policy = dict(auction_action=None, escalation_action=None,
        native_play_policy_id="observer_safe_trick_control_v1")
    limits = dict(maximum_iterations=1, maximum_queries=1, maximum_observations=100,
        maximum_branches=1, maximum_conflicts=250000, maximum_seconds=30,
        tighten_utility_ceiling=True, opponent_ceiling_policy=policy)
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._solve_symbolic_round_policy_bounds([source], Team.NORTH_SOUTH, **limits)
    assert stopped.value.reason == "maximum_queries"
    evidence = stopped.value.partial_receipt["evidence"]
    assert evidence["status"] == "incomplete" and evidence["strategicCoverage"] is False
    assert evidence["lowerBound"] is None and evidence["upperBound"] == 58
    assert evidence["queriesUsed"] == 1 and evidence["masters"] == []
    assert evidence["scope"]["opponentCeilingPolicy"] == policy
    assert len(evidence["queries"]) == 1  # No redundant cooperative ceiling query.
    query = evidence["queries"][0]
    assert query["purpose"] == "opponent_policy_upper_bound"
    assert query["branchModelHash"] == source._model_hash(search_branching="automatic")
    rebuilt = source.find_native_round_with_constant_policy_below(
        target=Team.WEST_EAST, policy_team=Team.WEST_EAST, **policy,
        threshold=query["receipt"]["threshold"], maximum_seconds=30, maximum_conflicts=250000)
    assert query["receipt"] == rebuilt
    assert rebuilt["query"]["solverStatus"] == "optimal"
    assert rebuilt["query"]["candidate"]["utility"] == -58
    assert rebuilt["policy"]["nativePlayPolicyBinding"]["policy"]["team"] == "west_east"
    assert str(source.model.proto) == before
    assert not source._observed_play_bindings
    for status in ("feasible", "unknown"):
        altered = copy.deepcopy(rebuilt)
        altered["query"]["solverStatus"] = status
        with monkeypatch.context() as patch:
            patch.setattr(OrToolsScenarioModelV1, "find_native_round_with_constant_policy_below",
                lambda *args, **kwargs: altered)
            with pytest.raises(InformationSetCompilationStopped) as incomplete:
                compiler._solve_symbolic_round_policy_bounds([source], Team.NORTH_SOUTH, **limits)
        unproved = incomplete.value.partial_receipt["evidence"]
        assert incomplete.value.reason == "opponent_ceiling_not_certified"
        assert unproved["upperBound"] > 58
        assert "opponentCeilingPolicy" not in unproved["scope"]
        assert unproved["strategicCoverage"] is False
    # The skipped cooperative solve must not allow a vacuous exact conclusion.
    source.model.add(False)
    with pytest.raises(ValueError, match="native opponent policy feasibility disagreement"):
        compiler._solve_symbolic_round_policy_bounds([source], Team.NORTH_SOUTH, **limits)
    for invalid in (None, True, "unregistered"):
        invalid_policy = {**policy, "native_play_policy_id": invalid}
        with pytest.raises(ValueError):
            compiler._solve_symbolic_round_policy_bounds([source], Team.NORTH_SOUTH,
                **{**limits, "opponent_ceiling_policy": invalid_policy})
        with monkeypatch.context() as patch:
            patch.setattr(compiler, "_prepare_native_bid_branches",
                lambda *args, **kwargs: pytest.fail("invalid policy reached construction"))
            with pytest.raises(ValueError):
                compiler._solve_native_bid_policy_bounds(exact, prefix, Team.NORTH_SOUTH,
                    visibility="complete_state", selected_root_action="sun", maximum_iterations=1,
                    maximum_queries=10, maximum_observations=100, maximum_branches=13,
                    maximum_model_bytes=100000000, maximum_conflicts=250000, maximum_seconds=30,
                    opponent_ceiling_policy=invalid_policy)


def test_native_incumbent_threshold_checks_cheapest_branch_before_hash_order(root, monkeypatch):
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1)

    rules, expensive, _, _ = _native_hokum_round_branch(root, "normal")
    cheap = OrToolsScenarioModelV1(rules=rules, dealer=Seat.NORTH)
    cheap.add_native_auction_fragment(prefix_length=0, allow_all_passed=True)
    for index, variable in enumerate(cheap.card_at_position):
        cheap.model.add(variable == index)
    cheap.add_native_redeal_continuation(match_score=(0, 0))
    # Isolate scheduling from arbitrary content-hash order; this is not admission.
    monkeypatch.setattr(expensive, "_model_hash", lambda **kwargs: "a" * 64)
    monkeypatch.setattr(cheap, "_model_hash", lambda **kwargs: "f" * 64)

    def unnecessary(**kwargs):
        raise AssertionError("expensive play was queried before the native zero-score branch")

    monkeypatch.setattr(expensive, "find_native_round_with_constant_policy_below", unnecessary)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._solve_symbolic_round_policy_bounds([expensive, cheap], Team.NORTH_SOUTH,
            maximum_iterations=1, maximum_queries=2, maximum_observations=100,
            maximum_branches=2, maximum_conflicts=100_000, maximum_seconds=5,
            incumbent_only=True, incumbent_threshold=1,
            incumbent_native_play_policy_id="observer_safe_partner_feed_compact_v1")
    assert stopped.value.reason == "incumbent_threshold_refuted"
    evidence = stopped.value.partial_receipt["evidence"]
    assert evidence["queriesUsed"] == 1
    assert evidence["queries"][0]["branchModelHash"] == "f" * 64
    assert evidence["queries"][0]["receipt"]["query"]["candidate"]["utility"] == 0
    assert evidence["strategicCoverage"] is False


@pytest.mark.real_clock
@pytest.mark.parametrize("policy_id,expected_floor,all_stage", [
    ("observer_safe_trick_control_v1", 54, False), ("observer_safe_partner_feed_v1", 58, False),
    ("observer_safe_partner_feed_compact_v1", 58, False),
    ("observer_safe_partner_feed_compact_v1", 58, True),
])
def test_native_tactical_incumbent_floor_is_owned_and_not_optimality(root, monkeypatch, policy_id, expected_floor, all_stage):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, canonical_sha256
    from baloot_platform.research.result_authority import (
        validate_native_incumbent_lower_bound_v2, validate_native_incumbent_threshold_floor_v1)

    compiler, _, prefix, _ = _native_suit_monopoly_bid(root)
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["deckOrder"] = [{0: 8, 8: 0}.get(card, card) for card in raw["deckOrder"]]
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    source = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
        branch=(Contract.SUN, None, Seat.NORTH, False), compact_known_initial_ownership=True)
    source.model.add(source.auction_fragment.choices[0]["sun"] == 1)
    nonplay = {"schemaVersion": 1, "auctionAction": "pass", "escalationAction": "decline",
        "projectPolicyId": "full_hand_project_max_value_v1"} if all_stage else None
    if all_stage:
        source.constrain_native_auction_root_choice("sun", actor=Seat.NORTH)
    before = str(source.model.proto)
    progress = []
    args = dict(maximum_iterations=1, maximum_queries=3 if all_stage else 1, maximum_observations=100,
        maximum_branches=1, maximum_conflicts=250000, maximum_seconds=30,
        incumbent_native_play_policy_id=policy_id, incumbent_only=True, progress_callback=progress.append)
    if all_stage:
        args["incumbent_nonplay_policy"] = nonplay
    receipt = compiler._solve_symbolic_round_policy_bounds([source], Team.NORTH_SOUTH, **args)
    owner = dict(target_team="north_south", native_play_policy_id=policy_id,
        rules_hash=compiler.rules.sha256, source_model_hashes=(source._model_hash(search_branching="automatic"),),
        source_models=(source,))
    if all_stage:
        owner["nonplay_policy"] = nonplay
    assert receipt["status"] == "scoped_incumbent_bound_complete"
    assert progress == [{"kind": "quality_solver_progress_v1", "phase": "preparation",
        "masterCount": 0, "queryCount": 1, "bindingCount": 2 if all_stage else 0, "solverStatus": "optimal"}]
    assert receipt["strategicCoverage"] is False and receipt["masters"] == []
    assert "incumbentPlayPoliciesHash" not in receipt["scope"]
    floor = validate_native_incumbent_lower_bound_v2(receipt, **owner)
    assert floor == expected_floor
    if all_stage:
        assert receipt["queriesUsed"] == 3 and receipt["bindingQueriesUsed"] == 2
        with pytest.raises(ValueError, match="ownership or coverage"):
            validate_native_incumbent_lower_bound_v2(receipt, **{**owner, "nonplay_policy": None})
        for field, replacement in (("kind", "wrong"), ("schemaVersion", True),
                                   ("policyId", "known_project_max_value_v1"),
                                   ("sourceCpModelProtoHash", "0" * 64)):
            changed = copy.deepcopy(receipt)
            row = changed["queries"][0]["receipt"]
            project_binding = row["policy"]["nativeProjectPolicyBinding"]
            project_binding[field] = replacement
            for document in (project_binding, row, changed):
                document["sha256"] = canonical_sha256({k: v for k, v in document.items() if k != "sha256"})
            with pytest.raises(ValueError, match="ownership or coverage"):
                validate_native_incumbent_lower_bound_v2(changed, **owner)
        from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
        with pytest.raises(InformationSetCompilationStopped) as exhausted:
            compiler._solve_symbolic_round_policy_bounds([source], Team.NORTH_SOUTH,
                **{**args, "maximum_queries": 2})
        assert exhausted.value.reason == "maximum_queries"
        assert exhausted.value.partial_receipt["evidence"]["queriesUsed"] == 0
    if policy_id == "observer_safe_partner_feed_v1":
        defensive = compiler._solve_symbolic_round_policy_bounds([source], Team.WEST_EAST,
            **{**args, "incumbent_native_play_policy_id": "observer_safe_trick_control_v1"})
        assert validate_native_incumbent_lower_bound_v2(defensive, **{
            **owner, "target_team": "west_east", "native_play_policy_id": "observer_safe_trick_control_v1",
        }) == -floor
    assert receipt["upperBound"] > floor  # A guaranteed floor is not an exact minimax value.
    threshold = compiler._solve_symbolic_round_policy_bounds([source], Team.NORTH_SOUTH,
        **args, incumbent_threshold=floor)
    assert validate_native_incumbent_threshold_floor_v1(threshold, **owner) == floor
    from baloot_platform.research.result_authority import validate_native_visible_policy_threshold_v1
    assert validate_native_visible_policy_threshold_v1(threshold["queries"][0]["receipt"],
        rules=compiler.rules, scenario=source, target=Team.NORTH_SOUTH, policy_team=Team.NORTH_SOUTH,
        native_play_policy_id=policy_id,
        auction_action=nonplay["auctionAction"] if nonplay else None,
        escalation_action=nonplay["escalationAction"] if nonplay else None,
        **({"native_project_policy_id": nonplay["projectPolicyId"], "record_policy_model_hashes": True} if nonplay else {}),
        required_threshold=floor, maximum_seconds=30, maximum_conflicts=250000) == floor
    assert str(source.model.proto) == before
    for override in ({"incumbent_native_play_policy_id": "unregistered"},
                     {"incumbent_play_policies": _native_ranked_play_policies(Team.NORTH_SOUTH)},
                     {"required_threshold": floor}):
        with pytest.raises(ValueError):
            compiler._solve_symbolic_round_policy_bounds([source], Team.NORTH_SOUTH, **{**args, **override})
    for field, value in (("policy", {}), ("sourceCpModelProtoHash", "bad"), ("kind", "wrong"),
                         ("sourceCpModelProtoHash", "0" * 64), ("boundCpModelProtoHash", "0" * 64)):
        changed = copy.deepcopy(receipt)
        query = changed["queries"][0]["receipt"]
        binding = query["policy"]["nativePlayPolicyBinding"]
        binding[field] = value
        for document in (binding, query, changed):
            document["sha256"] = canonical_sha256({k: v for k, v in document.items() if k != "sha256"})
        with pytest.raises(ValueError, match="ownership or coverage"):
            validate_native_incumbent_lower_bound_v2(changed, **owner)
    with pytest.raises(ValueError):
        validate_native_incumbent_lower_bound_v2(receipt,
            **{**owner, "native_play_policy_id": "observer_safe_tactical_category_v1"})
    with pytest.raises(ValueError):
        validate_native_incumbent_lower_bound_v2(receipt, **{**owner, "source_models": None})
    with pytest.raises(ValueError):
        validate_native_incumbent_lower_bound_v2(receipt,
            **{**owner, "native_play_policy_id": None,
               "play_policies": _native_ranked_play_policies(Team.NORTH_SOUTH)})
    all_pass = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state", branch=None)
    pass_receipt = compiler._solve_symbolic_round_policy_bounds([all_pass], Team.NORTH_SOUTH, **args)
    pass_owner = {**owner, "source_models": (all_pass,),
        "source_model_hashes": (all_pass._model_hash(search_branching="automatic"),)}
    assert validate_native_incumbent_lower_bound_v2(pass_receipt, **pass_owner) == 0
    assert "nativePlayPolicyBinding" not in pass_receipt["queries"][0]["receipt"]["policy"]
    pass_threshold = compiler._solve_symbolic_round_policy_bounds([all_pass], Team.NORTH_SOUTH,
        **args, incumbent_threshold=0)
    assert validate_native_incumbent_threshold_floor_v1(pass_threshold, **pass_owner) == 0
    if all_stage:
        # A policy-excluded purchased branch is not an impossible scenario.
        excluded = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
            branch=(Contract.SUN, None, Seat.NORTH, False), compact_known_initial_ownership=True)
        excluded.model.add(excluded.auction_fragment.buyer == int(Seat.NORTH))
        excluded.constrain_native_auction_root_choice("pass", actor=Seat.NORTH)
        mixed = compiler._solve_symbolic_round_policy_bounds([excluded, all_pass], Team.NORTH_SOUTH,
            **{**args, "maximum_queries": 6, "maximum_branches": 2})
        hashes = tuple(sorted(model._model_hash(search_branching="automatic") for model in (excluded, all_pass)))
        assert validate_native_incumbent_lower_bound_v2(mixed,
            **{**owner, "source_models": (excluded, all_pass), "source_model_hashes": hashes}) == 0
        assert mixed["status"] == "scoped_incumbent_bound_complete" and mixed["strategicCoverage"] is False
        assert any(row["receipt"]["query"]["solverStatus"] == "infeasible" for row in mixed["queries"])
    # Real root preparation must cover every partition before a floor can be
    # considered an opening-action bound; a supplied branch alone is insufficient.
    opening_args = {**args, "maximum_queries": 100, "maximum_branches": 13,
        "maximum_model_bytes": 100000000, "maximum_depth": 64,
        "visibility": "complete_state", "selected_root_action": "sun"}
    opening = compiler._solve_native_bid_policy_bounds(exact, prefix, Team.NORTH_SOUTH,
        **opening_args, incumbent_threshold=floor, compact_purchase_invariant_ownership=True)
    assert opening["status"] == "scoped_incumbent_threshold_complete"
    assert opening["strategicCoverage"] is False and "rootActionValues" not in opening
    assert opening["preparation"]["constructionComplete"] is True
    assert opening["preparation"]["scope"]["ownershipCompaction"] == "purchase_recipient_invariant_v1"
    assert len(opening["preparation"]["branches"]) == len(compiler._native_bid_branch_specs(Seat.EAST))
    assert all(row["retained"] or row["reachability"]["solverStatus"] == "infeasible"
               for row in opening["preparation"]["branches"])
    assert opening["strategy"]["scope"]["incumbentNativePlayPolicy"] == receipt["scope"]["incumbentNativePlayPolicy"]
    assert opening["queriesUsed"] == opening["preparation"]["queriesUsed"] + opening["strategy"]["queriesUsed"]
    assert opening["queriesUsed"] <= opening_args["maximum_queries"]
    import yaml
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1
    from baloot_platform.research.result_authority import reverify_native_bid_incumbent_floor_v1
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    profile = "offline_heavyweight_v1"
    budget = ScenarioProofBudgetV1.build(profile, settings["scenarioResearchV4"]["proofBudgets"]["profiles"][profile])
    verification = dict(rules=compiler.rules, source_prefix=prefix, normalized=exact,
        native_play_policy_id=policy_id, budget=budget, maximum_conflicts=250000, maximum_seconds=30)
    if all_stage:
        verification["nonplay_policy"] = nonplay
    assert reverify_native_bid_incumbent_floor_v1(opening, **verification) == floor
    changed = copy.deepcopy(opening)
    changed["strategy"]["lowerBound"] += 1
    for document in (changed["strategy"], changed):
        document["sha256"] = canonical_sha256({k: v for k, v in document.items() if k != "sha256"})
    with pytest.raises(ValueError, match="fresh reconstruction"):
        reverify_native_bid_incumbent_floor_v1(changed, **verification)
    with pytest.raises(InformationSetCompilationStopped) as cancelled:
        reverify_native_bid_incumbent_floor_v1(opening, **verification, cancel_check=lambda: True)
    assert cancelled.value.reason == "cancelled"
    with pytest.raises(ValueError, match="ownership or budget"):
        reverify_native_bid_incumbent_floor_v1(opening, **{**verification, "maximum_conflicts": 1})
    # Receipt validation consumes the same deadline as reconstruction.
    import time
    import baloot_platform.research.result_authority as authority
    original_semantics = authority._native_observed_bounds_semantics
    with monkeypatch.context() as patch:
        def slow_semantics(*args, **kwargs):
            result = original_semantics(*args, **kwargs)
            time.sleep(0.02)
            return result
        patch.setattr(authority, "_native_observed_bounds_semantics", slow_semantics)
        patch.setattr(OrToolsExactStateNormalizerV1, "normalize",
            lambda *args, **kwargs: pytest.fail("expired verification entered normalization"))
        with pytest.raises(InformationSetCompilationStopped) as expired:
            reverify_native_bid_incumbent_floor_v1(opening, **{**verification, "maximum_seconds": 0.001})
        assert expired.value.reason == "maximum_seconds"
    with monkeypatch.context() as patch:
        def unexpected(*args, **kwargs):
            pytest.fail("Invalid policy selection must fail before construction")
        patch.setattr(compiler, "_prepare_native_bid_branches", unexpected)
        for override in ({"incumbent_native_play_policy_id": "unregistered"},
                         {"incumbent_play_policies": _native_ranked_play_policies(Team.NORTH_SOUTH)},
                         {"required_threshold": floor}, {"enumerate_root_actions": True}):
            with pytest.raises(ValueError):
                compiler._solve_native_bid_policy_bounds(exact, prefix, Team.NORTH_SOUTH,
                    **{**opening_args, **override})


def test_native_incumbent_lower_bound_survives_incomplete_search(root, monkeypatch):
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.scenario_contracts import canonical_sha256

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    actor = Seat.parse(exact.next_actor)
    source = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
        branch=(Contract.SUN, None, actor, False))
    source.model.add(source.auction_fragment.choices[0]["sun"] == 1)
    policies = _native_ranked_play_policies(actor.team)
    args = dict(maximum_iterations=1, maximum_queries=1, maximum_observations=100,
        maximum_branches=2, maximum_conflicts=250000, maximum_seconds=30,
        incumbent_play_policies=policies)
    before = str(source.model.proto)
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._solve_symbolic_round_policy_bounds([source], actor.team, **args)
    assert stopped.value.reason == "maximum_queries"
    receipt = stopped.value.partial_receipt["evidence"]
    assert receipt["status"] == "incomplete" and not receipt["strategicCoverage"]
    assert receipt["lowerBound"] == receipt["queries"][0]["receipt"]["query"]["candidate"]["utility"]
    assert receipt["scope"]["incumbentPlayPoliciesHash"] == canonical_sha256(policies)
    assert receipt["queries"][0]["purpose"] == "incumbent_lower_bound"
    from baloot_platform.research.result_authority import validate_native_incumbent_lower_bound_v2
    assert validate_native_incumbent_lower_bound_v2(receipt,
        target_team=actor.team.name.lower(), play_policies=policies, rules_hash=source.rules.sha256,
        source_model_hashes=(source._model_hash(search_branching="automatic"),)) == receipt["lowerBound"]
    assert str(source.model.proto) == before
    second = copy.copy(source)
    threshold_args = {**args, "incumbent_only": True, "incumbent_threshold": receipt["lowerBound"]}
    threshold_proof = compiler._solve_symbolic_round_policy_bounds([source], actor.team, **threshold_args)
    assert threshold_proof["kind"] == "native_incumbent_threshold_floor_v1"
    assert threshold_proof["status"] == "scoped_incumbent_threshold_complete"
    assert threshold_proof["lowerBound"] == receipt["lowerBound"]
    assert not threshold_proof["strategicCoverage"]
    assert threshold_proof["queries"][0]["purpose"] == "incumbent_threshold_floor"
    assert threshold_proof["queries"][0]["receipt"]["query"]["solverStatus"] == "infeasible"
    from baloot_platform.research.result_authority import validate_native_incumbent_threshold_floor_v1
    threshold_owner = dict(target_team=actor.team.name.lower(), play_policies=policies,
        rules_hash=source.rules.sha256, source_model_hashes=(source._model_hash(search_branching="automatic"),))
    assert validate_native_incumbent_threshold_floor_v1(threshold_proof, **threshold_owner) == receipt["lowerBound"]
    for field in ("kind", "coverage", "utility"):
        changed = copy.deepcopy(threshold_proof)
        changed["scope"][field] = "unsupported"
        changed["scopeHash"] = canonical_sha256(changed["scope"])
        changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
        with pytest.raises(ValueError, match="ownership or coverage"):
            validate_native_incumbent_threshold_floor_v1(changed, **threshold_owner)
    with monkeypatch.context() as patch:
        patch.setattr(OrToolsScenarioModelV1, "find_native_round_with_constant_policy_below",
            lambda *a, **kw: copy.deepcopy(threshold_proof["queries"][0]["receipt"]))
        patch.setattr(compiler, "_prepare_native_bid_branches", lambda *a, **kw: ([source], {"queriesUsed": 0}))
        orchestrated = compiler._solve_native_bid_policy_bounds(exact, prefix, actor.team,
            visibility="complete_state", selected_root_action="sun", maximum_model_bytes=100000000,
            **threshold_args)
    assert orchestrated["status"] == "scoped_incumbent_threshold_complete"
    assert "cooperativeCeiling" not in orchestrated["strategy"]["scope"]
    assert validate_native_incumbent_threshold_floor_v1(orchestrated["strategy"], **threshold_owner) == receipt["lowerBound"]
    with monkeypatch.context() as patch:
        patch.setattr(compiler, "_prepare_native_bid_branches", lambda *a, **kw: ([source], {"queriesUsed": 3}))
        with pytest.raises(InformationSetCompilationStopped) as outside:
            compiler._solve_native_bid_policy_bounds(exact, prefix, actor.team,
                visibility="complete_state", selected_root_action="sun", maximum_model_bytes=100000000,
                **{**threshold_args, "incumbent_threshold": 10**20})
        assert outside.value.reason == "selected_threshold_out_of_domain"
        assert outside.value.partial_receipt["evidence"]["queriesUsed"] == 3
    for key, value in (("solverStatus", "unknown"), ("solverStatus", "optimal"),
            ("candidate", {"utility": 0}), ("threshold", receipt["lowerBound"] + 1),
            ("objectiveLowerBound", receipt["lowerBound"]), ("exhaustionReason", "maximum_seconds"),
            ("baseModelHash", "wrong-model"), ("queryMode", "optimization")):
        changed = copy.deepcopy(threshold_proof)
        policy_query = changed["queries"][0]["receipt"]
        outcome = policy_query["query"]
        outcome[key] = value
        for document in (outcome, policy_query, changed):
            document["sha256"] = canonical_sha256({k: v for k, v in document.items() if k != "sha256"})
        with pytest.raises(ValueError, match="ownership or coverage"):
            validate_native_incumbent_threshold_floor_v1(changed, **threshold_owner)
    # A threshold certificate must not masquerade as an optimized-value receipt.
    with pytest.raises(ValueError, match="ownership or coverage"):
        validate_native_incumbent_lower_bound_v2(threshold_proof,
            target_team=actor.team.name.lower(), play_policies=policies, rules_hash=source.rules.sha256,
            source_model_hashes=(source._model_hash(search_branching="automatic"),))
    with pytest.raises(InformationSetCompilationStopped) as threshold_miss:
        compiler._solve_symbolic_round_policy_bounds([source], actor.team,
            **{**threshold_args, "incumbent_threshold": receipt["lowerBound"] + 1})
    missed = threshold_miss.value.partial_receipt["evidence"]
    assert threshold_miss.value.partial_receipt["exhaustionReason"] == "incumbent_threshold_refuted"
    assert missed["status"] == "incomplete" and missed["lowerBound"] is None
    assert missed["queries"][0]["receipt"]["query"]["status"] == "witness_found"
    for invalid in (True, 1.5, 10**20):
        with pytest.raises(ValueError, match="incumbent threshold"):
            compiler._solve_symbolic_round_policy_bounds([source], actor.team,
                **{**threshold_args, "incumbent_threshold": invalid})
    # Reuse the real query to check orchestration without a second expensive solve.
    calls = []
    with monkeypatch.context() as patch:
        def cached_floor(*a, **kw):
            calls.append(kw)
            return copy.deepcopy(receipt["queries"][0]["receipt"])
        patch.setattr(OrToolsScenarioModelV1, "find_native_round_with_constant_policy_below", cached_floor)
        patch.setattr(compiler, "_prepare_native_bid_branches", lambda *a, **kw: ([source], {"queriesUsed": 0}))
        bounded = compiler._solve_native_bid_policy_bounds(exact, prefix, actor.team,
            visibility="complete_state", selected_root_action="sun", incumbent_only=True,
            maximum_model_bytes=100000000, **args)
    assert len(calls) == 1 and bounded["status"] == "scoped_incumbent_bound_complete"
    assert bounded["queriesUsed"] == 1 and not bounded["strategy"]["masters"]
    assert not bounded["strategy"]["cuts"] and bounded["strategy"]["lowerBound"] == receipt["lowerBound"]
    assert validate_native_incumbent_lower_bound_v2(bounded["strategy"], target_team=actor.team.name.lower(),
        play_policies=policies, rules_hash=source.rules.sha256,
        source_model_hashes=(source._model_hash(search_branching="automatic"),)) == receipt["lowerBound"]
    second.model = source.model.clone()
    second.model.add(second.model.new_bool_var("distinct_branch") == 1)
    # Coordinator arithmetic only: a complete union uses min, never max/first.
    replies = iter(({"status": "queried", "query": {"solverStatus": "optimal", "candidate": {"utility": value}}}
                    for value in (3, 7)))
    with monkeypatch.context() as patch:
        patch.setattr(OrToolsScenarioModelV1, "find_native_round_with_constant_policy_below", lambda *a, **kw: next(replies))
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            compiler._solve_symbolic_round_policy_bounds([source, second], actor.team,
                **{**args, "maximum_queries": 2})
    assert stopped.value.partial_receipt["evidence"]["lowerBound"] == 3
    for status in ("incomplete", "queried"):
        replies = iter(({"status": "queried", "query": {"solverStatus": "optimal", "candidate": {"utility": 3}}},
                        {"status": status, "query": {"solverStatus": "unknown", "candidate": None}}))
        with monkeypatch.context() as patch:
            patch.setattr(OrToolsScenarioModelV1, "find_native_round_with_constant_policy_below", lambda *a, **kw: next(replies))
            with pytest.raises(InformationSetCompilationStopped) as stopped:
                compiler._solve_symbolic_round_policy_bounds([source, second], actor.team,
                    **{**args, "maximum_queries": 2})
        assert stopped.value.partial_receipt["evidence"]["lowerBound"] is None
    replies = iter(({"status": "queried", "query": {"solverStatus": "optimal", "candidate": {"utility": 7}}},
                    {"status": "incomplete", "query": {"solverStatus": "unknown", "candidate": None,
                                                         "objectiveLowerBound": 3}}))
    with monkeypatch.context() as patch:
        patch.setattr(OrToolsScenarioModelV1, "find_native_round_with_constant_policy_below", lambda *a, **kw: next(replies))
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            compiler._solve_symbolic_round_policy_bounds([source, second], actor.team,
                **{**args, "maximum_queries": 2})
    evidence = stopped.value.partial_receipt["evidence"]
    assert evidence["lowerBound"] == 3 and evidence["status"] == "incomplete"
    assert not evidence["strategicCoverage"] and not evidence["masters"] and not evidence["cuts"]
    for invalid in (True, 1.5, float("nan"), 10**20):
        with monkeypatch.context() as patch:
            patch.setattr(OrToolsScenarioModelV1, "find_native_round_with_constant_policy_below",
                lambda *a, **kw: {"status": "incomplete", "query": {"solverStatus": "unknown",
                    "candidate": None, "objectiveLowerBound": invalid}})
            with pytest.raises(ValueError, match="incumbent objective bound is invalid"):
                compiler._solve_symbolic_round_policy_bounds([source], actor.team, **args)
    assert str(source.model.proto) == before


@pytest.mark.parametrize("tighten", [False, True])
def test_native_incumbent_world_reuse_retains_two_sided_bounds(root, tighten):
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    actor = Seat.parse(exact.next_actor)
    source = compiler._build_native_bid_branch(exact, prefix, visibility="actor_information_set",
        branch=(Contract.SUN, None, actor, False))
    source.model.add(source.auction_fragment.choices[0]["sun"] == 1)
    for variable, card in zip(source.card_at_position, prefix.deck_order, strict=True):
        source.model.add(variable == card)
    observation = source._auction_observation_hash(actor,
        [compiler._card(card) for card in exact.state["hands"][actor.name.lower()]],
        compiler._card(exact.state["purchaseCard"]), [])
    before = str(source.model.proto)
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._solve_symbolic_round_policy_bounds([source], actor.team,
            maximum_iterations=1, maximum_queries=2, maximum_observations=100,
            maximum_branches=1, maximum_conflicts=250000, maximum_seconds=30,
            incumbent_play_policies=_native_ranked_play_policies(actor.team),
            tighten_utility_ceiling=tighten,
            fixed_root=(observation, tuple(sorted(row["code"] for row in exact.legal_actions)), "sun"))
    assert stopped.value.reason == "maximum_queries"
    evidence = stopped.value.partial_receipt["evidence"]
    assert evidence["status"] == "incomplete" and not evidence["strategicCoverage"]
    assert [row["purpose"] for row in evidence["queries"]] == ([
        "incumbent_lower_bound", "incumbent_world_opponent_upper_bound"] if tighten else ["incumbent_lower_bound"])
    lower = evidence["queries"][0]["receipt"]["query"]
    if not tighten:
        assert len(evidence["masters"]) == len(evidence["cuts"]) == 1
        cut = evidence["cuts"][0]
        assert cut["utility"] == lower["candidate"]["utility"]
        assert cut["incumbentQuery"] == {
            "branchModelHash": evidence["queries"][0]["branchModelHash"],
            "ownedQueryHash": evidence["queries"][0]["receipt"]["sha256"],
            "queryHash": lower["sha256"]}
        expected = sorted({(row["observationHash"], row["selectedAction"])
            for kind in ("auction", "escalation", "project", "play")
            for row in lower["candidate"][kind + "Observations"]
            if not row.get("authoredPrefix", False) and Seat.parse(row["actor"]).team == actor.team})
        assert cut["targetActions"] == [list(row) for row in expected]
        assert all(key in evidence["domains"] for key, _ in expected)
        assert str(source.model.proto) == before
        return
    upper = evidence["queries"][1]["receipt"]
    assert lower["solverStatus"] == upper["solverStatus"] == "optimal"
    assert evidence["lowerBound"] == lower["candidate"]["utility"]
    assert evidence["upperBound"] == -upper["candidate"]["utility"]
    assert lower["candidate"]["deckOrder"] == upper["candidate"]["deckOrder"]
    assert evidence["scope"]["opponentCeilingWorldQueryHash"] == lower["sha256"]
    assert str(source.model.proto) == before
    assert not source._observed_escalation_bindings and not source._observed_play_bindings


def test_native_round_retains_only_solver_owned_objective_bounds(root, monkeypatch):
    import baloot_platform.research.ortools_scenario_model as module

    _, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    args = dict(target=Team.NORTH_SOUTH, threshold=1000,
        maximum_seconds=30, maximum_conflicts=250000)
    before = str(scenario.model.proto)
    exact = scenario.find_native_round_margin_below(**args)
    assert exact["solverStatus"] == "optimal"
    assert exact["objectiveLowerBound"] == exact["candidate"]["utility"]
    original = module._solve_cp_sat_with_cancel_v1
    bounds = []

    # CP-SAT's floating bound can sit just below its exact integer optimum.
    from math import nextafter, inf, floor
    bound_property = module.cp_model.CpSolver.best_objective_bound
    optimum = exact["candidate"]["utility"]
    for reported_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        def rounded_bound(solver, model, cancel):
            record = solver.best_bound_callback
            solver.best_bound_callback = lambda value: record(
                nextafter(value, -inf) if value == optimum else value)
            assert original(solver, model, cancel) == cp_model.OPTIMAL
            return reported_status

        with monkeypatch.context() as patch:
            patch.setattr(module, "_solve_cp_sat_with_cancel_v1", rounded_bound)
            patch.setattr(module.cp_model.CpSolver, "best_objective_bound", property(
                lambda solver: nextafter(bound_property.fget(solver), -inf)))
            rounded = scenario.find_native_round_margin_below(**args)
        assert rounded["candidate"]["utility"] == optimum
        assert rounded["objectiveLowerBound"] == (
            optimum if reported_status == cp_model.OPTIMAL else floor(nextafter(optimum, -inf)))

    def stop_on_bound(solver, model, cancel):
        record = solver.best_bound_callback

        def bounded(value):
            record(value)
            bounds.append(value)
            solver.stop_search()

        solver.best_bound_callback = bounded
        return original(solver, model, cancel)

    with monkeypatch.context() as patch:
        patch.setattr(module, "_solve_cp_sat_with_cancel_v1", stop_on_bound)
        partial = scenario.find_native_round_margin_below(**args)
    assert bounds
    assert partial["objectiveLowerBound"] <= exact["candidate"]["utility"]
    assert partial["queryModelHash"] == exact["queryModelHash"]
    assert partial["solverStatus"] == "unknown" and partial["status"] == "incomplete"
    assert partial["candidate"] is None
    def contradictory_bound(solver, model, cancel):
        status = original(solver, model, cancel)
        solver.best_bound_callback(exact["candidate"]["utility"] + 1)
        return status
    with monkeypatch.context() as patch:
        patch.setattr(module, "_solve_cp_sat_with_cancel_v1", contradictory_bound)
        with pytest.raises(ValueError, match="bound contradicts its witness"):
            scenario.find_native_round_margin_below(**args)
    # An uninitialized UNKNOWN response has a default zero, not a proof bound.
    with monkeypatch.context() as patch:
        patch.setattr(module, "_solve_cp_sat_with_cancel_v1", lambda *_: cp_model.UNKNOWN)
        assert "objectiveLowerBound" not in scenario.find_native_round_margin_below(**args)
    assert "objectiveLowerBound" not in scenario.find_native_round_margin_below(**args, cancel_check=lambda: True)
    assert "objectiveLowerBound" not in scenario.find_native_round_margin_below(**args, optimize_margin=False)
    for invalid in (float("nan"), float("inf"), 10**20):
        def invalid_bound(solver, model, cancel):
            solver.best_bound_callback(invalid)
            return cp_model.UNKNOWN
        with monkeypatch.context() as patch:
            patch.setattr(module, "_solve_cp_sat_with_cancel_v1", invalid_bound)
            with pytest.raises(ValueError, match="objective bound is invalid"):
                scenario.find_native_round_margin_below(**args)
    assert str(scenario.model.proto) == before


def test_auction_observation_identity_separates_complete_information(root):
    from baloot_platform.research.scenario_contracts import canonical_sha256

    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    arguments = (Seat.NORTH, [0, 1, 2, 3, 4], 5, [])
    actor_hash = scenario._auction_observation_hash(*arguments)
    assert actor_hash == canonical_sha256({"actor": int(Seat.NORTH), "cards": [0, 1, 2, 3, 4],
        "purchase": 5, "publicChoices": ()})
    scenario._complete_information_deck = tuple(range(32))
    complete_hash = scenario._auction_observation_hash(*arguments)
    assert complete_hash != actor_hash
    deck = list(scenario._complete_information_deck)
    deck[-2:] = reversed(deck[-2:])
    scenario._complete_information_deck = tuple(deck)
    assert scenario._auction_observation_hash(*arguments) != complete_hash


def test_native_boolean_empty_history_identities():
    from baloot_platform.research.ortools_scenario_model import _auction_logic_v1

    model = cp_model.CpModel()
    _, both, either = _auction_logic_v1(model, "empty_history")
    conjunction, disjunction = both("all"), either("any")
    solver = cp_model.CpSolver()
    assert solver.solve(model) == cp_model.OPTIMAL
    assert solver.boolean_value(conjunction)
    assert not solver.boolean_value(disjunction)


@pytest.mark.parametrize("level", ["normal", "double"])
def test_complete_project_recall_is_explicit_and_never_an_actor_payload(root, level):
    _, actor_model, _, _ = _native_hokum_round_branch(root, level)
    arguments = dict(target=Team.NORTH_SOUTH, threshold=1000,
        maximum_seconds=15, maximum_conflicts=250_000)
    actor = actor_model.find_native_round_margin_below(**arguments)["candidate"]
    assert actor is not None
    complete_model = copy.copy(actor_model)
    complete_model.model = actor_model.model.clone()
    complete_model._complete_information_deck = tuple(range(32))
    assert complete_model._model_hash(search_branching="automatic") != actor_model._model_hash(search_branching="automatic")
    complete = complete_model.find_native_round_margin_below(**arguments)["candidate"]
    assert complete is not None
    observed_declines = [[]]
    for row in complete["escalationObservations"]:
        information = row["information"]
        while len(observed_declines) < len(information["publicAnnouncements"]) + 1:
            observed_declines.append([])
        assert information["completeWindowDeclineOrder"] == observed_declines
        if row["selectedAction"] == complete_model._native_round_fragment.escalation.stage.domain.closure_action:
            observed_declines[-1].append(row["actor"])
        normalized, windows = complete_model._normalize_escalation_information(information)
        recall = normalized["completePriorWindowDeclines"]
        assert set(recall) == {seat.name.lower() for seat in Seat}
        assert recall[row["actor"]] == information["ownDeclinedWindows"]
        assert all(window in windows for values in recall.values() for window in values)
        with pytest.raises(ValueError):
            actor_model._normalize_escalation_information(information)
        malformed = copy.deepcopy(information)
        malformed["completePriorWindowDeclines"].pop("north")
        with pytest.raises(ValueError, match="complete escalation recall"):
            complete_model._normalize_escalation_information(malformed)
        malformed = copy.deepcopy(information)
        malformed["completeWindowDeclineOrder"][-1].append(row["actor"])
        with pytest.raises(ValueError, match="complete escalation order"):
            complete_model._normalize_escalation_information(malformed)
    for kind in ("escalationObservations", "projectObservations", "playObservations"):
        assert all("completePriorWindowDeclines" not in row["information"] for row in actor[kind])
        assert all("completePriorWindowDeclines" in row["information"] for row in complete[kind])
        assert all("completeWindowDeclineOrder" not in row["information"] for row in actor[kind])
    for kind in ("projectObservations", "playObservations"):
        assert all(row["information"]["completeWindowDeclineOrder"] == observed_declines for row in complete[kind])
    choices = [row["selectedAction"] for row in complete["projectObservations"]]
    for index, row in enumerate(complete["projectObservations"]):
        assert row["information"]["completeProjectActions"] == choices[:index]
        assert "completeProjectActions" not in actor["projectObservations"][index]["information"]
    for index, row in enumerate(complete["playObservations"]):
        assert row["information"]["completeProjectActions"] == choices[:min(index + 1, 4)]
        assert "completeProjectActions" not in actor["playObservations"][index]["information"]
    info = complete["projectObservations"][1]["information"]
    with pytest.raises(ValueError):
        actor_model._normalize_project_information(info)
    with pytest.raises(ValueError):
        actor_model._normalize_play_information(complete["playObservations"][1]["information"])
    changed = copy.deepcopy(info)
    changed["completeProjectActions"] = []
    with pytest.raises(ValueError, match="complete project history"):
        complete_model._normalize_project_information(changed)
    changed["completeProjectActions"] = [next(code for code in
        complete_model._native_round_fragment.projects.declaration_actions[int(Seat.NORTH)] if code != choices[0])]
    assert complete_model._project_observation_hash(changed) != complete_model._project_observation_hash(info)
    normalized, windows = complete_model._normalize_project_information(info)
    ordered_model = copy.copy(complete_model)
    ordered_model.model = complete_model.model.clone()
    ordered = ordered_model._native_escalation_information_literal(normalized, windows, "ordered_recall")
    ordered_model.model.add(ordered == 1)
    for index, row in enumerate(complete["escalationObservations"]):
        observed, prior_windows = ordered_model._normalize_escalation_information(row["information"])
        guard = ordered_model._native_escalation_information_literal(observed, prior_windows, f"prefix_recall_{index}")
        ordered_model.model.add(guard == 1)
    order_solver = cp_model.CpSolver()
    order_solver.parameters.max_time_in_seconds = 10
    order_solver.parameters.num_search_workers = 1
    assert order_solver.solve(ordered_model.model) == cp_model.OPTIMAL
    reversed_info = copy.deepcopy(normalized)
    if level == "normal":  # This fixture has two declines in one window; double does not.
        reversible = next(rows for rows in reversed_info["completeWindowDeclineOrder"] if len(rows) > 1)
        reversible.reverse()
        assert complete_model._escalation_observation_hash(reversed_info) != complete_model._escalation_observation_hash(normalized)
        reversed_guard = ordered_model._native_escalation_information_literal(reversed_info, windows, "reversed_recall")
        ordered_model.model.add(reversed_guard == 1)
        assert order_solver.solve(ordered_model.model) == cp_model.INFEASIBLE
    recall_model = copy.copy(complete_model)
    recall_model.model = complete_model.model.clone()
    original = recall_model._native_escalation_information_literal(normalized, windows, "original_recall")
    contradictory = copy.deepcopy(normalized)
    other = next(seat.name.lower() for seat in Seat if seat.name.lower() != normalized["actor"])
    remembered = contradictory["completePriorWindowDeclines"][other]
    contradictory["completePriorWindowDeclines"][other] = [window for window in windows
        if (window in remembered) != (window == windows[0])]
    opposite = recall_model._native_escalation_information_literal(contradictory, windows, "opposite_recall")
    recall_model.model.add(original == 1)
    recall_model.model.add(opposite == 1)
    recall_solver = cp_model.CpSolver()
    recall_solver.parameters.max_time_in_seconds = 10
    recall_solver.parameters.num_search_workers = 1
    assert recall_solver.solve(recall_model.model) == cp_model.INFEASIBLE
    literals = complete_model._native_round_history_literals(normalized, windows, "complete_recall_test")
    for literal in literals:
        complete_model.model.add(literal == 1)
    first = choices[0]
    projects = [project for project in complete_model._native_round_fragment.projects.projects
                if project.owner == Seat.NORTH]
    if first == "pass":
        complete_model.model.add_bool_or([project.declared for project in projects])
    else:
        selected = next(project.declared for project in projects if "declare:" + project.project_id == first)
        complete_model.model.add(selected == 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_search_workers = 1
    assert solver.solve(complete_model.model) == cp_model.INFEASIBLE


def _native_hokum_round_branch(root, level, *, rules=None, deck_order=None, opportunity_order=None, fixed_play=True):
    from baloot_platform.research.ortools_play_tail import TailPrefixActionV1
    from baloot_platform.research.ortools_scoring_stage import scoring_level_id_v1

    rules = rules or RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    auction = scenario.add_native_auction_fragment(prefix_length=0, allow_all_passed=True)
    line = "north:S7 west:S10 south:SK east:D9 east:H8 north:HJ west:HK south:H7 west:SJ south:SA east:D10 north:S8 east:H9 north:HQ west:HA south:D7 south:D8 east:H10 north:DJ west:DA north:S9 west:SQ south:C9 east:CA west:C7 south:C10 east:CQ north:DK north:DQ west:C8 south:CJ east:CK"
    path = tuple(TailPrefixActionV1.build(*action.split(":")) for action in line.split())
    branch = scenario.add_native_round_continuation(contract="hokum", trump="diamonds",
        first_leader="north", locked=False, match_score=(0, 0), prefix=path if fixed_play else (),
        opportunity_order=opportunity_order)
    # Bind values only after the whole model exists; no deal-derived truth was
    # supplied to the native project, transition, or scoring compilers.
    deck_order = tuple(range(32)) if deck_order is None else deck_order
    for index, variable in enumerate(scenario.card_at_position):
        scenario.model.add(variable == deck_order[index])
    for index, code in enumerate(("hokum:diamonds", "pass", "pass", "pass")):
        scenario.model.add(auction.choices[index][code] == 1)
    if level is not None:
        scenario.model.add(branch.escalation.native["level"] == scoring_level_id_v1(level))
    return rules, scenario, auction, branch


def test_native_guarded_escalation_and_ranked_policy_composition(root):
    from baloot_platform.core.models import LegalPlayActionV1
    from baloot_platform.research.ortools_scoring_stage import scoring_level_id_v1

    _, source, _, branch = _native_hokum_round_branch(root, None, fixed_play=False)
    witness = copy.copy(source)
    witness.model = source.model.clone()
    witness.model.add(branch.escalation.native["level"] == scoring_level_id_v1("double"))
    found = witness.find_native_round_margin_below(target=Team.NORTH_SOUTH, threshold=1000,
        maximum_seconds=30, maximum_conflicts=250000)
    assert found["status"] == "witness_found"
    observation = next(row for row in found["candidate"]["escalationObservations"]
                       if row["selectedAction"].split(":")[0] == "double")
    row = {"information": observation["information"], "action": observation["selectedAction"]}
    team = Seat.parse(observation["actor"]).team
    args = dict(target=team, policy_team=team, auction_action=None, escalation_action=None,
        escalation_observations=[row], play_policies=_native_ranked_play_policies(team),
        threshold=1000, maximum_seconds=30, maximum_conflicts=250000)
    before = str(source.model.proto)
    universal = source.find_native_round_with_constant_policy_below(**{
        **args, "escalation_observations": None, "escalation_action": row["action"]})
    assert universal["status"] == "invalid_policy"
    result = source.find_native_round_with_constant_policy_below(**args)
    assert result["status"] == "queried" and result["query"]["solverStatus"] == "optimal"
    assert result["policy"]["escalationObservations"] == [row]
    assert result["bindings"][0]["observationHash"] == observation["observationHash"]
    assert result["bindings"][0]["status"] == "bound"
    target_choice = next(item for item in result["query"]["candidate"]["playObservations"]
        if Seat.parse(item["actor"]).team != team and len(item["legalActions"]) > 1)
    action = LegalPlayActionV1.from_code(next(code for code in target_choice["legalActions"]
        if code != target_choice["selectedAction"]))
    alternate = copy.copy(source)
    alternate.model = source.model.clone()
    index = target_choice["playIndex"]
    slot = branch.play.play_state_context.play_slots[index]
    alternate.model.add(branch.play._extract.actors[index] == int(Seat.parse(target_choice["actor"])))
    alternate.model.add(slot.card == action.card.id)
    alternate.model.add(slot.ekka == int(action.ekka))
    assert alternate.find_native_round_with_constant_policy_below(**args)["status"] == "queried"
    for changes in ({"escalation_action": "double"},
                    {"escalation_observations": [{**row, "unknown": True}]},
                    {"escalation_observations": [row, {**row, "action": "decline"}]},
                    {"policy_team": team.other, "play_policies": None}):
        with pytest.raises(ValueError):
            source.find_native_round_with_constant_policy_below(**{**args, **changes})
    assert str(source.model.proto) == before
    assert not source._observed_escalation_bindings and not source._observed_play_bindings


@pytest.mark.parametrize("order", [("east", "north", "west", "south"),
    ("west", "south", "east", "north")])
def test_native_round_binds_frozen_escalation_order(root, order):
    _, scenario, _, branch = _native_hokum_round_branch(root, "normal", opportunity_order=order)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(scenario.model) == cp_model.OPTIMAL
    assert branch.scope["escalationOpportunityOrder"] == list(order)
    rows = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
    assert rows
    for window in {row["window"] for row in branch.escalation_opportunities}:
        slots = [row for row in branch.escalation_opportunities if row["window"] == window]
        assert [solver.value(row["actor"]) for row in slots] == [int(Seat.parse(seat)) for seat in order]


@pytest.mark.parametrize("declaration_mode", ("optional", "mandatory"))
def test_native_project_prefix_fixes_history_and_rejects_reordered_actors(root, declaration_mode):
    from baloot_platform.core.project_ir import ProjectDeclarationActionV1
    data = copy.deepcopy(RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml").data)
    data["projects"]["declaration_mode"] = declaration_mode
    rules, scenario, _, branch = _native_hokum_round_branch(root, "normal", rules=RuleBook(data))
    order = rules.project_domain.declaration_seats(scenario.dealer)
    actions = tuple(ProjectDeclarationActionV1.build("pass", actor=seat) for seat in order)
    before = str(scenario.model.proto)
    for invalid in (tuple(reversed(actions)), (*actions, actions[0]), ({"action": "pass"},)):
        with pytest.raises(ValueError, match="ordered declaration"):
            scenario.constrain_native_project_prefix(invalid)
        assert str(scenario.model.proto) == before
    scenario.constrain_native_project_prefix(actions)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    status = solver.solve(scenario.model)
    if declaration_mode == "mandatory":
        assert status == cp_model.INFEASIBLE
        return
    assert status == cp_model.OPTIMAL
    assert all(solver.value(project.declared) == 0 for project in branch.projects.projects)
    assert all(solver.value(points) == 0 for points in branch.projects.combined_awarded_value)


@pytest.mark.parametrize("level", ("normal", "double"))
def test_native_completed_escalation_fixes_real_opportunity_history(root, level):
    _, scenario, _, branch = _native_hokum_round_branch(root, level,
        opportunity_order=("west", "east", "north", "south"))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(scenario.model) == cp_model.OPTIMAL
    rows = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
    events = [{"actor": row["actor"], "actionCode": row["selectedAction"]} for row in rows]
    assert events
    original = str(scenario.model.proto)
    for invalid in ([{"actor": "unknown", "actionCode": "decline"}], [{"actor": "north", "actionCode": "unknown"}]):
        with pytest.raises(ValueError, match="ordered events"):
            scenario.constrain_native_completed_escalation(invalid)
        assert str(scenario.model.proto) == original
    scenario.constrain_native_completed_escalation(events)
    assert solver.solve(scenario.model) == cp_model.OPTIMAL
    actual = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
    assert [{"actor": row["actor"], "actionCode": row["selectedAction"]} for row in actual] == events
    changed = scenario.model.clone()
    first = next(slot for slot in branch.escalation_opportunities if solver.value(slot["active"]))
    changed.add(changed.get_int_var_from_proto_index(first["actor"].index) != int(Seat.parse(events[0]["actor"])))
    assert solver.solve(changed) == cp_model.INFEASIBLE


@pytest.mark.parametrize("recorded_count", (0, 1))
def test_native_escalation_prefix_keeps_future_opportunities_open(root, recorded_count):
    _, scenario, _, branch = _native_hokum_round_branch(root, "double",
        opportunity_order=("west", "east", "north", "south"))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(scenario.model) == cp_model.OPTIMAL
    rows = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
    events = [{"actor": row["actor"], "actionCode": row["selectedAction"]} for row in rows]
    assert len(events) > recorded_count
    original = str(scenario.model.proto)
    with pytest.raises(ValueError, match="ordered events"):
        scenario.constrain_native_escalation_prefix(events[:recorded_count], complete=1)
    assert str(scenario.model.proto) == original
    scenario.constrain_native_escalation_prefix(events[:recorded_count])
    assert solver.solve(scenario.model) == cp_model.OPTIMAL
    actual = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
    assert [{"actor": row["actor"], "actionCode": row["selectedAction"]}
            for row in actual[:recorded_count]] == events[:recorded_count]
    assert len(actual) > recorded_count
    assert scenario._recorded_escalation_count == recorded_count
    # The same history remains impossible as a completed window.
    closed = scenario.model.clone()
    closed.add(sum(closed.get_bool_var_from_proto_index(slot["active"].index)
                   for slot in branch.escalation_opportunities) == recorded_count)
    assert solver.solve(closed) == cp_model.INFEASIBLE


@pytest.mark.parametrize("locked", (False, True))
def test_native_active_escalation_root_keeps_lock_partitions_and_private_declines(root, locked):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [{"action": "hokum"}, {"action": "pass"},
            {"action": "pass"}, {"action": "pass"}], "escalationActions": [],
        "projectDeclarations": [], "projectActions": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix,
        decision_point={"kind": "escalation", "selector": {"actingSeat": "east"}})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    events = [{"actor": "west", "actionCode": "decline"}]
    kwargs = dict(opportunity_order=("west", "east", "north", "south"), escalation_events=events)
    with pytest.raises(ValueError, match="final lock partition"):
        compiler._native_post_auction_root_model(exact, prefix, **kwargs)
    model = compiler._native_post_auction_root_model(exact, prefix, final_locked=locked, **kwargs)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 15
    assert solver.solve(model.model) == cp_model.OPTIMAL
    branch = model._native_round_fragment
    path = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
    assert path[0]["actor"] == "west" and path[0]["selectedAction"] == "decline"
    assert path[1]["actor"] == "east"
    assert bool(solver.value(branch.escalation.native["locked"])) is locked
    observations = model._extract_native_escalation_observations(solver, path)
    assert observations[0]["authoredPrefix"] is True
    assert not observations[1].get("authoredPrefix", False)
    assert sorted(observations[1]["legalActions"]) == sorted(row["code"] for row in exact.legal_actions)
    wrong_actor = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix,
        decision_point={"kind": "escalation", "selector": {"actingSeat": "west"}})
    wrong = compiler._native_post_auction_root_model(wrong_actor, prefix, final_locked=locked, **kwargs)
    assert solver.solve(wrong.model) == cp_model.INFEASIBLE


@pytest.mark.parametrize("visibility", ["complete_state", "actor_information_set"])
def test_native_active_escalation_preparation_retains_both_partition_receipts(root, visibility):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    compiler, _, source, _ = _native_suit_monopoly_bid(root)
    raw = source.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[{"action": "sun"}], projectActions=[], matchScore=[101, 100])
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(prefix,
        decision_point={"kind": "escalation", "selector": {"actingSeat": "west"}})
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._solve_native_post_auction_policy_bounds(exact, prefix, Team.WEST_EAST,
            visibility=visibility,
            selected_root_action=None, opportunity_order=("west", "east", "north", "south"),
            escalation_events=[], maximum_iterations=1,
            maximum_queries=6 if visibility == "actor_information_set" else 4, maximum_observations=64,
            maximum_model_bytes=100_000_000, maximum_conflicts=250_000, maximum_depth=64, maximum_seconds=45)
    assert stopped.value.reason == "maximum_queries"
    receipt = stopped.value.partial_receipt["evidence"]
    assert receipt["status"] == "incomplete" and receipt["strategicCoverage"] is False
    assert receipt["queriesUsed"] == (6 if visibility == "actor_information_set" else 4)
    partitions = receipt["preparation"]["partitions"]
    assert [row["finalLocked"] for row in partitions] == [False, True]
    assert partitions[0]["retained"] is True
    assert partitions[1]["retained"] is False
    assert partitions[1]["rootQuery"]["solverStatus"] == "infeasible"
    assert all(row["depthQuery"]["status"] == "depth_bound_proven" for row in partitions)
    if visibility == "actor_information_set":
        assert all(row["sourceObservationQuery"]["status"] == "witness_found" for row in partitions)
        assert all(row.get("rootObservationSource") != "source_observation_projection" for row in partitions)
        assert partitions[0]["rootQuery"]["scope"]["escalationRoot"]["observationHash"] == partitions[0]["rootObservationHash"]
        assert partitions[0]["rootQuery"]["scope"]["escalationOpportunityOrder"] == "symbolic_per_window"
    assert receipt["strategy"] is None


@pytest.mark.real_clock
@pytest.mark.parametrize("receipt_version", (1, 2))
def test_native_active_escalation_constant_bounds_cover_the_full_partition(root, monkeypatch, receipt_version):
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofBudgetV1, ScenarioProofRequestV1, canonical_sha256
    from baloot_platform.research.result_authority import (
        build_native_constant_escalation_quality_artifact_v1, validate_native_constant_escalation_bounds_v1,
        validate_native_constant_escalation_budget_v1)
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped

    compiler, _, source, _ = _native_suit_monopoly_bid(root)
    raw = source.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[{"action": "sun"}], projectActions=[], matchScore=[101, 100])
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(prefix,
        decision_point={"kind": "escalation", "selector": {"actingSeat": "west"}})
    result = compiler._solve_native_post_auction_policy_bounds(exact, prefix, Team.WEST_EAST,
        escalation_receipt_version=receipt_version,
        selected_root_action=None, opportunity_order=("west", "east", "north", "south"),
        escalation_events=[], maximum_iterations=1, maximum_queries=7, maximum_observations=64,
        maximum_model_bytes=100_000_000, maximum_conflicts=250_000, maximum_depth=64, maximum_seconds=45)
    assert result["status"] == "scoped_constant_bound_converged"
    assert result["strategicCoverage"] is False  # Still requires independent admission validation.
    assert result["schemaVersion"] == receipt_version
    assert result["queriesUsed"] == (7 if receipt_version == 1 else 6)
    proof = result["strategy"]
    assert proof["selectedAction"] == "decline"
    assert proof["lowerBound"] == proof["upperBound"] == -92
    assert proof["sha256"] == canonical_sha256({key: value for key, value in proof.items() if key != "sha256"})
    assert [row["retained"] for row in result["preparation"]["partitions"]] == [True, False]
    row = proof["partitions"][0]
    if receipt_version == 2:
        assert row["upperQuery"] == result["preparation"]["partitions"][0]["rootQuery"]
    assert row["upperQuery"]["solverStatus"] == row["lowerQuery"]["query"]["solverStatus"] == "optimal"
    assert all(binding["status"] == "bound" for binding in row["lowerQuery"]["bindings"])
    validation = dict(rules=compiler.rules, source_prefix=prefix, normalized=exact,
        opportunity_order=("west", "east", "north", "south"), escalation_events=[], maximum_seconds=45)
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
        "claim": "outcome_reachable", "targetTeam": "west_east", "utility": "awarded_point_margin",
        "successThreshold": -100000, "budgetProfile": "test"})
    profile = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component(
        "challenge")["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    budget = ScenarioProofBudgetV1.build("test", profile)
    artifact = build_native_constant_escalation_quality_artifact_v1(result, request=request, budget=budget,
        maximum_conflicts=250000, **validation)
    caps = {"iterations": budget.native_observed.maximum_iterations,
        "queries": budget.native_observed.maximum_queries, "observations": budget.native_observed.maximum_observations,
        "modelBytes": budget.maximum_cache_bytes, "conflictsPerSolve": 250000,
        "depth": min(budget.maximum_depth, budget.maximum_horizon_actions),
        "seconds": min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds)}
    for field, cap in caps.items():
        for invalid in (True, 0, cap + 1):
            forged_budget = {**result, "limits": {**result["limits"], field: invalid}}
            with pytest.raises(ValueError, match="frozen"):
                validate_native_constant_escalation_budget_v1(forged_budget, budget, maximum_conflicts=250000)
    quality = artifact["proofModel"]["actionQuality"]
    assert quality["actualActionOptimal"] is True and quality["regret"] == 0
    assert quality["bestValue"] == quality["actualValue"] == -92
    assert quality["optimalActionsComplete"] is False
    assert artifact["proof"]["claimSatisfied"] is artifact["proof"]["winningRootActionCount"] is None
    assert artifact["proof"]["completionScope"] == "selected_escalation_current_deal"
    assert artifact["allocationModel"]["priorNonAnnouncingSeats"] == []
    assert artifact["allocationModel"]["priorWindowRecall"] == {}
    assert quality["sha256"] == canonical_sha256({**{key: value for key, value in quality.items() if key != "sha256"},
        "modelProto": artifact["proofModel"]["modelProto"]})
    from baloot_platform.research import result_authority
    from baloot_platform.research.ortools_research_executor import _complete_state_quality_artifact_v1
    context = dict(maximum_nodes=100000, maximum_depth=64, maximum_seconds=45, cancel_check=None,
        opportunity_order=("west", "east", "north", "south"), budget=budget,
        maximum_conflicts=250000, escalation_events=[])
    with monkeypatch.context() as routing:
        def solve(_self, *_args, **kwargs):
            assert kwargs["escalation_events"] == []
            assert kwargs["maximum_conflicts"] == 250000
            assert kwargs["maximum_queries"] == budget.native_observed.maximum_queries
            return result
        routing.setattr(type(compiler), "_solve_native_post_auction_policy_bounds", solve)
        routing.setattr(result_authority, "build_native_constant_escalation_quality_artifact_v1",
            lambda *_args, **_kwargs: artifact)
        produced = _complete_state_quality_artifact_v1(compiler.rules, exact, prefix,
            {"kind": "escalation"}, request, **context)
        assert produced == artifact
        with pytest.raises(ValueError, match="actual private history"):
            _complete_state_quality_artifact_v1(compiler.rules, exact, prefix,
                {"kind": "escalation"}, request, prior_non_announcing_seats=("east",), **context)
        routing.setattr(type(compiler), "_solve_native_post_auction_policy_bounds",
            lambda *_args, **_kwargs: {**result, "status": "incomplete"})
        with pytest.raises(InformationSetCompilationStopped) as incomplete:
            _complete_state_quality_artifact_v1(compiler.rules, exact, prefix,
                {"kind": "escalation"}, request, **context)
        assert incomplete.value.reason == "native_escalation_quality_incomplete"
    unsigned = {key: value for key, value in result.items() if key != "sha256"}
    with pytest.raises(ValueError, match="checksum"):
        validate_native_constant_escalation_bounds_v1(unsigned, **validation)
    for overrides, reason in (({"maximum_seconds": 1e-9}, "maximum_seconds"),
                              ({"cancel_check": lambda: True}, "cancelled")):
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            validate_native_constant_escalation_bounds_v1(result, **{**validation, **overrides})
        assert stopped.value.reason == reason
    for key, changed in (("status", "incomplete"), ("sourcePrefixHash", "0" * 64)):
        invalid = {**result, key: changed}
        invalid["sha256"] = canonical_sha256({name: value for name, value in invalid.items() if name != "sha256"})
        with pytest.raises(ValueError, match="ownership or completion"):
            validate_native_constant_escalation_bounds_v1(invalid, **validation)
    forged = copy.deepcopy(result)
    forged["strategy"]["lowerBound"] += 1
    forged["strategy"]["upperBound"] += 1
    forged["strategy"]["sha256"] = canonical_sha256({key: value for key, value in forged["strategy"].items() if key != "sha256"})
    forged["sha256"] = canonical_sha256({key: value for key, value in forged.items() if key != "sha256"})
    with pytest.raises(ValueError, match="native reconstruction"):
        validate_native_constant_escalation_bounds_v1(forged, **validation)


def test_native_escalation_quality_envelope_binds_every_section(monkeypatch):
    from baloot_platform.research import result_authority

    expected = {key: {"identity": key} for key in
        ("request", "proof", "proofModel", "graph", "allocationModel")}
    frozen_context = object()
    calls = []

    def rebuild(graph, **context):
        assert graph == expected["graph"]
        assert context == {"rules": frozen_context}
        calls.append(graph)
        return expected

    # Native reconstruction is covered by the integration test above; isolate envelope binding here.
    monkeypatch.setattr(result_authority, "build_native_constant_escalation_quality_artifact_v1", rebuild)
    validate = result_authority.validate_native_constant_escalation_quality_artifact_v1
    validate(expected, rules=frozen_context)
    for section in ("request", "proof", "proofModel", "allocationModel"):
        with pytest.raises(ValueError, match="binding mismatch"):
            validate({**expected, section: {"identity": "substituted"}}, rules=frozen_context)
    with pytest.raises(ValueError, match="binding mismatch"):
        validate({**expected, "extra": True}, rules=frozen_context)
    for malformed in (None, {}, {"graph": None}):
        with pytest.raises(ValueError, match="malformed"):
            validate(malformed, rules=frozen_context)
    assert len(calls) == 6


@pytest.mark.real_clock
def test_native_observed_escalation_quality_uses_symbolic_history_and_fresh_proof(root, monkeypatch, record_property):
    import time
    from baloot_platform.config import build_bundle
    from baloot_platform.research import ortools_research_executor as executor
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofRequestV1
    from baloot_platform.research.result_authority import validate_native_observed_play_quality_artifact_v1
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1

    compiler, _, source, hands = _native_suit_monopoly_bid(root)
    raw = source.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[{"actor": "north", "action": "pass"},
                               {"actor": "west", "action": "sun"},
                               {"actor": "north", "action": "pass"}],
               projectActions=[], matchScore=[100, 101])
    layout = DealLayoutV1.build(compiler.rules.structural_domain, dealer=Seat.EAST, buyer=Seat.WEST)
    deck = [None] * 32
    for seat in Seat:
        for position, card in zip(layout.positions_for(seat, final=True), sorted(hands[seat]), strict=True):
            deck[position] = card
    raw["deckOrder"] = deck
    prefix = ExactTracePrefixV1.build(raw)
    settings = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component(
        "challenge")["scenarioResearchV4"]
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "exact",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "outcome_reachable", "targetTeam": "north_south", "utility": "awarded_point_margin",
        "successThreshold": 0, "budgetProfile": "offline_heavyweight_v1"})
    def forbidden(*args, **kwargs):
        pytest.fail("native observed escalation must not enumerate hidden worlds")
    monkeypatch.setattr(executor, "compile_observed_decision_v1", forbidden)
    order = ("north", "south", "west", "east")
    generation_start = time.perf_counter()
    continuation = executor.resolve_quality_escalation_continuation_v1(compiler.rules, prefix,
        {"kind": "escalation", "selector": {"actingSeat": "north"}}, request,
        maximum_nodes=100000, maximum_depth=64, maximum_worlds=1, maximum_seconds=180,
        maximum_decision_seconds=180, maximum_decisions=1, maximum_evidence_bytes=100000000,
        opportunity_order=order,
        proof_budget_profiles=settings["proofBudgets"]["profiles"],
        search_budget_profiles=settings["searchBudgets"]["profiles"])
    record_property("generationSeconds", time.perf_counter() - generation_start)
    assert len(continuation["decisions"]) == len(continuation["events"]) == 1, continuation.get("partialReceipt")
    result = continuation["artifacts"][continuation["decisions"][0]["artifactHash"]]
    assert result["status"] == "recorded", result.get("partialReceipt")
    quality = result["proofModel"]["actionQuality"]
    assert result["proofModel"]["kind"] == "native_observed_escalation_quality_model_v1"
    assert quality["actualActionOptimal"] is True and quality["regret"] == 0
    assert result["transition"]["event"]["actionCode"] == quality["actualAction"]
    bounds = result["graph"]["bestProof"]
    assert result["graph"]["schemaVersion"] == 2
    assert result["graph"]["queryCountScope"] == "certificate_construction_only"
    assert result["graph"]["queriesUsed"] == bounds["queriesUsed"] + 1
    record_property("constructionQueries", result["graph"]["queriesUsed"])
    assert bounds["strategy"]["lowerBound"] == bounds["strategy"]["upperBound"]
    assert bounds["preparation"]["partitions"][0]["rootQuery"]["scope"][
        "escalationOpportunityOrder"] == "symbolic_per_window"
    assert bounds["preparation"]["partitions"][1]["retained"] is False
    exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(prefix,
        decision_point={"kind": "escalation", "selector": {"actingSeat": "north"}})
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1
    budget = ScenarioProofBudgetV1.build(request.budget_profile,
        settings["proofBudgets"]["profiles"][request.budget_profile])
    context = dict(rules=compiler.rules, source_prefix=prefix, normalized=exact, request=request,
        budget=budget, opportunity_order=order, escalation_events=[], decision_kind="escalation",
        maximum_conflicts=settings["searchBudgets"]["profiles"][request.budget_profile]["maximumConflicts"],
        maximum_seconds=180)
    artifact = {key: result[key] for key in ("request", "proof", "proofModel", "graph", "allocationModel")}
    from baloot_platform.research.result_authority import validate_native_escalation_quality_evidence_v1
    verification_start = time.perf_counter()
    validate_native_escalation_quality_evidence_v1(continuation, source_prefix=prefix,
        rules_hash=compiler.rules.sha256, request=request, rules=compiler.rules,
        search_budget_profiles=settings["searchBudgets"]["profiles"], maximum_seconds=180)
    record_property("externalVerificationSeconds", time.perf_counter() - verification_start)
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    with pytest.raises(InformationSetCompilationStopped, match="cancelled"):
        validate_native_escalation_quality_evidence_v1(continuation, source_prefix=prefix,
            rules_hash=compiler.rules.sha256, request=request, cancel_check=lambda: True)
    with pytest.raises(ValueError, match="owned exact selected-action"):
        validate_native_observed_play_quality_artifact_v1(artifact, **{**context, "decision_kind": "project"})


def test_native_active_escalation_driver_uses_complete_history_and_frozen_bounds(root):
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_research_executor import resolve_quality_escalation_continuation_v1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofRequestV1

    compiler, _, source, _ = _native_suit_monopoly_bid(root)
    raw = source.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[{"action": "sun"}], projectActions=[], matchScore=[101, 100])
    prefix = ExactTracePrefixV1.build(raw)
    settings = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component(
        "challenge")["scenarioResearchV4"]
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
        "claim": "outcome_reachable", "targetTeam": "west_east", "utility": "awarded_point_margin",
        "successThreshold": -100000, "budgetProfile": "offline_heavyweight_v1"})
    result = resolve_quality_escalation_continuation_v1(compiler.rules, prefix,
        {"kind": "escalation", "selector": {}}, request,
        opportunity_order=("west", "east", "north", "south"), maximum_nodes=100000,
        maximum_depth=64, maximum_worlds=1, maximum_seconds=180, maximum_decision_seconds=90,
        maximum_decisions=8, maximum_evidence_bytes=100000000,
        proof_budget_profiles=settings["proofBudgets"]["profiles"],
        search_budget_profiles=settings["searchBudgets"]["profiles"])
    assert result["status"] == "complete", result["partialReceipt"]
    assert result["events"] == [{"actor": seat, "actionCode": "decline"} for seat in ("west", "east")]
    for index, row in enumerate(result["decisions"]):
        artifact = result["artifacts"][row["artifactHash"]]
        assert artifact["proofModel"]["kind"] == "native_escalation_selected_action_quality_model_v1"
        assert artifact["allocationModel"]["escalationEvents"] == result["events"][:index]
        assert artifact["proofModel"]["actionQuality"]["actualActionOptimal"] is True
        assert artifact["proofModel"]["actionQuality"]["optimalActionsComplete"] is False
        assert artifact["proof"]["completionStatus"] == "exact_complete"


@pytest.mark.parametrize("decision_kind", ["play", "project"])
def test_native_observed_play_root_is_source_owned_without_fixing_hidden_deal(root, decision_kind, monkeypatch):
    from baloot_platform.core.project_ir import ProjectDeclarationActionV1
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    order = ("west", "east", "north", "south")
    rules, source, _, _ = _native_hokum_round_branch(root, "normal", opportunity_order=order)
    witness = source.find_native_round_margin_below(target=Team.NORTH_SOUTH, threshold=1000,
        maximum_seconds=15, maximum_conflicts=250_000)
    candidate = witness["candidate"]
    project = candidate["projectObservations"][0]
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": candidate["deckOrder"],
        "dealer": "east", "auctionActions": [
            {key: row[key] for key in ("actor", "action", "trumpSuit") if row.get(key) is not None}
            for row in candidate["auctionActions"]],
        "escalationActions": [], "projectDeclarations": [],
        "projectActions": [ProjectDeclarationActionV1.build(project["selectedAction"],
            actor=Seat.parse(project["actor"])).to_dict()] if decision_kind == "play" else [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    fixed_rows = []
    original = OrToolsScenarioModelV1.add_native_round_continuation
    def capture(self, **kwargs):
        fixed_rows.append(kwargs.get("fixed_card_owners"))
        return original(self, **kwargs)
    monkeypatch.setattr(OrToolsScenarioModelV1, "add_native_round_continuation", capture)
    events = [{"actor": row["actor"], "actionCode": row["selectedAction"]}
              for row in candidate["escalationObservations"]]
    with pytest.raises(ValueError, match="positive finite"):
        compiler._native_post_auction_root_model(exact, prefix, opportunity_order=order,
            escalation_events=events, visibility="actor_information_set")
    model = compiler._native_post_auction_root_model(exact, prefix, opportunity_order=order,
        escalation_events=events, visibility="actor_information_set",
        maximum_seconds=45, maximum_conflicts=250_000, compact_public_ownership=True)
    public = compiler._public_fixed_ownership(exact)
    assert all(owner is not None for owner in fixed_rows[0])  # Complete source witness only.
    assert fixed_rows[-1] == tuple(public.get(Card(index)) for index in range(32))
    assert any(owner is None for owner in fixed_rows[-1])
    assert model._complete_information_deck is None
    assert model._native_play_source_query["queryMode"] == "feasibility_witness"
    if decision_kind == "play":
        assert model._native_play_source_policy["policyId"] == "observer_safe_trick_control_v1"
        assert model._native_play_source_policy["scope"] == "source_observation_only"
        assert model._native_play_source_policy["probe"]["status"] == "witness_found"
        assert (model._native_play_source_policy["probe"]["queryModelHash"]
                == model._native_play_source_query["queryModelHash"])
    else:
        assert model._native_play_source_policy is None
    observed_root = model._native_play_root if decision_kind == "play" else model._native_project_root
    assert observed_root["playIndex"] == 0
    assert model._recorded_project_count == int(decision_kind == "play")
    if decision_kind == "project":
        information = model._native_play_source_query["candidate"]["projectObservations"][0]["information"]
        information = {key: value for key, value in information.items() if not key.startswith("complete")}
        before = str(model.model.proto)
        returned = model.constrain_native_project_observation(information=information)
        returned["playIndex"] = 99
        assert model._native_project_root["playIndex"] == 0
        assert str(model.model.proto) == before
        depth = model.find_native_continuation_depth_above(maximum_depth=36,
            maximum_seconds=15, maximum_conflicts=250_000)
        assert depth["status"] == "depth_bound_proven"
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 30
    assert solver.solve(model.model) == cp_model.OPTIMAL
    first = [solver.value(value) for value in model.owner_of_card]
    # Require different hidden ownership, not merely another deck ordering.
    model.model.add_forbidden_assignments(model.owner_of_card, [first])
    assert solver.solve(model.model) == cp_model.OPTIMAL
    second = [solver.value(value) for value in model.owner_of_card]
    assert second != first
    actor = int(Seat.parse(exact.next_actor))
    assert [card for card, owner in enumerate(first) if owner == actor] == [
        card for card, owner in enumerate(second) if owner == actor]
    with pytest.raises(ValueError, match="compaction must be boolean"):
        compiler._native_post_auction_root_model(exact, prefix, opportunity_order=order,
            escalation_events=events, compact_public_ownership=1)


@pytest.mark.real_clock
def test_native_observed_early_play_root_uses_bounded_source_policy_witness(root, monkeypatch):
    import json
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, canonical_sha256

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json")
                         .read_text(encoding="utf-8"))
    raw = dict(fixture["exactTracePrefix"])
    raw.pop("sha256", None)
    raw.update(projectDeclarations=[], projectAssertions=[],
        projectActions=[{"actor": seat.name.lower(), "action": "pass"}
                        for seat in rules.project_domain.declaration_seats(Seat.parse(raw["dealer"]))],
        playActions=raw["playActions"][:4], stateAssertions={})
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "play"})
    observed = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._native_post_auction_root_model(
        exact, prefix, opportunity_order=("west", "east", "north", "south"),
        escalation_events=[], visibility="actor_information_set",
        maximum_seconds=20, maximum_conflicts=250_000)
    assert observed._native_play_source_policy["probe"]["status"] == "witness_found"
    assert observed._native_play_source_query["status"] == "witness_found"
    assert observed._native_play_root["playIndex"] == 4
    assert observed._complete_information_deck is None
    assert observed._native_round_fragment.scope["projectVisibilitySpecialization"] == "all_pass_v1"
    assert not any(variable.name.startswith("round_public_projects_")
                   for variable in observed.model.proto.variables)
    for index, view in enumerate(observed._native_round_fragment.public_project_bindings):
        assert list(view["compared"].proto.domain) == [int(index >= 4), int(index >= 4)]
        assert all(list(value.proto.domain) == [0, 0] for value in view["revealed"].values())
    assert len(exact.legal_actions) == 7
    source = observed._native_play_source_query["candidate"]
    before = str(observed.model.proto)
    from baloot_platform.research import ortools_scenario_model as module
    native_solve = module._solve_cp_sat_with_cancel_v1
    solves = []
    def traced_solve(solver, model, cancelled):
        solves.append((solver.parameters.fix_variables_to_their_hinted_value,
            solver.parameters.max_number_of_conflicts,
            solver.parameters.repair_hint, solver.parameters.hint_conflict_limit))
        return native_solve(solver, model, cancelled)
    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", traced_solve)
    hint_args = dict(target=Seat.parse(exact.next_actor).team,
        maximum_seconds=20, maximum_conflicts=250_000, optimize_margin=False,
        deck_hint=source["deckOrder"],
        play_card_hint=[Card.parse(row["card"]).id for row in source["play"]["plays"]])
    adverse = observed.find_native_round_margin_below(
        threshold=source["utility"], **hint_args)
    assert adverse["status"] == "witness_found"
    assert adverse["fixedTraceProbe"]["solverStatus"] == "infeasible"
    assert adverse["fixedTraceProbe"]["accepted"] is False
    assert adverse["fixedWorldProbe"]["scope"] == "fixed_hint_world_witness"
    assert [row[0] for row in solves] == ([True, True] if adverse["fixedWorldProbe"]["accepted"]
                                       else [True, True, False])
    assert solves[1][1] == max(1, (250_000 - adverse["fixedTraceProbe"]["conflicts"]) // 2)
    assert solves[0][2] is False and solves[1][2] is True
    assert solves[1][3] == solves[1][1]
    assert adverse["fixedWorldProbe"]["hintRepair"] == {"enabled": True, "conflictLimit": solves[1][1]}
    if adverse["fixedWorldProbe"]["accepted"]:
        assert adverse["candidate"]["deckOrder"] == source["deckOrder"]
    else:
        assert solves[2][2] is True and solves[2][3] == solves[2][1]
        assert solves[2][1] == (250_000 - adverse["fixedTraceProbe"]["conflicts"]
                               - adverse["fixedWorldProbe"]["conflicts"])
    assert adverse["candidate"]["utility"] < source["utility"]
    assert adverse["deckHintHash"] == canonical_sha256(source["deckOrder"])
    assert adverse["playCardHintHash"] == canonical_sha256(
        [Card.parse(row["card"]).id for row in source["play"]["plays"]])
    assert str(observed.model.proto) == before

    solves.clear()
    proposed = observed.find_native_round_margin_below(threshold=source["utility"] + 1, **hint_args)
    assert proposed["status"] == "witness_found"
    assert proposed["fixedTraceProbe"]["accepted"] is True
    assert proposed["fixedTraceProbe"]["scope"] == "fixed_hint_witness"
    assert proposed["candidate"]["deckOrder"] == source["deckOrder"]
    assert proposed["candidate"]["utility"] == source["utility"]
    assert "objectiveLowerBound" not in proposed
    assert "fixedWorldProbe" not in proposed
    assert [row[0] for row in solves] == [True]
    solves.clear()
    absent = observed.find_native_round_margin_below(threshold=-10_000, **hint_args)
    assert absent["status"] == "no_matching_outcome"
    assert absent["fixedTraceProbe"]["accepted"] is False
    assert absent["fixedWorldProbe"]["accepted"] is False
    assert [row[0] for row in solves] == [True, True, False]
    assert solves[1][2] is True and solves[2][2] is True
    assert absent["hintRepair"] == {"enabled": True, "conflictLimit": solves[2][1]}

    # An owner may already hint the deck before this query receives an updated
    # deck/line hint. Compose those hints without mutating the owner or adding
    # duplicate indices, which CP-SAT would reject as MODEL_INVALID.
    observed.model.add_hint(observed.card_at_position[0], source["deckOrder"][0])
    hinted_before = str(observed.model.proto)
    composed = observed.find_native_round_margin_below(threshold=source["utility"] + 1, **hint_args)
    assert composed["status"] == "witness_found" and composed["fixedTraceProbe"]["accepted"]
    assert composed["queryModelHash"] == proposed["queryModelHash"]
    assert str(observed.model.proto) == hinted_before
    observed.model.clear_hints()

    # A rejected proposal cannot certify absence, consume a second allowance,
    # bypass cancellation, or pin an optimization query.
    for stop in ("seconds", "conflicts", "cancelled", "shared_conflicts", "optimized"):
        clock, cancelled, calls = [100.0], [False], []
        class StoppedSolver:
            def __init__(self):
                self.parameters = type("Parameters", (), {"fix_variables_to_their_hinted_value": False,
                    "repair_hint": False, "hint_conflict_limit": 10})()
                self.num_conflicts = self.num_branches = 0
                self.best_bound_callback = None
            def status_name(self, status):
                return {cp_model.INFEASIBLE: "INFEASIBLE", cp_model.UNKNOWN: "UNKNOWN"}[status]
        def stopped_solve(solver, model, check):
            calls.append((solver.parameters.fix_variables_to_their_hinted_value,
                solver.parameters.max_number_of_conflicts))
            clock[0] += 21 if stop == "seconds" else 1
            cancelled[0] = stop == "cancelled"
            solver.num_conflicts = 7 if stop == "conflicts" else 3 if len(calls) == 1 else 4
            return cp_model.INFEASIBLE if len(calls) == 1 and stop != "optimized" else cp_model.UNKNOWN
        with monkeypatch.context() as patch:
            patch.setattr(module.time, "monotonic", lambda: clock[0])
            patch.setattr(module.cp_model, "CpSolver", StoppedSolver)
            patch.setattr(module, "_solve_cp_sat_with_cancel_v1", stopped_solve)
            stopped = observed.find_native_round_margin_below(threshold=source["utility"],
                cancel_check=lambda: cancelled[0], **{**hint_args, "maximum_conflicts": 7,
                    "optimize_margin": stop == "optimized"})
        assert stopped["status"] == "incomplete" and stopped["candidate"] is None
        assert stopped["solverStatus"] == "unknown"
        assert stopped["exhaustionReason"] == {"seconds": "maximum_seconds", "conflicts": "maximum_conflicts",
            "cancelled": "cancelled", "shared_conflicts": "maximum_conflicts", "optimized": "solver_unknown"}[stop]
        assert calls == ([(True, 7), (True, 2)] if stop == "shared_conflicts" else
                         [(False, 7)] if stop == "optimized" else [(True, 7)])
        assert ("fixedTraceProbe" in stopped) == (stop != "optimized")
    assert str(observed.model.proto) == before

    with pytest.raises(ValueError, match="complete legal card sequence"):
        observed.find_native_round_margin_below(
            target=Seat.parse(exact.next_actor).team, threshold=source["utility"],
            maximum_seconds=20, maximum_conflicts=250_000, play_card_hint=[True] * 32)
    refined = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._refine_symbolic_round_policy(
        observed, Seat.parse(exact.next_actor).team, {}, threshold=source["utility"] + 1,
        maximum_queries=10, maximum_observations=2000, maximum_conflicts=250_000,
        maximum_seconds=20)
    assert refined["status"] == "policy_extension_required"
    assert refined["queries"][0]["deckHintHash"] == canonical_sha256(source["deckOrder"])
    assert str(observed.model.proto) == before

    with pytest.raises(ValueError, match="complete legal deck"):
        observed.find_native_round_margin_below(
            target=Seat.parse(exact.next_actor).team, threshold=source["utility"],
            maximum_seconds=20, maximum_conflicts=250_000,
            deck_hint=source["deckOrder"][:-1])

    calls = []
    def policy_miss_then_unrestricted(self, **kwargs):
        calls.append(self)
        if len(calls) == 1:
            assert any(var.name.startswith("visible_policy_") for var in self.model.proto.variables)
            return {"status": "no_matching_outcome", "solverStatus": "infeasible"}
        assert not any(var.name.startswith("visible_policy_") for var in self.model.proto.variables)
        raise RuntimeError("unrestricted fallback reached")

    monkeypatch.setattr(OrToolsScenarioModelV1, "find_native_round_margin_below",
                        policy_miss_then_unrestricted)
    with pytest.raises(RuntimeError, match="unrestricted fallback reached"):
        OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._native_post_auction_root_model(
            exact, prefix, opportunity_order=("west", "east", "north", "south"),
            escalation_events=[], visibility="actor_information_set",
            maximum_seconds=20, maximum_conflicts=250_000)
    assert len(calls) == 2


def test_native_post_auction_root_preserves_doubled_hokum_history(root):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    order = ("west", "east", "north", "south")
    rules, source, _, branch = _native_hokum_round_branch(root, "double", opportunity_order=order)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(source.model) == cp_model.OPTIMAL
    rows = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
    events = [{"actor": row["actor"], "actionCode": row["selectedAction"]} for row in rows]
    actions = []
    for event in events:
        action, _, mode = event["actionCode"].partition(":")
        if action != "decline":
            actions.append({"actor": event["actor"], "action": action, "mode": mode or None})
    assert actions and any(row["actionCode"] == "decline" for row in events)
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [{"actor": "north", "action": "hokum", "trumpSuit": "diamonds"},
            {"actor": "west", "action": "pass"}, {"actor": "south", "action": "pass"},
            {"actor": "east", "action": "pass"}],
        "escalationActions": actions, "projectDeclarations": [], "projectActions": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    model = compiler._native_post_auction_root_model(exact, prefix,
        opportunity_order=order, escalation_events=events)
    assert solver.solve(model.model) == cp_model.OPTIMAL
    actual_branch = model._native_round_fragment
    actual = actual_branch.escalation.extract_opportunity_path(solver, actual_branch.escalation_opportunities)
    assert [{"actor": row["actor"], "actionCode": row["selectedAction"]} for row in actual] == events
    actor = int(Seat.parse(exact.next_actor))
    assert {code for code, literal in actual_branch.projects.declaration_actions[actor].items()
            if solver.value(literal)} == {row["code"] for row in exact.legal_actions}
    truncated = compiler._native_post_auction_root_model(exact, prefix,
        opportunity_order=order, escalation_events=events[:-1])
    assert solver.solve(truncated.model) == cp_model.INFEASIBLE


@pytest.mark.parametrize("actual,optimum", ((0, 0), (0, 1), (2, 1)))
@pytest.mark.parametrize("decision_kind", ("play", "project"))
@pytest.mark.parametrize("receipt_version", (1, 2))
def test_observed_optimality_composes_verified_values_and_reserves_replays(root, rules, monkeypatch, actual, optimum, decision_kind, receipt_version):
    import yaml
    from types import SimpleNamespace
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1

    profiles = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]["proofBudgets"]["profiles"]
    budget = ScenarioProofBudgetV1.build("offline_heavyweight_v1", profiles["offline_heavyweight_v1"])
    checked = []
    monkeypatch.setattr(authority, f"reverify_native_observed_{decision_kind}_bounds_v1",
        lambda value, **context: checked.append(context["selected_root_action"]))
    # Isolate numerical composition here; the real native integration is below.
    selected = {"queriesUsed": 1, "limits": {"queries": 5}, "strategy": {"lowerBound": actual}, "sha256": "a" * 64}
    best = {"queriesUsed": 1, "limits": {"queries": 5}, "strategy": {"upperBound": optimum}, "sha256": "b" * 64}
    context = dict(rules=rules, source_prefix=SimpleNamespace(sha256="c" * 64),
        normalized=SimpleNamespace(legal_actions=[{"code": "C7"}, {"code": "C8"}],
            exact_state_id="test-state", next_actor="north"), selected_root_action="C7",
        opportunity_order=tuple(Seat), escalation_events=[], budget=budget,
        maximum_conflicts=1000, maximum_seconds=10, decision_kind=decision_kind, receipt_version=receipt_version)
    tight = replace(budget, native_observed=replace(budget.native_observed, maximum_queries=12))
    with pytest.raises(ValueError, match="combined"):
        authority.compose_native_observed_play_optimality_v1(selected, best, **{**context, "budget": tight})
    assert checked == []
    if actual > optimum:
        with pytest.raises(ValueError, match="proof values disagree"):
            authority.compose_native_observed_play_optimality_v1(selected, best, **context)
    else:
        result = authority.compose_native_observed_play_optimality_v1(selected, best, **context)
        assert result["actualActionOptimal"] is (actual == optimum)
        assert result["status"] == ("selected_action_optimal" if actual == optimum else "selected_action_suboptimal")
        assert result["regret"] == optimum - actual
        assert result["bestValue"] == optimum and result["actualValue"] == actual
        assert result["queriesUsed"] == (5 if receipt_version == 1 else 3)
        assert result["schemaVersion"] == receipt_version
        assert (result.get("queryCountScope") == "certificate_construction_only") is (receipt_version == 2)
        assert result["kind"] == f"native_observed_{decision_kind}_optimality_v1"
        assert result["scope"] == f"selected_{decision_kind}_current_deal"
    assert checked == ["C7", None]
    if actual == optimum:
        shared = {**best, "status": "scoped_bound_converged", "strategy": {"lowerBound": optimum, "upperBound": optimum,
            "policy": {"root": "C7"}, "finalBranchProofs": [{"branchModelHash": "native-test"}]},
            "preparation": {"rootObservationHash": "root"}}
        checked.clear()
        result = authority.compose_native_observed_play_optimality_v1(shared, shared,
            **{**context, "budget": tight})
        assert checked == [None] and result["queriesUsed"] == (3 if receipt_version == 1 else 2)
        assert result["actualActionOptimal"] and result["regret"] == 0
        assert result["selectionSource"] == "verified_best_strategy"
        with pytest.raises(ValueError, match="does not certify"):
            authority.compose_native_observed_play_optimality_v1(shared, shared,
                **{**context, "selected_root_action": "C8"})
        assert checked == [None]


@pytest.mark.parametrize("decision_kind", ("play", "project", "escalation"))
def test_observed_optimality_owned_bounds_keep_external_verification(root, rules, monkeypatch, decision_kind):
    import yaml
    from types import SimpleNamespace
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, canonical_sha256

    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    budget = ScenarioProofBudgetV1.build("offline_heavyweight_v1", profile)
    normalized = SimpleNamespace(legal_actions=[{"code": "C7"}, {"code": "C8"}], next_actor="north",
        exact_state_id="test-state", state_hash="test-state-hash", prefix_hash="test-prefix", require_integrity=lambda: None)
    preparation = {"rootObservationHash": "root", "rootLegalActions": ["C7", "C8"]}
    source_context = {"opportunityOrder": [seat.name.lower() for seat in Seat], "escalationEvents": []}
    preparation.update({"partitions": [source_context]} if decision_kind == "escalation" else source_context)
    bounds = {"kind": "native_post_auction_policy_bounds_v1",
        "rulesHash": rules.sha256, "sourcePrefixHash": "test-prefix", "prefixHash": "test-prefix",
        "stateHash": normalized.state_hash, "exactStateId": normalized.exact_state_id,
        "decisionKind": decision_kind, "visibility": "actor_information_set", "targetTeam": "north_south",
        "selectedRootAction": None, "utility": "awarded_point_margin", "status": "scoped_bound_converged",
        "strategicCoverage": False, "preparation": preparation, "queriesUsed": 1,
        "limits": {"queries": 5, "iterations": 5, "observations": 5, "modelBytes": 1000,
            "conflictsPerSolve": 1000, "depth": 64, "seconds": 10},
        "strategy": {"lowerBound": 0, "upperBound": 0, "policy": {"root": "C7"},
            "finalBranchProofs": [{"branchModelHash": "native-test"}]}}
    bounds["sha256"] = canonical_sha256(bounds)
    verified = frozenset({bounds["sha256"]})
    context = dict(rules=rules, source_prefix=SimpleNamespace(sha256="test-prefix"), normalized=normalized,
        selected_root_action="C7", opportunity_order=tuple(Seat), escalation_events=[], budget=budget,
        maximum_conflicts=1000, maximum_seconds=10, decision_kind=decision_kind, receipt_version=2)
    checked = []
    verifier = (f"reverify_native_observed_{decision_kind}_bounds_v1" if decision_kind != "escalation"
                else "_reverify_native_observed_post_auction_bounds_v1")
    monkeypatch.setattr(authority, verifier, lambda value, **kwargs: checked.append(value["sha256"]))
    owned = authority._compose_native_observed_play_optimality_v1(bounds, bounds,
        **context, verified_bounds=verified)
    assert checked == [] and owned["queriesUsed"] == 2
    external = authority.compose_native_observed_play_optimality_v1(bounds, bounds, **context)
    assert external == owned and checked == [bounds["sha256"]]
    with pytest.raises(TypeError, match="verified_bounds"):
        authority.compose_native_observed_play_optimality_v1(bounds, bounds, **context, verified_bounds=verified)
    for substituted in ({"source_prefix": SimpleNamespace(sha256="wrong-owner")},
                        {"opportunity_order": tuple(reversed(tuple(Seat)))},
                        {"escalation_events": [{"actor": "north", "actionCode": "decline"}]}):
        with pytest.raises(ValueError, match="owned proof binding"):
            authority._compose_native_observed_play_optimality_v1(bounds, bounds,
                **{**context, **substituted}, verified_bounds=verified)
    corrupt = {**bounds, "queriesUsed": 2}
    with pytest.raises(ValueError, match="owned proof binding"):
        authority._compose_native_observed_play_optimality_v1(corrupt, corrupt,
            **context, verified_bounds=verified)
    authority._compose_native_observed_play_optimality_v1(bounds, bounds, **context)
    assert checked == [bounds["sha256"], bounds["sha256"]]


@pytest.mark.parametrize("fixed_root", (False, True))
@pytest.mark.parametrize("visibility", ("complete_state", "actor_information_set"))
def test_native_strategy_refinement_excludes_recorded_round_history(root, fixed_root, visibility, monkeypatch):
    from baloot_platform.core.project_ir import ProjectDeclarationActionV1
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
    from baloot_platform.research.result_authority import validate_native_post_auction_bounds_v1
    from baloot_platform.research.scenario_contracts import canonical_sha256

    rules, scenario, auction, branch = _native_hokum_round_branch(root, "double",
        opportunity_order=("west", "east", "north", "south"))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(scenario.model) == cp_model.OPTIMAL
    opportunities = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
    scenario.constrain_native_completed_escalation([
        {"actor": row["actor"], "actionCode": row["selectedAction"]} for row in opportunities])
    declarations = [branch.projects.extract_declaration_decision(solver, actor)
        for actor in rules.project_domain.declaration_seats(scenario.dealer)]
    scenario.constrain_native_project_prefix(tuple(ProjectDeclarationActionV1.build(
        row["selectedAction"], actor=Seat.parse(row["actor"])) for row in declarations))
    scenario.auction_fragment = replace(auction, prefix_length=4)
    scenario._recorded_play_count = 31
    scenario._complete_information_deck = tuple(range(32))
    history_hash = scenario._model_hash(search_branching="automatic")
    scenario._recorded_play_count = 30
    assert scenario._model_hash(search_branching="automatic") != history_hash
    scenario._recorded_play_count = 31
    original = str(scenario.model.proto)
    result = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._refine_symbolic_round_policy(
        scenario, Team.WEST_EAST, {}, threshold=1000, maximum_queries=5,
        maximum_observations=32, maximum_conflicts=250_000, maximum_seconds=15)
    assert result["status"] == "policy_counterexample_in_branch"
    candidate = result["candidate"]
    for stage in ("auction", "escalation", "project"):
        assert candidate[stage + "Observations"]
        assert all(row.get("authoredPrefix") for row in candidate[stage + "Observations"])
    plays = candidate["playObservations"]
    assert all(row.get("authoredPrefix") for row in plays[:31])
    assert not plays[31].get("authoredPrefix", False)
    assert len(plays[31]["legalActions"]) == 1
    assert result["domains"] == {plays[31]["observationHash"]: tuple(plays[31]["legalActions"])}
    assert str(scenario.model.proto) == original
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": candidate["deckOrder"],
        "dealer": "east", "auctionActions": [
            {key: row[key] for key in ("actor", "action", "trumpSuit") if row.get(key) is not None}
            for row in candidate["auctionActions"]],
        "escalationActions": [
            {key: row[key] for key in ("actor", "action", "requestedMode") if row.get(key) is not None}
            for row in candidate["escalation"]["observation"]["actions"]],
        "projectDeclarations": [], "projectActions": [ProjectDeclarationActionV1.build(
            row["selectedAction"], actor=Seat.parse(row["actor"])).to_dict() for row in declarations],
        "playActions": candidate["play"]["plays"][:31]})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    bounds = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._solve_native_post_auction_policy_bounds(
        exact, prefix, Team.WEST_EAST, selected_root_action=plays[31]["selectedAction"] if fixed_root else None,
        visibility=visibility,
        opportunity_order=("west", "east", "north", "south"),
        escalation_events=[{"actor": row["actor"], "actionCode": row["selectedAction"]} for row in opportunities],
        maximum_iterations=10, maximum_queries=30, maximum_observations=32,
        maximum_model_bytes=100_000_000, maximum_conflicts=250_000, maximum_depth=1, maximum_seconds=30)
    assert bounds["status"] == "scoped_bound_converged"
    assert bounds["strategy"]["lowerBound"] == bounds["strategy"]["upperBound"] == candidate["utility"]
    assert bounds["queriesUsed"] == 2 + bounds["strategy"]["queriesUsed"]
    assert bounds["strategicCoverage"] is False
    validation = dict(rules=rules, source_prefix=prefix, normalized=exact,
        opportunity_order=("west", "east", "north", "south"),
        escalation_events=[{"actor": row["actor"], "actionCode": row["selectedAction"]} for row in opportunities])
    if visibility == "actor_information_set":
        assert bounds["visibility"] == visibility
        assert bounds["preparation"]["sourceObservationQuery"]["queryMode"] == "feasibility_witness"
        assert bounds["preparation"]["rootQuery"]["queryMode"] == "feasibility_witness"
        assert bounds["preparation"]["rootQuery"] == bounds["preparation"]["sourceObservationQuery"]
        assert bounds["preparation"]["rootObservationSource"] == "source_observation_projection"
        assert bounds["preparation"]["rootObservationHash"] != bounds["preparation"]["rootQuery"][
            "candidate"]["playObservations"][31]["observationHash"]
        # This internal certificate must not enter complete-state admission.
        with pytest.raises(ValueError, match="ownership or completion"):
            validate_native_post_auction_bounds_v1(bounds, **validation)
        from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
        compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
        def source_stopped(*args, **kwargs):
            raise InformationSetCompilationStopped("maximum_seconds", stage="native_observed_play_root",
                evidence={"sourceObservationQuery": {"status": "incomplete"}})
        monkeypatch.setattr(compiler, "_native_post_auction_root_model", source_stopped)
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            compiler._solve_native_post_auction_policy_bounds(exact, prefix, Team.WEST_EAST,
                selected_root_action=None, visibility=visibility,
                opportunity_order=validation["opportunity_order"], escalation_events=validation["escalation_events"],
                maximum_iterations=10, maximum_queries=1, maximum_observations=32,
                maximum_model_bytes=100_000_000, maximum_conflicts=250_000,
                maximum_depth=1, maximum_seconds=30)
        partial = stopped.value.partial_receipt["evidence"]
        assert partial["queriesUsed"] == 1
        assert partial["status"] == "incomplete"
        assert partial["cause"]["evidence"]["sourceObservationQuery"]["status"] == "incomplete"
        import yaml
        from baloot_platform.research.result_authority import reverify_native_observed_play_bounds_v1
        from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1
        profiles = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
            encoding="utf-8"))["scenarioResearchV4"]["proofBudgets"]["profiles"]
        context = {**validation, "selected_root_action": bounds["selectedRootAction"],
            "budget": ScenarioProofBudgetV1.build("offline_heavyweight_v1", profiles["offline_heavyweight_v1"]),
            "maximum_conflicts": 250_000, "maximum_seconds": 45}
        reverify_native_observed_play_bounds_v1(bounds, **context)
        if not fixed_root:
            legacy = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._solve_native_post_auction_policy_bounds(
                exact, prefix, Team.WEST_EAST, selected_root_action=None, visibility=visibility,
                reuse_source_observation=False, observed_play_bounds_version=1, path_floor_version=1,
                opportunity_order=validation["opportunity_order"], escalation_events=validation["escalation_events"],
                maximum_iterations=10, maximum_queries=30, maximum_observations=32,
                maximum_model_bytes=100_000_000, maximum_conflicts=250_000,
                maximum_depth=1, maximum_seconds=30)
            assert "rootObservationSource" not in legacy["preparation"]
            assert legacy["queriesUsed"] == 3 + legacy["strategy"]["queriesUsed"]
            assert legacy["preparation"]["rootObservationHash"] == bounds["preparation"]["rootObservationHash"]
            assert legacy["strategy"]["lowerBound"] == bounds["strategy"]["lowerBound"]
            reverify_native_observed_play_bounds_v1(legacy, **context)
        with pytest.raises(InformationSetCompilationStopped, match="cancelled"):
            reverify_native_observed_play_bounds_v1(bounds, **context, cancel_check=lambda: True)
        if fixed_root:
            for mutation in ("bound", "coercion"):
                forged = copy.deepcopy(bounds)
                if mutation == "bound":
                    forged["strategy"]["lowerBound"] += 1
                    forged["strategy"]["upperBound"] += 1
                else:
                    forged["strategy"]["scope"]["cooperativeCeiling"] = 1
                for document in (forged["strategy"], forged):
                    document["sha256"] = canonical_sha256({key: value for key, value in document.items() if key != "sha256"})
                with pytest.raises(ValueError, match="fresh native reconstruction"):
                    reverify_native_observed_play_bounds_v1(forged, **context)
            from baloot_platform.research.result_authority import (
                build_native_observed_play_quality_artifact_v1, validate_native_observed_play_quality_artifact_v1)
            from baloot_platform.research.scenario_contracts import ScenarioProofRequestV1
            best_bounds = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._solve_native_post_auction_policy_bounds(
                exact, prefix, Team.WEST_EAST, selected_root_action=None, visibility=visibility,
                opportunity_order=validation["opportunity_order"], escalation_events=validation["escalation_events"],
                maximum_iterations=10, maximum_queries=30, maximum_observations=32,
                maximum_model_bytes=100_000_000, maximum_conflicts=250_000,
                maximum_depth=1, maximum_seconds=30)
            request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "exact",
                "visibility": visibility, "strategyClass": "pure_deterministic",
                "claim": "winning_root_action_count", "targetTeam": "west_east",
                "utility": "awarded_point_margin", "successThreshold": 0,
                "selectedAction": bounds["selectedRootAction"], "budgetProfile": context["budget"].profile_id})
            artifact_context = {key: value for key, value in context.items() if key != "selected_root_action"}
            artifact_context.update(request=request, maximum_seconds=60)
            artifact = build_native_observed_play_quality_artifact_v1(bounds, best_bounds, **artifact_context)
            quality = artifact["graph"]
            assert quality["status"] == "selected_action_optimal"
            assert quality["actualValue"] == quality["bestValue"] == candidate["utility"]
            assert quality["regret"] == 0
            assert quality["queriesUsed"] == 2 * (bounds["queriesUsed"] + best_bounds["queriesUsed"]) + 1
            assert artifact["proof"]["winningRootActionCount"] is None
            assert artifact["proof"]["claimSatisfied"] is None
            assert artifact["proofModel"]["actionQuality"]["optimalActionsComplete"] is False
            validate_native_observed_play_quality_artifact_v1(artifact, **artifact_context)
            with pytest.raises(ValueError, match="malformed"):
                validate_native_observed_play_quality_artifact_v1({**artifact, "graph": {}}, **artifact_context)
            forged_artifact = copy.deepcopy(artifact)
            forged_artifact["proof"]["winningRootActionCount"] = 1
            forged_artifact["proof"]["sha256"] = canonical_sha256({
                key: value for key, value in forged_artifact["proof"].items() if key != "sha256"})
            with pytest.raises(ValueError, match="artifact binding mismatch"):
                validate_native_observed_play_quality_artifact_v1(forged_artifact, **artifact_context)
        return
    validate_native_post_auction_bounds_v1(bounds, **validation)
    # Identity-reader compatibility only; native threshold truth and fresh
    # reconstruction are exercised by the prefix-bound tests below.
    threshold_bounds = copy.deepcopy(bounds)
    owned_strategy = threshold_bounds["strategy"]
    threshold = owned_strategy["upperBound"] + 1
    diagnostic = {"kind": "native_counterexample_prefix_threshold_v1",
        "scope": "fixed_world_cooperative_upper_bound", "rulesHash": rules.sha256,
        "targetTeam": bounds["targetTeam"], "sourceQueryHash": owned_strategy["queries"][0]["receipt"]["sha256"],
        "prefixPlayCount": len(prefix.play_actions), "requiredThreshold": threshold,
        "complementThreshold": 1 - threshold,
        "complementTargetTeam": Seat.parse(exact.next_actor).team.other.name.lower(),
        "queryMode": "feasibility_v1", "status": "upper_bound_complete",
        "solverStatus": "infeasible", "upperBound": threshold - 1}
    for mutation in (None, {"upperBound": threshold}, {"solverStatus": "unknown"},
                     {"complementThreshold": threshold}, {"requiredThreshold": True}):
        changed = copy.deepcopy(threshold_bounds)
        strategy = changed["strategy"]
        receipt = {**diagnostic, **(mutation or {})}
        receipt["sha256"] = canonical_sha256(receipt)
        strategy["queries"].insert(1, {"branchModelHash": strategy["queries"][0]["branchModelHash"],
            "receipt": receipt})
        strategy["queriesUsed"] += 1
        changed["queriesUsed"] += 1
        strategy["sha256"] = canonical_sha256({k: v for k, v in strategy.items() if k != "sha256"})
        changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
        if mutation is None:
            validate_native_post_auction_bounds_v1(changed, **validation)
        else:
            with pytest.raises(ValueError, match="prefix threshold evidence"):
                validate_native_post_auction_bounds_v1(changed, **validation)
    for mutation in ("value", "model", "incomplete", "queries", "source"):
        changed = copy.deepcopy(bounds)
        strategy = changed["strategy"]
        if mutation == "value":
            strategy["upperBound"] += 1
            strategy["lowerBound"] += 1
        elif mutation == "model":
            strategy["scope"]["branches"][0]["modelHash"] = "0" * 64
            strategy["scopeHash"] = canonical_sha256(strategy["scope"])
        elif mutation == "incomplete":
            strategy["status"] = "incomplete"
        elif mutation == "queries":
            changed["queriesUsed"] += 1
        else:
            changed["sourcePrefixHash"] = "0" * 64
        strategy["sha256"] = canonical_sha256({key: value for key, value in strategy.items() if key != "sha256"})
        changed["sha256"] = canonical_sha256({key: value for key, value in changed.items() if key != "sha256"})
        with pytest.raises(ValueError, match="post-auction"):
            validate_native_post_auction_bounds_v1(changed, **validation)

def _native_mixed_tail_quality_context(root, play_count=20):
    from baloot_platform.core.project_ir import ProjectDeclarationActionV1
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_research_executor import _complete_state_quality_artifact_v1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofRequestV1

    order = ("west", "east", "north", "south")
    rules, scenario, auction, branch = _native_hokum_round_branch(root, "double", opportunity_order=order)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(scenario.model) == cp_model.OPTIMAL
    plays = branch.play._extract(solver, cp_model.OPTIMAL, scenario._model_hash(search_branching="automatic"))
    events = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
    announcements = []
    for row in events:
        action, _, mode = row["selectedAction"].partition(":")
        if action != "decline":
            announcements.append({"actor": row["actor"], "action": action, "mode": mode or None})
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)), "dealer": "east",
        "auctionActions": [{key: value for key, value in row.to_dict().items()
            if key in {"actor", "action", "trumpSuit"} and value is not None}
            for row in auction.extract_actions(solver)],
        "escalationActions": announcements, "projectDeclarations": [],
        "projectActions": [ProjectDeclarationActionV1.build(
            branch.projects.extract_declaration_decision(solver, actor)["selectedAction"], actor=actor).to_dict()
            for actor in rules.project_domain.declaration_seats(scenario.dealer)],
        "playActions": [row.to_dict() for row in plays.plays[:play_count]]})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    assert len(exact.legal_actions) > 1
    target = Seat.parse(exact.next_actor).team
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
        "strategyClass": "pure_deterministic", "claim": "winning_root_action_count", "successThreshold": 0,
        "targetTeam": target.name.lower(), "utility": "awarded_point_margin", "budgetProfile": "test"})
    oracle = _complete_state_quality_artifact_v1(rules, exact, prefix, {"kind": "play"}, request,
        maximum_nodes=100_000, maximum_depth=12, maximum_seconds=30, cancel_check=None, opportunity_order=order)
    values = {row["action"]: row["value"] for row in oracle["proof"]["rootActions"]}
    return rules, exact, prefix, request, values, order, events


def test_native_post_auction_quality_matches_exact_mixed_tail(root):
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_research_executor import _complete_state_quality_artifact_v1
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, ScenarioProofRequestV1
    rules, exact, prefix, request, values, order, _ = _native_mixed_tail_quality_context(root)
    worst = min(values, key=lambda action: (values[action], action))
    profile = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component(
        "challenge")["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    budget = ScenarioProofBudgetV1.build("offline_heavyweight_v1", profile)
    raw = request.to_dict()
    raw.pop("sha256")
    raw.update(selectedAction=worst, budgetProfile=budget.profile_id)
    request = ScenarioProofRequestV1.build(raw)
    artifact = _complete_state_quality_artifact_v1(rules, exact, prefix, {"kind": "play"}, request,
        maximum_nodes=100_000, maximum_depth=12, maximum_seconds=60, cancel_check=None,
        opportunity_order=order, budget=budget)
    result = artifact["proofModel"]["actionQuality"]
    assert result["bestValue"] == max(values.values())
    assert result["actualValue"] == values[worst]
    assert all(values[action] == max(values.values()) for action in result["optimalActions"])
    assert result["regret"] == max(values.values()) - values[worst]
    assert result["actualActionOptimal"] is (values[worst] == max(values.values()))


def test_native_compact_complete_state_mixed_tail_scope_and_values(root, monkeypatch):
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1,
    )
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofBudgetV1, ScenarioProofRequestV1, canonical_sha256
    from baloot_platform.research.result_authority import build_native_complete_state_play_quality_artifact_v1, native_featured_play_proof_supported_v1
    from baloot_platform.research.ortools_research_executor import _complete_state_quality_artifact_v1
    rules, exact, prefix, request, values, _, _ = _native_mixed_tail_quality_context(root)
    profile = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component(
        "challenge")["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    budget = ScenarioProofBudgetV1.build("offline_heavyweight_v1", profile)
    raw = request.to_dict()
    raw.pop("sha256")
    raw.update(budgetProfile=budget.profile_id, selectedAction=min(values, key=lambda action: (values[action], action)))
    request = ScenarioProofRequestV1.build(raw)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    args = dict(budget=budget, maximum_seconds=60)
    result = compiler._prove_complete_state_play(exact, prefix, request, **args)
    assert {row["action"]: row["value"] for row in result["certificate"]["roots"]} == values
    preparation = result["preparation"]
    assert preparation["coverage"] == "exact_fixed_deal_terminal_tail"
    assert preparation["remainingActions"] == 12
    assert preparation["sourcePrefixHash"] == prefix.sha256
    assert preparation["exactStateId"] == exact.exact_state_id
    assert result["strategicCoverage"] is False
    for proof in result["roots"]["roots"].values():
        assert proof["scope"]["worlds"] == [preparation["worldId"]]
        scope = proof["scope"]["utilityScope"]
        assert scope["playStepCount"] == scope["cardCount"]
        assert scope["boundaryEvaluator"] is None
    artifact = build_native_complete_state_play_quality_artifact_v1(result, rules=rules,
        source_prefix=prefix, normalized=exact, request=request, budget=budget)
    assert artifact["proof"]["completionStatus"] == "exact_complete"
    wrapped = copy.deepcopy(result)
    query = {"schemaVersion": 3, "kind": "native_observed_path_query_v3",
        "baseModelHash": preparation["modelHash"], "queryModelHash": "1" * 64,
        "status": "infeasible", "counterexampleObjective": "minimum_utility"}
    query = {**query, "sha256": canonical_sha256(query)}
    first_root = next(iter(wrapped["roots"]["roots"].values()))
    first_root["queries"] = [*first_root.get("queries", ()),
        {"worldId": preparation["worldId"], "receipt": query}]

    def seal_wrapped():
        first_root["sha256"] = canonical_sha256({k: v for k, v in first_root.items() if k != "sha256"})
        roots = wrapped["roots"]
        roots["sha256"] = canonical_sha256({k: v for k, v in roots.items() if k != "sha256"})
        wrapped["certificate"] = compiler._certify_observed_root_actions(roots,
            success_threshold=0, claim=request.claim, selected_action=request.selected_action,
            maximum_seconds=60, action_quality=True, quality_scope="compiled_complete_state")
        wrapped["status"] = wrapped["certificate"]["status"]
        wrapped["sha256"] = canonical_sha256({k: v for k, v in wrapped.items() if k != "sha256"})

    seal_wrapped()
    repaired = build_native_complete_state_play_quality_artifact_v1(wrapped, rules=rules,
        source_prefix=prefix, normalized=exact, request=request, budget=budget)
    assert repaired["proof"]["rootActions"] == artifact["proof"]["rootActions"]
    for corrupt in ("world", "model", "checksum", "extra_field", "strip_wrapper", "direct_wrong_model", "unhashable_branch"):
        wrapper = first_root["queries"][-1]
        original = copy.deepcopy(wrapper)
        if corrupt == "world":
            wrapper["worldId"] = "0" * 64
        elif corrupt == "model":
            wrapper["receipt"]["baseModelHash"] = "0" * 64
            wrapper["receipt"]["sha256"] = canonical_sha256({k:v for k,v in wrapper["receipt"].items() if k != "sha256"})
        elif corrupt == "checksum":
            wrapper["receipt"]["sha256"] = "0" * 64
        elif corrupt == "extra_field":
            wrapper["unowned"] = True
        elif corrupt in {"strip_wrapper", "direct_wrong_model"}:
            first_root["queries"][-1] = copy.deepcopy(wrapper["receipt"])
            if corrupt == "direct_wrong_model":
                direct = first_root["queries"][-1]
                direct.update(kind="native_path_extremum_v1", worldId=preparation["worldId"], baseModelHash="0" * 64)
                direct["sha256"] = canonical_sha256({k:v for k,v in direct.items() if k != "sha256"})
        else:
            wrapper["branchId"] = []
        seal_wrapped()
        with pytest.raises(ValueError, match="root ownership"):
            build_native_complete_state_play_quality_artifact_v1(wrapped, rules=rules,
                source_prefix=prefix, normalized=exact, request=request, budget=budget)
        first_root["queries"][-1] = original
    assert "hiddenWorlds" not in artifact["proof"]
    assert artifact["proofModel"]["actionQuality"]["scope"] == "compiled_complete_state"
    assert artifact["proofModel"]["actionQuality"]["actualValue"] == values[request.selected_action]
    assert native_featured_play_proof_supported_v1(artifact) is True
    with monkeypatch.context() as routing:
        routing.setattr(OrToolsActorInformationSetCompilerV1, "_prove_complete_state_play", lambda *args, **kwargs: result)
        produced = _complete_state_quality_artifact_v1(rules, exact, prefix, {"kind": "play"}, request,
            maximum_nodes=100_000, maximum_depth=12, maximum_seconds=60, cancel_check=None, budget=budget)
        assert produced == artifact
    for field in ("modelHash", "worldId", "sourcePrefixHash", "budgetHash"):
        mutated = copy.deepcopy(result)
        mutated["preparation"][field] = "0" * 64
        mutated["preparation"]["sha256"] = canonical_sha256({k: v for k, v in mutated["preparation"].items() if k != "sha256"})
        mutated["sha256"] = canonical_sha256({k: v for k, v in mutated.items() if k != "sha256"})
        with pytest.raises(ValueError, match="compact quality"):
            build_native_complete_state_play_quality_artifact_v1(mutated, rules=rules,
                source_prefix=prefix, normalized=exact, request=request, budget=budget)
    mutated = copy.deepcopy(result)
    mutated["certificate"]["actionQuality"]["regret"] += 1
    mutated["certificate"]["sha256"] = canonical_sha256({k: v for k, v in mutated["certificate"].items() if k != "sha256"})
    mutated["sha256"] = canonical_sha256({k: v for k, v in mutated.items() if k != "sha256"})
    with pytest.raises(ValueError, match="action-quality model"):
        build_native_complete_state_play_quality_artifact_v1(mutated, rules=rules,
            source_prefix=prefix, normalized=exact, request=request, budget=budget)
    changed = prefix.to_dict()
    changed.pop("sha256")
    changed["playActions"] = changed["playActions"][:-1]
    with pytest.raises(ValueError, match="ownership"):
        compiler._prove_complete_state_play(exact, ExactTracePrefixV1.build(changed), request, **args)
    changed["projectActions"] = changed["projectActions"][:-1]
    with pytest.raises(ValueError, match="ownership"):
        compiler._prove_complete_state_play(exact, ExactTracePrefixV1.build(changed), request, **args)
    with pytest.raises(InformationSetCompilationStopped, match="maximum_depth"):
        compiler._prove_complete_state_play(exact, prefix, request, **{**args, "budget": replace(budget, maximum_depth=11)})
    with pytest.raises(InformationSetCompilationStopped, match="cancelled"):
        compiler._prove_complete_state_play(exact, prefix, request, **args, cancel_check=lambda: True)
    with pytest.raises(ValueError, match="exact play"):
        compiler._prove_complete_state_play(exact, prefix, replace(request, visibility="actor_information_set"), **args)
    def exhausted(*args, **kwargs):
        assert kwargs["native_limits"].maximum_observations <= budget.maximum_nodes
        raise InformationSetCompilationStopped("maximum_queries", evidence={"completedRoots": 1})
    monkeypatch.setattr(compiler, "_solve_observed_root_actions", exhausted)
    with pytest.raises(InformationSetCompilationStopped, match="maximum_queries") as stopped:
        compiler._prove_complete_state_play(exact, prefix, request, **args)
    partial = stopped.value.partial_receipt["evidence"]
    assert partial["preparation"]["exactStateId"] == exact.exact_state_id
    assert partial["causeReceipt"]["evidence"]["completedRoots"] == 1
    with pytest.raises(InformationSetCompilationStopped) as capped:
        compiler._prove_complete_state_play(exact, prefix, request, **args, maximum_nodes=1)
    assert capped.value.partial_receipt["evidence"]["preparation"]["maximumNodeObservations"] == 1
    wrong_world = copy.deepcopy(result["roots"])
    next(iter(wrong_world["roots"].values()))["scope"]["worlds"] = ["0" * 64]
    monkeypatch.setattr(compiler, "_solve_observed_root_actions", lambda *args, **kwargs: wrong_world)
    with pytest.raises(ValueError, match="root coverage"):
        compiler._prove_complete_state_play(exact, prefix, request, **args)
    monkeypatch.setattr(compiler, "_solve_observed_root_actions", lambda *args, **kwargs: result["roots"])
    monkeypatch.setattr(compiler, "_certify_observed_root_actions", lambda *args, **kwargs: {"status": "incomplete"})
    with pytest.raises(InformationSetCompilationStopped, match="qualification_incomplete") as stopped:
        compiler._prove_complete_state_play(exact, prefix, request, **args)
    assert canonical_sha256(stopped.value.partial_receipt["evidence"]["roots"]) == canonical_sha256(result["roots"])


@pytest.mark.real_clock
@pytest.mark.parametrize("auction", [False, True])
@pytest.mark.parametrize("stop", [None, "maximum_seconds", "cancelled", "declared_allowance", "readback_exhausted", "substitution"])
def test_native_quality_wrapper_readback_shares_generation_deadline(monkeypatch, auction, stop):
    from baloot_platform.research import ortools_research_executor as executor
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.scenario_contracts import canonical_sha256

    clock, cancelled, reads = [100.0], [False], []
    monkeypatch.setattr(executor.time, "monotonic", lambda: clock[0])
    def cancel_check():
        return cancelled[0]
    def generate(*args, **kwargs):
        assert kwargs["auction_only"] is auction
        clock[0] += 6.0 if stop == "maximum_seconds" else 3.0
        cancelled[0] = stop == "cancelled"
        core = {"status": "complete", "decisions": [{"action": "retained"}], "partialReceipt": None}
        if stop == "cancelled":
            core.update(status="incomplete", partialReceipt=InformationSetCompilationStopped(
                "maximum_nodes", stage="action_analysis").partial_receipt)
        return {**core, "sha256": canonical_sha256(core), "executionReceipt": {}}
    def validate(*args, **kwargs):
        assert kwargs["cancel_check"] is cancel_check
        assert kwargs["maximum_seconds"] == (7.0 if stop == "declared_allowance" else 3.0)
        reads.append(True)
        if stop == "readback_exhausted":
            raise InformationSetCompilationStopped("maximum_seconds", stage="quality_readback")
        if stop == "substitution":
            raise ValueError("source state substitution")
    monkeypatch.setattr(executor, "_resolve_quality_continuation_v1", generate)
    monkeypatch.setattr(authority, "_validate_native_decision_quality_evidence_v1", validate)
    wrapper = executor.resolve_quality_auction_continuation_v1 if auction else executor.resolve_quality_play_continuation_v1
    from types import SimpleNamespace
    arguments = dict(maximum_seconds=10.0, cancel_check=cancel_check,
        **({} if stop == "declared_allowance" else {"runtime_maximum_seconds": 6.0}))
    if stop == "substitution":
        with pytest.raises(ValueError, match="source state substitution"):
            wrapper(SimpleNamespace(sha256="rules"), None, object(), **arguments)
        assert reads == [True]
        return
    result = wrapper(SimpleNamespace(sha256="rules"), None, object(), **arguments)
    assert reads == ([] if stop in {"maximum_seconds", "cancelled"} else [True])
    assert result["decisions"] == [{"action": "retained"}]
    assert result["sha256"] == canonical_sha256({key: value for key, value in result.items()
        if key not in {"sha256", "executionReceipt"}})
    incomplete = stop in {"maximum_seconds", "cancelled", "readback_exhausted"}
    assert result["status"] == ("incomplete" if incomplete else "complete")
    assert result["executionReceipt"]["elapsedSeconds"] == (6.0 if stop == "maximum_seconds" else 3.0)
    if incomplete:
        assert result["partialReceipt"]["exhaustionReason"] == ("cancelled" if stop == "cancelled" else "maximum_seconds")
    if stop == "cancelled":
        cause = result["partialReceipt"]["evidence"]
        assert cause["generationReceipt"]["exhaustionReason"] == "maximum_nodes"
        assert cause["readbackReceipt"]["exhaustionReason"] == "cancelled"


def test_native_compact_complete_state_production_generate_audit_readback(root, monkeypatch):
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_research_executor import resolve_quality_play_continuation_v1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofRequestV1
    from baloot_platform.research.result_authority import validate_native_decision_quality_evidence_v1
    rules, exact, prefix, request, _, _, _ = _native_mixed_tail_quality_context(root, play_count=24)
    profile = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component(
        "challenge")["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    raw = request.to_dict()
    raw.pop("sha256")
    raw["budgetProfile"] = "offline_heavyweight_v1"
    request = ScenarioProofRequestV1.build(raw)
    limits = dict(maximum_nodes=100_000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=60, maximum_decision_seconds=20, maximum_evidence_bytes=30_000_000,
        proof_budget_profiles={request.budget_profile: profile})
    result = resolve_quality_play_continuation_v1(rules, prefix, request, **limits)
    assert result["status"] == "complete", result.get("partialReceipt")
    assert len(result["decisions"]) == 8
    assert len(result["finalPrefix"]["playActions"]) == 32
    assert any(artifact["proofModel"].get("kind") == "native_complete_state_root_outcome_v1"
        for artifact in result["artifacts"].values())
    assert all(row["actualActionOptimal"] is True and row["regret"] == 0 for row in result["decisions"])
    context = dict(source_prefix=prefix, rules_hash=rules.sha256, rules=rules, request=request)
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    initializers = []
    initialize = OrToolsExactStateNormalizerV1.__init__
    def tracked_initialize(self, *args, **kwargs):
        initializers.append(self)
        initialize(self, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(OrToolsExactStateNormalizerV1, "__init__", tracked_initialize)
        validate_native_decision_quality_evidence_v1(result, **context)
    assert len(initializers) == 1
    audited = resolve_quality_play_continuation_v1(rules, prefix, request,
        recorded_trace=ExactTracePrefixV1.build(result["finalPrefix"]), **limits)
    assert audited["status"] == "complete", audited.get("partialReceipt")
    assert audited["finalPrefix"] == result["finalPrefix"]
    with monkeypatch.context() as patch:
        patch.setattr(OrToolsExactStateNormalizerV1, "__init__", tracked_initialize)
        validate_native_decision_quality_evidence_v1(audited, **context)
    assert len(initializers) == 2
    assert initializers[0] is not initializers[1]  # Stored reads cannot inherit prior certificates.


@pytest.mark.real_clock
@pytest.mark.parametrize("native_policy_id", ["observer_safe_partner_feed_v1", "observer_safe_partner_feed_compact_v1"])
def test_native_complete_state_opposite_strategies_match_independent_tail_value(root, monkeypatch, native_policy_id):
    """Establish both native guarantees before attempting paired-path admission."""
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1

    rules, exact, prefix, request, values, _, _ = _native_mixed_tail_quality_context(root, play_count=24)
    profile = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component(
        "challenge")["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    budget = ScenarioProofBudgetV1.build("offline_heavyweight_v1", profile)
    request = replace(request, budget_profile=budget.profile_id)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    start = compiler.graph_compiler._parse_start(exact, expected_rules_hash=rules.sha256, request=request)
    actor_team = Seat.parse(exact.next_actor).team
    synthesized = compiler._synthesize_complete_state_policy_pair(exact, prefix, request,
        budget=budget, maximum_seconds=30, maximum_evidence_bytes=30_000_000)
    assert synthesized["status"] == "complete"
    assert synthesized["queriesUsed"] == sum(proof["resources"]["usage"]["counterexampleQueries"]
        for proof in synthesized["proofs"].values())
    assert synthesized["queriesUsed"] <= budget.native_observed.maximum_queries
    assert synthesized["iterationsUsed"] <= budget.native_observed.maximum_iterations
    assert synthesized["modelBytes"] <= budget.maximum_cache_bytes
    results = {}
    for target in (actor_team, actor_team.other):
        proof = synthesized["proofs"][target.name.lower()]
        assert proof["status"] == "converged"
        expected = max(values.values()) * (1 if target == actor_team else -1)
        assert proof["value"] == expected
        assert "selectedRootAction" not in proof["scope"]
        assert proof["scope"]["utilityScope"]["boundaryEvaluator"] is None
        assert proof["scope"]["utilityScope"]["playStepCount"] == len(start.cards)
        # Rebuild the native model: neither the old query cache nor an opposite
        # numeric value establishes a unilateral policy guarantee.
        fresh = compiler.graph_compiler._path_oracle(start, target=target,
            utility=request.utility, cancel_check=None)
        check = compiler._find_observed_policy_counterexample(exact.state, fresh, target,
            dict(proof["policy"]), {}, threshold=expected, known_domains=proof["domains"],
            maximum_queries=budget.native_observed.maximum_queries,
            maximum_prefixes=budget.native_observed.maximum_observations, maximum_seconds=30)
        assert check["status"] == "no_counterexample"
        results[target] = proof
    assert results[actor_team]["value"] == -results[actor_team.other]["value"]
    assert results[actor_team]["scope"]["utilityScope"] == results[actor_team.other]["scope"]["utilityScope"]
    # Materialize only the joint reached path, never turn master cuts into an
    # invented opponent policy. Free observations remain universally covered by
    # the fresh no-counterexample queries above.
    joint = compiler.graph_compiler._path_oracle(start, target=actor_team,
        utility=request.utility, cancel_check=None)
    path = ()
    for _ in range(len(start.cards) - joint.base_length):
        witness, _, _ = joint.find_path_below(joint.query_utility_bounds()[1] + 1,
            path=path, conditional_actions={}, maximum_seconds=10, cancelled=None,
            feasibility_only=True)
        assert witness is not None
        team = witness.at_prefix(joint.base_length + len(path)).next_actor.team
        proof = results[team]
        bindings, _, _ = compiler._observed_policy_bindings(exact.state, joint, witness,
            team, dict(proof["policy"]), {}, known_domains=proof["domains"], maximum_seconds=10)
        action = bindings.get(path, witness.plays[joint.base_length + len(path)])
        assert action is not None
        path = (*path, action)
    witness, value, _ = joint.find_path_below(joint.query_utility_bounds()[1] + 1,
        path=path, conditional_actions={}, maximum_seconds=10, cancelled=None,
        feasibility_only=True)
    assert witness is not None and value == max(values.values())
    assert len(witness.plays) == len(start.cards)

    from baloot_platform.research.result_authority import verify_native_complete_state_policy_pair_v1
    import json
    pair = json.loads(json.dumps({team.name.lower(): proof for team, proof in results.items()}))
    verified = verify_native_complete_state_policy_pair_v1(pair, rules=rules,
        normalized=exact, source_prefix=prefix, request=request, budget=budget)
    assert verified["wholeRoundCertified"] is False
    assert verified["coverage"] == "fixed_state_terminal_play_suffix"
    assert set(verified["guarantees"]) == set(pair)
    from baloot_platform.research.ortools_research_executor import resolve_quality_play_continuation_v1
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1, ExactTracePrefixV1
    execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
        "maximumWallSeconds": 60, "maximumEvidenceBytes": 30_000_000,
        "maximumEscalationDecisions": 20, "completeStatePlayPolicyPair": True})
    from baloot_platform.research import result_authority as authority
    native_verify = authority.verify_native_complete_state_policy_pair_v1
    generation_verifications = []
    def counted_verify(*args, **kwargs):
        generation_verifications.append(kwargs["source_prefix"].sha256)
        return native_verify(*args, **kwargs)
    monkeypatch.setattr(authority, "verify_native_complete_state_policy_pair_v1", counted_verify)
    generated = resolve_quality_play_continuation_v1(rules, prefix, request,
        execution_budget=execution, maximum_nodes=100_000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=60, maximum_decision_seconds=30, maximum_evidence_bytes=30_000_000,
        proof_budget_profiles={request.budget_profile: profile})
    assert generated["status"] == "complete", generated.get("partialReceipt")
    assert len(generated["decisions"]) == 8
    assert len(generated["finalPrefix"]["playActions"]) == 32
    assert len(generated["pairedPolicyEvidence"]) == 1
    assert generation_verifications == [prefix.sha256]
    owned_pair = next(iter(generated["pairedPolicyEvidence"].values()))
    assert owned_pair["synthesis"]["status"] == "complete"
    assert (owned_pair["synthesis"]["queriesUsed"] + owned_pair["receipt"]["counterexampleAndReachabilityQueries"]
        <= budget.native_observed.maximum_queries)
    assert all(row["actualActionOptimal"] is True and row["regret"] == 0
        and row["optimalActionCount"] is None for row in generated["decisions"])
    assert all(row["selectionReason"] == "native_complete_state_equilibrium_path" for row in generated["decisions"])
    assert all(artifact["proofModel"]["actionQuality"]["optimalActionsComplete"] is False
        and artifact["proof"]["claimSatisfied"] is None for artifact in generated["artifacts"].values())
    assert all(artifact["proof"]["rootActions"] and all(action["value"] is None
        and action["satisfies"] is None for action in artifact["proof"]["rootActions"])
        for artifact in generated["artifacts"].values())
    from baloot_platform.research.result_authority import validate_native_decision_quality_evidence_v1
    from baloot_platform.research.scenario_contracts import canonical_sha256
    validate_native_decision_quality_evidence_v1(generated, source_prefix=prefix,
        rules_hash=rules.sha256, rules=rules, request=request)
    assert generation_verifications == [prefix.sha256, prefix.sha256]
    # Audit the immutable recorded path through the same owned pair, without
    # recomputing every root at every card or substituting a recorded action.
    from baloot_platform.research import ortools_research_executor as executor_module
    original_quality = executor_module._complete_state_quality_artifact_v1
    def no_root_reproof(*args, **kwargs):
        pytest.fail("matching certified recorded path repeated all-root proof")
    with monkeypatch.context() as recorded_patch:
        recorded_patch.setattr(executor_module, "_complete_state_quality_artifact_v1", no_root_reproof)
        audited = resolve_quality_play_continuation_v1(rules, prefix, request,
            recorded_trace=ExactTracePrefixV1.build(generated["finalPrefix"]),
            execution_budget=execution, maximum_nodes=100_000, maximum_depth=8, maximum_worlds=1,
            maximum_seconds=60, maximum_decision_seconds=30, maximum_evidence_bytes=30_000_000,
            proof_budget_profiles={request.budget_profile: profile})
    assert audited["status"] == "complete", audited.get("partialReceipt")
    assert audited["finalPrefix"] == generated["finalPrefix"]
    assert all(row["selectionReason"] == "recorded_action" and row["actualActionOptimal"] is True
               for row in audited["decisions"])
    validate_native_decision_quality_evidence_v1(audited, source_prefix=prefix,
        rules_hash=rules.sha256, rules=rules, request=request)
    # Recorded card codes and integer IDs describe the same legal choices,
    # but their immutable authored prefix hashes must remain distinct.
    authored = generated["finalPrefix"].copy()
    authored.pop("sha256")
    authored["playActions"] = [*authored["playActions"][:len(prefix.play_actions)], *[
        {**row, "card": Card.parse(row["card"]).code}
        for row in authored["playActions"][len(prefix.play_actions):]]]
    authored_prefix = ExactTracePrefixV1.build(authored)
    coded_audit = resolve_quality_play_continuation_v1(rules, prefix, request,
        recorded_trace=authored_prefix,
        execution_budget=execution, maximum_nodes=100_000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=60, maximum_decision_seconds=30, maximum_evidence_bytes=30_000_000,
        proof_budget_profiles={request.budget_profile: profile})
    assert coded_audit["status"] == "complete", coded_audit.get("partialReceipt")
    assert coded_audit["finalPrefix"] == authored_prefix.to_dict()
    validate_native_decision_quality_evidence_v1(coded_audit, source_prefix=prefix,
        rules_hash=rules.sha256, rules=rules, request=request)
    focal_index = next((row["playIndex"] for row in generated["decisions"][1:]
                        if row["optimalActionCountUpperBound"] > 1), 24)
    focal_calls = []
    def focal_roots(*args, **kwargs):
        assert len(args[2].play_actions) >= focal_index, "focal count disabled reuse before its decision"
        focal_calls.append(len(args[2].play_actions))
        return original_quality(*args, **kwargs)
    with monkeypatch.context() as focal_patch:
        focal_patch.setattr(executor_module, "_complete_state_quality_artifact_v1", focal_roots)
        focal_audit = resolve_quality_play_continuation_v1(rules, prefix, request,
            recorded_trace=ExactTracePrefixV1.build(generated["finalPrefix"]),
            featured_play_index=focal_index,
            featured_decision_point={"kind": "play", "selector": {
                "trickIndex": focal_index // 4, "position": focal_index % 4}},
            execution_budget=execution, maximum_nodes=100_000, maximum_depth=8, maximum_worlds=1,
            maximum_seconds=60, maximum_decision_seconds=30, maximum_evidence_bytes=30_000_000,
            proof_budget_profiles={request.budget_profile: profile})
    assert focal_audit["status"] == "complete", focal_audit.get("partialReceipt")
    assert focal_audit["finalPrefix"] == generated["finalPrefix"]
    assert focal_index in focal_calls
    focal_row = next(row for row in focal_audit["decisions"] if row["playIndex"] == focal_index)
    focal_artifact = focal_audit["artifacts"][focal_row["proofArtifactHash"]]
    assert focal_artifact["proof"]["winningRootActionCount"] is not None
    assert focal_artifact["proofModel"]["kind"] != "native_complete_state_policy_pair_quality_v1"
    validate_native_decision_quality_evidence_v1(focal_audit, source_prefix=prefix,
        rules_hash=rules.sha256, rules=rules, request=request)
    # Generated Foundry lines need the same exact pair before the focal
    # decision; its root count must still come from an independent proof.
    focal_calls.clear()
    focal_syntheses = []
    original_synthesis = OrToolsActorInformationSetCompilerV1._synthesize_complete_state_policy_pair
    def synthesize_once(self, normalized, source, *args, **kwargs):
        focal_syntheses.append(source.sha256)
        return original_synthesis(self, normalized, source, *args, **kwargs)
    with monkeypatch.context() as focal_patch:
        focal_patch.setattr(executor_module, "_complete_state_quality_artifact_v1", focal_roots)
        focal_patch.setattr(OrToolsActorInformationSetCompilerV1,
            "_synthesize_complete_state_policy_pair", synthesize_once)
        focal_generated = resolve_quality_play_continuation_v1(rules, prefix, request,
            featured_play_index=focal_index,
            featured_decision_point={"kind": "play", "selector": {
                "trickIndex": focal_index // 4, "position": focal_index % 4}},
            execution_budget=execution, maximum_nodes=100_000, maximum_depth=8, maximum_worlds=1,
            maximum_seconds=60, maximum_decision_seconds=30, maximum_evidence_bytes=30_000_000,
            proof_budget_profiles={request.budget_profile: profile})
    assert focal_generated["status"] == "complete", focal_generated.get("partialReceipt")
    assert len(focal_generated["finalPrefix"]["playActions"]) == 32
    assert focal_syntheses == [prefix.sha256]
    assert focal_index in focal_calls
    assert all((focal_generated["artifacts"][row["proofArtifactHash"]]["proofModel"]["kind"]
        == "native_complete_state_policy_pair_quality_v1") == (row["playIndex"] < focal_index)
        for row in focal_generated["decisions"])
    generated_focal = next(row for row in focal_generated["decisions"] if row["playIndex"] == focal_index)
    generated_artifact = focal_generated["artifacts"][generated_focal["proofArtifactHash"]]
    assert generated_artifact["proof"]["winningRootActionCount"] is not None
    assert generated_artifact["proofModel"]["kind"] != "native_complete_state_policy_pair_quality_v1"
    validate_native_decision_quality_evidence_v1(focal_generated, source_prefix=prefix,
        rules_hash=rules.sha256, rules=rules, request=request)
    # A different legal card is not repaired or rejected merely by pair mismatch.
    # The independent root oracle must assess that exact submitted choice.
    other = next(action for action in values if action != generated["decisions"][0]["action"])
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["playActions"] = [*raw["playActions"], {"actor": exact.next_actor, "card": other}]
    alternate = ExactTracePrefixV1.build(raw)
    fallback_calls = []
    def audit_other(*args, **kwargs):
        assert args[4].selected_action == other
        artifact = original_quality(*args, **kwargs)
        assert artifact["proofModel"]["actionQuality"]["actualAction"] == other
        fallback_calls.append(args[4].selected_action)
        return artifact
    with monkeypatch.context() as alternate_patch:
        alternate_patch.setattr(executor_module, "_complete_state_quality_artifact_v1", audit_other)
        alternate_result = resolve_quality_play_continuation_v1(rules, prefix, request,
            recorded_trace=alternate, continue_recorded_prefix=True,
            execution_budget=execution, maximum_nodes=100_000, maximum_depth=8, maximum_worlds=1,
            maximum_seconds=60, maximum_decision_seconds=30, maximum_evidence_bytes=30_000_000,
            proof_budget_profiles={request.budget_profile: profile},
            cancel_check=lambda: bool(fallback_calls))
    assert fallback_calls == [other]
    assert alternate_result["status"] == "incomplete"
    assert alternate.to_dict() == raw | {"sha256": alternate.sha256}
    def require_fresh_verification(*args, **kwargs):
        raise RuntimeError("stale pair token requires fresh verification")
    with monkeypatch.context() as nested:
        nested.setattr(authority, "validate_native_complete_state_policy_pair_v1", require_fresh_verification)
        with pytest.raises(RuntimeError, match="stale pair token"):
            authority._validate_native_decision_quality_evidence_v1(generated, source_prefix=prefix,
                rules_hash=rules.sha256, rules=rules, request=request,
                verified_artifacts=frozenset({canonical_sha256({
                    "kind": "verified_native_complete_state_policy_pair_v1", "evidenceHash": "a" * 64})}))
    # Paired proof work must stop at the caller's deadline, not its larger tier.
    import time
    from baloot_platform.research import result_authority as authority
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    clock, calls = [0.0], []
    def normalize_pair(self, supplied, **kwargs):
        assert supplied.sha256 == prefix.sha256 and kwargs["maximum_seconds"] == 10
        assert kwargs["cancel_check"]() is False
        calls.append("normalization")
        return exact
    def verify_pair(*args, **kwargs):
        assert kwargs["cancel_check"]() is False
        clock[0] = 10
        assert kwargs["cancel_check"]() is True
        calls.append("proof")
        raise RuntimeError("paired proof respects parent deadline")
    with monkeypatch.context() as nested:
        nested.setattr(time, "monotonic", lambda: clock[0])
        nested.setattr(OrToolsExactStateNormalizerV1, "normalize_play_or_project", normalize_pair)
        nested.setattr(authority, "validate_native_complete_state_policy_pair_v1", verify_pair)
        with pytest.raises(RuntimeError, match="paired proof respects parent deadline"):
            validate_native_decision_quality_evidence_v1(generated, source_prefix=prefix,
                rules_hash=rules.sha256, rules=rules, request=request, maximum_seconds=10)
        assert calls == ["normalization", "proof"]
    # Exercise the real audit projection with the generated artifacts. The
    # stubbed outer result is transport scaffolding, NOT whole-round evidence.
    from types import SimpleNamespace
    import baloot_platform.research.service as service_module
    from baloot_platform.research.result_authority import ORTOOLS_RESULT_IDENTITY_V3_KIND
    envelope = {"scope": "all_decisions", "status": "complete", "operation": "generate",
        "rulesHash": rules.sha256, "postAuction": {"projectPlay": generated}}
    envelope["sha256"] = canonical_sha256(envelope)
    trace = {"playingQualityEvidence": {"receipt": envelope}}
    snapshot = {"bundleHash": "b" * 64, "components": {
        "rules": {"payload": rules.data, "sha256": rules.sha256}}}
    snapshot["sha256"] = canonical_sha256(snapshot)
    result = {"resultCode": "LAF-PAIR-AUDIT", "resultHash": "c" * 64,
        "provenance": {"resultIdentity": {"kind": ORTOOLS_RESULT_IDENTITY_V3_KIND,
            "executorSemanticHash": "a" * 64, "configurationSnapshotHash": snapshot["sha256"]}},
        "researchAuthority": {"traceHash": canonical_sha256(trace), "bundleHash": snapshot["bundleHash"]},
        "artifacts": {"completeTrace": {"uri": "cas://trace"}},
        "authorityArtifacts": {"configurationSnapshot": {"uri": "cas://snapshot"}}}
    with monkeypatch.context() as audit_patch:
        audit_patch.setattr(service_module, "_load_owned_result_json_artifact_v1",
            lambda store, reference, **kwargs: (snapshot, result["resultHash"])
            if reference["uri"] == "cas://snapshot" else (trace, "a" * 64))
        audit = service_module.ResearchFoundryServiceV1.get_playing_quality_audit(
            SimpleNamespace(get_result=lambda code: result, artifact_store=object()), result["resultCode"])
    assert audit["decisionCount"] == audit["optimalDecisionCount"] == audit["zeroRegretDecisionCount"] == 8
    assert [row["legalActionCount"] for row in audit["decisions"]] == [
        row["optimalActionCountUpperBound"] for row in generated["decisions"]]
    claimed_unique = json.loads(json.dumps(generated))
    claimed_unique["decisions"][0]["optimalActionCount"] = 1
    changed_row = claimed_unique["decisions"][0]
    changed_row["sha256"] = canonical_sha256({key: value for key, value in changed_row.items() if key != "sha256"})
    claimed_unique["sha256"] = canonical_sha256({key: value for key, value in claimed_unique.items()
        if key not in {"sha256", "executionReceipt"}})
    with pytest.raises(ValueError, match="paired quality selected-action"):
        validate_native_decision_quality_evidence_v1(claimed_unique, source_prefix=prefix,
            rules_hash=rules.sha256, rules=rules, request=request)
    with pytest.raises(TimeoutError, match="cancelled"):
        verify_native_complete_state_policy_pair_v1(pair, rules=rules,
            normalized=exact, source_prefix=prefix, request=request, budget=budget,
            cancel_check=lambda: True)
    continuation = verified["continuation"]
    assert len(continuation["decisions"]) == 8
    assert continuation["terminalValue"] == max(values.values())
    assert len(continuation["plays"]) == len(start.cards)
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_research_executor import _complete_state_quality_artifact_v1
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    current = prefix
    for decision in continuation["decisions"]:
        team = Seat.parse(decision["actor"]).team
        assert decision["policyReceiptHash"] == results[team]["sha256"]
        assert decision["observationHash"]
        state = normalizer.normalize_play_or_project(current)
        action = next(row["code"] for row in state.legal_actions
            if Card.parse(row["card"]) == Card.parse(decision["action"]["card"])
            and row.get("ekka", False) == decision["action"].get("ekka", False))
        owned_request = replace(request, target_team=team.name.lower(), selected_action=action)
        # No compact budget: this independent reference builds the entire small
        # subgame, rather than trusting the paired-policy optimality assertion.
        independent = _complete_state_quality_artifact_v1(rules, state, current,
            {"kind": "play"}, owned_request, maximum_nodes=100_000,
            maximum_depth=12, maximum_seconds=30, cancel_check=None)
        quality = independent["proofModel"]["actionQuality"]
        assert quality["actualActionOptimal"] is True
        assert quality["regret"] == 0
        current = normalizer.append_normalized_action(current, state, action)
    import copy
    from baloot_platform.research.scenario_contracts import canonical_sha256
    from baloot_platform.research.result_authority import validate_native_complete_state_policy_pair_v1
    validate_native_complete_state_policy_pair_v1(json.loads(json.dumps(verified)), pair,
        rules=rules, normalized=exact, source_prefix=prefix, request=request, budget=budget)
    for field in ("sourcePrefixHash", "proofRequestHash", "proofBudgetHash", "continuation"):
        altered = copy.deepcopy(verified)
        if field == "continuation":
            altered[field]["decisions"][0]["action"]["card"] = "forged"
            altered[field]["sha256"] = canonical_sha256({k: v for k, v in altered[field].items() if k != "sha256"})
        else:
            altered[field] = "0" * 64
        altered["sha256"] = canonical_sha256({k: v for k, v in altered.items() if k != "sha256"})
        with pytest.raises(ValueError, match="paired policy stored"):
            validate_native_complete_state_policy_pair_v1(altered, pair, rules=rules,
                normalized=exact, source_prefix=prefix, request=request, budget=budget)
    for mutation in ("root_restriction", "world_substitution", "duplicate_policy"):
        altered = copy.deepcopy(pair)
        receipt = altered[actor_team.name.lower()]
        if mutation == "root_restriction":
            receipt["scope"]["selectedRootAction"] = "forged"
        elif mutation == "world_substitution":
            receipt["scope"]["worlds"] = ["0" * 64]
        else:
            receipt["policy"] = [("forged", "C7"), ("forged", "C7")]
        receipt["scopeHash"] = canonical_sha256(receipt["scope"])
        receipt["sha256"] = canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})
        with pytest.raises(ValueError, match="paired policy"):
            verify_native_complete_state_policy_pair_v1(altered, rules=rules,
                normalized=exact, source_prefix=prefix, request=request, budget=budget)

    # Registered visible programs use the same proof owner, not stored numeric
    # constants. This non-forced eight-card suffix has an independent oracle.
    programs = {team.name.lower(): native_policy_id for team in Team}
    source = compiler.graph_compiler._path_oracle(start, target=actor_team,
        utility=request.utility, cancel_check=None)
    source_proto, source_hash = str(source.build.model.proto), source.model_hash
    bound = source.with_native_play_policy(team=actor_team, policy_id=programs[actor_team.name.lower()])
    assert bound.build.model is not source.build.model and bound.model_hash != source_hash
    assert str(source.build.model.proto) == source_proto and source.model_hash == source_hash
    native_pair = compiler._synthesize_complete_state_policy_pair(exact, prefix, request,
        budget=budget, maximum_seconds=30, native_play_policy_ids=programs)
    assert native_pair["queriesUsed"] == 2 and native_pair["iterationsUsed"] == 0
    assert {key: proof["value"] for key, proof in native_pair["proofs"].items()} == {
        team.name.lower(): max(values.values()) * (1 if team == actor_team else -1) for team in Team}
    native_verified = verify_native_complete_state_policy_pair_v1(native_pair["proofs"], rules=rules,
        normalized=exact, source_prefix=prefix, request=request, budget=budget)
    assert native_verified["wholeRoundCertified"] is False
    assert native_verified["continuation"]["terminalValue"] == max(values.values())
    assert len(native_verified["continuation"]["decisions"]) == 8
    assert all(row["selection"] == "native_policy" for row in native_verified["continuation"]["decisions"])
    assert native_verified["jointWitnessReuse"] is True
    assert native_verified["counterexampleAndReachabilityQueries"] == 3
    assert len(native_verified["continuation"]["queryHashes"]) == 1
    legacy_pair = verify_native_complete_state_policy_pair_v1(native_pair["proofs"], rules=rules,
        normalized=exact, source_prefix=prefix, request=request, budget=budget, reuse_joint_witness=False)
    assert "jointWitnessReuse" not in legacy_pair
    assert legacy_pair["continuation"]["decisions"] == native_verified["continuation"]["decisions"]
    assert legacy_pair["counterexampleAndReachabilityQueries"] == 11
    validate_native_complete_state_policy_pair_v1(legacy_pair, native_pair["proofs"], rules=rules,
        normalized=exact, source_prefix=prefix, request=request, budget=budget)
    native_generated = resolve_quality_play_continuation_v1(rules, prefix, request,
        paired_policy_proofs=native_pair["proofs"], execution_budget=execution,
        maximum_nodes=100_000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=60, maximum_decision_seconds=30, maximum_evidence_bytes=30_000_000,
        proof_budget_profiles={request.budget_profile: profile})
    assert native_generated["status"] == "complete"
    assert len(native_generated["decisions"]) == 8 and len(native_generated["finalPrefix"]["playActions"]) == 32
    validate_native_decision_quality_evidence_v1(native_generated, source_prefix=prefix,
        rules_hash=rules.sha256, rules=rules, request=request, maximum_seconds=60)
    for field in ("nativePolicyModelHash", "queryHashes", "value"):
        changed = copy.deepcopy(native_pair["proofs"])
        proof = changed[actor_team.name.lower()]
        proof[field] = ["0" * 64] if field == "queryHashes" else proof[field] + 1 if field == "value" else "0" * 64
        proof["sha256"] = canonical_sha256({key: value for key, value in proof.items() if key != "sha256"})
        with pytest.raises(ValueError, match="paired native policy"):
            verify_native_complete_state_policy_pair_v1(changed, rules=rules,
                normalized=exact, source_prefix=prefix, request=request, budget=budget)
    with pytest.raises(ValueError, match="supported policy"):
        compiler._synthesize_complete_state_policy_pair(exact, prefix, request, budget=budget,
            maximum_seconds=30, native_play_policy_ids={actor_team.name.lower(): "unknown"})


@pytest.mark.parametrize("completed_teams", [0, 1])
def test_native_pair_synthesis_retains_aggregate_interrupted_usage(root, monkeypatch, completed_teams):
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1)
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1
    rules, exact, prefix, request, _, _, _ = _native_mixed_tail_quality_context(root, play_count=24)
    profile = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component(
        "challenge")["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    budget = ScenarioProofBudgetV1.build("offline_heavyweight_v1", profile)
    request = replace(request, budget_profile=budget.profile_id)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    calls = 0

    def stop_during_team(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert 0 < kwargs["maximum_evidence_bytes"] < 1_000_000
        if calls <= completed_teams:
            return {"value": 0, "resources": {"usage": {"counterexampleQueries": 3, "masterIterations": 2}}}
        raise InformationSetCompilationStopped("maximum_seconds", stage="native_observed_worlds",
            evidence={"resources": {"usage": {"counterexampleQueries": 4, "masterIterations": 1}}})

    monkeypatch.setattr(compiler, "_solve_observed_worlds", stop_during_team)
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._synthesize_complete_state_policy_pair(exact, prefix, request, budget=budget, maximum_seconds=30,
            maximum_evidence_bytes=1_000_000)
    evidence = stopped.value.partial_receipt["evidence"]
    assert len(evidence["proofs"]) == completed_teams
    assert evidence["queriesUsed"] == completed_teams * 3 + 4
    assert evidence["iterationsUsed"] == completed_teams * 2 + 1
    assert evidence["modelBytes"] > 0
    assert evidence["causeReceipt"]["evidence"]["resources"]["usage"]["counterexampleQueries"] == 4
    before = calls
    with pytest.raises(InformationSetCompilationStopped) as capped:
        compiler._synthesize_complete_state_policy_pair(exact, prefix, request, budget=budget,
            maximum_seconds=30, maximum_evidence_bytes=1)
    assert calls == before and capped.value.reason == "maximum_evidence_bytes"
    assert capped.value.partial_receipt["evidence"]["queriesUsed"] == 0
    if completed_teams == 0:
        from baloot_platform.research.ortools_complete_state_graph import _CpSatPathOracle
        from baloot_platform.research.ortools_play_tail import PlayTailSolveStopped
        def stop_native_query(*args, **kwargs):
            raise PlayTailSolveStopped("cancelled", evidence={"queryModelHash": "a" * 64})
        monkeypatch.setattr(_CpSatPathOracle, "find_path_below", stop_native_query)
        with pytest.raises(InformationSetCompilationStopped) as native_stopped:
            compiler._synthesize_complete_state_policy_pair(exact, prefix, request,
                budget=budget, maximum_seconds=30, native_play_policy_ids={
                    team.name.lower(): "observer_safe_partner_feed_v1" for team in Team})
        native_evidence = native_stopped.value.partial_receipt["evidence"]
        assert native_stopped.value.reason == "cancelled"
        assert native_evidence["proofs"] == {} and native_evidence["queriesUsed"] == 1
        assert native_evidence["modelBytes"] > native_evidence["preparation"]["modelBytes"]
        assert native_evidence["causeReceipt"]["evidence"]["queryModelHash"] == "a" * 64


def test_native_ranked_pair_certifies_retained_mixed_round(root):
    from baloot_platform.config import build_bundle
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.result_authority import verify_native_complete_state_policy_pair_v1
    from baloot_platform.research.scenario_contracts import (
        ExactTracePrefixV1, ScenarioProofBudgetV1, ScenarioProofRequestV1, canonical_sha256)

    bundle = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml")
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    prefix = ExactTracePrefixV1.build(dict(schemaVersion=1, dealer="east",
        deckOrder=[7,24,25,0,4,6,1,3,5,10,11,12,27,31,13,30,9,29,16,18,2,26,28,19,21,23,14,15,17,8,20,22],
        auctionActions=[{"actor": "north", "action": "sun"}], projectActions=[], playActions=[]))
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    declaration = normalizer.normalize_play_or_project(prefix)
    assert len(declaration.legal_actions) == 1
    prefix = normalizer.append_normalized_action(prefix, declaration, declaration.legal_actions[0]["code"])
    exact = normalizer.normalize_play_or_project(prefix)
    assert exact.decision_kind == "play" and not prefix.play_actions
    profile = bundle.component("challenge")["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    budget = ScenarioProofBudgetV1.build("offline_heavyweight_v1", profile)
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="complete_state",
        strategyClass="pure_deterministic", claim="winning_root_action_count",
        targetTeam="north_south", utility="awarded_point_margin", budgetProfile=budget.profile_id))
    policies = {}
    for seat, seed in (("north", 47), ("south", 9), ("west", 0), ("east", 0)):
        core = dict(schemaVersion=1, seat=seat, policyId="fixed_action_map_v1", version="1.0.0",
            seed=seed, choices=[], materialization="seeded_legal_action_order_v1")
        policies[seat] = {key: value for key, value in core.items() if key != "seat"}
        policies[seat]["profileHash"] = canonical_sha256(core)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    synthesis = compiler._synthesize_complete_state_policy_pair(exact, prefix, request,
        budget=budget, maximum_seconds=30, play_policies=policies)
    assert {team: proof["value"] for team, proof in synthesis["proofs"].items()} == {
        "north_south": 64, "west_east": -64}
    verified = verify_native_complete_state_policy_pair_v1(synthesis["proofs"], rules=rules,
        normalized=exact, source_prefix=prefix, request=request, budget=budget)
    assert len(verified["continuation"]["decisions"]) == 32
    assert verified["continuation"]["terminalValue"] == 64
    assert verified["wholeRoundCertified"] is False
    from baloot_platform.research.ortools_research_executor import resolve_quality_play_continuation_v1
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1, ScenarioPlayingQualityPolicyV1
    execution = ScenarioQualityExecutionBudgetV1.build(dict(schemaVersion=1,
        maximumWallSeconds=90, maximumEvidenceBytes=30_000_000, maximumEscalationDecisions=20,
        completeStatePlayPolicyPair=True, bidIncumbentProof=dict(schemaVersion=1,
            maximumSeconds=10, maximumQueries=24, playPolicies=policies)))
    policy = ScenarioPlayingQualityPolicyV1.build(dict(schemaVersion=2, mode="exact", visibility="complete_state",
        decisions={kind: dict(utility="awarded_point_margin", budgetProfile=budget.profile_id)
                   for kind in ("bid", "escalation", "project", "play")}))
    generated = resolve_quality_play_continuation_v1(rules, prefix, policy,
        execution_budget=execution, maximum_nodes=100_000, maximum_depth=36, maximum_worlds=1,
        maximum_seconds=90, maximum_decision_seconds=90, maximum_evidence_bytes=30_000_000,
        proof_budget_profiles={request.budget_profile: profile})
    assert generated["status"] == "complete", generated.get("partialReceipt")
    assert len(generated["finalPrefix"]["playActions"]) == 32
    assert all(row["actualActionOptimal"] is True and row["regret"] == 0 for row in generated["decisions"])
    assert next(iter(generated["pairedPolicyEvidence"].values()))["synthesis"]["proofs"] == synthesis["proofs"]
    forged = copy.deepcopy(synthesis["proofs"])
    forged["north_south"]["rankedPlayPolicies"]["north"]["seed"] += 1
    changed = forged["north_south"]["rankedPlayPolicies"]["north"]
    changed["profileHash"] = canonical_sha256({"seat": "north", **{k: v for k, v in changed.items() if k != "profileHash"}})
    forged["north_south"]["sha256"] = canonical_sha256({k: v for k, v in forged["north_south"].items() if k != "sha256"})
    with pytest.raises(ValueError):
        verify_native_complete_state_policy_pair_v1(forged, rules=rules,
            normalized=exact, source_prefix=prefix, request=request, budget=budget)


@pytest.mark.real_clock
@pytest.mark.parametrize("stop", ["timeout", "evidence"])
def test_native_paired_quality_timeout_retains_incomplete_receipt(root, monkeypatch, stop):
    from baloot_platform.config import build_bundle
    from baloot_platform.research import result_authority
    from baloot_platform.research.ortools_research_executor import resolve_quality_play_continuation_v1
    rules, _, prefix, request, _, _, _ = _native_mixed_tail_quality_context(root, play_count=24)
    profile = build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component(
        "challenge")["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    request = replace(request, budget_profile="offline_heavyweight_v1")

    def exhausted(*args, **kwargs):
        if stop == "timeout":
            raise TimeoutError("paired policy verification budget exhausted")
        return {"proofBudgetHash": kwargs["budget"].sha256}

    def reject_oversized_pair(*args, **kwargs):
        pytest.fail("oversized shared evidence reached action artifact construction")

    monkeypatch.setattr(result_authority, "verify_native_complete_state_policy_pair_v1", exhausted)
    monkeypatch.setattr(result_authority, "build_native_paired_play_action_artifact_v1", reject_oversized_pair)
    result = resolve_quality_play_continuation_v1(rules, prefix, request,
        paired_policy_proofs={}, maximum_nodes=100_000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=30, maximum_decision_seconds=20,
        maximum_evidence_bytes=1 if stop == "evidence" else 30_000_000,
        proof_budget_profiles={request.budget_profile: profile})
    assert result["status"] == "incomplete"
    assert result["decisions"] == []
    assert result["finalPrefix"] == prefix.to_dict()
    if stop == "timeout":
        assert result["partialReceipt"]["reason"] == "paired policy verification budget exhausted"
    else:
        assert result["partialReceipt"]["exhaustionReason"] == "maximum_evidence_bytes"
        assert result["executionReceipt"]["evidenceBytes"] == 0
        assert "pairedPolicyEvidence" not in result


def test_native_project_prefix_preserves_a_real_declared_project(root):
    from baloot_platform.core.project_ir import ProjectDeclarationActionV1
    rules, scenario, _, branch = _native_hokum_round_branch(root, "normal")
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(scenario.model) == cp_model.OPTIMAL
    actions = []
    selected = None
    for seat in rules.project_domain.declaration_seats(scenario.dealer):
        project = next((row for row in branch.projects.projects
            if row.owner == seat and solver.value(row.declared)), None)
        actions.append(ProjectDeclarationActionV1.build(
            "pass" if project is None else "declare:" + project.project_id, actor=seat))
        if project is not None:
            selected = project
            break
    assert selected is not None
    scenario.constrain_native_project_prefix(actions)
    assert solver.solve(scenario.model) == cp_model.OPTIMAL
    assert solver.value(selected.declared) == 1
    changed = scenario.model.clone()
    changed.add(changed.get_bool_var_from_proto_index(selected.declared.index) == 0)
    assert solver.solve(changed) == cp_model.INFEASIBLE


@pytest.mark.parametrize("authored_bid", [False, True])
def test_native_constant_policy_query_is_owned_and_preserves_history(root, authored_bid):
    from baloot_platform.research.scenario_contracts import canonical_sha256

    _, scenario, auction, _ = _native_hokum_round_branch(root, "normal")
    if authored_bid:
        scenario.auction_fragment = replace(auction, prefix_length=1)
    before = str(scenario.model.proto)
    source_hash = scenario._model_hash(search_branching="automatic")
    result = scenario.find_native_round_with_constant_policy_below(
        target=Team.NORTH_SOUTH, policy_team=Team.NORTH_SOUTH if authored_bid else Team.WEST_EAST,
        auction_action="pass", escalation_action="decline", threshold=1000,
        maximum_seconds=30, maximum_conflicts=250_000)
    assert result["status"] == "queried" and result["strategicCoverage"] is False
    assert result["sourceModelHash"] == source_hash
    assert result["rulesHash"] == scenario.rules.sha256 and result["threshold"] == 1000
    assert result["policy"]["auctionPrefixLength"] == int(authored_bid)
    assert len(result["bindings"]) == 2
    assert all(row["status"] == "bound" and row["solverStatus"] == "INFEASIBLE" for row in result["bindings"])
    query = result["query"]
    assert query["solverStatus"] == "optimal" and query["candidate"] is not None
    assert query["candidate"]["auctionActions"][0]["action"] == "hokum"
    assert len(query["candidate"]["play"]["plays"]) == 32
    assert query["baseModelHash"] != source_hash
    assert result["sha256"] == canonical_sha256({k: v for k, v in result.items() if k != "sha256"})
    assert str(scenario.model.proto) == before
    assert scenario._observed_auction_bindings == scenario._observed_escalation_bindings == {}


@pytest.mark.real_clock
@pytest.mark.parametrize("case", ("first_purchase", "unreachable_all_passed", "illegal_second_round", "purchase_recall", "pass"))
def test_native_constant_auction_policy_checks_its_reachable_history(root, case):
    import hashlib
    import time
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, canonical_sha256
    from baloot_platform.research.ortools_scenario_model import _auction_logic_v1

    compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
    if case == "illegal_second_round":
        raw = prefix.to_dict()
        raw.pop("sha256")
        raw["auctionActions"] = [{"action": "pass"}] * 7
        prefix = ExactTracePrefixV1.build(raw)
        exact = OrToolsExactStateNormalizerV1(compiler.rules, native_cp_sat=True).normalize(
            prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    scenario = compiler._native_bid_root_model(exact, prefix, visibility="complete_state")
    if case != "illegal_second_round":
        scenario.constrain_native_auction_root_choice("pass", actor=Seat.parse(exact.next_actor))
    if case == "unreachable_all_passed":
        scenario.model.add(scenario.auction_fragment.states[-1]["allPassed"] == 1)
    purchase = compiler._card(exact.state["purchaseCard"])
    action = ("pass" if case == "pass" else "sun" if case == "first_purchase"
              else f"hokum:{purchase.suit.name.lower()}")
    choices = scenario._native_constant_nonplay_choices(Team.WEST_EAST, "auction", action)
    original = str(scenario.model.proto)
    if case in {"pass", "unreachable_all_passed"}:
        legacy_probe = scenario.model.clone()
        _, both, _ = _auction_logic_v1(legacy_probe, "constant_auction_policy_legality")
        legacy_probe.add_bool_or([both(str(i), matches, legal.Not())
            for i, (matches, _, legal) in enumerate(choices)])
        legacy_hash = hashlib.sha256(str(legacy_probe.proto).encode()).hexdigest()
        if case == "unreachable_all_passed":
            old_solver = cp_model.CpSolver()
            old_solver.parameters.num_search_workers = 1
            old_solver.parameters.max_time_in_seconds = 10
            # The old probe falsely rejected this policy on a history its
            # first purchase excludes. Preserve that actual regression.
            assert old_solver.solve(legacy_probe) == cp_model.OPTIMAL
    bindings = {}
    receipt = scenario._bind_native_policy_choice(observation_hash=canonical_sha256({"case": case}),
        action=action, choices=choices, kind="constant_auction", bindings=bindings,
        deadline=time.monotonic() + 10, maximum_conflicts=250000, cancel_check=None)
    assert receipt["sha256"] == canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})
    if case in {"illegal_second_round", "purchase_recall"}:
        assert receipt["status"] == "invalid" and receipt["solverStatus"] == "OPTIMAL"
        assert str(scenario.model.proto) == original and not bindings
        if case == "illegal_second_round":
            assert action not in {row["code"] for row in exact.legal_actions}
        return
    assert receipt["status"] == "bound" and receipt["solverStatus"] == "INFEASIBLE"
    assert bindings[receipt["observationHash"]] == receipt
    if case == "pass":
        assert receipt["legalityModelHash"] == legacy_hash
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    status = solver.solve(scenario.model)
    if case == "unreachable_all_passed":
        assert status == cp_model.INFEASIBLE
    else:
        assert status == cp_model.OPTIMAL
        actions = scenario.auction_fragment.extract_actions(solver)
        assert actions[0].action == "pass"
        if case == "first_purchase":
            assert len(actions) >= 2 and actions[1].action == "sun"
            assert actions[1].actor.team == Team.WEST_EAST


@pytest.mark.parametrize("dealer", tuple(Seat))
def test_native_root_actor_sun_policy_preserves_buyer_and_fixed_root(root, dealer):
    from baloot_platform.research.ortools_scenario_model import _CONTRACT_IDS

    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    actor = rules.structural_domain.dealer_right(dealer)
    scenario = OrToolsScenarioModelV1(rules=rules, dealer=dealer)
    fragment = scenario.add_native_auction_fragment(
        required_purchase_card=Card.of(Suit.SPADES, Rank.ACE), prefix_length=0)
    policy_id = "prefer_root_actor_sun_partner_pass_v1"
    with pytest.raises(ValueError, match="fixed root owned"):
        scenario._native_visible_auction_policy_choices(actor.team, policy_id)
    scenario.constrain_native_auction_root_choice("pass", actor=actor)
    with pytest.raises(ValueError, match="fixed root owned"):
        scenario._native_visible_auction_policy_choices(actor.team.other, policy_id)
    for matches, selected, available in scenario._native_visible_auction_policy_choices(actor.team, policy_id):
        scenario.model.add(selected == 1).only_enforce_if(matches)
        scenario.model.add(available == 1).only_enforce_if(matches)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(scenario.model) == cp_model.OPTIMAL
    actions = fragment.extract_actions(solver)
    assert actions[0].action == "pass"
    assert all(row.action == "pass" for row in actions if row.actor == actor.partner)
    expected = {"contract": _CONTRACT_IDS[Contract.SUN], "buyer": int(actor),
        "recipient": int(actor), "declarer": int(actor)}
    counterexample = scenario.model.clone()
    alternatives = []
    for key, value in expected.items():
        different = counterexample.new_bool_var("different_" + key)
        counterexample.add(fragment.states[-1][key] != value).only_enforce_if(different)
        alternatives.append(different)
    counterexample.add_bool_or(alternatives)
    assert solver.solve(counterexample) == cp_model.INFEASIBLE


def test_native_constant_auction_continuation_preserves_selected_root(root):
    _, scenario, auction, _ = _native_hokum_round_branch(root, "normal")
    scenario.constrain_native_auction_root_choice("hokum:diamonds", actor=Seat.NORTH)
    source = str(scenario.model.proto)
    args = dict(target=Team.NORTH_SOUTH, policy_team=Team.NORTH_SOUTH,
        auction_action="pass", escalation_action=None, threshold=1000,
        maximum_seconds=10, maximum_conflicts=250000, optimize_margin=False)
    receipt = scenario.find_native_round_with_constant_policy_below(**args)
    assert receipt["status"] == "queried"
    assert receipt["policy"]["auctionPrefixLength"] == 0
    assert receipt["policy"]["auctionContinuationStart"] == 1
    assert receipt["query"]["status"] == "witness_found"
    assert receipt["query"]["candidate"]["auctionActions"][0]["action"] == "hokum"
    assert str(scenario.model.proto) == source
    scenario.constrain_native_auction_root_choice("hokum:diamonds", actor=Seat.NORTH)
    assert str(scenario.model.proto) == source
    with pytest.raises(ValueError, match="cannot be replaced"):
        scenario.constrain_native_auction_root_choice("sun", actor=Seat.NORTH)
    assert str(scenario.model.proto) == source
    # Only the chosen decision is excluded: later matching actors remain bound.
    assert len(scenario._native_constant_nonplay_choices(Team.NORTH_SOUTH, "auction", "pass")) == len(auction.choices) - 1
    assert len(scenario._native_constant_nonplay_choices(Team.WEST_EAST, "auction", "pass")) == len(auction.choices)
    # Matching constants are harmless and retain their historical lowering.
    assert len(scenario._native_constant_nonplay_choices(Team.NORTH_SOUTH, "auction", "hokum:diamonds")) == len(auction.choices)


@pytest.mark.parametrize("stage", ("escalation", "project"))
def test_native_constant_future_policy_does_not_rewrite_recorded_actions(root, stage):
    from baloot_platform.core.project_ir import ProjectDeclarationActionV1

    _, scenario, _, branch = _native_hokum_round_branch(root, "double")
    original = scenario.find_native_round_margin_below(target=Team.NORTH_SOUTH, threshold=10000,
        maximum_seconds=30, maximum_conflicts=250_000)
    assert original["status"] == "witness_found"
    events = [{"actor": row["actor"], "actionCode": row["selectedAction"]}
              for row in original["candidate"]["escalationObservations"]]
    scenario.constrain_native_completed_escalation(events)
    project_actions = None
    if stage == "project":
        order = scenario.rules.project_domain.declaration_seats(scenario.dealer)
        declarations = {row["actor"]: row["selectedAction"] for row in original["candidate"]["projectObservations"]}
        scenario.constrain_native_project_prefix(tuple(ProjectDeclarationActionV1.build(
            declarations[seat.name.lower()], actor=seat) for seat in order))
        declared = next(code for code in branch.projects.declaration_actions[int(Seat.NORTH)]
                        if code != declarations["north"])
        project_actions = {"north": declared}
        policy_team = Team.NORTH_SOUTH
    else:
        announced = next(row for row in events if row["actionCode"] != "decline")
        policy_team = Seat.parse(announced["actor"]).team
    before = str(scenario.model.proto)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(scenario.model) == cp_model.OPTIMAL
    result = scenario.find_native_round_with_constant_policy_below(target=Team.NORTH_SOUTH,
        policy_team=policy_team, auction_action=None,
        escalation_action="decline" if stage == "escalation" else None,
        project_actions=project_actions, threshold=10000, maximum_seconds=30, maximum_conflicts=250_000)
    assert result["status"] == "queried"
    assert result["query"]["status"] == "witness_found"
    actual = result["query"]["candidate"]
    assert [{"actor": row["actor"], "actionCode": row["selectedAction"]}
            for row in actual["escalationObservations"]] == events
    if stage == "project":
        assert {row["actor"]: row["selectedAction"] for row in actual["projectObservations"]} == declarations
    assert str(scenario.model.proto) == before


def test_native_constant_policy_rejects_invalid_and_incomplete_legality(root, monkeypatch):
    import baloot_platform.research.ortools_scenario_model as module

    _, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    before = str(scenario.model.proto)
    args = dict(target=Team.NORTH_SOUTH, policy_team=Team.WEST_EAST,
        auction_action="pass", escalation_action="gahwa", threshold=1000,
        maximum_seconds=30, maximum_conflicts=250_000)
    invalid = scenario.find_native_round_with_constant_policy_below(**args)
    assert invalid["status"] == "invalid_policy" and invalid["query"] is None
    assert invalid["bindings"][-1]["status"] == "invalid"
    assert str(scenario.model.proto) == before
    for changes in ({"auction_action": "unknown"}, {"escalation_action": "unknown"},
                    {"auction_action": None, "escalation_action": None}, {"maximum_seconds": True},
                    {"maximum_conflicts": False}, {"policy_team": "west_east"}, {"threshold": True}):
        with pytest.raises(ValueError):
            scenario.find_native_round_with_constant_policy_below(**{**args, **changes})
        assert str(scenario.model.proto) == before
    args["escalation_action"] = "decline"
    branch = scenario._native_round_fragment
    scenario._native_round_fragment = replace(branch, scope={**branch.scope, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="scope integrity"):
        scenario.find_native_round_with_constant_policy_below(**args)
    scenario._native_round_fragment = branch
    cancelled = scenario.find_native_round_with_constant_policy_below(**args, cancel_check=lambda: True)
    assert cancelled["status"] == "incomplete" and cancelled["bindings"] == []
    assert cancelled["exhaustionReason"] == "cancelled"
    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", lambda *a, **kw: cp_model.UNKNOWN)
    incomplete = scenario.find_native_round_with_constant_policy_below(**args)
    assert incomplete["status"] == "incomplete" and incomplete["query"] is None
    assert incomplete["bindings"][0]["solverStatus"] == "UNKNOWN"
    assert incomplete["exhaustionReason"] == "policy_legality_incomplete"
    statuses = iter((cp_model.INFEASIBLE, cp_model.UNKNOWN))
    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", lambda *a, **kw: next(statuses))
    partial = scenario.find_native_round_with_constant_policy_below(**args)
    assert partial["status"] == "incomplete" and partial["query"] is None
    assert [row["status"] for row in partial["bindings"]] == ["bound", "incomplete"]
    assert str(scenario.model.proto) == before
    scenario._observed_auction_bindings["already-bound"] = {}
    with pytest.raises(ValueError, match="unbound"):
        scenario.find_native_round_with_constant_policy_below(**args)


@pytest.mark.real_clock
@pytest.mark.parametrize("level", ["normal", "double"])
def test_native_round_branch_connects_symbolic_auction_to_terminal_score(root, level):
    from baloot_platform.research.reference_play import ReferencePlayInterpreterV1, ReferencePlayV1
    from baloot_platform.research.reference_scoring import ReferenceScoringInterpreterV1

    rules, scenario, auction, branch = _native_hokum_round_branch(root, level)
    solver = cp_model.CpSolver()
    result = branch.play.solve(solver=solver, maximum_time_seconds=15)
    assert len(result.plays) == 32
    assert branch.play.model is scenario.model is branch.scoring.model
    assert branch.scope["allPassedCovered"] is False
    assert branch.scope["strategicCoverage"] is False
    reference = ReferencePlayInterpreterV1.from_rules(rules.data)
    hands = [list(hand) for hand in result.initial_hands]
    for step, action in enumerate(result.plays):
        trick = [ReferencePlayV1(row.actor, row.card, row.ekka)
                 for row in result.plays[(step // 4) * 4:step]]
        assert action.card in reference.legal_moves(hands[int(action.actor)], trick,
            actor=action.actor, contract=Contract.HOKUM, trump=Suit.DIAMONDS)
        hands[int(action.actor)].remove(action.card)
        if step % 4 == 3:
            trick.append(ReferencePlayV1(action.actor, action.card, action.ekka))
            assert result.trick_winners[step // 4] == reference.trick_winner(
                trick, contract=Contract.HOKUM, trump=Suit.DIAMONDS)
    assert all(not hand for hand in hands)
    inputs = branch.scoring.inputs
    counting = branch.scoring.counting_team
    expected = ReferenceScoringInterpreterV1.from_rules(rules.data).score(
        contract=Contract.HOKUM, declarer=Seat(solver.value(auction.buyer)), escalation=level,
        trick_raw_points=result.trick_raw_points, trick_winners=result.trick_winners,
        project_points=tuple(solver.value(value) for value in inputs.project_points),
        baloot_points=tuple(solver.value(value) for value in inputs.baloot_points),
        counting_team=solver.value(counting))
    assert tuple(solver.value(value) for value in branch.scoring.awarded_points) == expected.awarded_points
    assert solver.value(branch.scoring.round_winner) == int(expected.winner)
    before = str(scenario.model.proto)
    margin = expected.awarded_points[0] - expected.awarded_points[1]
    query = dict(target=Team.NORTH_SOUTH, maximum_seconds=15, maximum_conflicts=250_000)
    witness = scenario.find_native_round_margin_below(threshold=margin + 1, **query)
    assert witness["status"] == "witness_found"
    assert witness["candidate"]["utility"] == margin
    assert len(witness["candidate"]["play"]["plays"]) == 32
    from baloot_platform.core.models import LegalPlayActionV1
    decisions = witness["candidate"]["playDecisions"]
    assert len(decisions) == 32
    for index, (decision, action) in enumerate(zip(decisions, result.plays, strict=True)):
        assert decision["playIndex"] == index
        assert decision["actor"] == action.actor.name.lower()
        assert decision["selectedAction"] == LegalPlayActionV1(action.card, action.ekka).code
        assert set(decision["legalActions"]) == {
            LegalPlayActionV1(row.card, row.ekka).code
            for row in result.legal_play_actions_by_step[index]}
    # North owns both diamond honours in this fixed deal, and plays the
    # second at index 28: the public declaration must not appear earlier.
    assert witness["candidate"]["publicBalootEvents"] == [
        {"playIndex": 28, "actor": "north", "kind": "declared"}]
    play_observations = witness["candidate"]["playObservations"]
    assert len(play_observations) == 32
    assert len(branch.public_project_bindings) == 9
    for index, observation in enumerate(play_observations):
        info = observation["information"]
        assert set(info) == {"actor", "initialCards", "visibleCards", "purchaseCard", "publicChoices",
            "publicAnnouncements", "ownDeclinedWindows", "ownProjectAction", "publicProjects",
            "publicPlays", "projectComparison", "projectPoints", "revealedProjects", "publicBalootEvents"}
        assert info["actor"] == decisions[index]["actor"]
        assert info["visibleCards"] == sorted(card.id for card in result.initial_hands[int(Seat.parse(info["actor"]))])
        assert info["publicPlays"] == [row.to_dict() for row in result.plays[:index]]
        assert len(info["publicProjects"]) == min(index + 1, 4)
        assert (info["projectComparison"] is None) == (index < 4)
        assert (info["projectPoints"] is None) == (index < 4)
        assert info["publicBalootEvents"] == ([] if index <= 28 else witness["candidate"]["publicBalootEvents"])
        assert not info["revealedProjects"] if index <= 4 else True
        assert observation["observationHash"] == scenario._round_observation_hash("play", info)
    # The observation key cannot depend on an unplayed future card, even though
    # the privileged native witness retains the complete line.
    altered_play = replace(result, plays=(*result.plays[:16], *reversed(result.plays[16:])))
    altered = scenario._extract_native_play_observations(solver, altered_play, decisions[:17],
        witness["candidate"]["projectObservations"], witness["candidate"]["publicBalootEvents"])
    assert altered[:17] == play_observations[:17]
    project_decisions = witness["candidate"]["projectDecisions"]
    assert [row["actor"] for row in project_decisions] == [
        seat.name.lower() for seat in rules.project_domain.declaration_seats(scenario.dealer)]
    assert all(row["selectedAction"] in row["legalActions"] for row in project_decisions)
    declared_ids = {row["projectId"] for row in witness["candidate"]["projects"]["projects"] if row["declared"]}
    assert {row["selectedAction"].split(":", 1)[1] for row in project_decisions
            if row["selectedAction"] != "pass"} == declared_ids
    observations = witness["candidate"]["projectObservations"]
    assert len(observations) == 4
    for index, observation in enumerate(observations):
        info = observation["information"]
        assert set(info) == {"actor", "initialCards", "visibleCards", "purchaseCard", "publicChoices",
                            "publicAnnouncements", "ownDeclinedWindows", "publicProjects", "publicPlays"}
        actor = Seat.parse(observation["actor"])
        assert info["visibleCards"] == sorted(card.id for card in result.initial_hands[int(actor)])
        assert info["publicPlays"] == [row.to_dict() for row in result.plays[:index]]
        assert len(info["publicProjects"]) == index
        assert all(not ({"projectId", "cards", "legalActions"} & set(row)) for row in info["publicProjects"])
        assert info["ownDeclinedWindows"] == [row["windowHash"] for row in
            witness["candidate"]["escalationOpportunities"] if row["actor"] == actor.name.lower()
            and row["selectedAction"] == branch.escalation.stage.domain.closure_action]
    opportunities = witness["candidate"]["escalationOpportunities"]
    assert opportunities
    assert all(row["selectedAction"] in row["legalActions"] for row in opportunities)
    assert sum(row["selectedAction"] != "decline" for row in opportunities) == (level == "double")
    assert branch.scope["escalationOpportunityOrder"] == "symbolic_per_window"
    assert witness["scope"]["strategicCoverage"] is False
    absent = scenario.find_native_round_margin_below(threshold=margin, **query)
    assert absent["status"] == "no_matching_outcome"
    assert absent["solverStatus"] == "infeasible"
    assert absent["candidate"] is None
    cancelled = scenario.find_native_round_margin_below(
        threshold=margin + 1, cancel_check=lambda: True, **query)
    assert cancelled["status"] == "incomplete"
    assert cancelled["exhaustionReason"] == "cancelled"
    assert cancelled["candidate"] is None
    expired = scenario.find_native_round_margin_below(threshold=margin + 1,
        **{**query, "maximum_seconds": 1e-12})
    assert expired["status"] == "incomplete"
    assert expired["exhaustionReason"] == "maximum_seconds"
    assert expired["candidate"] is None
    assert str(scenario.model.proto) == before
    scenario.model.add(branch.scoring.awarded_points[0] != expected.awarded_points[0])
    assert solver.solve(scenario.model) == cp_model.INFEASIBLE


@pytest.mark.real_clock
def test_native_play_policy_is_observation_local_and_fail_closed(root, monkeypatch):
    import baloot_platform.research.ortools_scenario_model as module

    _, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    candidate = scenario.find_native_round_margin_below(target=Team.NORTH_SOUTH, threshold=1000,
        maximum_seconds=15, maximum_conflicts=250_000)["candidate"]
    observation = candidate["playObservations"][0]
    chosen = observation["selectedAction"]
    alternative = next(code for code in observation["legalActions"] if code != chosen)
    original = str(scenario.model.proto)
    for case in ("matching", "different_hand", "illegal", "unknown", "cancelled", "deadline"):
        owned = copy.copy(scenario)
        owned.model = scenario.model.clone()
        owned._observed_play_bindings = {}
        info = copy.deepcopy(observation["information"])
        if case == "different_hand":
            index = next(i for i, card in enumerate(info["visibleCards"]) if card not in info["initialCards"])
            info["visibleCards"][index] = next(card for card in range(32) if card not in info["visibleCards"])
        action = (next(code for code in owned._native_round_fragment.play.native_legal_action_literals(0)
                       if code not in observation["legalActions"]) if case == "illegal" else alternative)
        arguments = dict(information=info, action=action, maximum_conflicts=250_000,
            maximum_seconds=1e-12 if case == "deadline" else 10,
            cancel_check=(lambda: True) if case == "cancelled" else None)
        with monkeypatch.context() as patch:
            if case == "unknown":
                patch.setattr(module, "_solve_cp_sat_with_cancel_v1", lambda *_: cp_model.UNKNOWN)
            receipt = owned.bind_native_play_observed_choice(**arguments)
        expected = "invalid" if case == "illegal" else "incomplete" if case in {"unknown", "cancelled", "deadline"} else "bound"
        assert receipt["status"] == expected
        assert bool(owned._observed_play_bindings) == (expected == "bound")
        if expected == "bound":
            before = str(owned.model.proto)
            assert owned.bind_native_play_observed_choice(**arguments) == receipt
            with pytest.raises(ValueError, match="conflicting actions"):
                owned.bind_native_play_observed_choice(**{**arguments, "action": chosen})
            assert str(owned.model.proto) == before
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        assert solver.solve(owned.model) == (cp_model.INFEASIBLE if case == "matching" else cp_model.OPTIMAL)
    for change in ({"hiddenHands": []}, {"publicPlays": [{"actor": "north", "card": True, "ekka": False}]},
                   {"projectPoints": [0, 0]}, {"publicBalootEvents": [{"playIndex": 0, "actor": "north", "kind": "declared"}]},
                   {"publicProjects": [{**observation["information"]["publicProjects"][0], "actor": False}]}):
        with pytest.raises((ValueError, TypeError)):
            scenario.bind_native_play_observed_choice(information={**observation["information"], **change},
                action=alternative, maximum_seconds=10, maximum_conflicts=250_000)
        assert str(scenario.model.proto) == original
    assert scenario._observed_play_bindings == {}

    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    compiler = OrToolsActorInformationSetCompilerV1(scenario.rules, native_cp_sat=True)
    policy = {row["observationHash"]: row["selectedAction"]
              for kind in ("auction", "escalation", "project", "play")
              for row in candidate[kind + "Observations"] if Seat.parse(row["actor"]).team == Team.NORTH_SOUTH}
    policy[observation["observationHash"]] = alternative
    absent = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, policy,
        threshold=1000, maximum_queries=4, maximum_observations=32, maximum_conflicts=250_000, maximum_seconds=30)
    assert absent["status"] == "no_adverse_round_in_branch" and not absent["strategicCoverage"]
    assert len(absent["bindings"]) == 1 and absent["bindings"][0]["kind"] == "native_play_policy_binding_v1"
    assert absent["queries"][-1]["playPolicyBindings"] == absent["bindings"]
    assert scenario._observed_play_bindings == {} and str(scenario.model.proto) == original


@pytest.mark.real_clock
@pytest.mark.parametrize("play_index", [0, 17, 29])
@pytest.mark.parametrize("contract", ["sun", "hokum"])
def test_native_prefix_legality_preserves_owner_and_fails_closed(root, monkeypatch, play_index, contract):
    import baloot_platform.research.ortools_scenario_model as module

    if contract == "hokum":
        _, source, _, _ = _native_hokum_round_branch(root, "normal")
    else:
        compiler, exact, prefix, _ = _native_suit_monopoly_bid(root)
        source = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state",
            branch=(Contract.SUN, None, Seat.parse(exact.next_actor), False), cancel_check=None)
        source.model.add(source.auction_fragment.choices[0]["sun"] == 1)
    candidate = source.find_native_round_margin_below(target=Team.NORTH_SOUTH, threshold=1000,
        maximum_seconds=15, maximum_conflicts=250000)["candidate"]
    row = candidate["playObservations"][play_index]
    original = str(source.model.proto)
    solve = module._solve_cp_sat_with_cancel_v1
    sizes = []
    def measured(solver, model, cancel_check):
        sizes.append(len(model.proto.constraints))
        return solve(solver, model, cancel_check)
    receipts, owners = [], []
    with monkeypatch.context() as patch:
        patch.setattr(module, "_solve_cp_sat_with_cancel_v1", measured)
        for prefix_legality in (False, True):
            owned = copy.copy(source)
            owned.model, owned._observed_play_bindings = source.model.clone(), {}
            receipt = owned.bind_native_play_observed_choice(information=row["information"],
                action=row["selectedAction"], maximum_seconds=10, maximum_conflicts=250000,
                _prefix_legality=prefix_legality)
            assert receipt["status"] == "bound" and receipt["solverStatus"] == "INFEASIBLE"
            receipts.append(receipt)
            owners.append(str(owned.model.proto))
    assert len(sizes) == 2 and sizes[1] < sizes[0]
    assert owners[0] == owners[1] and receipts[0]["modelHash"] == receipts[1]["modelHash"]
    assert receipts[1]["legalityScope"] == "actor_visible_play_prefix_relaxation_v1"
    assert "legalityScope" not in receipts[0]
    assert receipts[0]["legalityModelHash"] != receipts[1]["legalityModelHash"]
    if play_index == 0:
        rows = [{"information": item["information"], "action": item["selectedAction"]}
                for item in candidate["playObservations"][:2]]
        legacy, batched = copy.copy(source), copy.copy(source)
        for owner in (legacy, batched):
            owner.model, owner._observed_play_bindings = source.model.clone(), {}
        expected = []
        for item in rows:
            expected.append(legacy.bind_native_play_observed_choice(**item,
                maximum_seconds=10, maximum_conflicts=250000, _prefix_legality=True))
            legacy.native_play_observation_literal(information=item["information"])
        actual, matches = batched._bind_native_play_overrides(rows,
            maximum_seconds=20, maximum_conflicts=250000)
        assert len(actual) == sum(map(len, matches.values())) == 2
        assert str(batched.model.proto) == str(legacy.model.proto)
        for observed, direct in zip(actual, expected, strict=True):
            assert observed["status"] == direct["status"] == "bound"
            assert observed["modelHash"] == batched._model_hash(search_branching="fixed")
            assert all(observed[key] == direct[key] for key in
                ("rulesHash", "observationHash", "action", "legalityModelHash", "legalityScope", "solverStatus"))
        stopped = copy.copy(source)
        stopped.model, stopped._observed_play_bindings = source.model.clone(), {}
        cancelled, matches = stopped._bind_native_play_overrides(rows,
            maximum_seconds=10, maximum_conflicts=250000, cancel_check=lambda: True)
        assert len(cancelled) == 1 and cancelled[0]["status"] == "incomplete" and not matches
        assert not stopped._observed_play_bindings
    # A removed card or a wrong-suit card must not become a certified choice.
    illegal = next(code for code in source._native_round_fragment.play.native_legal_action_literals(play_index)
        if code not in row["legalActions"] and not code.endswith("+ekka"))
    owned = copy.copy(source)
    owned.model, owned._observed_play_bindings = source.model.clone(), {}
    invalid = owned.bind_native_play_observed_choice(information=row["information"],
        action=illegal, maximum_seconds=10, maximum_conflicts=250000, _prefix_legality=True)
    assert invalid["status"] == "incomplete" and not owned._observed_play_bindings
    for status in (cp_model.UNKNOWN, cp_model.FEASIBLE):
        owned = copy.copy(source)
        owned.model, owned._observed_play_bindings = source.model.clone(), {}
        with monkeypatch.context() as patch:
            patch.setattr(module, "_solve_cp_sat_with_cancel_v1", lambda *args: status)
            receipt = owned.bind_native_play_observed_choice(information=row["information"],
                action=row["selectedAction"], maximum_seconds=10, maximum_conflicts=250000,
                _prefix_legality=True)
        assert receipt["status"] == "incomplete" and not owned._observed_play_bindings
    if play_index == 0:
        # An illegal Ekka in the relaxation is not a source-world refutation.
        illegal_ekka = row["selectedAction"] + "+ekka"
        assert illegal_ekka not in row["legalActions"]
        owned = copy.copy(source)
        owned.model, owned._observed_play_bindings = source.model.clone(), {}
        receipt = owned.bind_native_play_observed_choice(information=row["information"],
            action=illegal_ekka, maximum_seconds=10, maximum_conflicts=250000, _prefix_legality=True)
        assert receipt["status"] == "incomplete" and not owned._observed_play_bindings
        for seconds, cancel in ((1e-12, None), (10, lambda: True)):
            receipt = owned.bind_native_play_observed_choice(information=row["information"],
                action=row["selectedAction"], maximum_seconds=seconds, maximum_conflicts=250000,
                _prefix_legality=True, cancel_check=cancel)
            assert receipt["status"] == "incomplete" and not owned._observed_play_bindings
    assert str(source.model.proto) == original and not source._observed_play_bindings


def test_native_play_policy_guards_late_public_history(root):
    _, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    candidate = scenario.find_native_round_margin_below(target=Team.NORTH_SOUTH, threshold=1000,
        maximum_seconds=15, maximum_conflicts=250_000)["candidate"]
    observation = next(row for row in reversed(candidate["playObservations"]) if len(row["legalActions"]) > 1)
    assert observation["playIndex"] >= 16
    alternative = next(code for code in observation["legalActions"] if code != observation["selectedAction"])
    for matching in (True, False):
        owned = copy.copy(scenario)
        owned.model = scenario.model.clone()
        owned._observed_play_bindings = {}
        info = copy.deepcopy(observation["information"])
        if not matching:
            public = {Card.parse(row["card"]).id for row in info["publicPlays"]}
            replacement = next(card for card in range(32) if card not in public and card not in info["visibleCards"])
            row = next(row for row in info["publicPlays"] if row["actor"] != info["actor"])
            row["card"] = replacement
        receipt = owned.bind_native_play_observed_choice(information=info, action=alternative,
            maximum_seconds=10, maximum_conflicts=250_000)
        assert receipt["status"] == "bound"
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        assert solver.solve(owned.model) == (cp_model.INFEASIBLE if matching else cp_model.OPTIMAL)
    assert scenario._observed_play_bindings == {}


@pytest.mark.parametrize("play_index", [0, 17, 29])
def test_native_play_observation_literal_is_reified_and_rejects_hidden_input(root, play_index):
    _, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    candidate = scenario.find_native_round_margin_below(target=Team.NORTH_SOUTH, threshold=1000,
        maximum_seconds=15, maximum_conflicts=250_000)["candidate"]
    information = candidate["playObservations"][play_index]["information"]
    before = str(scenario.model.proto)
    with pytest.raises(ValueError):
        scenario.native_play_observation_literal(information={**information, "hiddenHands": []})
    assert str(scenario.model.proto) == before
    for matching in (True, False):
        for required in (True, False):
            owned = copy.copy(scenario)
            owned.model = scenario.model.clone()
            info = copy.deepcopy(information)
            if not matching:
                # Keep the initial five-card memory intact while changing a later own card.
                index = next(i for i, card in enumerate(info["visibleCards"])
                    if card not in info["initialCards"] and card not in {
                        Card.parse(row["card"]).id for row in info["publicPlays"]}) if play_index == 0 else None
                if index is not None:
                    info["visibleCards"][index] = next(card for card in range(32) if card not in info["visibleCards"])
                else:
                    info["projectPoints"] = [value + 1 for value in info["projectPoints"]]
            literal = owned.native_play_observation_literal(information=info)
            owned.model.add(literal == int(required))
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.max_time_in_seconds = 15
            assert solver.solve(owned.model) == (cp_model.OPTIMAL if matching == required else cp_model.INFEASIBLE)
    assert str(scenario.model.proto) == before


@pytest.mark.parametrize("play_index", [0, 17, 29])
def test_native_observed_play_root_preserves_exact_remaining_depth(root, play_index):
    _, scenario, auction, _ = _native_hokum_round_branch(root, "normal")
    query = dict(target=Team.NORTH_SOUTH, threshold=1000, maximum_seconds=15, maximum_conflicts=250_000)
    candidate = scenario.find_native_round_margin_below(**query)["candidate"]
    info = candidate["playObservations"][play_index]["information"]
    before = str(scenario.model.proto)
    with pytest.raises(ValueError, match="complete public auction"):
        scenario.constrain_native_play_observation(information=info)
    assert str(scenario.model.proto) == before
    scenario.auction_fragment = replace(auction, prefix_length=len(info["publicChoices"]))
    receipt = scenario.constrain_native_play_observation(information=info)
    fixed = str(scenario.model.proto)
    assert scenario.constrain_native_play_observation(information=info) == receipt
    assert str(scenario.model.proto) == fixed
    different = candidate["playObservations"][play_index + 1]["information"]
    with pytest.raises(ValueError, match="already fixed"):
        scenario.constrain_native_play_observation(information=different)
    assert str(scenario.model.proto) == fixed
    receipt["playIndex"] = -1
    assert scenario.constrain_native_play_observation(information=info)["playIndex"] == play_index
    receipt = scenario.constrain_native_play_observation(information=info)
    found = scenario.find_native_round_margin_below(**query)
    assert found["status"] == "witness_found"
    assert found["scope"]["playRoot"] == receipt
    assert all(row["authoredPrefix"] for row in found["candidate"]["auctionObservations"])
    assert all(row["authoredPrefix"] for row in found["candidate"]["escalationObservations"])
    for kind, count in (("project", min(play_index + 1, 4)), ("play", play_index)):
        assert [row.get("authoredPrefix", False) for row in found["candidate"][kind + "Observations"]] == [
            index < count for index in range(4 if kind == "project" else 32)]
    assert scenario._native_constant_nonplay_choices(Team.NORTH_SOUTH, "escalation", "decline") == []
    remaining = 32 - play_index + 4 - min(play_index + 1, 4)
    limits = dict(maximum_seconds=15, maximum_conflicts=250_000)
    assert scenario.find_native_continuation_depth_above(maximum_depth=remaining, **limits)["status"] == "depth_bound_proven"
    exceeded = scenario.find_native_continuation_depth_above(maximum_depth=remaining - 1, **limits)
    assert exceeded["status"] == "over_depth_witness"
    assert exceeded["witnessDepth"] == remaining


@pytest.mark.parametrize("level", ["normal", "triple"])
def test_native_observed_escalation_root_uses_world_relative_past(root, level):
    data = copy.deepcopy(RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml").data)
    data["escalation"]["response_actor_policy"] = "team"
    _, source, auction, branch = _native_hokum_round_branch(root, level, rules=RuleBook(data))
    window = 0 if level == "normal" else 2
    if window:
        for index, actor in enumerate((Seat.WEST, Seat.NORTH)):
            source.model.add(branch.escalation.native["actionSlots"][index][0] == int(actor))
        first = next(slot for slot in branch.escalation_opportunities if slot["window"] == 0 and slot["ordinal"] == 0)
        source.model.add(first["actor"] == int(Seat.EAST))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 15
    assert solver.solve(source.model) == cp_model.OPTIMAL
    opportunities = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
    observations = source._extract_native_escalation_observations(solver, opportunities)
    info = next(row["information"] for row, slot in zip(observations, opportunities, strict=True)
        if row["actor"] == "east" and slot["window"] == window)
    before = str(source.model.proto)
    with pytest.raises(ValueError, match="complete public auction"):
        source.constrain_native_escalation_observation(information=info)
    assert str(source.model.proto) == before
    hashes, positions = [], []
    for first_actor in (Seat.EAST, Seat.WEST):
        owned = copy.copy(source)
        owned.model = source.model.clone()
        owned.auction_fragment = replace(auction, prefix_length=len(info["publicChoices"]))
        root_receipt = owned.constrain_native_escalation_observation(information=info)
        fixed = str(owned.model.proto)
        assert owned.constrain_native_escalation_observation(information=info) == root_receipt
        assert str(owned.model.proto) == fixed
        for invalid in ({**info, "ordinal": 0}, {**info, "completeWindowDeclineOrder": []},
                        {**info, "visibleCards": [True, *info["visibleCards"][1:]]}):
            with pytest.raises(ValueError):
                owned.constrain_native_escalation_observation(information=invalid)
            assert str(owned.model.proto) == fixed
        current = [slot for slot in branch.escalation_opportunities if slot["window"] == window]
        owned.model.add(current[0]["actor"] == int(first_actor))
        owned.model.add(current[1]["actor"] == int(Seat.WEST if first_actor == Seat.EAST else Seat.EAST))
        choices = owned._native_constant_nonplay_choices(Team.WEST_EAST, "escalation", "decline",
            escalation=branch.escalation, opportunities=branch.escalation_opportunities)
        assert solver.solve(owned.model) == cp_model.OPTIMAL
        position = solver.value(owned._native_escalation_root_position)
        positions.append(position)
        path = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
        rows = owned._extract_native_escalation_observations(solver, path)
        hashes.append(root_receipt["observationHash"])
        for slot, row in zip(path, rows, strict=True):
            assert row.get("authoredPrefix", False) == (slot["window"] * 4 + slot["ordinal"] < position)
        for index, (matches, _, _) in enumerate(choices):
            if index < position:
                assert not solver.boolean_value(matches)
        remaining = 36 + sum(not row.get("authoredPrefix", False) for row in rows)
        limits = dict(maximum_seconds=15, maximum_conflicts=250_000)
        assert owned.find_native_continuation_depth_above(maximum_depth=remaining, **limits)["status"] == "depth_bound_proven"
        over = owned.find_native_continuation_depth_above(maximum_depth=remaining - 1, **limits)
        assert over["status"] == "over_depth_witness" and over["witnessDepth"] == remaining
        query = owned.find_native_round_with_constant_policy_below(target=Team.WEST_EAST,
            policy_team=Team.WEST_EAST, auction_action=None, escalation_action="decline", threshold=1000, **limits)
        assert query["status"] == "queried"
        assert query["query"]["scope"]["escalationRoot"] == root_receipt
        assert query["query"]["status"] == "witness_found"
    assert hashes[0] == hashes[1] and positions[1] == positions[0] + 1
    assert str(source.model.proto) == before


@pytest.mark.real_clock
def test_native_escalation_binding_is_observer_local_and_fail_closed(root, monkeypatch):
    import baloot_platform.research.ortools_scenario_model as module

    _, scenario, _, branch = _native_hokum_round_branch(root, "normal")
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(scenario.model) == cp_model.OPTIMAL
    opportunities = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
    observations = scenario._extract_native_escalation_observations(solver, opportunities)
    info = next(row["information"] for row in observations if row["actor"] == "east")
    assert len(info["visibleCards"]) == 8 and len(info["initialCards"]) == 5
    for case in ("matching", "different_hand", "illegal", "unknown", "cancelled", "deadline"):
        owned = copy.copy(scenario)
        owned.model = scenario.model.clone()
        owned._observed_escalation_bindings = {}
        information = copy.deepcopy(info)
        if case == "different_hand":
            index = next(i for i, card in enumerate(information["visibleCards"]) if card not in information["initialCards"])
            information["visibleCards"][index] = next(card for card in range(32) if card not in information["visibleCards"])
        arguments = dict(information=information, action="triple" if case == "illegal" else "double:open",
            maximum_seconds=1e-12 if case == "deadline" else 10, maximum_conflicts=250_000,
            cancel_check=(lambda: True) if case == "cancelled" else None)
        with monkeypatch.context() as patch:
            if case == "unknown":
                patch.setattr(module, "_solve_cp_sat_with_cancel_v1", lambda *_: cp_model.UNKNOWN)
            receipt = owned.bind_native_escalation_observed_choice(**arguments)
        expected = "invalid" if case == "illegal" else "incomplete" if case in {"unknown", "cancelled", "deadline"} else "bound"
        assert receipt["status"] == expected
        assert bool(owned._observed_escalation_bindings) == (expected == "bound")
        if expected == "bound":
            before = str(owned.model.proto)
            assert owned.bind_native_escalation_observed_choice(**arguments) == receipt
            with pytest.raises(ValueError, match="conflicting actions"):
                owned.bind_native_escalation_observed_choice(**{**arguments, "action": "decline"})
            assert str(owned.model.proto) == before
        assert solver.solve(owned.model) == (cp_model.INFEASIBLE if case == "matching" else cp_model.OPTIMAL)
    before = str(scenario.model.proto)
    for malformed in ({**info, "ordinal": 0}, {**info, "ownDeclinedWindows": ["a" * 64]},
                      {**info, "initialCards": [True, *info["initialCards"][1:]]}):
        with pytest.raises(ValueError):
            scenario.bind_native_escalation_observed_choice(information=malformed, action="decline",
                maximum_seconds=10, maximum_conflicts=250_000)
        assert str(scenario.model.proto) == before
    assert scenario._observed_escalation_bindings == {}
    # Neither another seat's private decline nor the opportunity ordinal may
    # change what East sees before any public announcement.
    visible = []
    for first in (Seat.EAST, Seat.WEST):
        model = scenario.model.clone()
        model.add(branch.escalation_opportunities[0]["actor"] == int(first))
        assert solver.solve(model) == cp_model.OPTIMAL
        rows = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
        visible.append(next(row for row in scenario._extract_native_escalation_observations(solver, rows)
                            if row["actor"] == "east"))
    assert visible[0] == visible[1]


def test_escalation_policy_preserves_own_decline_recall_across_windows(root):
    data = copy.deepcopy(RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml").data)
    data["escalation"]["response_actor_policy"] = "team"
    _, scenario, _, branch = _native_hokum_round_branch(root, "triple", rules=RuleBook(data))
    for index, actor in enumerate((Seat.WEST, Seat.NORTH)):
        scenario.model.add(branch.escalation.native["actionSlots"][index][0] == int(actor))
    for window in (0, 2):
        slot = next(row for row in branch.escalation_opportunities if row["window"] == window and row["ordinal"] == 0)
        scenario.model.add(slot["actor"] == int(Seat.EAST))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(scenario.model) == cp_model.OPTIMAL
    rows = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
    observations = scenario._extract_native_escalation_observations(solver, rows)
    east = [row for row in observations if row["actor"] == "east"]
    assert len(east) == 2 and len(east[-1]["information"]["ownDeclinedWindows"]) == 1
    remembered_mode = solver.value(branch.escalation.native["stateLocked"][1])
    for recalls_match in (False, True):
        owned = copy.copy(scenario)
        owned.model = scenario.model.clone()
        owned._observed_escalation_bindings = {}
        info = copy.deepcopy(east[-1]["information"])
        if not recalls_match:
            info["ownDeclinedWindows"] = []
        binding = owned.bind_native_escalation_observed_choice(information=info, action="four:open",
            maximum_seconds=10, maximum_conflicts=250_000)
        assert binding["status"] == "bound"
        assert solver.solve(owned.model) == cp_model.OPTIMAL
        if recalls_match:
            # A different public Double mode legitimately has a different
            # information key; the policy must not constrain that other history.
            assert solver.value(branch.escalation.native["stateLocked"][1]) != remembered_mode
        owned.model.add(branch.escalation.native["stateLocked"][1] == remembered_mode)
        assert solver.solve(owned.model) == (cp_model.INFEASIBLE if recalls_match else cp_model.OPTIMAL)


def test_escalation_observation_remembers_initial_hand_without_hidden_or_future_leak(root):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST, buyer=Seat.NORTH)
    initial = layout.positions_for(Seat.NORTH, final=False)
    late = next(position for position in layout.positions_for(Seat.NORTH, final=True)
                if position not in initial and position != layout.purchase_position)
    receipts, project_receipts = [], []
    for swap in (False, True):
        deck = list(range(32))
        if swap:
            deck[initial[0]], deck[late] = deck[late], deck[initial[0]]
        _, scenario, _, branch = _native_hokum_round_branch(root, "double", deck_order=deck)
        scenario.model.add(branch.escalation_opportunities[0]["actor"] == int(Seat.EAST))
        scenario.model.add(branch.escalation.native["actionSlots"][0][0] == int(Seat.EAST))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        assert solver.solve(scenario.model) == cp_model.OPTIMAL
        rows = branch.escalation.extract_opportunity_path(solver, branch.escalation_opportunities)
        observed = {row["actor"]: row for row in scenario._extract_native_escalation_observations(solver, rows)}
        receipts.append(observed)
        play = branch.play._extract(solver, cp_model.OPTIMAL, branch.play.model_hash)
        escalation = branch.escalation.extract_stage_result(solver, model_hash=branch.play.model_hash,
            solver_status="optimal", branches=solver.num_branches, conflicts=solver.num_conflicts)
        decisions = [branch.projects.extract_declaration_decision(solver, seat)
            for seat in rules.project_domain.declaration_seats(scenario.dealer)]
        project_observations = scenario._extract_native_project_observations(solver, rows, escalation, play, decisions)
        project_receipts.append({row["actor"]: row for row in project_observations})
        assert project_receipts[-1]["north"]["information"]["ownDeclinedWindows"]
        # Neither a future play nor its outcome may affect a declaration key.
        altered_future = replace(play, plays=(*play.plays[:4], *reversed(play.plays[4:])))
        assert scenario._extract_native_project_observations(solver, rows, escalation, altered_future, decisions) == project_observations
        info = observed["east"]["information"]
        key = scenario._escalation_observation_hash(info)
        # A hypothetical later lock/order cannot split this information set.
        branch.scope["locked"] = True
        branch.scope["escalationOpportunityOrder"] = "different_unobserved_order"
        assert scenario._escalation_observation_hash(info) == key
    assert receipts[0]["east"] == receipts[1]["east"]
    assert receipts[0]["north"]["information"]["visibleCards"] == receipts[1]["north"]["information"]["visibleCards"]
    assert receipts[0]["north"]["observationHash"] != receipts[1]["north"]["observationHash"]
    assert project_receipts[0]["east"] == project_receipts[1]["east"]
    assert project_receipts[0]["north"]["information"]["visibleCards"] == project_receipts[1]["north"]["information"]["visibleCards"]
    assert project_receipts[0]["north"]["observationHash"] != project_receipts[1]["north"]["observationHash"]


@pytest.mark.real_clock
def test_project_policy_binds_only_matching_actor_information(root, monkeypatch):
    import baloot_platform.research.ortools_scenario_model as module

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    _, scenario, _, branch = _native_hokum_round_branch(root, "normal", rules=rules)
    candidate = scenario.find_native_round_margin_below(target=Team.NORTH_SOUTH, threshold=1000,
        maximum_seconds=15, maximum_conflicts=250_000)["candidate"]
    north = next(row for row in candidate["projectObservations"] if row["actor"] == "north")
    assert len(north["legalActions"]) == 2
    chosen = north["selectedAction"]
    alternative = next(code for code in north["legalActions"] if code != chosen)
    for project in branch.projects.projects:
        if project.owner == Seat.NORTH:
            scenario.model.add(project.declared == int(chosen == "declare:" + project.project_id))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    for case in ("matching", "different_hand", "illegal", "unknown", "cancelled", "deadline"):
        owned = copy.copy(scenario)
        owned.model = scenario.model.clone()
        owned._observed_project_bindings = {}
        info = copy.deepcopy(north["information"])
        if case == "different_hand":
            index = next(i for i, card in enumerate(info["visibleCards"]) if card not in info["initialCards"])
            info["visibleCards"][index] = next(card for card in range(32) if card not in info["visibleCards"])
        action = (next(code for code in branch.projects.declaration_actions[int(Seat.NORTH)]
                       if code not in north["legalActions"]) if case == "illegal" else alternative)
        arguments = dict(information=info, action=action, maximum_conflicts=250_000,
            maximum_seconds=1e-12 if case == "deadline" else 10,
            cancel_check=(lambda: True) if case == "cancelled" else None)
        with monkeypatch.context() as patch:
            if case == "unknown":
                patch.setattr(module, "_solve_cp_sat_with_cancel_v1", lambda *_: cp_model.UNKNOWN)
            receipt = owned.bind_native_project_observed_choice(**arguments)
        expected = "invalid" if case == "illegal" else "incomplete" if case in {"unknown", "cancelled", "deadline"} else "bound"
        assert receipt["status"] == expected
        assert bool(owned._observed_project_bindings) == (expected == "bound")
        if expected == "bound":
            before = str(owned.model.proto)
            assert owned.bind_native_project_observed_choice(**arguments) == receipt
            with pytest.raises(ValueError, match="conflicting actions"):
                owned.bind_native_project_observed_choice(**{**arguments, "action": chosen})
            assert str(owned.model.proto) == before
        assert solver.solve(owned.model) == (cp_model.INFEASIBLE if case == "matching" else cp_model.OPTIMAL)
    original = str(scenario.model.proto)
    for malformed in ({**north["information"], "hiddenHands": []},
                      {**north["information"], "publicPlays": [{"actor": "west", "card": "S7", "ekka": False}]},
                      {**north["information"], "initialCards": [True, *north["information"]["initialCards"][1:]]}):
        with pytest.raises((ValueError, TypeError)):
            scenario.bind_native_project_observed_choice(information=malformed, action=alternative,
                maximum_seconds=10, maximum_conflicts=250_000)
        assert str(scenario.model.proto) == original
    assert scenario._observed_project_bindings == {}

    west_info = copy.deepcopy(next(row["information"] for row in candidate["projectObservations"] if row["actor"] == "west"))
    west_info["publicProjects"][0]["actor"] = False
    with pytest.raises(ValueError, match="declaration is invalid"):
        scenario.bind_native_project_observed_choice(information=west_info, action="pass",
            maximum_seconds=10, maximum_conflicts=250_000)
    assert str(scenario.model.proto) == original

    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    arguments = dict(threshold=1000, maximum_queries=4, maximum_observations=32,
        maximum_conflicts=250_000, maximum_seconds=30)
    source_hash_calls = []
    model_hash = OrToolsScenarioModelV1._model_hash
    def counted_hash(model, **kwargs):
        if model is scenario:
            source_hash_calls.append(kwargs)
        return model_hash(model, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(OrToolsScenarioModelV1, "_model_hash", counted_hash)
        unknown = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {}, **arguments)
    assert len(source_hash_calls) == 1
    assert unknown["sourceModelHash"] == model_hash(scenario, search_branching="automatic")
    policy = {row["observationHash"]: row["selectedAction"]
              for kind in ("auction", "escalation", "project")
              for row in unknown["candidate"][kind + "Observations"] if Seat.parse(row["actor"]).team == Team.NORTH_SOUTH}
    policy[north["observationHash"]] = alternative
    absent = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, policy, **arguments)
    assert absent["status"] == "no_adverse_round_in_branch" and not absent["strategicCoverage"]
    assert len(absent["bindings"]) == 1 and absent["bindings"][0]["kind"] == "native_project_policy_binding_v1"
    assert absent["queries"][-1]["projectPolicyBindings"] == absent["bindings"]
    assert str(scenario.model.proto) == original and scenario._observed_project_bindings == {}


def test_symbolic_round_refinement_enforces_escalation_policy(root):
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1

    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    before = str(scenario.model.proto)
    arguments = dict(threshold=1000, maximum_queries=4, maximum_observations=32,
        maximum_conflicts=250_000, maximum_seconds=30)
    unknown = compiler._refine_symbolic_round_policy(scenario, Team.WEST_EAST, {}, **arguments)
    rows = [row for kind in ("auction", "escalation") for row in unknown["candidate"][kind + "Observations"]
            if Seat.parse(row["actor"]).team == Team.WEST_EAST]
    assert len(rows) == 4
    policy = {row["observationHash"]: row["selectedAction"] for row in rows}
    policy.update({row["observationHash"]: row["selectedAction"] for row in unknown["candidate"]["projectObservations"]
                   if Seat.parse(row["actor"]).team == Team.WEST_EAST})
    policy.update({row["observationHash"]: row["selectedAction"] for row in unknown["candidate"]["playObservations"]
                   if Seat.parse(row["actor"]).team == Team.WEST_EAST})
    compliant = compiler._refine_symbolic_round_policy(scenario, Team.WEST_EAST, policy, **arguments)
    assert compliant["status"] == "policy_counterexample_in_branch" and not compliant["strategicCoverage"]
    # Forcing Double eliminates this normal-only fixture, not all round branches.
    alternative = {**policy, rows[-1]["observationHash"]: "double:open"}
    absent = compiler._refine_symbolic_round_policy(scenario, Team.WEST_EAST, alternative, **arguments)
    assert absent["status"] == "no_adverse_round_in_branch" and not absent["strategicCoverage"]
    assert len(absent["bindings"]) == 1
    assert absent["bindings"][0]["kind"] == "native_escalation_policy_binding_v1"
    assert absent["queries"][-1]["escalationPolicyBindings"] == absent["bindings"]
    assert str(scenario.model.proto) == before


@pytest.mark.parametrize("changed_domain", [False, True])
def test_symbolic_fixed_root_domain_ignores_only_order(root, monkeypatch, changed_domain):
    from baloot_platform.research.ortools_information_set_compiler import (
        OrToolsActorInformationSetCompilerV1, InformationSetCompilationStopped)
    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    actions = ("pass", "other" if changed_domain else "declare")
    monkeypatch.setattr(compiler, "_refine_symbolic_round_policy", lambda *args, **kwargs: {
        "queries": [], "bindings": [], "domains": {"root": actions},
        "status": "policy_extension_required", "candidate": {"utility": 0,
            "auctionObservations": [], "escalationObservations": [], "playObservations": [],
            "projectObservations": [{"actor": "north", "observationHash": "root", "selectedAction": "pass"}]}})
    with pytest.raises(ValueError if changed_domain else InformationSetCompilationStopped,
            match="conflicting legal domains" if changed_domain else "maximum_iterations"):
        compiler._solve_symbolic_round_policy_bounds([scenario], Team.NORTH_SOUTH,
            maximum_iterations=1, maximum_queries=10, maximum_observations=10,
            maximum_branches=1, maximum_conflicts=1000, maximum_seconds=15,
            fixed_root=("root", ("declare", "pass"), "pass"))


@pytest.mark.parametrize("case", ["converged", "incomplete", "contradiction", "legacy"])
@pytest.mark.parametrize("policy_id", [None, "observer_safe_tactical_category_v1"])
def test_symbolic_retained_floor_preserves_policy_and_partial_evidence(root, monkeypatch, case, policy_id):
    from types import SimpleNamespace
    from baloot_platform.research.scenario_contracts import canonical_sha256
    from baloot_platform.research.ortools_information_set_compiler import (
        OrToolsActorInformationSetCompilerV1, InformationSetCompilationStopped,
        OrToolsActorInformationSetProofV1)
    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    # Orchestration only: independent native fixtures own mathematical soundness.
    scenario._native_project_root = {"fixture": True}
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    queries = []
    def query(**kwargs):
        core = {"status": "witness_found", "solverStatus": "optimal",
            "candidate": {"utility": 0 if not queries else -10, "deckOrder": list(range(32))}}
        result = {**core, "sha256": canonical_sha256(core)}
        queries.append(result)
        return result
    monkeypatch.setattr(scenario, "find_native_round_margin_below", query)
    def policy_query(**kwargs):
        core = {"status": "queried", "query": query(**kwargs)}
        return {**core, "sha256": canonical_sha256(core)}
    monkeypatch.setattr(scenario, "find_native_round_with_constant_policy_below", policy_query)
    monkeypatch.setattr(OrToolsActorInformationSetProofV1, "_solve_master", lambda *args, **kwargs:
        SimpleNamespace(model_hash="master", status="unknown" if case == "incomplete" else "optimal",
            value=-1 if case == "contradiction" else 0, policy={"unproved": "action"}))
    def refine(*args, **kwargs):
        assert case == "legacy", "must not validate an optimistic policy after bound equality"
        raise InformationSetCompilationStopped("maximum_queries", stage="fixture")
    monkeypatch.setattr(compiler, "_refine_symbolic_round_policy", refine)
    def run():
        return compiler._solve_symbolic_round_policy_bounds([scenario], Team.NORTH_SOUTH,
            maximum_iterations=2, maximum_queries=10, maximum_observations=10,
            maximum_branches=1, maximum_conflicts=1000, maximum_seconds=15,
            tighten_utility_ceiling=True, retain_path_floor=case != "legacy",
            opponent_ceiling_policy=({"auction_action": None, "escalation_action": None,
                "native_play_policy_id": policy_id} if policy_id is not None else None))
    if case == "contradiction":
        with pytest.raises(ValueError, match="path floor exceeds"):
            run()
    elif case == "converged":
        result = run()
        assert result["lowerBound"] == result["upperBound"] == 0
        assert result["policy"] == result["pathFloor"]["policy"] == {}
        assert result["pathFloor"]["queryHash"] == queries[0]["sha256"]
        assert result["scope"]["utilityBounds"][1] == 10
        assert len(queries) == 2
        assert not any(row.get("purpose") == "cooperative_upper_bound" for row in result["queries"])
        assert result["strategicCoverage"] is False
    else:
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            run()
        result = stopped.value.partial_receipt["evidence"]
        assert result["lowerBound"] == (None if case == "legacy" else 0)
        assert ("pathFloor" in result) == (case != "legacy")
        assert result["scope"]["utilityBounds"][1] == 10
        assert len(queries) == 2
        assert result["status"] == "incomplete"
    if case != "contradiction":
        if policy_id is None:
            assert result["scope"]["fixedWorldCooperativeCeiling"]["value"] == 10
        else:
            assert result["scope"]["opponentCeilingWorldQueryHash"] == queries[0]["sha256"]
            assert result["scope"]["opponentCeilingPolicy"]["native_play_policy_id"] == policy_id
            assert "fixedWorldCooperativeCeiling" not in result["scope"]


@pytest.mark.parametrize("policy_bound", [False, True])
def test_fixed_world_ceiling_receipt_must_bind_compatible_world(policy_bound):
    from baloot_platform.research.result_authority import _validate_native_final_branch_proofs_v2
    from baloot_platform.research.scenario_contracts import canonical_sha256

    def owned(core):
        return {**core, "sha256": canonical_sha256(core)}

    branch = "a" * 64
    branch_scope = owned({"kind": "test_branch"})
    cards = list(range(32))
    floor = owned({"status": "witness_found", "solverStatus": "optimal",
        "targetTeam": "north_south", "scope": branch_scope,
        "candidate": {"utility": 0, "deckOrder": cards}})
    ceiling = owned({"status": "witness_found", "solverStatus": "optimal",
        "targetTeam": "west_east", "scope": branch_scope,
        "candidate": {"utility": -10, "deckOrder": cards}})
    terminal = owned({"status": "no_matching_outcome", "solverStatus": "infeasible",
        "targetTeam": "north_south", "scope": branch_scope, "threshold": 0,
        "baseModelHash": branch, "auctionPolicyBindings": [],
        "escalationPolicyBindings": [], "projectPolicyBindings": [], "playPolicyBindings": []})
    refinement = {"sourceModelHash": branch, "rulesHash": "r" * 64,
        "scope": branch_scope, "targetTeam": "north_south",
        "status": "no_adverse_round_in_branch", "threshold": 0,
        "bindings": [], "boundModelHash": branch, "queries": [terminal]}
    refinement_hash = canonical_sha256(refinement)
    scope = {"rulesHash": "r" * 64, "targetTeam": "north_south",
        "branches": [{"modelHash": branch, "scope": branch_scope}],
        "utilityBounds": [-100, 10], "fixedWorldCooperativeCeiling": {
            "value": 10, "queryHash": ceiling["sha256"],
            "sourceWorldQueryHash": floor["sha256"]}}
    core = {"schemaVersion": 1, "kind": "native_symbolic_round_policy_bounds_v2",
        "strategicCoverage": False, "scope": scope, "scopeHash": canonical_sha256(scope),
        "upperBound": 0, "policy": {},
        "queries": [{"branchModelHash": branch, "purpose": "universal_path_floor",
            "receipt": floor}, {"branchModelHash": branch,
            "purpose": "adversarial_world_upper_bound",
            "sourceWorldQueryHash": floor["sha256"], "receipt": ceiling},
            {"branchModelHash": branch, "receipt": terminal}],
        "finalBranchProofs": [{"branchModelHash": branch,
            "refinementHash": refinement_hash, "policyIndependent": True}],
        "refinements": {refinement_hash: {"core": {key: value for key, value in
            refinement.items() if key != "queries"}, "queryHashes": [canonical_sha256(terminal)]}}}
    if policy_bound:
        from baloot_platform.research.ortools_play_tail import native_observer_safe_policy_identity_v1
        identity = native_observer_safe_policy_identity_v1("observer_safe_tactical_category_v1", Team.WEST_EAST)
        binding = owned({"schemaVersion": 1, "kind": "native_visible_play_policy_binding_v1",
            "policy": identity, "sourceCpModelProtoHash": "c" * 64, "boundCpModelProtoHash": "d" * 64})
        ceiling = owned({**{key: value for key, value in ceiling.items() if key != "sha256"},
            "baseModelHash": "e" * 64})
        policy_receipt = owned({"schemaVersion": 1, "kind": "native_constant_policy_round_query_v1",
            "strategicCoverage": False, "status": "queried", "rulesHash": "r" * 64, "bindings": [],
            "scope": branch_scope, "targetTeam": "west_east", "query": ceiling,
            "policy": {"team": "west_east", "auctionAction": None, "escalationAction": None,
                "scope": "future_nonplay_constants_and_native_visible_play", "playPrefixLength": 0,
                "nativePlayPolicy": identity, "nativePlayPolicyBinding": binding,
                "playPolicyModelHash": "e" * 64}})
        scope.pop("fixedWorldCooperativeCeiling")
        scope.update(opponentCeilingPolicy={"auction_action": None, "escalation_action": None,
            "native_play_policy_id": identity["policyId"]}, opponentCeilingWorldQueryHash=floor["sha256"],
            opponentCeilingProjectActions={})
        core["scopeHash"] = canonical_sha256(scope)
        core["queries"][1].update(purpose="fixed_world_opponent_upper_bound", receipt=policy_receipt)
    result = owned(core)
    _validate_native_final_branch_proofs_v2(result, target_team="north_south")
    forged = copy.deepcopy(result)
    if policy_bound:
        forged["scope"]["utilityBounds"][1] = 11
    else:
        forged["scope"]["fixedWorldCooperativeCeiling"]["value"] = 11
    forged["scopeHash"] = canonical_sha256(forged["scope"])
    forged["sha256"] = canonical_sha256({key: value for key, value in forged.items() if key != "sha256"})
    with pytest.raises(ValueError, match="ownership or coverage"):
        _validate_native_final_branch_proofs_v2(forged, target_team="north_south")
    if policy_bound:
        for case in ("world_reference", "missing_bound", "policy_identity", "world_deck",
                "malformed_projects", "wrong_projects", "binding_model", "upper_status"):
            forged = copy.deepcopy(result)
            world = forged["queries"][0]["receipt"]
            bound = forged["queries"][1]["receipt"]
            if case == "world_reference":
                forged["scope"]["opponentCeilingWorldQueryHash"] = "0" * 64
            elif case == "missing_bound":
                del forged["queries"][1]
            elif case == "policy_identity":
                bound["policy"]["nativePlayPolicy"]["policyId"] = "unknown_policy"
            elif case == "world_deck":
                world["candidate"]["deckOrder"] = [1, 0, *cards[2:]]
            elif case == "malformed_projects":
                world["candidate"]["projectObservations"] = [None]
            elif case == "wrong_projects":
                forged["scope"]["opponentCeilingProjectActions"] = {"west": "pass"}
            elif case == "binding_model":
                bound["policy"]["nativePlayPolicyBinding"]["boundCpModelProtoHash"] = "c" * 64
            else:
                bound["query"]["solverStatus"] = "feasible"

            def refresh(document):
                if isinstance(document, dict):
                    for value in document.values():
                        refresh(value)
                    if "sha256" in document:
                        document["sha256"] = canonical_sha256({key: value for key, value in document.items()
                            if key != "sha256"})
                elif isinstance(document, list):
                    for value in document:
                        refresh(value)

            refresh(forged)
            if case in {"world_deck", "malformed_projects"}:
                forged["scope"]["opponentCeilingWorldQueryHash"] = world["sha256"]
                forged["queries"][1]["sourceWorldQueryHash"] = world["sha256"]
            forged["scopeHash"] = canonical_sha256(forged["scope"])
            refresh(forged)
            with pytest.raises(ValueError, match="ownership or coverage"):
                _validate_native_final_branch_proofs_v2(forged, target_team="north_south")


@pytest.mark.parametrize("decision_kind", ("play", "project"))
@pytest.mark.parametrize("first_witness_infeasible", (False, True))
def test_native_adversarial_witness_does_not_restrict_source(root, monkeypatch,
        decision_kind, first_witness_infeasible):
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import canonical_sha256

    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    scenario._complete_information_deck = None
    scenario._native_project_root = {"fixture": True} if decision_kind == "project" else None
    scenario._native_play_root = {"fixture": True} if decision_kind == "play" else None
    scenario._native_play_source_query = {"candidate": {
        "playObservations": [{"actor": "north"}], "projectObservations": [{"actor": "north"}]}}
    scenario._recorded_play_count = scenario._recorded_project_count = 0
    before = str(scenario.model.proto)
    calls = []

    def query(self, **kwargs):
        calls.append(self)
        assert str(scenario.model.proto) == before
        if len(calls) == 3 and first_witness_infeasible:
            core = {"status": "no_matching_outcome", "solverStatus": "infeasible"}
        else:
            core = {"status": "witness_found", "solverStatus": "optimal", "candidate": {
                "utility": -10 if len(calls) == 2 else 0, "deckOrder": list(range(32)),
                "projectObservations": [{"actor": seat, "selectedAction": "pass"}
                    for seat in ("west", "east")]}}
        return {**core, "sha256": canonical_sha256(core)}

    def policy_query(self, **kwargs):
        assert self is not scenario
        assert kwargs["policy_team"] == Team.WEST_EAST
        assert kwargs["project_actions"] == {"west": "pass", "east": "pass"}
        core = {"status": "queried", "query": {"status": "witness_found", "solverStatus": "optimal",
            "candidate": {"utility": 0}}, "bindings": [{"status": "bound"}] * 2}
        return {**core, "sha256": canonical_sha256(core)}

    monkeypatch.setattr(OrToolsScenarioModelV1, "find_native_round_margin_below", query)
    monkeypatch.setattr(OrToolsScenarioModelV1, "find_native_round_with_constant_policy_below", policy_query)
    result = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._solve_symbolic_round_policy_bounds(
        [scenario], Team.NORTH_SOUTH, maximum_iterations=1, maximum_queries=10,
        maximum_observations=10, maximum_branches=1, maximum_conflicts=1000, maximum_seconds=15,
        tighten_utility_ceiling=True, observed_play_path_bounds=True, retain_path_floor=True)
    assert result["lowerBound"] == result["upperBound"] == 0
    assert result["strategicCoverage"] is False
    witnesses = [row for row in result["queries"] if row.get("purpose") == "adversarial_world_witness"]
    expected = ["partner_without_projects_v1", "single_suit_hands_v1"]
    if decision_kind == "project":
        expected.reverse()
    assert [row["construction"] for row in witnesses] == expected[:1 + first_witness_infeasible]
    assert result["scope"]["opponentCeilingWorldQueryHash"] == witnesses[-1]["receipt"]["sha256"]
    assert result["bindingQueriesUsed"] == 2
    assert result["queriesUsed"] == len(result["queries"]) + 2
    assert str(scenario.model.proto) == before
    assert all(not getattr(scenario, "_observed_" + kind + "_bindings")
        for kind in ("auction", "escalation", "project", "play"))


@pytest.mark.parametrize("policy_bound", (False, True))
def test_symbolic_proof_reuse_excludes_policy_dependent_bounds(root, monkeypatch, policy_bound):
    # Isolate reuse orchestration; native proof semantics are checked separately.
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import canonical_sha256

    rules, excluded, _, _ = _native_hokum_round_branch(root, "normal")
    _, changing, _, _ = _native_hokum_round_branch(root, "double")
    # Prove absence before changing policy, regardless of native model-size drift.
    excluded, changing = sorted((excluded, changing), key=lambda scenario: (
        len(scenario.model.proto.constraints), len(scenario.model.proto.variables),
        scenario._model_hash(search_branching="automatic")))
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    calls = {id(excluded): 0, id(changing): 0}
    proof_queries = {}
    def refine(scenario, target, policy, **kwargs):
        calls[id(scenario)] += 1
        scope = scenario._native_round_query_context()[0]
        model_hash = scenario._model_hash(search_branching="automatic")
        bindings = [{"status": "bound"}] if policy_bound else []
        query = {"status": "no_matching_outcome", "solverStatus": "infeasible",
            "scope": scope, "baseModelHash": model_hash, "targetTeam": target.name.lower(),
            "threshold": kwargs["threshold"], "auctionPolicyBindings": bindings,
            "escalationPolicyBindings": [], "projectPolicyBindings": [], "playPolicyBindings": []}
        query["sha256"] = canonical_sha256(query)
        proof_queries.setdefault(id(scenario), query)
        receipt = {"queries": [query], "domains": {}, "bindings": bindings,
            "scope": scope, "rulesHash": rules.sha256, "targetTeam": target.name.lower(),
            "sourceModelHash": model_hash, "boundModelHash": model_hash}
        if scenario is excluded or calls[id(scenario)] > 1:
            core = {**receipt, "status": "no_adverse_round_in_branch",
                "policyHash": canonical_sha256(policy), "threshold": kwargs["threshold"],
                "bindings": [{"status": "bound"}] if policy_bound else []}
            return {**core, "sha256": canonical_sha256(core)}
        return {**receipt, "status": "policy_extension_required", "domains": {"root": ("a",)},
            "candidate": {"utility": 0, "auctionObservations": [{"actor": "north",
                "observationHash": "root", "selectedAction": "a"}],
                "escalationObservations": [], "projectObservations": [], "playObservations": []}}
    monkeypatch.setattr(compiler, "_refine_symbolic_round_policy", refine)
    result = compiler._solve_symbolic_round_policy_bounds([excluded, changing], Team.NORTH_SOUTH,
        maximum_iterations=3, maximum_queries=10, maximum_observations=10,
        maximum_branches=2, maximum_conflicts=1000, maximum_seconds=15)
    assert result["status"] == "scoped_bound_converged" and not result["strategicCoverage"]
    assert {row["branchModelHash"] for row in result["finalBranchProofs"]} == {
        model._model_hash(search_branching="automatic") for model in (excluded, changing)}
    queries_by_hash = {canonical_sha256(row["receipt"]): row["receipt"] for row in result["queries"]}
    for row in result["finalBranchProofs"]:
        compact = result["refinements"][row["refinementHash"]]
        restored = {**compact["core"], "queries": [queries_by_hash[key] for key in compact["queryHashes"]]}
        assert canonical_sha256(restored) == row["refinementHash"]
        assert restored["threshold"] >= result["upperBound"]
        assert not restored["bindings"] if row["policyIndependent"] else restored["policyHash"] == canonical_sha256(result["policy"])
    assert calls[id(changing)] == 2
    assert calls[id(excluded)] == (2 if policy_bound else 1)
    if policy_bound:
        assert "reusedBranchProofs" not in result
    else:
        reuse, = result["reusedBranchProofs"]
        assert reuse["requestedThreshold"] == 0 < reuse["provedThreshold"]
        assert reuse["queryHash"] == canonical_sha256(proof_queries[id(excluded)])
        assert reuse["branchModelHash"] == excluded._model_hash(search_branching="automatic")
    assert result["sha256"] == canonical_sha256({key: value for key, value in result.items() if key != "sha256"})
    for mutation in ("policy", "threshold", "unknown", "binding", "checksum", "rules", "scope",
                     "team", "source_model", "query_model", "query_threshold", "query_team",
                     "query_checksum", "stripped_bindings"):
        calls.update({id(excluded): 0, id(changing): 0})
        def corrupt(*args, **kwargs):
            row = copy.deepcopy(refine(*args, **kwargs))
            if row["status"] != "no_adverse_round_in_branch":
                return row
            if mutation == "policy":
                row["policyHash"] = "stale-policy"
            elif mutation == "threshold":
                row["threshold"] -= 1
            elif mutation == "unknown":
                row["queries"][-1]["solverStatus"] = "unknown"
            elif mutation == "binding":
                row["bindings"] = [{"status": "incomplete"}]
            elif mutation == "rules":
                row["rulesHash"] = "different-rules"
            elif mutation == "scope":
                row["scope"] = {}
            elif mutation == "team":
                row["targetTeam"] = Team.NORTH_SOUTH.other.name.lower()
            elif mutation == "source_model":
                row["sourceModelHash"] = "different-model"
            elif mutation == "query_model":
                row["queries"][-1]["baseModelHash"] = "different-model"
            elif mutation == "query_threshold":
                row["queries"][-1]["threshold"] -= 1
            elif mutation == "query_team":
                row["queries"][-1]["targetTeam"] = Team.NORTH_SOUTH.other.name.lower()
            elif mutation == "stripped_bindings":
                row["bindings"] = []
                row["queries"][-1]["playPolicyBindings"] = [{"status": "bound"}]
            query = row["queries"][-1]
            query["sha256"] = ("corrupted" if mutation == "query_checksum" else
                canonical_sha256({key: value for key, value in query.items() if key != "sha256"}))
            if mutation != "checksum":
                row["sha256"] = canonical_sha256({key: value for key, value in row.items() if key != "sha256"})
            else:
                row["sha256"] = "corrupted"
            return row
        monkeypatch.setattr(compiler, "_refine_symbolic_round_policy", corrupt)
        with pytest.raises(ValueError, match="branch absence evidence mismatch"):
            compiler._solve_symbolic_round_policy_bounds([excluded, changing], Team.NORTH_SOUTH,
                maximum_iterations=3, maximum_queries=10, maximum_observations=10,
                maximum_branches=2, maximum_conflicts=1000, maximum_seconds=15)


@pytest.mark.parametrize("target", tuple(Team))
def test_symbolic_cooperative_ceiling_is_native_and_not_strategic(root, target):
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1,
    )
    from baloot_platform.research.scenario_contracts import canonical_sha256

    rules, normal, _, _ = _native_hokum_round_branch(root, "normal")
    _, doubled, _, _ = _native_hokum_round_branch(root, "double")
    before = [str(model.model.proto) for model in (normal, doubled)]
    impossible = copy.copy(normal)
    impossible.model = normal.model.clone()
    impossible.model.add_bool_or([])
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    limits = dict(maximum_iterations=4, maximum_queries=20, maximum_observations=100,
        maximum_branches=3, maximum_conflicts=250_000, maximum_seconds=60,
        tighten_utility_ceiling=True)
    result = compiler._solve_symbolic_round_policy_bounds([normal, doubled, impossible], target, **limits)
    from baloot_platform.research.result_authority import _validate_native_final_branch_proofs_v2
    _validate_native_final_branch_proofs_v2(result, target_team=target.name.lower())
    assert len(result["finalBranchProofs"]) == 3
    assert sum(row.get("infeasible") is True for row in result["finalBranchProofs"]) == 1
    bounds = [row for row in result["queries"] if row.get("purpose") == "cooperative_upper_bound"]
    assert len(bounds) == 3
    assert sum(row["receipt"]["solverStatus"] == "infeasible" for row in bounds) == 1
    bounds = [row for row in bounds if row["receipt"]["candidate"] is not None]
    assert all(row["receipt"]["solverStatus"] == "optimal" for row in bounds)
    assert all(row["receipt"]["targetTeam"] == target.other.name.lower() for row in bounds)
    ceiling = max(-row["receipt"]["candidate"]["utility"] for row in bounds)
    assert result["scope"]["utilityBounds"][1] == ceiling
    assert result["upperBound"] <= ceiling < max(normal._native_round_query_context()[1][0].proto.domain)
    assert result["masters"][-1]["upperBound"] == result["upperBound"]
    for cut in result["cuts"]:
        if "prefixBoundQueryHash" in cut:
            proof = next(row["receipt"] for row in result["queries"]
                         if row["receipt"]["sha256"] == cut["prefixBoundQueryHash"])
            assert proof["status"] == "upper_bound_complete"
            assert proof["upperBound"] == cut["utility"]
            assert proof["targetActions"] == cut["targetActions"]
    assert result["strategicCoverage"] is False
    assert result["scopeHash"] == canonical_sha256(result["scope"])
    assert result["sha256"] == canonical_sha256({key: value for key, value in result.items() if key != "sha256"})
    assert [str(model.model.proto) for model in (normal, doubled)] == before
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._solve_symbolic_round_policy_bounds([normal, doubled], target,
            **{**limits, "maximum_queries": 1})
    partial = stopped.value.partial_receipt["evidence"]
    assert stopped.value.reason == "maximum_queries"
    assert len(partial["queries"]) == 1 and not partial["masters"]
    assert partial["scope"]["utilityBounds"][1] > ceiling  # Partial bounds cannot replace the ceiling.
    with pytest.raises(ValueError, match="no feasible supplied branch"):
        compiler._solve_symbolic_round_policy_bounds([impossible], target, **limits)
    for invalid in (1, None, "true"):
        with pytest.raises(ValueError, match="ceiling selection"):
            compiler._solve_symbolic_round_policy_bounds([normal], target,
                **{**limits, "tighten_utility_ceiling": invalid})
    policy = {"auction_action": "pass", "escalation_action": "decline",
        "native_play_policy_id": "observer_safe_trick_control_v1"}
    direct = {**limits, "tighten_utility_ceiling": False, "maximum_queries": 1,
        "_unavailable_opponent_ceiling_policy": policy}
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._solve_symbolic_round_policy_bounds([normal, doubled], target, **direct)
    partial = stopped.value.partial_receipt["evidence"]
    assert stopped.value.reason == "maximum_queries" and len(partial["masters"]) == 1
    assert not partial["queries"] and partial["status"] == "incomplete"
    assert "cooperativeCeiling" not in partial["scope"]
    assert partial["scope"]["opponentCeilingUnavailable"] == {
        "reason": "multiple_contract_branches", "policy": policy}
    assert partial["scopeHash"] == canonical_sha256(partial["scope"])
    with pytest.raises(ValueError, match="direct multi-branch refinement"):
        compiler._solve_symbolic_round_policy_bounds([normal], target, **direct)


@pytest.mark.parametrize("complete_information", (False, True))
@pytest.mark.parametrize("level", ("normal", "double"))
def test_native_counterexample_prefix_bound_owns_a_fresh_legal_suffix(root, complete_information, level, monkeypatch, record_property):
    import time
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import canonical_sha256

    rules, scenario, _, _ = _native_hokum_round_branch(root, level)
    if complete_information:
        scenario._complete_information_deck = tuple(range(32))
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    query = scenario.find_native_round_margin_below(target=Team.NORTH_SOUTH, threshold=10000,
        maximum_seconds=15, maximum_conflicts=250_000)
    assert query["status"] == "witness_found"
    before = str(scenario.model.proto)
    args = dict(prefix_play_count=24, maximum_seconds=15, maximum_conflicts=250_000)
    started = time.perf_counter()
    result = compiler._bound_native_counterexample_prefix(scenario, query, Team.NORTH_SOUTH, **args)
    record_property("exactMaximumSeconds", time.perf_counter() - started)
    assert result["status"] == "upper_bound_complete"
    assert result["upperBound"] >= query["candidate"]["utility"]
    assert result["scope"] == "fixed_world_cooperative_upper_bound" and not result["strategicCoverage"]
    assert result["nativePlay"]["modelHash"] == result["modelHash"]
    assert result["sourceQueryHash"] == query["sha256"]
    assert result["sha256"] == canonical_sha256({key: value for key, value in result.items() if key != "sha256"})
    expected = sorted({(row["observationHash"], row["selectedAction"])
        for kind in ("auction", "escalation", "project", "play")
        for row in (query["candidate"][kind + "Observations"][:24]
                    if kind == "play" else query["candidate"][kind + "Observations"])
        if not row.get("authoredPrefix", False) and Seat.parse(row["actor"]).team == Team.NORTH_SOUTH})
    assert result["targetActions"] == [list(row) for row in expected]
    for threshold in (result["upperBound"], result["upperBound"] + 1):
        started = time.perf_counter()
        bound = compiler._bound_native_counterexample_prefix(scenario, query, Team.NORTH_SOUTH,
            **args, required_threshold=threshold)
        record_property("thresholdWitnessSeconds" if threshold == result["upperBound"] else "thresholdAbsenceSeconds",
            time.perf_counter() - started)
        assert bound["kind"] == "native_counterexample_prefix_threshold_v1"
        assert bound["complementThreshold"] == 1 - threshold
        assert bound["complementTargetTeam"] == "west_east"
        if threshold > result["upperBound"]:
            assert bound["status"] == "upper_bound_complete" and bound["solverStatus"] == "infeasible"
            assert bound["upperBound"] == threshold - 1
            assert bound["targetActions"] == result["targetActions"] and bound["nativePlay"] is None
            from baloot_platform.research.result_authority import validate_native_prefix_threshold_v1
            validation = dict(rules=rules, scenario=scenario, source_query=query, target=Team.NORTH_SOUTH,
                **args, required_threshold=threshold)
            assert validate_native_prefix_threshold_v1(bound, **validation) == threshold - 1
            # After a real native reconstruction, exercise receipt substitution
            # against that same independently produced expected document.
            with monkeypatch.context() as patch:
                patch.setattr(OrToolsActorInformationSetCompilerV1, "_bound_native_counterexample_prefix",
                    lambda *a, **kw: bound)
                for field, value in (("upperBound", threshold), ("requiredThreshold", threshold + 1),
                        ("complementThreshold", threshold), ("complementTargetTeam", "north_south"),
                        ("modelHash", "0" * 64), ("prefixHash", "0" * 64),
                        ("sourceQueryHash", "0" * 64), ("targetActions", []), ("schemaVersion", True)):
                    corrupt_bound = {**bound, field: value}
                    corrupt_bound["sha256"] = canonical_sha256({k: v for k, v in corrupt_bound.items() if k != "sha256"})
                    with pytest.raises(ValueError, match="evidence mismatch"):
                        validate_native_prefix_threshold_v1(corrupt_bound, **validation)
            with pytest.raises(ValueError, match="reverification"):
                validate_native_prefix_threshold_v1(bound, **{**validation, "cancel_check": lambda: True})
        else:
            assert bound["status"] == "threshold_reachable"
            assert "upperBound" not in bound and "targetActions" not in bound
        assert bound["sha256"] == canonical_sha256({key: value for key, value in bound.items() if key != "sha256"})
    future = {row["observationHash"] for row in query["candidate"]["playObservations"][24:]}
    assert not future.intersection(row[0] for row in result["targetActions"])
    assert str(scenario.model.proto) == before
    for change in ({"cancel_check": lambda: True}, {"maximum_seconds": 1e-12}):
        stopped = compiler._bound_native_counterexample_prefix(scenario, query, Team.NORTH_SOUTH, **{**args, **change})
        assert stopped["status"] == "incomplete"
        assert "upperBound" not in stopped and "targetActions" not in stopped
    for status in (cp_model.UNKNOWN, cp_model.FEASIBLE):
        with monkeypatch.context() as patch:
            patch.setattr("baloot_platform.research.ortools_complete_state_graph._solve_cp_sat_with_cancel_v1",
                          lambda *args, **kwargs: status)
            stopped = compiler._bound_native_counterexample_prefix(scenario, query, Team.NORTH_SOUTH, **args)
        assert stopped["status"] == "incomplete" and "exactState" in stopped
        assert "upperBound" not in stopped and "targetActions" not in stopped
        cause = stopped["cause"]
        assert cause["evidence"]["queryPhase"] == "solver"
        assert len(cause["evidence"]["queryModelHash"]) == 64
        assert cause["sha256"] == canonical_sha256({key: value for key, value in cause.items() if key != "sha256"})
    for invalid in (True, 0, 3, 32):
        with pytest.raises(ValueError, match="post-project prefix"):
            compiler._bound_native_counterexample_prefix(scenario, query, Team.NORTH_SOUTH,
                **{**args, "prefix_play_count": invalid})
    corrupt = copy.deepcopy(query)
    corrupt["candidate"]["utility"] += 1
    with pytest.raises(ValueError, match="ownership or checksum"):
        compiler._bound_native_counterexample_prefix(scenario, corrupt, Team.NORTH_SOUTH, **args)
    other = copy.copy(scenario)
    other.model = scenario.model.clone()
    other.model.add(0 == 0)
    with pytest.raises(ValueError, match="ownership or checksum"):
        compiler._bound_native_counterexample_prefix(other, query, Team.NORTH_SOUTH, **args)


@pytest.mark.real_clock
@pytest.mark.parametrize("target", (Team.NORTH_SOUTH, Team.WEST_EAST))
@pytest.mark.parametrize("complete_information", (False, True))
def test_native_minimax_prefix_bound_reconstructs_and_rejects_substitution(root, monkeypatch, target, complete_information):
    import yaml
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.result_authority import validate_native_prefix_minimax_v1
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, canonical_sha256

    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    if complete_information:
        scenario._complete_information_deck = tuple(range(32))
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
    budget = ScenarioProofBudgetV1.build("offline_heavyweight_v1",
        settings["proofBudgets"]["profiles"]["offline_heavyweight_v1"])
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    query = scenario.find_native_round_margin_below(target=target, threshold=10000,
        maximum_seconds=15, maximum_conflicts=250_000)
    assert query["status"] == "witness_found"
    args = dict(prefix_play_count=24, maximum_seconds=15, maximum_conflicts=250_000, minimax_budget=budget)
    before = str(scenario.model.proto)
    from baloot_platform.research import ortools_complete_state_graph as graph_module
    original_solve = graph_module._solve_cp_sat_with_cancel_v1
    graph_limits = []
    def observed_solve(solver, model, cancelled, *extra):
        graph_limits.append(solver.parameters.max_number_of_conflicts)
        return original_solve(solver, model, cancelled, *extra)
    with monkeypatch.context() as patch:
        patch.setattr(graph_module, "_solve_cp_sat_with_cancel_v1", observed_solve)
        bound = compiler._bound_native_counterexample_prefix(scenario, query, target, **args)
    assert graph_limits and set(graph_limits) == {250_000}
    assert bound["status"] == "upper_bound_complete"
    assert bound["kind"] == "native_counterexample_prefix_minimax_v1"
    assert bound["scope"] == "complete_state_minimax_upper_bound" and not bound["strategicCoverage"]
    proof = bound["proof"]
    assert proof["completionStatus"] in {"exact_complete", "proof_failed"}
    assert (proof["completionStatus"] == "exact_complete") == proof["claimSatisfied"]
    assert bound["upperBound"] == next(row["value"] for row in proof["rootActions"]
        if row["action"] == proof["selectedAction"])
    validation = dict(rules=rules, scenario=scenario, source_query=query, target=target, **args)
    assert validate_native_prefix_minimax_v1(bound, **validation) == bound["upperBound"]
    assert str(scenario.model.proto) == before
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    warm = compiler._bound_native_counterexample_prefix(scenario, query, target, **args, _normalizer=normalizer)
    reused = compiler._bound_native_counterexample_prefix(scenario, query, target, **args, _normalizer=normalizer)
    for row in (warm, reused):
        assert row["status"] == "upper_bound_complete" and row["upperBound"] == bound["upperBound"]
        assert row["exactState"]["sha256"] == bound["exactState"]["sha256"]
        assert row["modelHash"] == bound["modelHash"]
    key, values, digest = normalizer._deal_prefix_certificate
    normalizer._deal_prefix_certificate = (key, values, "0" * len(digest))
    with pytest.raises(ValueError, match="certificate integrity"):
        compiler._bound_native_counterexample_prefix(scenario, query, target, **args, _normalizer=normalizer)
    for invalid in (object(), OrToolsExactStateNormalizerV1(rules)):
        with pytest.raises(ValueError, match="matching native normalizer"):
            compiler._bound_native_counterexample_prefix(scenario, query, target, **args, _normalizer=invalid)
    small_budget = ScenarioProofBudgetV1.build("diagnostic_nodes",
        {**settings["proofBudgets"]["profiles"]["offline_heavyweight_v1"], "maximumNodes": 1})
    exhausted = compiler._bound_native_counterexample_prefix(scenario, query, target,
        **{**args, "minimax_budget": small_budget})
    assert exhausted["status"] == "incomplete" and "upperBound" not in exhausted
    cpu_budget = ScenarioProofBudgetV1.build("diagnostic_cpu",
        {**settings["proofBudgets"]["profiles"]["offline_heavyweight_v1"], "maximumCpuSeconds": 1e-9})
    exhausted = compiler._bound_native_counterexample_prefix(scenario, query, target,
        **{**args, "minimax_budget": cpu_budget})
    assert exhausted["status"] == "incomplete" and "upperBound" not in exhausted
    with monkeypatch.context() as patch:
        patch.setattr(OrToolsActorInformationSetCompilerV1, "_bound_native_counterexample_prefix", lambda *a, **kw: bound)
        for key, value in (("upperBound", bound["upperBound"] + 1), ("graphHash", "0" * 64),
                ("proofBudgetHash", "0" * 64), ("sourceQueryHash", "0" * 64), ("targetActions", [])):
            corrupt = {**bound, key: value}
            corrupt["sha256"] = canonical_sha256({k:v for k,v in corrupt.items() if k != "sha256"})
            with pytest.raises(ValueError, match="evidence mismatch"):
                validate_native_prefix_minimax_v1(corrupt, **validation)
    stopped = compiler._bound_native_counterexample_prefix(scenario, query, target, **args, cancel_check=lambda: True)
    assert stopped["status"] == "incomplete" and "upperBound" not in stopped
    assert scenario._complete_information_deck == (tuple(range(32)) if complete_information else None)
    with pytest.raises(ValueError, match="typed proof budget"):
        compiler._bound_native_counterexample_prefix(scenario, query, target,
            **{**args, "minimax_budget": settings["proofBudgets"]["profiles"]["offline_heavyweight_v1"]})


@pytest.mark.real_clock
@pytest.mark.parametrize("complete_information", (False, True))
@pytest.mark.parametrize("exhaustion_reason", ("maximum_nodes", "maximum_seconds"))
def test_native_refinement_uses_owned_minimax_suffix_and_falls_back(root, monkeypatch, complete_information, exhaustion_reason):
    import yaml
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1, canonical_sha256

    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    if complete_information:
        scenario._complete_information_deck = tuple(range(32))
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
    budget = ScenarioProofBudgetV1.build("suffix_test",
        {**settings["proofBudgets"]["profiles"]["offline_heavyweight_v1"], "maximumDepth": 8})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    limits = dict(threshold=10000, maximum_queries=3, maximum_observations=100,
        maximum_conflicts=250_000, maximum_seconds=30, shorten_counterexamples=True, minimax_budget=budget)
    result = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {}, **limits)
    bound = next(row for row in result["queries"] if row["sha256"] == result["prefixBoundQueryHash"])
    assert bound["kind"] == "native_counterexample_prefix_threshold_v1"
    assert len(result["queries"]) == 2  # No graph is built when the cheap proof suffices.
    assert bound["status"] == "upper_bound_complete" and bound["prefixPlayCount"] == 4
    assert bound["upperBound"] < limits["threshold"]
    assert result["minimaxProofBudgetHash"] == budget.sha256
    assert set(result["domains"]) == {key for key, _ in bound["targetActions"]}
    original_bound = compiler._bound_native_counterexample_prefix
    visited = []
    normalizers = []
    shares = []
    def node_limited(*args, **kwargs):
        normalizers.append(kwargs["_normalizer"])
        shares.append((kwargs["prefix_play_count"], kwargs.get("minimax_budget") is not None, kwargs["maximum_seconds"]))
        assert 0 < kwargs["maximum_seconds"] < limits["maximum_seconds"] / 2
        if kwargs.get("minimax_budget") is None:
            # Force the orchestration branch only; the eventual bound below is
            # a real native proof, not this synthetic reachability receipt.
            reachable = {"kind": "native_counterexample_prefix_threshold_v1", "status": "threshold_reachable"}
            return {**reachable, "sha256": canonical_sha256(reachable)}
        visited.append(kwargs["prefix_play_count"])
        assert 32 - kwargs["prefix_play_count"] <= budget.maximum_depth
        if len(visited) == 1:
            stopped = {"kind": "native_counterexample_prefix_minimax_v1", "status": "incomplete",
                "graphEvidence": {"exhaustionReason": exhaustion_reason}}
            return {**stopped, "sha256": canonical_sha256(stopped)}
        return original_bound(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(compiler, "_bound_native_counterexample_prefix", node_limited)
        smaller = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {},
            **{**limits, "maximum_queries": 7})
    assert len(visited) == 2 and visited[0] == 24 and visited[1] > 24
    assert normalizers and all(row is normalizers[0] for row in normalizers)
    paired = [row for row in shares if row[0] == visited[0]]
    assert paired[0][1] is False and paired[1][1] is True
    assert paired[1][2] > paired[0][2]  # Do not charge the completed cheap query twice.
    certified = next(row for row in smaller["queries"] if row["sha256"] == smaller["prefixBoundQueryHash"])
    assert certified["status"] == "upper_bound_complete" and certified["prefixPlayCount"] == visited[1]
    failed = {"kind": "native_counterexample_prefix_minimax_v1", "status": "incomplete"}
    failed["sha256"] = canonical_sha256(failed)
    monkeypatch.setattr(compiler, "_bound_native_counterexample_prefix", lambda *a, **kw: failed)
    fallback = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {}, **limits)
    assert "prefixBoundQueryHash" not in fallback and fallback["candidate"] == result["candidate"]


def test_native_refinement_does_not_extend_forced_choices(root, monkeypatch):
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1

    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    limits = dict(threshold=10000, maximum_queries=1, maximum_observations=100,
        maximum_conflicts=250000, maximum_seconds=30)
    witness = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {}, **limits)
    query = witness["queries"][0]  # A real, owned native counterexample, not mocked truth.
    rows = [row for stage in ("auction", "escalation", "project", "play")
        for row in query["candidate"][stage + "Observations"]
        if not row.get("authoredPrefix", False) and Seat.parse(row["actor"]).team == Team.NORTH_SOUTH]
    forced = [row for row in rows if len(row["legalActions"]) == 1]
    assert forced and any(len(row["legalActions"]) > 1 for row in rows)
    policy = {row["observationHash"]: row["selectedAction"] for row in rows if len(row["legalActions"]) > 1}
    monkeypatch.setattr(OrToolsScenarioModelV1, "find_native_round_margin_below", lambda *a, **kw: query)
    result = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, policy, **limits)
    assert result["status"] == "policy_counterexample_in_branch"
    assert result["candidate"] == query["candidate"] and result["bindings"] == []
    missing = dict(policy)
    missing.pop(next(iter(missing)))
    assert compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, missing,
        **limits)["status"] == "policy_extension_required"
    illegal = {**policy, forced[0]["observationHash"]: "invalid"}
    with pytest.raises(ValueError, match="illegal visible action"):
        compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, illegal, **limits)


@pytest.mark.parametrize("complete_information", (False, True))
@pytest.mark.parametrize("optimize_prefix_bounds", (False, True))
def test_native_refinement_shortens_only_proved_counterexamples(root, monkeypatch, complete_information, optimize_prefix_bounds):
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import canonical_sha256

    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    if complete_information:
        scenario._complete_information_deck = tuple(range(32))
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    limits = dict(threshold=10000, maximum_queries=3, maximum_observations=100,
        maximum_conflicts=250_000, maximum_seconds=30, shorten_counterexamples=True,
        optimize_prefix_bounds=optimize_prefix_bounds)
    result = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {}, **limits)
    bound = next(row for row in result["queries"] if row["sha256"] == result["prefixBoundQueryHash"])
    assert len(result["queries"]) == (2 if optimize_prefix_bounds else 3)
    assert bound["status"] == "upper_bound_complete"
    if optimize_prefix_bounds:
        assert bound["kind"] == "native_counterexample_prefix_bound_v1"
        assert bound["nativePlay"]["solverStatus"] == "optimal"
        assert result["candidate"]["utility"] <= bound["upperBound"] < limits["threshold"]
    else:
        assert bound["kind"] == "native_counterexample_prefix_threshold_v1"
        assert bound["solverStatus"] == "infeasible"
        assert bound["upperBound"] == limits["threshold"] - 1
    from baloot_platform.research.result_authority import _native_observed_bounds_semantics
    replayed = copy.deepcopy(result)
    changed_bound = next(row for row in replayed["queries"] if row["sha256"] == bound["sha256"])
    changed_bound["limits"]["seconds"] /= 2
    changed_bound["sha256"] = canonical_sha256({k: v for k, v in changed_bound.items() if k != "sha256"})
    replayed["prefixBoundQueryHash"] = changed_bound["sha256"]
    replayed["sha256"] = canonical_sha256({k: v for k, v in replayed.items() if k != "sha256"})
    def normalize(document):
        return _native_observed_bounds_semantics(document, maximum_seconds=30)
    assert normalize(result) == normalize(replayed)
    for invalid in (0, True, 31, float("inf")):
        corrupted = copy.deepcopy(replayed)
        row = next(row for row in corrupted["queries"] if row["sha256"] == changed_bound["sha256"])
        row["limits"]["seconds"] = invalid
        row["sha256"] = canonical_sha256({k: v for k, v in row.items() if k != "sha256"}) if invalid != float("inf") else "bad"
        with pytest.raises(ValueError):
            normalize(corrupted)
    assert bound["sourceQueryHash"] == result["queries"][0]["sha256"]
    assert bound["prefixPlayCount"] < 32 and bound["upperBound"] < limits["threshold"]
    full = {(row["observationHash"], row["selectedAction"])
        for kind in ("auction", "escalation", "project", "play")
        for row in result["candidate"][kind + "Observations"]
        if not row.get("authoredPrefix", False) and Seat.parse(row["actor"]).team == Team.NORTH_SOUTH}
    assert set(map(tuple, bound["targetActions"])) < full
    assert set(result["domains"]) == {key for key, _ in bound["targetActions"]}
    unchanged = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {},
        **{**limits, "maximum_queries": 1})
    assert len(unchanged["queries"]) == 1 and "prefixBoundQueryHash" not in unchanged
    # Optional strengthening failure preserves the original valid witness.
    failed = {"kind": "native_counterexample_prefix_bound_v1", "status": "incomplete"}
    failed["sha256"] = canonical_sha256(failed)
    monkeypatch.setattr(compiler, "_bound_native_counterexample_prefix", lambda *args, **kwargs: failed)
    fallback = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {}, **limits)
    assert len(fallback["queries"]) == 2 and "prefixBoundQueryHash" not in fallback
    assert fallback["candidate"] == unchanged["candidate"]
    assert fallback["domains"] == unchanged["domains"]
    with pytest.raises(ValueError, match="shortening selection"):
        compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {},
            **{**limits, "shorten_counterexamples": 1})
    with pytest.raises(ValueError, match="shortening selection"):
        compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {},
            **{**limits, "optimize_prefix_bounds": 1})


@pytest.mark.parametrize("cut_position", (0, 1, 2))
def test_native_shortening_finds_earliest_completed_threshold_cut(root, monkeypatch, cut_position):
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import canonical_sha256

    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    limits = dict(threshold=10000, maximum_observations=100,
        maximum_conflicts=250_000, maximum_seconds=30)
    witness = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {},
        maximum_queries=1, **limits)
    lengths = sorted({row["playIndex"] + 1 for row in witness["candidate"]["playObservations"]
        if 4 <= row["playIndex"] + 1 < 32 and len(row["legalActions"]) > 1
        and Seat.parse(row["actor"]).team == Team.NORTH_SOUTH})
    assert len(lengths) >= 3
    first_cut = lengths[(len(lengths) - 1) * cut_position // 2]
    visited = []

    def bound(*args, prefix_play_count, **kwargs):
        visited.append(prefix_play_count)
        assert kwargs["required_threshold"] == limits["threshold"]
        choice = witness["candidate"]["playObservations"][prefix_play_count - 1]
        assert choice["playIndex"] + 1 == prefix_play_count
        assert Seat.parse(choice["actor"]).team == Team.NORTH_SOUTH and len(choice["legalActions"]) > 1
        value = {"status": "upper_bound_complete" if prefix_play_count >= first_cut else "threshold_reachable",
            "prefixPlayCount": prefix_play_count,
            **({"upperBound": limits["threshold"] - 1,
                "targetActions": [[row["observationHash"], row["selectedAction"]]
                    for kind in ("auction", "escalation", "project", "play")
                    for row in witness["candidate"][kind + "Observations"]
                    if not row.get("authoredPrefix", False) and Seat.parse(row["actor"]).team == Team.NORTH_SOUTH
                    and (kind != "play" or row["playIndex"] < prefix_play_count)]}
               if prefix_play_count >= first_cut else {})}
        return {**value, "sha256": canonical_sha256(value)}

    # Isolate monotone search control; native bound soundness has separate tests.
    monkeypatch.setattr(compiler, "_bound_native_counterexample_prefix", bound)
    result = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {},
        maximum_queries=32, shorten_counterexamples=True, **limits)
    selected = next(row for row in result["queries"] if row["sha256"] == result["prefixBoundQueryHash"])
    assert selected["prefixPlayCount"] == first_cut
    assert len(visited) <= len(lengths).bit_length()
    assert len(set(visited)) == len(visited)


@pytest.mark.parametrize("status", ("feasible", "unknown"))
def test_symbolic_cooperative_ceiling_rejects_unproved_bounds(root, monkeypatch, status):
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1,
    )

    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    # Orchestration fault injection only; the positive test uses native solves.
    monkeypatch.setattr(scenario, "find_native_round_margin_below", lambda **kwargs: {
        "status": "witness_found" if status == "feasible" else "incomplete",
        "solverStatus": status, "exhaustionReason": None})
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._solve_symbolic_round_policy_bounds([scenario], Team.NORTH_SOUTH,
            maximum_iterations=1, maximum_queries=2, maximum_observations=32,
            maximum_branches=1, maximum_conflicts=1000, maximum_seconds=10,
            tighten_utility_ceiling=True)
    partial = stopped.value.partial_receipt["evidence"]
    assert stopped.value.reason == "cooperative_bound_not_optimal"
    assert partial["status"] == "incomplete" and not partial["strategicCoverage"]
    assert len(partial["queries"]) == 1 and not partial["masters"]


@pytest.mark.real_clock
@pytest.mark.parametrize("fixed_world", (False, True))
def test_symbolic_round_policy_master_retains_native_cuts_and_scope(root, fixed_world):
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1,
    )

    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    before = str(scenario.model.proto)
    limits = dict(maximum_iterations=1, maximum_queries=4, maximum_observations=32,
        maximum_branches=2, maximum_conflicts=250_000, maximum_seconds=30)
    if fixed_world:
        scenario._complete_information_deck = tuple(range(32))
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._solve_symbolic_round_policy_bounds([scenario], Team.NORTH_SOUTH, **limits)
    assert stopped.value.reason == "maximum_iterations"
    partial = stopped.value.partial_receipt["evidence"]
    assert partial["status"] == "incomplete" and not partial["strategicCoverage"]
    assert partial["scope"]["coverage"] == "supplied_deal_branches_only"
    assert len(partial["cuts"]) == len(partial["masters"]) == len(partial["queries"]) == 1
    candidate = partial["queries"][0]["receipt"]["candidate"]
    expected = sorted({(row["observationHash"], row["selectedAction"])
        for kind in ("auction", "escalation", "project", "play")
        for row in candidate[kind + "Observations"] if Seat.parse(row["actor"]).team == Team.NORTH_SOUTH})
    assert partial["cuts"][0]["targetActions"] == [list(row) for row in expected]
    assert partial["cuts"][0]["utility"] == candidate["utility"]
    assert all(tuple(partial["domains"][key]) for key, _ in expected)
    # This intentionally fixed-line test is not a complete legal universe.
    # Escaping its line can close the scoped query, but must not certify Baloot.
    result = compiler._solve_symbolic_round_policy_bounds([scenario], Team.NORTH_SOUTH,
        **{**limits, "maximum_iterations": 3, "maximum_queries": 6})
    assert result["status"] == "scoped_bound_converged"
    assert result["queriesUsed"] == 5
    assert result["bindingQueriesUsed"] == 1
    assert result["strategicCoverage"] is False
    assert result["cuts"][0] == partial["cuts"][0]
    assert str(scenario.model.proto) == before
    assert not any((scenario._observed_auction_bindings, scenario._observed_escalation_bindings,
                    scenario._observed_project_bindings, scenario._observed_play_bindings))
    for changes, reason in (({"maximum_queries": 1}, "maximum_queries"),
                            ({"maximum_queries": 4}, "maximum_queries"),
                            ({"maximum_seconds": 1e-12}, "maximum_seconds"),
                            ({"cancel_check": lambda: True}, "cancelled"),
                            ({"maximum_observations": 1}, "maximum_observations")):
        with pytest.raises(InformationSetCompilationStopped) as exhausted:
            compiler._solve_symbolic_round_policy_bounds([scenario], Team.NORTH_SOUTH,
                **{**limits, "maximum_iterations": 3, **changes})
        assert exhausted.value.reason == reason
        assert exhausted.value.partial_receipt["evidence"]["status"] == "incomplete"
        if changes.get("maximum_queries") == 4:
            assert exhausted.value.partial_receipt["evidence"]["queriesUsed"] == 4
    for changes in ({"maximum_branches": False}, {"maximum_iterations": 0},
                    {"maximum_seconds": float("nan")}):
        with pytest.raises(ValueError):
            compiler._solve_symbolic_round_policy_bounds([scenario], Team.NORTH_SOUTH, **{**limits, **changes})
    with pytest.raises(ValueError, match="distinct"):
        compiler._solve_symbolic_round_policy_bounds([scenario, scenario], Team.NORTH_SOUTH, **limits)
    assert scenario._observed_escalation_bindings == {}


def test_symbolic_master_reuses_auction_context_from_bound_witness(root, monkeypatch, record_property):
    from baloot_platform.research.ortools_information_set_compiler import (
        OrToolsActorInformationSetCompilerV1,
    )
    from baloot_platform.research.scenario_contracts import canonical_sha256

    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    arguments = dict(threshold=1000, maximum_queries=8, maximum_observations=64,
        maximum_conflicts=250_000, maximum_seconds=30)
    witness = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {}, **arguments)
    observation = next(row for row in witness["candidate"]["auctionObservations"]
        if Seat.parse(row["actor"]).team == Team.NORTH_SOUTH)
    owned = copy.copy(scenario)
    owned.model, owned._observed_auction_bindings = scenario.model.clone(), {}
    binding = owned.bind_native_auction_observed_choice(actor=Seat.parse(observation["actor"]),
        visible_cards=observation["visibleCards"], purchase_card=observation["purchaseCard"],
        public_choices=observation["publicChoices"], action=observation["selectedAction"],
        maximum_seconds=10, maximum_conflicts=250_000, _prefix_legality=True)
    assert binding["status"] == "bound"
    query = owned.find_native_round_margin_below(target=Team.NORTH_SOUTH, threshold=1000,
        maximum_seconds=15, maximum_conflicts=250_000)
    assert query["status"] == "witness_found"
    assert query["baseModelHash"] != witness["sourceModelHash"]
    # This is a real conditional witness, not reusable branch-absence evidence.
    core = {**witness, "queries": [query], "bindings": [binding], "queriesUsed": 2,
        "candidate": query["candidate"], "boundModelHash": query["baseModelHash"]}
    core.pop("sha256")
    first = {**core, "sha256": canonical_sha256(core)}
    refine, calls = compiler._refine_symbolic_round_policy, []

    def measured(model, target, policy, **kwargs):
        calls.append(kwargs["_auction_contexts"])
        if len(calls) == 1:
            return first
        expected = {row["observationHash"] for row in query["candidate"]["auctionObservations"]
            if Seat.parse(row["actor"]).team == target and row["observationHash"] in policy
            and policy[row["observationHash"]] != row["selectedAction"]}
        assert expected and expected <= kwargs["_auction_contexts"].keys()
        # A known context still goes through fresh native legality and reachability.
        return refine(model, target, policy, **kwargs)

    monkeypatch.setattr(compiler, "_refine_symbolic_round_policy", measured)
    before = str(scenario.model.proto)
    result = compiler._solve_symbolic_round_policy_bounds([scenario], Team.NORTH_SOUTH,
        maximum_iterations=3, maximum_queries=10, maximum_observations=64,
        maximum_branches=1, maximum_conflicts=250_000, maximum_seconds=30)
    assert len(calls) == 2 and not calls[0]
    assert result["status"] == "scoped_bound_converged" and not result["strategicCoverage"]
    assert result["bindingQueriesUsed"] == 2
    assert len(result["queries"]) == 2  # Conditional witness, then fresh absence query.
    cold = refine(scenario, Team.NORTH_SOUTH, result["policy"],
        **{**arguments, "threshold": result["upperBound"]})
    assert cold["status"] == "no_adverse_round_in_branch" and len(cold["queries"]) == 2
    assert cold["boundModelHash"] == result["queries"][-1]["receipt"]["baseModelHash"]
    record_property("nativeOutcomeQueriesWithContext", 1)
    record_property("nativeOutcomeQueriesWithoutContext", len(cold["queries"]))
    assert str(scenario.model.proto) == before and not scenario._observed_auction_bindings


def test_symbolic_fixed_world_ceiling_preserves_adversary_model(root, monkeypatch):
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1)
    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    before = str(scenario.model.proto)
    witness = scenario.find_native_round_margin_below(target=Team.NORTH_SOUTH, threshold=1000,
        maximum_seconds=15, maximum_conflicts=250000)
    observation = next(row for row in witness["candidate"]["auctionObservations"]
                       if Seat.parse(row["actor"]).team == Team.NORTH_SOUTH)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    evidence = compiler._solve_symbolic_round_policy_bounds([scenario], Team.NORTH_SOUTH,
        maximum_iterations=1, maximum_queries=20, maximum_observations=64,
        maximum_branches=1, maximum_conflicts=250000, maximum_seconds=30,
        fixed_root=(observation["observationHash"], tuple(observation["legalActions"]),
                    observation["selectedAction"]), tighten_utility_ceiling=True,
        opponent_ceiling_policy={"auction_action": "pass", "escalation_action": "decline"})
    assert evidence["status"] == "scoped_bound_converged"
    queries = {row.get("purpose"): row["receipt"] for row in evidence["queries"]}
    assert "cooperative_upper_bound" not in queries
    world = queries["upper_bound_world_witness"]
    bound = queries["fixed_world_opponent_upper_bound"]
    assert world["baseModelHash"] == scenario._model_hash(search_branching="automatic")
    assert evidence["scope"]["opponentCeilingWorldQueryHash"] == world["sha256"]
    assert bound["query"]["solverStatus"] == "optimal"
    assert evidence["scope"]["utilityBounds"][1] <= -bound["query"]["candidate"]["utility"]
    assert evidence["strategicCoverage"] is False
    assert str(scenario.model.proto) == before
    assert not scenario._observed_auction_bindings and not scenario._observed_play_bindings
    limits = dict(maximum_iterations=1, maximum_queries=20, maximum_observations=64,
        maximum_branches=1, maximum_conflicts=250000, maximum_seconds=30,
        fixed_root=(observation["observationHash"], tuple(observation["legalActions"]),
                    observation["selectedAction"]), tighten_utility_ceiling=True,
        opponent_ceiling_policy={"auction_action": "pass", "escalation_action": "decline"})
    # Isolate negative admission paths; the positive path above uses real solves.
    for case in ("wrong_root", "unknown_world", "feasible_bound", "infeasible_bound"):
        altered = copy.deepcopy(world)
        if case == "wrong_root":
            altered["candidate"]["auctionObservations"] = []
        if case == "unknown_world":
            altered.update(status="incomplete", exhaustionReason="solver_unknown")
        with monkeypatch.context() as patch:
            patch.setattr(OrToolsScenarioModelV1, "find_native_round_margin_below", lambda *a, **kw: altered)
            patch.setattr(OrToolsScenarioModelV1, "find_native_round_with_constant_policy_below",
                lambda *a, **kw: {"status": "queried", "bindings": [], "query": {
                    "solverStatus": "feasible" if case == "feasible_bound" else "infeasible",
                    "candidate": {"utility": -999}}})
            if case in {"wrong_root", "unknown_world"}:
                with pytest.raises(InformationSetCompilationStopped) as stopped:
                    compiler._solve_symbolic_round_policy_bounds([scenario], Team.NORTH_SOUTH, **limits)
                assert stopped.value.reason == ("world_witness_root_mismatch" if case == "wrong_root" else "solver_unknown")
            else:
                def absent(*a, **kw):
                    from baloot_platform.research.scenario_contracts import canonical_sha256
                    scope = a[0]._native_round_query_context()[0]
                    model_hash = a[0]._model_hash(search_branching="automatic")
                    query = {"status": "no_matching_outcome", "solverStatus": "infeasible",
                        "scope": scope, "baseModelHash": model_hash,
                        "targetTeam": a[1].name.lower(), "threshold": kw["threshold"],
                        **{kind + "PolicyBindings": [] for kind in ("auction", "escalation", "project", "play")}}
                    query["sha256"] = canonical_sha256(query)
                    core = {"status": "no_adverse_round_in_branch",
                        "queries": [query], "rulesHash": rules.sha256, "scope": scope,
                        "sourceModelHash": model_hash, "boundModelHash": model_hash,
                        "targetTeam": a[1].name.lower(),
                        "policyHash": canonical_sha256(a[2]), "threshold": kw["threshold"],
                        "bindings": [], "domains": {}}
                    return {**core, "sha256": canonical_sha256(core)}
                patch.setattr(compiler, "_refine_symbolic_round_policy", absent)
                result = compiler._solve_symbolic_round_policy_bounds([scenario], Team.NORTH_SOUTH, **limits)
                assert "opponentCeilingWorldQueryHash" not in result["scope"]
                assert "opponentCeilingPolicy" not in result["scope"]
                assert result["strategicCoverage"] is False
        assert str(scenario.model.proto) == before


def test_symbolic_policy_branch_order_is_deterministic(root):
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1,
    )

    rules, normal, _, _ = _native_hokum_round_branch(root, "normal")
    _, doubled, _, _ = _native_hokum_round_branch(root, "double")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    expected_first = min((normal, doubled), key=lambda model: (
        len(model.model.proto.constraints), len(model.model.proto.variables),
        model._model_hash(search_branching="automatic")))._model_hash(search_branching="automatic")
    limits = dict(maximum_iterations=1, maximum_queries=4, maximum_observations=64,
        maximum_branches=2, maximum_conflicts=250_000, maximum_seconds=30)
    receipts = []
    for branches in ([normal, doubled], [doubled, normal]):
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            compiler._solve_symbolic_round_policy_bounds(branches, Team.NORTH_SOUTH, **limits)
        assert stopped.value.reason == "maximum_iterations"
        receipt = stopped.value.partial_receipt["evidence"]
        assert len(receipt["scope"]["branches"]) == 2
        assert len(receipt["masters"]) == len(receipt["queries"]) == len(receipt["cuts"]) == 1
        assert receipt["queries"][0]["branchModelHash"] == expected_first
        assert [row["modelHash"] for row in receipt["scope"]["branches"]] == sorted(
            row["modelHash"] for row in receipt["scope"]["branches"])
        assert not receipt["strategicCoverage"]
        receipts.append(receipt)
    assert receipts[0] == receipts[1]
    with pytest.raises(ValueError, match="resource limits"):
        compiler._solve_symbolic_round_policy_bounds([normal, doubled], Team.NORTH_SOUTH,
            **{**limits, "maximum_branches": 1})


@pytest.mark.real_clock
@pytest.mark.parametrize("fixed_world", (False, True))
def test_symbolic_outcome_cache_reuses_only_identical_complete_native_queries(root, monkeypatch, fixed_world):
    from baloot_platform.research.ortools_information_set_compiler import (
        OrToolsActorInformationSetCompilerV1, InformationSetCompilationStopped)
    from baloot_platform.research.scenario_contracts import canonical_sha256

    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    arguments = dict(threshold=1000, maximum_queries=4, maximum_observations=64,
        maximum_conflicts=250_000, maximum_seconds=30)
    if fixed_world:
        # The fixture already fixes every deck variable to this exact deal.
        scenario._complete_information_deck = tuple(range(32))
    query = OrToolsScenarioModelV1.find_native_round_margin_below
    calls, cache, legality_cache = [], {}, {}
    model_hash = OrToolsScenarioModelV1._model_hash
    serialized_models = []

    def counted_hash(owned, **kwargs):
        if kwargs.get("proto_hash") is None:
            serialized_models.append(owned)
        return model_hash(owned, **kwargs)

    monkeypatch.setattr(OrToolsScenarioModelV1, "_model_hash", counted_hash)

    def counted(owned, **kwargs):
        assert kwargs["optimize_margin"] is False
        calls.append(owned._model_hash(search_branching="automatic"))
        return query(owned, **kwargs)

    monkeypatch.setattr(OrToolsScenarioModelV1, "find_native_round_margin_below", counted)
    first = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {},
        **arguments, _outcome_cache=cache)
    # One source serialization plus the counted query's explicit instrumentation.
    # An untouched clone has the same hash; do not serialize it for cache/evidence.
    assert len(serialized_models) == 2
    assert serialized_models[0] is scenario
    assert first["boundModelHash"] == first["sourceModelHash"] == model_hash(
        scenario, search_branching="automatic")
    second = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {},
        **arguments, _outcome_cache=cache)
    assert first == second and len(calls) == len(cache) == 1
    hint = first["queries"][0]
    observation = next(row for row in first["candidate"]["auctionObservations"]
        if Seat.parse(row["actor"]).team == Team.NORTH_SOUTH)
    policy = {observation["observationHash"]: "sun"}
    serialized_before = len(serialized_models)
    absent = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, policy,
        **arguments, observation_hint=hint, _outcome_cache=cache, _legality_cache=legality_cache)
    # The binding already hashed its owner. Only source and query instrumentation
    # serialize here; cache lookup/evidence reuse that same unmodified digest.
    assert len(serialized_models) - serialized_before == 2
    assert absent["status"] == "no_adverse_round_in_branch" and len(calls) == 2
    assert absent["bindings"] and absent["boundModelHash"] != absent["sourceModelHash"]
    assert absent["boundModelHash"] == absent["queries"][-1]["baseModelHash"]
    serialized_before = len(serialized_models)
    assert compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, policy,
        **arguments, observation_hint=hint, _outcome_cache=cache, _legality_cache=legality_cache) == absent
    assert len(legality_cache) == 1
    assert not hasattr(scenario, "_native_policy_legality_cache")
    assert len(serialized_models) - serialized_before == 1
    assert len(calls) == len(cache) == 2
    bind = OrToolsScenarioModelV1.bind_native_auction_observed_choice
    def corrupt_digest(owned, **kwargs):
        receipt = bind(owned, **kwargs)
        owned._last_policy_binding_proto_hash = "0" * 64
        return receipt
    with monkeypatch.context() as corruption:
        corruption.setattr(OrToolsScenarioModelV1, "bind_native_auction_observed_choice", corrupt_digest)
        with pytest.raises(ValueError, match="digest ownership"):
            compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, policy,
                **arguments, observation_hint=hint, _outcome_cache=cache)
    compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {},
        **{**arguments, "threshold": 999}, _outcome_cache=cache)
    compiler._refine_symbolic_round_policy(scenario, Team.WEST_EAST, {},
        **arguments, _outcome_cache=cache)
    assert len(calls) == len(cache) == 4
    for overrides, reason in (({"cancel_check": lambda: True}, "cancelled"),
                              ({"maximum_seconds": 1e-12}, "maximum_seconds")):
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {},
                **{**arguments, **overrides}, _outcome_cache=cache)
        assert stopped.value.reason == reason
    key = next(iter(cache))
    feasible = {**cache[key], "solverStatus": "feasible"}
    feasible["sha256"] = canonical_sha256({k: v for k, v in feasible.items() if k != "sha256"})
    count_before = len(calls)
    reused = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {},
        **arguments, _outcome_cache={key: feasible})
    assert len(calls) == count_before
    assert reused["candidate"]["utility"] == feasible["candidate"]["utility"] < arguments["threshold"]
    assert reused["status"] == "policy_extension_required"
    for change in ({"sha256": "0" * 64}, {"status": "incomplete"},
                   {"baseModelHash": "0" * 64}, {"queryMode": "wrong"}):
        altered = {**cache[key], **change}
        if "sha256" not in change:
            altered["sha256"] = canonical_sha256({k: v for k, v in altered.items() if k != "sha256"})
        with pytest.raises(ValueError, match="cache identity"):
            compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {},
                **arguments, _outcome_cache={key: altered})
    incomplete = {**hint, "status": "incomplete", "solverStatus": "unknown",
                  "exhaustionReason": "solver_unknown"}
    incomplete["sha256"] = canonical_sha256({k: v for k, v in incomplete.items() if k != "sha256"})
    monkeypatch.setattr(OrToolsScenarioModelV1, "find_native_round_margin_below", lambda *a, **kw: incomplete)
    empty = {}
    with pytest.raises(InformationSetCompilationStopped):
        compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {},
            **arguments, _outcome_cache=empty)
    assert not empty
    assert not scenario._observed_auction_bindings


@pytest.mark.real_clock
def test_symbolic_auction_refinement_retains_policy_and_continuation_boundaries(root, monkeypatch):
    from baloot_platform.research.scenario_contracts import canonical_sha256
    from baloot_platform.research.ortools_information_set_compiler import (
        OrToolsActorInformationSetCompilerV1, InformationSetCompilationStopped)

    rules, scenario, _, _ = _native_hokum_round_branch(root, "normal")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    before = str(scenario.model.proto)
    arguments = dict(threshold=1000, maximum_queries=4, maximum_observations=32,
        maximum_conflicts=250_000, maximum_seconds=30)
    unknown = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {}, **arguments)
    assert unknown["status"] == "policy_extension_required"
    observations = [row for row in unknown["candidate"]["auctionObservations"]
                    if Seat.parse(row["actor"]).team == Team.NORTH_SOUTH]
    assert len(observations) == 2
    assert all(len(row["visibleCards"]) == 5 and row["selectedAction"] in row["legalActions"]
               for row in observations)
    policy = {row["observationHash"]: row["selectedAction"] for row in observations}
    policy.update({row["observationHash"]: row["selectedAction"] for row in unknown["candidate"]["projectObservations"]
                   if Seat.parse(row["actor"]).team == Team.NORTH_SOUTH})
    assert compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, policy, **arguments)["status"] == "policy_extension_required"
    policy.update({row["observationHash"]: row["selectedAction"] for row in unknown["candidate"]["playObservations"]
                   if Seat.parse(row["actor"]).team == Team.NORTH_SOUTH})
    compliant = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, policy, **arguments)
    assert compliant["status"] == "policy_counterexample_in_branch"
    assert compliant["strategicCoverage"] is False
    assert compliant["bindings"] == []
    alternative = {**policy, observations[0]["observationHash"]: "sun"}
    absent = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, alternative, **arguments)
    # The test pins a Hokum trace. Sun excludes this branch; it does not prove
    # that Sun is globally winning, or that the other stages played optimally.
    assert absent["status"] == "no_adverse_round_in_branch"
    assert absent["strategicCoverage"] is False
    assert len(absent["queries"]) == 2 and len(absent["bindings"]) == 1
    assert absent["bindings"][0]["status"] == "bound"
    # A different escalation branch may share an auction observation, not its
    # outcome witness. Bind that choice with a fresh native legality query.
    _, foreign, _, _ = _native_hokum_round_branch(root, "double")
    foreign_before = str(foreign.model.proto)
    contexts = {observations[0]["observationHash"]: observations[0]}
    prebound = compiler._refine_symbolic_round_policy(foreign, Team.NORTH_SOUTH, alternative,
        **arguments, _auction_contexts=contexts)
    assert prebound["status"] == "no_adverse_round_in_branch"
    assert not prebound["strategicCoverage"]
    assert prebound["queriesUsed"] == 2
    assert len(prebound["queries"]) == len(prebound["bindings"]) == 1
    assert prebound["bindings"][0]["status"] == "bound"
    assert prebound["bindings"][0]["solverStatus"] == "INFEASIBLE"
    assert prebound["queries"][0]["baseModelHash"] == prebound["boundModelHash"]
    assert prebound["boundModelHash"] != prebound["sourceModelHash"]
    own_hint = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, alternative,
        **arguments, _auction_contexts=contexts, observation_hint=unknown["queries"][0])
    assert own_hint["queriesUsed"] == 2 and len(own_hint["bindings"]) == 1
    with pytest.raises(ValueError, match="hint ownership or checksum"):
        compiler._refine_symbolic_round_policy(foreign, Team.NORTH_SOUTH, alternative,
            **arguments, observation_hint=unknown["queries"][0])
    with pytest.raises(ValueError, match="context ownership"):
        compiler._refine_symbolic_round_policy(foreign, Team.NORTH_SOUTH, alternative,
            **arguments, _auction_contexts={next(iter(contexts)): {**observations[0], "actor": "west"}})
    with pytest.raises(InformationSetCompilationStopped) as prebinding_budget:
        compiler._refine_symbolic_round_policy(foreign, Team.NORTH_SOUTH, alternative,
            **{**arguments, "maximum_queries": 1}, _auction_contexts=contexts)
    partial = prebinding_budget.value.partial_receipt["evidence"]
    assert prebinding_budget.value.reason == "maximum_queries"
    assert partial["queriesUsed"] == 1 and not partial["queries"]
    assert len(partial["bindings"]) == 1 and not partial["strategicCoverage"]
    assert str(foreign.model.proto) == foreign_before
    assert not foreign._observed_auction_bindings
    # The first changed bid excludes this pinned branch; do not spend another
    # binding query on a later observation that is now unreachable.
    batched = dict(alternative)
    later = observations[1]
    batched[later["observationHash"]] = next(
        action for action in later["legalActions"] if action != later["selectedAction"])
    result = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, batched, **arguments)
    assert result["status"] == "no_adverse_round_in_branch"
    assert len(result["queries"]) == 2 and len(result["bindings"]) == 1
    assert all(row["status"] == "bound" for row in result["bindings"])
    assert result["strategicCoverage"] is False
    hint = unknown["queries"][0]
    hinted = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, batched,
        **arguments, observation_hint=hint)
    assert hinted["status"] == result["status"]
    assert len(hinted["queries"]) == 1 and len(hinted["bindings"]) == 1
    assert hinted["observationHintHash"] == hint["sha256"]
    chained = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, batched,
        **arguments, observation_hints=[hint])
    assert chained["status"] == result["status"]
    assert chained["observationHintHashes"] == [hint["sha256"]]
    assert len(chained["queries"]) == 1 and len(chained["bindings"]) == 1
    # A matching hint is not itself a counterexample, even for an unchanged policy.
    fresh = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, policy,
        **arguments, observation_hint=hint)
    assert fresh["status"] == "policy_counterexample_in_branch"
    assert len(fresh["queries"]) == 1 and not fresh["bindings"]
    fresh_chain = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, policy,
        **arguments, observation_hints=[hint])
    assert fresh_chain["status"] == "policy_counterexample_in_branch"
    assert len(fresh_chain["queries"]) == 1 and not fresh_chain["bindings"]
    forged = copy.deepcopy(hint)
    forged["candidate"]["utility"] = arguments["threshold"] + 1
    forged["sha256"] = canonical_sha256({k: v for k, v in forged.items() if k != "sha256"})
    checked = compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, policy,
        **arguments, observation_hints=[forged])
    assert checked["status"] == "policy_counterexample_in_branch"
    assert checked["candidate"]["utility"] < arguments["threshold"]
    assert len(checked["queries"]) == 1  # A rehashed hint cannot replace a fresh solve.
    for change in ({"sha256": "0" * 64}, {"baseModelHash": "0" * 64},
                   {"targetTeam": "west_east"}, {"status": "no_matching_outcome"}):
        with pytest.raises(ValueError, match="hint ownership or checksum"):
            compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, policy,
                **arguments, observation_hint={**hint, **change})
        with pytest.raises(ValueError, match="hint chain ownership or checksum"):
            compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, policy,
                **arguments, observation_hints=[{**hint, **change}])
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, batched,
            **{**arguments, "maximum_queries": 2})
    assert stopped.value.reason == "maximum_queries"
    assert len(stopped.value.partial_receipt["evidence"]["bindings"]) == 1
    bound = []
    bind = OrToolsScenarioModelV1.bind_native_auction_observed_choice

    def cancel_after_binding(owned, **kwargs):
        receipt = bind(owned, **kwargs)
        bound.append(receipt)
        return receipt

    with monkeypatch.context() as patch:
        patch.setattr(OrToolsScenarioModelV1, "bind_native_auction_observed_choice", cancel_after_binding)
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, batched,
                **arguments, cancel_check=lambda: bool(bound))
    assert stopped.value.reason == "cancelled"
    assert len(stopped.value.partial_receipt["evidence"]["bindings"]) == 1
    for overrides, reason in (({"maximum_queries": 1}, "maximum_queries"),
                              ({"cancel_check": lambda: True}, "cancelled"),
                              ({"maximum_seconds": 1e-12}, "maximum_seconds")):
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, alternative,
                **{**arguments, **overrides})
        assert stopped.value.reason == reason
        assert stopped.value.partial_receipt["evidence"]["strategicCoverage"] is False
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, {},
            **{**arguments, "maximum_observations": 1})
    assert stopped.value.reason == "maximum_observations"
    with pytest.raises(ValueError, match="illegal visible action"):
        compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH,
            {observations[0]["observationHash"]: "invalid"}, **arguments)
    for overrides in ({"maximum_queries": 0}, {"maximum_seconds": float("nan")},
                      {"maximum_conflicts": True}, {"maximum_observations": 1}):
        with pytest.raises(ValueError):
            compiler._refine_symbolic_round_policy(scenario, Team.NORTH_SOUTH, policy,
                **{**arguments, **overrides})
    assert str(scenario.model.proto) == before
    assert scenario._observed_auction_bindings == {}


@pytest.mark.parametrize("change", [
    {"trump": None}, {"locked": 1}, {"match_score": (True, 0)},
    {"match_score": (-1, 0)}, {"match_score": (152, 0)}, {"first_leader": "east"},
    *({'decision_prefix_length': value} for value in (-1, 33, True, 1.5, '24')),
])
def test_native_round_branch_rejects_invalid_scope_before_mutation(rules, change):
    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    scenario.add_native_auction_fragment(prefix_length=0)
    before = str(scenario.model.proto)
    arguments = dict(contract="hokum", trump="diamonds", first_leader="north",
        locked=False, match_score=(0, 0))
    with pytest.raises((TypeError, ValueError)):
        scenario.add_native_round_continuation(**{**arguments, **change})
    assert str(scenario.model.proto) == before


@pytest.mark.parametrize("cancel_at", (1, 10))
def test_native_round_construction_cancels_before_play_model(rules, monkeypatch, cancel_at):
    from baloot_platform.research import ortools_project_fragment, ortools_play_tail

    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    scenario.add_native_auction_fragment(prefix_length=0)
    before = str(scenario.model.proto)
    checks = 0

    def stopped():
        nonlocal checks
        checks += 1
        return checks >= cancel_at

    def unexpected_play(*args, **kwargs):
        pytest.fail("cancelled project construction reached the play model")

    original = ortools_project_fragment.attach_project_lifecycle_fragment_v1
    projects = []

    def attach(*args, **kwargs):
        projects.append(kwargs)
        assert kwargs["cancel_check"] is stopped
        return original(*args, **kwargs)

    monkeypatch.setattr(ortools_project_fragment, "attach_project_lifecycle_fragment_v1", attach)
    monkeypatch.setattr(ortools_play_tail.OrToolsPlayTailStageV1, "build_model", unexpected_play)
    with pytest.raises(ortools_play_tail.PlayTailSolveStopped) as exc:
        scenario.add_native_round_continuation(contract="sun", trump=None,
            first_leader="north", locked=False, match_score=(0, 0), cancel_check=stopped)
    assert exc.value.solver_status == "cancelled"
    assert scenario._native_round_fragment is None
    assert checks == cancel_at
    assert len(projects) == (0 if cancel_at == 1 else 1)
    if cancel_at == 1:
        assert str(scenario.model.proto) == before


def test_native_model_digest_reuse_preserves_identity_without_reserializing(rules, monkeypatch):
    import builtins
    import hashlib
    from baloot_platform.research import ortools_scenario_model as module
    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    proto_type = type(scenario.model.proto)
    calls = []
    def counted(value):
        if isinstance(value, proto_type):
            calls.append(value)
        return builtins.str(value)
    monkeypatch.setattr(module, "str", counted, raising=False)
    for changed in (False, True):
        if changed:
            scenario.model.add(scenario.card_at_position[0] == 0)
        digest = hashlib.sha256(builtins.str(scenario.model.proto).encode("utf-8")).hexdigest()
        before = len(calls)
        cached = scenario._model_hash(search_branching="automatic", proto_hash=digest)
        assert len(calls) == before
        assert scenario._model_hash(search_branching="automatic") == cached
        assert len(calls) == before + 1
        if not changed:
            initial = cached
        else:
            assert cached != initial


def test_native_depth_query_counts_every_stage_without_truncating(root, rules, monkeypatch):
    import builtins
    import json
    import yaml
    from baloot_platform.research import ortools_scenario_model as module
    from baloot_platform.research.ortools_play_tail import TailPrefixActionV1
    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.parse(raw["dealer"]))
    auction = scenario.add_native_auction_fragment(prefix_length=0, allow_all_passed=True)
    scenario.model.add(auction.choices[0]["sun"] == 1)
    for index, card in enumerate(raw["deckOrder"]):
        scenario.model.add(scenario.card_at_position[index] == card)
    branch = scenario.add_native_round_continuation(contract="sun", trump=None,
        first_leader="north", locked=False, match_score=(0, 0),
        prefix=tuple(TailPrefixActionV1.build(row["actor"], row["card"]) for row in raw["playActions"]))
    scenario.model.add(branch.escalation.native["level"] == 0)
    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    budget = profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    slot_bound = (len(auction.choices) + len(branch.escalation_opportunities)
        + len(branch.projects.declaration_actions) + len(branch.play._extract.actors))
    assert slot_bound <= min(budget["maximumDepth"], budget["maximumHorizonActions"])
    before = str(scenario.model.proto)
    args = dict(maximum_seconds=15, maximum_conflicts=250_000)
    witness = scenario.find_native_continuation_depth_above(maximum_depth=36, **args)
    assert witness["status"] == "over_depth_witness", witness
    assert witness["witnessDepth"] >= 37  # One bid, four project decisions and 32 plays, plus escalation.
    assert "queryScope" not in witness  # A relaxed witness is not a legal continuation.
    bound = scenario.find_native_continuation_depth_above(maximum_depth=budget["maximumDepth"],
        record_model_bytes=True, **args)
    assert bound["status"] == "depth_bound_proven", bound
    assert bound["witnessDepth"] is None
    assert "modelBytes" not in witness
    assert bound["modelBytes"]["owner"] == len(before.encode("utf-8"))
    assert bound["queryScope"] == "decision_slot_domain_relaxation_v1"
    assert bound["modelBytes"]["depthQuery"] < bound["modelBytes"]["owner"]
    boundary = scenario.find_native_continuation_depth_above(maximum_depth=slot_bound,
        record_model_bytes=True, **args)
    assert boundary["status"] == "depth_bound_proven"
    assert boundary["queryScope"] == bound["queryScope"]
    assert boundary["sourceModelHash"] == bound["sourceModelHash"] == witness["sourceModelHash"]
    serializations = []
    def counted(value):
        if value is scenario.model.proto:
            serializations.append(value)
        return builtins.str(value)
    with monkeypatch.context() as patch:
        patch.setattr(module, "str", counted, raising=False)
        reused = scenario.find_native_continuation_depth_above(maximum_depth=budget["maximumDepth"],
            record_model_bytes=True, _source_proto=before.encode("utf-8"), **args)
        assert reused == bound
        assert not serializations
        with pytest.raises(TypeError, match="serialized bytes"):
            scenario.find_native_continuation_depth_above(maximum_depth=64, _source_proto=before, **args)
    assert scenario.find_native_continuation_depth_above(maximum_depth=64,
        cancel_check=lambda: True, **args)["status"] == "incomplete"
    assert str(scenario.model.proto) == before


@pytest.mark.real_clock
@pytest.mark.parametrize("optimize", (True, False))
def test_native_round_query_distinguishes_witness_from_optimization(rules, monkeypatch, optimize):
    from baloot_platform.research import ortools_scenario_model as module
    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    scenario.add_native_auction_fragment(allow_all_passed=True)
    scenario.add_native_redeal_continuation(match_score=(0, 0))
    original = str(scenario.model.proto)
    native_solve = module._solve_cp_sat_with_cancel_v1
    objectives, probing_levels = [], []
    def solve(solver, model, cancelled):
        objectives.append(bool(model.has_objective()))
        probing_levels.append(solver.parameters.cp_model_probing_level)
        return native_solve(solver, model, cancelled)
    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", solve)
    limits = dict(target=Team.NORTH_SOUTH, maximum_seconds=10, maximum_conflicts=250000,
        optimize_margin=optimize)
    witness = scenario.find_native_round_margin_below(threshold=1, **limits)
    assert witness["status"] == "witness_found" and witness["candidate"]["utility"] == 0
    assert witness.get("queryMode") == (None if optimize else "feasibility_witness")
    absent = scenario.find_native_round_margin_below(threshold=0, **limits)
    assert absent["status"] == "no_matching_outcome" and absent["solverStatus"] == "infeasible"
    assert objectives == [optimize, optimize]
    assert probing_levels == [0, 0]
    assert str(scenario.model.proto) == original
    with pytest.raises(TypeError, match="boolean"):
        scenario.find_native_round_margin_below(threshold=1, **{**limits, "optimize_margin": 1})

    policy_limits = {**limits, "policy_team": Team.NORTH_SOUTH,
        "auction_action": "pass", "escalation_action": None}
    policy_witness = scenario.find_native_round_with_constant_policy_below(threshold=1, **policy_limits)
    assert policy_witness["status"] == "queried"
    assert policy_witness["query"]["candidate"]["utility"] == 0
    assert policy_witness["query"].get("queryMode") == (None if optimize else "feasibility_witness")
    policy_absent = scenario.find_native_round_with_constant_policy_below(threshold=0, **policy_limits)
    assert policy_absent["status"] == "queried"
    assert policy_absent["query"]["solverStatus"] == "infeasible"
    assert policy_absent["query"]["status"] == "no_matching_outcome"
    assert not policy_absent["strategicCoverage"]
    assert str(scenario.model.proto) == original
    with pytest.raises(TypeError, match="boolean"):
        scenario.find_native_round_with_constant_policy_below(threshold=1,
            **{**policy_limits, "optimize_margin": 1})

    native_query = OrToolsScenarioModelV1.find_native_round_margin_below
    for stop in ("expired", "cancelled", "incomplete"):
        clock, cancelled = [100.0], [False]
        def query_then_stop(self, **kwargs):
            result = native_query(self, **kwargs)
            clock[0] += policy_limits["maximum_seconds"] + 1
            cancelled[0] = stop == "cancelled"
            if stop == "incomplete":
                result = {**result, "status": "incomplete", "solverStatus": "unknown", "candidate": None}
                result.pop("sha256")
                result["sha256"] = module.canonical_sha256(result)
            return result
        with monkeypatch.context() as patch:
            patch.setattr(module.time, "monotonic", lambda: clock[0])
            patch.setattr(OrToolsScenarioModelV1, "find_native_round_margin_below", query_then_stop)
            returned = scenario.find_native_round_with_constant_policy_below(threshold=1,
                cancel_check=lambda: cancelled[0], **policy_limits)
        assert returned["status"] == ("queried" if stop == "expired" else "incomplete")
        if stop == "cancelled":
            assert returned["exhaustionReason"] == "cancelled"
        assert str(scenario.model.proto) == original


def test_native_round_query_reports_reached_conflict_limit(rules, monkeypatch):
    from baloot_platform.research import ortools_scenario_model as module

    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    scenario.add_native_auction_fragment(allow_all_passed=True)
    scenario.add_native_redeal_continuation(match_score=(0, 0))

    class UnknownSolver:
        def __init__(self, conflicts):
            self.parameters = type("Parameters", (), {})()
            self.num_conflicts = conflicts

        def status_name(self, _status):
            return "UNKNOWN"

    for conflicts, expected in ((7, "maximum_conflicts"), (6, "solver_unknown")):
        with monkeypatch.context() as patch:
            patch.setattr(module.cp_model, "CpSolver", lambda: UnknownSolver(conflicts))
            patch.setattr(module, "_solve_cp_sat_with_cancel_v1", lambda *_: cp_model.UNKNOWN)
            result = scenario.find_native_round_margin_below(
                target=Team.NORTH_SOUTH, threshold=1, maximum_seconds=10,
                maximum_conflicts=7, optimize_margin=False)
        assert result["status"] == "incomplete"
        assert result["exhaustionReason"] == expected


def test_native_round_query_searches_unresolved_sun_ownership(root, rules):
    import json
    from baloot_platform.research.ortools_play_tail import TailPrefixActionV1
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, canonical_sha256

    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    scenario = OrToolsScenarioModelV1(rules=rules, dealer=Seat.parse(raw["dealer"]))
    scenario.add_native_auction_fragment(prefix_length=0, allow_all_passed=True)
    branch = scenario.add_native_round_continuation(contract="sun", trump=None,
        first_leader="north", locked=False, match_score=(0, 0),
        prefix=tuple(TailPrefixActionV1.build(row["actor"], row["card"]) for row in raw["playActions"]))
    # Two unplayed opposing Aces remain symbolically allocated; query extraction
    # must carry the selected deal, not the source fixture's hidden ownership.
    unknown_positions = (25, 27)
    for index, card in enumerate(raw["deckOrder"]):
        if index not in unknown_positions:
            scenario.model.add(scenario.card_at_position[index] == card)
    scenario.model.add(branch.escalation.native["level"] == 0)
    # The inverse-encoded allocation used to report false infeasibility only
    # with presolve enabled. Both paths must accept this same legal universe.
    for presolve in (True, False):
        parity_solver = cp_model.CpSolver()
        parity_solver.parameters.num_search_workers = 1
        parity_solver.parameters.cp_model_presolve = presolve
        parity_solver.parameters.max_time_in_seconds = 15
        assert parity_solver.solve(scenario.model) == cp_model.OPTIMAL
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.parse(raw["dealer"]))
    visible = [raw["deckOrder"][index] for index in layout.positions_for(Seat.NORTH, final=False)]
    binding = scenario.bind_native_auction_observed_choice(actor=Seat.NORTH,
        visible_cards=visible, purchase_card=raw["deckOrder"][scenario.purchase_position],
        action="sun", public_choices=(), maximum_seconds=10, maximum_conflicts=250_000)
    assert binding["status"] == "bound"
    before = str(scenario.model.proto)
    receipts, selectors = [], []
    for swapped in (False, True):
        position = unknown_positions[0]
        card = raw["deckOrder"][unknown_positions[int(swapped)]]
        selector = scenario.model.new_bool_var(f"query_deal_{swapped}")
        selectors.append(selector)
        scenario.model.add(scenario.card_at_position[position] == card).only_enforce_if(selector)
        scenario.model.clear_assumptions()
        scenario.model.add_assumption(selector)
        receipt = scenario.find_native_round_margin_below(target=Team.NORTH_SOUTH,
            threshold=1000, maximum_seconds=15, maximum_conflicts=250_000)
        assert receipt["status"] == "witness_found"
        assert receipt["auctionPolicyBindings"] == [binding]
        assert receipt["candidate"]["auctionActions"][0]["action"] == "sun"
        assert receipt["candidate"]["deckOrder"][position] == card
        candidate = receipt["candidate"]
        replay = dict(raw, deckOrder=candidate["deckOrder"], stateAssertions={},
            playActions=[dict(actor=row["actor"], card=row["card"], ekka=row["ekka"])
                         for row in candidate["play"]["plays"]])
        exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
            ExactTracePrefixV1.build(replay), decision_point={"kind": "play"}, maximum_seconds=30)
        assert exact.state["phase"] == "scored"
        assert candidate["awardedPoints"] == exact.state["roundScore"]["awardedPoints"]
        assert receipt["sha256"] == canonical_sha256({key: value for key, value in receipt.items() if key != "sha256"})
        receipts.append(receipt)
    assert receipts[0]["candidate"]["deckOrder"] != receipts[1]["candidate"]["deckOrder"]
    assert receipts[0]["candidate"]["auctionObservations"] == receipts[1]["candidate"]["auctionObservations"]
    first_project = receipts[0]["candidate"]["projectObservations"][0]
    second_project = receipts[1]["candidate"]["projectObservations"][0]
    assert first_project == second_project
    first_play = receipts[0]["candidate"]["playObservations"][0]
    assert first_play == receipts[1]["candidate"]["playObservations"][0]
    scenario.model.clear_assumptions()
    scenario.model.add(scenario.native_play_observation_literal(information=first_play["information"]) == 1)
    project_binding = scenario.bind_native_project_observed_choice(information=first_project["information"],
        action=first_project["selectedAction"], maximum_seconds=10, maximum_conflicts=250_000)
    assert project_binding["status"] == "bound"
    play_binding = scenario.bind_native_play_observed_choice(information=first_play["information"],
        action=first_play["selectedAction"], maximum_seconds=10, maximum_conflicts=250_000)
    assert play_binding["status"] == "bound"
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    for selector in selectors:
        scenario.model.clear_assumptions()
        scenario.model.add_assumption(selector)
        assert solver.solve(scenario.model) == cp_model.OPTIMAL
    scenario.model.clear_assumptions()
    assert receipts[0]["baseModelHash"] != receipts[1]["baseModelHash"]
    assert before != str(scenario.model.proto)  # Only explicit test assumptions and policy binding changed it.
    branch.scope["locked"] = True
    with pytest.raises(ValueError, match="integrity"):
        scenario.find_native_round_margin_below(target=Team.NORTH_SOUTH,
            threshold=1000, maximum_seconds=1, maximum_conflicts=100)


@pytest.mark.parametrize("indices", [(), tuple(range(5)), tuple(range(32)), (3, 7, 11, 15, 23), (2, 4, 10, 12, 28)])
def test_native_auction_hand_values_match_reference(rules, indices):
    from baloot_platform.core.bidding import _sun_score, _hokum_score

    model = cp_model.CpModel()
    visible = tuple(model.new_bool_var(f"visible_{index}") for index in range(32))
    fragment = attach_auction_hand_values_v1(model, rules, visible_cards=visible)
    # Fix membership after lowering: scores must depend on variables, not Python truth.
    for index, literal in enumerate(visible):
        model.add(literal == int(index in indices))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    cards = tuple(Card(index) for index in indices)
    assert solver.value(fragment.sun) / fragment.scale == pytest.approx(_sun_score(cards, rules))
    for suit in Suit:
        assert solver.value(fragment.hokum[suit]) / fragment.scale == pytest.approx(_hokum_score(cards, suit, rules))
    for key, value in fragment.thresholds.items():
        assert value / fragment.scale == rules.data["bidding"]["heuristics"][key]


def test_native_auction_hand_values_are_configured_and_symbolic(rules):
    changed = copy.deepcopy(rules)
    changed.data["bidding"]["heuristics"]["sun_rank_weights"][int(Rank.ACE)] = 30.125
    model = cp_model.CpModel()
    visible = tuple(model.new_bool_var(f"visible_{index}") for index in range(32))
    original = attach_auction_hand_values_v1(model, rules, visible_cards=visible, name="original")
    fragment = attach_auction_hand_values_v1(model, changed, visible_cards=visible, name="changed")
    model.add(sum(visible) == 1)
    model.maximize(fragment.sun)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    selected = [Card(index) for index, literal in enumerate(visible) if solver.boolean_value(literal)]
    assert selected[0].rank == Rank.ACE
    assert solver.value(fragment.sun) / fragment.scale == 30.125
    assert solver.value(original.sun) / original.scale == 3.2
    assert fragment.profile_hash != original.profile_hash


@pytest.mark.parametrize("value", [True, "3.2", -1, float("nan"), float("inf"), 1e30, 1e-30])
def test_native_auction_hand_values_reject_invalid_coefficients(rules, value):
    changed = copy.deepcopy(rules)
    changed.data["bidding"]["heuristics"]["sun_rank_weights"][0] = value
    model = cp_model.CpModel()
    with pytest.raises(ValueError):
        attach_auction_hand_values_v1(model, changed, visible_cards=[model.new_bool_var(f"v{i}") for i in range(32)])


@pytest.mark.parametrize("round_number", [1, 2])
@pytest.mark.parametrize("codes", [
    ("S7", "H7", "C7", "D7", "S8"),
    ("SJ", "S9", "SA", "S10", "SK"),
    ("HA", "CA", "DA", "H10", "C10"),
])
def test_native_auction_action_margins_use_own_and_public_cards(rules, round_number, codes):
    from baloot_platform.core.bidding import _sun_score, _hokum_score

    cards = tuple(Card.parse(code) for code in codes)
    purchase = Card.parse("DQ")
    model = cp_model.CpModel()
    visible = [model.new_bool_var(f"v{i}") for i in range(32)]
    round_var = model.new_int_var(1, 2, "round")
    hand, margins = attach_auction_action_values_v1(
        model, rules, visible_cards=visible, purchase_card=purchase, round_index=round_var,
    )
    for index, var in enumerate(visible):
        model.add(var == int(Card(index) in cards))
    model.add(round_var == round_number)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    profile = rules.data["bidding"]["heuristics"]
    prefix = "second_" if round_number == 2 else ""
    expected = {"pass": 0, "sun": _sun_score(cards, rules) - profile[f"{prefix}sun_threshold"],
                "ashkal": _sun_score((*cards, purchase), rules) - profile["ashkal_partner_threshold"]}
    expected.update({f"hokum:{suit.name.lower()}": _hokum_score(cards, suit, rules) - profile[f"{prefix}hokum_threshold"]
                     for suit in Suit})
    assert {code: solver.value(value) / hand.scale for code, value in margins.items()} == pytest.approx(expected)


@pytest.mark.parametrize("indices", [tuple(range(4)), tuple(range(6)), (0, 1, 2, 3, 31)])
def test_native_auction_action_values_reject_wrong_hand_or_owned_purchase(rules, indices):
    model = cp_model.CpModel()
    hand, margins = attach_auction_action_values_v1(
        model, rules, visible_cards=[model.new_constant(int(i in indices)) for i in range(32)],
        purchase_card=Card(31), round_index=model.new_constant(1),
    )
    assert hand.scale > 0 and margins
    assert cp_model.CpSolver().solve(model) == cp_model.INFEASIBLE


@pytest.mark.parametrize("codes,expected", [
    (("S7", "H7", "C7", "D7", "S8"), "pass"),
    (("SJ", "S9", "SA", "S10", "SK"), "hokum:spades"),
])
def test_native_auction_action_values_rank_reachable_choices(rules, codes, expected):
    model = cp_model.CpModel()
    cards = tuple(Card.parse(code) for code in codes)
    purchase = Card.parse("SQ")
    states, choices = attach_auction_path_v1(
        model, AuctionDomainIRV1.from_rules(rules.data), dealer=model.new_constant(int(Seat.EAST)),
        purchase_suit=model.new_constant(int(purchase.suit)), purchase_rank=model.new_constant(int(purchase.rank)),
        maximum_actions=8,
    )
    hand, values = attach_auction_action_values_v1(
        model, rules, visible_cards=[model.new_constant(int(Card(i) in cards)) for i in range(32)],
        purchase_card=purchase, round_index=states[0]["round"],
    )
    selected_value = model.new_int_var(-1000 * hand.scale, 1000 * hand.scale, "selected_value")
    for code, choice in choices[0].items():
        if code == "closed":
            model.add(choice == 0)
        else:
            model.add(selected_value == values[code]).only_enforce_if(choice)
    model.maximize(selected_value)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    assert [code for code, choice in choices[0].items() if solver.boolean_value(choice)] == [expected]


def test_native_search_can_construct_an_optimal_sun_auction(rules):
    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    model.add_native_auction_fragment(
        allowed_contracts=(Contract.SUN,), require_optimal_actions=True,
    )
    result = model.solve(maximum_seconds=15, search_branching="automatic")
    assert result.status == "satisfied"
    assert result.auction is not None
    assert result.auction.completion_contract == Contract.SUN
    assert result.auction.completion_actions[-1].action == "sun"


def test_auction_hand_heuristic_can_exclude_a_legal_sun_deal(root):
    compiler, _exact, prefix, _hands = _native_suit_monopoly_bid(root)
    statuses = []
    for require_hand_policy in (False, True):
        model = OrToolsScenarioModelV1(rules=compiler.rules, dealer=Seat.EAST,
            required_buyer=Seat.NORTH)
        model.add_native_auction_fragment(allowed_contracts=(Contract.SUN,),
            require_optimal_actions=require_hand_policy)
        for variable, card in zip(model.card_at_position, prefix.deck_order, strict=True):
            model.model.add(variable == card)
        result = model.solve(maximum_seconds=5, search_branching="automatic")
        statuses.append(result.status)
    # Feasibility is not a strategic proof; the optional hand policy narrows it.
    assert statuses == ["satisfied", "infeasible"]


@pytest.mark.parametrize("purchase", [False, True])
def test_native_action_extraction_handles_all_pass_and_closed_padding(rules, purchase):
    model = cp_model.CpModel()
    domain = AuctionDomainIRV1.from_rules(rules.data)
    states, choices = attach_auction_path_v1(
        model, domain, dealer=model.new_constant(int(Seat.EAST)),
        purchase_suit=model.new_constant(int(Suit.HEARTS)),
        purchase_rank=model.new_constant(int(Rank.KING)), maximum_actions=8,
    )
    if purchase:
        model.add(choices[0]["sun"] == 1)
    else:
        for row in choices:
            model.add(row["pass"] == 1)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    fragment = OrToolsNativeAuctionFragmentV1(states, choices, 0, {}, {}, domain.sha256)
    records = fragment.extract_actions(solver)
    assert len(records) == (1 if purchase else 8)
    assert fragment.extract_actions(solver, length=0) == ()
    assert fragment.extract_actions(solver, length=1) == records[:1]
    assert records[0].actor == Seat.NORTH
    assert solver.value(states[-1]["allPassed"]) == int(not purchase)
    if not purchase:
        assert all(row.action == "pass" and row.buyer is None for row in records)
        assert [row.round_index for row in records] == [1] * 4 + [2] * 4
    for invalid in (True, -1, 9, "1"):
        with pytest.raises(ValueError, match="extraction length"):
            fragment.extract_actions(solver, length=invalid)


@pytest.mark.parametrize("official", [False, True, "one_round"])
@pytest.mark.parametrize("rank", [Rank.ACE, Rank.KING])
@pytest.mark.parametrize("dealer", tuple(Seat))
def test_native_auction_paths_match_complete_reference_universe(root, rules, monkeypatch, official, rank, dealer):
    from baloot_platform.core.bidding_machine import BiddingMachine
    if official is True:
        rules = RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    domain = AuctionDomainIRV1.from_rules(rules.data)
    if official == "one_round":
        domain = replace(domain, rounds=1)
    expected, intermediate = {}, {}
    contracts = {None: -1, Contract.SUN: 0, Contract.HOKUM: 1, Contract.ASHKAL: 2}
    def visit(machine, path):
        candidate = machine._candidate
        intermediate[path] = {"round": machine.round_index, "terminal": int(machine.finished),
            "contract": contracts[candidate.contract if candidate else None],
            "actor": int(machine.actor) if not machine.finished else None}
        if machine.finished:
            result = machine.result
            expected[path] = {"allPassed": int(result.all_passed), "contract": contracts[result.contract],
                "trump": int(result.trump) if result.trump is not None else len(Suit),
                **{key: int(getattr(result, field)) if getattr(result, field) is not None else -1
                   for key, field in {"buyer": "buyer", "caller": "caller", "declarer": "declarer",
                                      "recipient": "purchase_recipient"}.items()}}
            return
        for action in machine.legal_actions():
            trumps = ([Suit.HEARTS] if machine.round_index == 1 else [s for s in Suit if s != Suit.HEARTS]) if action == "hokum" else [None]
            for trump in trumps:
                child = copy.deepcopy(machine)
                child.submit(action, trump=trump)
                visit(child, (*path, f"hokum:{trump.name.lower()}" if trump is not None else action))
    visit(BiddingMachine(dealer, Suit.HEARTS, domain, rank), ())
    def forbidden(*args, **kwargs):
        raise AssertionError("native auction path called a Python semantic evaluator")
    for method in ("legal_actions", "resolve_action", "preselected_bid"):
        monkeypatch.setattr(AuctionDomainIRV1, method, forbidden)
    model = cp_model.CpModel()
    states, actions = attach_auction_path_v1(model, domain, dealer=model.new_constant(int(dealer)),
        purchase_suit=model.new_constant(int(Suit.HEARTS)), purchase_rank=model.new_constant(int(rank)), maximum_actions=8)
    model.add(states[-1]["terminal"] == 1)
    seen = set()
    class Check(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            path = ()
            for state, choices in zip(states, actions):
                for key, value in intermediate[path].items():
                    if value is not None:
                        assert self.value(state[key]) == value, (path, key)
                code, = [code for code, literal in choices.items() if self.value(literal)]
                if code != "closed":
                    path = (*path, code)
            assert path not in seen
            seen.add(path)
            assert {key: self.value(states[-1][key]) for key in expected[path]} == expected[path]
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 30
    assert solver.solve(model, Check()) == cp_model.OPTIMAL
    assert seen == set(expected)


@pytest.mark.parametrize("limit", [True, 0, 9, 1.5])
def test_native_auction_path_rejects_invalid_length(rules, limit):
    model = cp_model.CpModel()
    with pytest.raises(ValueError, match="maximum_actions"):
        attach_auction_path_v1(model, AuctionDomainIRV1.from_rules(rules.data),
            dealer=model.new_constant(0), purchase_suit=model.new_constant(0),
            purchase_rank=model.new_constant(0), maximum_actions=limit)


@pytest.mark.parametrize("contract", tuple(Contract))
@pytest.mark.parametrize("dealer", tuple(Seat))
@pytest.mark.parametrize("partner", [False, True])
@pytest.mark.parametrize("pass_allowed", [False, True])
def test_native_auction_path_preselected_history_and_roles(rules, monkeypatch, contract, dealer, partner, pass_allowed):
    from baloot_platform.core.bidding_machine import BiddingMachine
    model = cp_model.CpModel()
    domain = replace(AuctionDomainIRV1.from_rules(rules.data), require_preselected=True,
                     default_contract=contract, ashkal_contract_owner="taker",
                     ashkal_taker="bidder_partner" if partner else "bidder", ashkal_priority_positions=(1, 3),
                     pass_allowed=pass_allowed)
    if not pass_allowed:
        domain = replace(domain, first_round_allowed=tuple(x for x in domain.first_round_allowed if x != "pass"),
                         second_round_allowed=tuple(x for x in domain.second_round_allowed if x != "pass"))
    reference = BiddingMachine(dealer, Suit.HEARTS, domain, Rank.ACE)
    def forbidden(*args, **kwargs):
        raise AssertionError("preselection called the Python interpreter")
    for method in ("legal_actions", "resolve_action", "preselected_bid"):
        monkeypatch.setattr(AuctionDomainIRV1, method, forbidden)
    states, choices = attach_auction_path_v1(model, domain, dealer=model.new_constant(int(dealer)),
        purchase_suit=model.new_constant(int(Suit.HEARTS)), purchase_rank=model.new_constant(int(Rank.ACE)), maximum_actions=8)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(model) == cp_model.OPTIMAL
    actual = [(solver.value(state["actor"]), code) for state, row in zip(states, choices)
              for code, literal in row.items() if code != "closed" and solver.value(literal)]
    assert actual == [(int(record.seat), f"hokum:{record.trump.name.lower()}" if record.action == "hokum" else record.action)
                      for record in reference.records]
    assert solver.value(states[-1]["terminal"]) == 1
    for key, field in {"buyer": "buyer", "caller": "caller", "recipient": "purchase_recipient", "declarer": "declarer"}.items():
        assert solver.value(states[-1][key]) == int(getattr(reference.result, field))
    assert solver.value(states[-1]["trump"]) == (int(Suit.HEARTS) if contract == Contract.HOKUM else len(Suit))


@pytest.mark.parametrize("invalid", ["missing_default", "ineligible", "incompatible_owner"])
def test_native_preselection_refuses_invalid_configuration(rules, invalid):
    domain = replace(AuctionDomainIRV1.from_rules(rules.data), require_preselected=True,
                     default_contract=Contract.ASHKAL, ashkal_taker="bidder_partner", ashkal_contract_owner="taker")
    if invalid == "missing_default":
        domain = replace(domain, default_contract=None)
    elif invalid == "ineligible":
        domain = replace(domain, ashkal_priority_positions=(1,))
    else:
        domain = replace(domain, ashkal_contract_owner="bidder")
    model = cp_model.CpModel()
    def compile():
        return attach_auction_path_v1(model, domain, dealer=model.new_constant(0),
            purchase_suit=model.new_constant(0), purchase_rank=model.new_constant(0), maximum_actions=8)
    if invalid == "incompatible_owner":
        compile()
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = 5
        assert solver.solve(model) == cp_model.INFEASIBLE
    else:
        with pytest.raises(ValueError, match="preselected"):
            compile()


@pytest.mark.parametrize("profile", ["legacy", "official", "restricted"])
def test_native_auction_legality_exhaustive_symbolic_parity(root, rules, monkeypatch, profile):
    if profile == "official":
        rules = RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    domain = AuctionDomainIRV1.from_rules(rules.data)
    if profile == "restricted":
        domain = replace(domain, rounds=1, pass_allowed=False, ashkal_enabled=False,
                         first_round_allowed=("sun", "hokum"), ashkal_priority_positions=())
    reference = domain.legal_actions
    def forbidden(*args, **kwargs):
        raise AssertionError("native legality called the reference interpreter")
    monkeypatch.setattr(AuctionDomainIRV1, "legal_actions", forbidden)
    model = cp_model.CpModel()
    ranges = {"round_index": (1, domain.rounds), "position": (1, 4), "in_recall": (0, 1),
              "terminal": (0, 1), "candidate_contract": (-1, 2), "purchase_rank": (0, 7),
              "purchase_suit": (0, 3), "prior_nonpass_count": (0, 1)}
    inputs = {key: model.new_int_var(low, high, key) for key, (low, high) in ranges.items()}
    actions, choices = attach_auction_legal_actions_v1(model, domain, **inputs)
    contracts = {-1: None, 0: Contract.SUN, 1: Contract.HOKUM, 2: Contract.ASHKAL}
    class Check(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.count = 0
        def on_solution_callback(self):
            values = {key: self.value(variable) for key, variable in inputs.items()}
            expected = set() if values["terminal"] else set(reference(
                round_index=values["round_index"], position=values["position"],
                in_recall=bool(values["in_recall"]), candidate_contract=contracts[values["candidate_contract"]],
                purchase_rank=Rank(values["purchase_rank"]),
                prior_round_actions=("sun",) if values["prior_nonpass_count"] else ("pass",)))
            actual = {key for key, variable in actions.items() if self.value(variable)}
            assert actual == expected, (values, actual, expected)
            expected_choices = expected - {"hokum"}
            if "hokum" in expected:
                expected_choices |= {f"hokum:{suit.name.lower()}" for suit in Suit
                    if (int(suit) == values["purchase_suit"]) == (values["round_index"] == 1)}
            assert {key for key, variable in choices.items() if self.value(variable)} == expected_choices
            self.count += 1
    checker = Check()
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.max_time_in_seconds = 30
    assert solver.solve(model, checker) == cp_model.OPTIMAL
    assert checker.count == domain.rounds * 4 * 2 * 2 * 4 * 8 * 4 * 2


def _leaf(predicate_id: str) -> dict:
    return {
        "predicateId": predicate_id,
        "kind": "test.literal",
        "version": 1,
        "hardness": "hard",
        "arguments": {},
    }


@pytest.mark.parametrize("buyer", tuple(Seat))
def test_integrated_deal_model_matches_every_buyer_layout(rules, buyer: Seat):
    first = OrToolsScenarioModelV1(
        rules=rules,
        dealer=Seat.EAST,
        required_buyer=buyer,
    ).solve(seed=17, maximum_seconds=5)
    repeated = OrToolsScenarioModelV1(
        rules=rules,
        dealer=Seat.EAST,
        required_buyer=buyer,
    ).solve(seed=17, maximum_seconds=5)

    assert first == repeated
    assert first.status == "satisfied"
    assert first.solver_status == "optimal"
    assert first.buyer == buyer
    assert first.deck_order is not None
    assert first.position_of_card is not None
    assert first.owner_at_position is not None
    assert first.owner_of_card is not None
    assert sorted(first.deck_order) == list(range(32))
    assert all(
        first.position_of_card[card] == position
        for position, card in enumerate(first.deck_order)
    )

    layout = DealLayoutV1.build(
        rules.structural_domain,
        dealer=Seat.EAST,
        buyer=buyer,
    )
    expected_owner_at_position = [None] * 32
    for seat in Seat:
        for position in layout.positions_for(seat, final=True):
            expected_owner_at_position[position] = seat
    assert list(first.owner_at_position) == expected_owner_at_position
    assert all(
        first.owner_of_card[card] == first.owner_at_position[position]
        for position, card in enumerate(first.deck_order)
    )
    assert [first.owner_of_card.count(seat) for seat in Seat] == [8, 8, 8, 8]
    assert first.purchase_position == layout.purchase_position
    assert first.purchase_card == Card(first.deck_order[layout.purchase_position])
    assert first.purchase_suit == first.purchase_card.suit
    assert first.owner_of_card[first.purchase_card.id] == buyer
    assert first.model_hash and len(first.model_hash) == 64
    assert first.solution_hash and len(first.solution_hash) == 64


def test_full_deck_no_good_excludes_only_the_exact_assignment(rules):
    first = OrToolsScenarioModelV1(
        rules=rules,
        dealer=Seat.NORTH,
        required_buyer=Seat.SOUTH,
    ).solve(maximum_seconds=5)
    assert first.deck_order is not None

    second = OrToolsScenarioModelV1(
        rules=rules,
        dealer=Seat.NORTH,
        required_buyer=Seat.SOUTH,
        excluded_deck_orders=[first.deck_order, first.deck_order],
    ).solve(maximum_seconds=5)

    assert second.status == "satisfied"
    assert second.deck_order is not None
    assert second.deck_order != first.deck_order
    assert len(second.no_good_hashes) == 1
    assert second.model_hash != first.model_hash


def test_canonical_deal_packet_no_good_changes_semantic_deal(rules):
    options = {
        "rules": rules,
        "dealer": Seat.NORTH,
        "required_buyer": Seat.SOUTH,
        "canonicalize_deal_packets": True,
    }
    first = OrToolsScenarioModelV1(**options).solve(maximum_seconds=5)
    assert first.deck_order is not None
    assert first.owner_of_card is not None

    layout = DealLayoutV1.build(
        rules.structural_domain,
        dealer=Seat.NORTH,
        buyer=Seat.SOUTH,
    )
    for seat in Seat:
        initial = layout.positions_for(seat, final=False)
        additions = tuple(
            position
            for position in layout.positions_for(seat, final=True)
            if position not in initial and position != layout.purchase_position
        )
        assert [first.deck_order[position] for position in initial] == sorted(
            first.deck_order[position] for position in initial
        )
        assert [first.deck_order[position] for position in additions] == sorted(
            first.deck_order[position] for position in additions
        )

    second = OrToolsScenarioModelV1(
        **options,
        excluded_deck_orders=[first.deck_order],
    ).solve(maximum_seconds=5)
    assert second.deck_order is not None
    assert second.owner_of_card is not None
    assert (second.owner_of_card, second.purchase_card) != (
        first.owner_of_card,
        first.purchase_card,
    )


@pytest.mark.parametrize("dealer", list(Seat))
@pytest.mark.parametrize("buyer", list(Seat))
def test_symbolic_buyer_canonicalization_preserves_every_purchase_branch(rules, dealer, buyer):
    original = OrToolsScenarioModelV1(rules=rules, dealer=dealer, required_buyer=buyer).solve(maximum_seconds=5)
    assert original.status == "satisfied"
    layout = DealLayoutV1.build(rules.structural_domain, dealer=dealer)
    groups = (*layout.initial_positions, *layout.purchase_invariant_undealt_groups(rules.structural_domain))
    canonical = list(original.deck_order)
    for group in groups:
        for position, card in zip(group, sorted(canonical[index] for index in group), strict=True):
            canonical[position] = card
    model = OrToolsScenarioModelV1(rules=rules, dealer=dealer, canonicalize_deal_packets=True)
    # The buyer remains a solver variable; each possible branch stays feasible.
    model.model.add(model.buyer == int(buyer))
    for position, card in enumerate(canonical):
        model.model.add(model.card_at_position[position] == card)
    result = model.solve(maximum_seconds=5)
    assert result.status == "satisfied"
    assert result.owner_of_card == original.owner_of_card
    assert result.purchase_card == original.purchase_card
    assert result.deck_order == tuple(canonical)


@pytest.mark.parametrize("dealer", list(Seat))
@pytest.mark.parametrize("contract,caller_offset", [("sun", i) for i in range(4)] + [("ashkal", 2), ("ashkal", 3)])
def test_native_bid_completion_packets_preserve_world_and_purchase_roles(root, dealer, contract, caller_offset):
    import random
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    layout = DealLayoutV1.build(rules.structural_domain, dealer=dealer)
    deck = list(range(32))
    random.Random(42 + int(dealer)).shuffle(deck)
    purchase_index = deck.index(Card.parse("H7").id)
    deck[purchase_index], deck[layout.purchase_position] = deck[layout.purchase_position], deck[purchase_index]
    prefix = ExactTracePrefixV1.build(dict(schemaVersion=1, dealer=dealer.name.lower(), deckOrder=deck,
        auctionActions=[], escalationActions=[], projectDeclarations=[], playActions=[]))
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=10)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)

    def fix_auction(model):
        for index, choices in enumerate(model.auction_fragment.choices):
            if index <= caller_offset:
                model.model.add(choices[contract if index == caller_offset else "pass"] == 1)
            else:
                model.model.add_bool_or([choices["closed"], choices["pass"]])

    original = compiler._native_bid_root_model(exact, prefix, visibility="complete_state")
    fix_auction(original)
    witness = original.solve(maximum_seconds=5)
    assert witness.status == "satisfied"
    recipient = witness.purchase_recipient
    final_layout = DealLayoutV1.build(rules.structural_domain, dealer=dealer, buyer=recipient)
    caller = rules.structural_domain.recipient_order(dealer)[caller_offset]
    assert witness.buyer == caller
    assert recipient == (rules.structural_domain.partner(caller) if contract == "ashkal" else caller)
    canonical = list(deck)
    for group in (*layout.initial_positions, *(packet.positions for packet in final_layout.final_packets)):
        for position, card in zip(group, sorted(canonical[index] for index in group), strict=True):
            canonical[position] = card
    assert canonical[layout.purchase_position] == deck[layout.purchase_position]

    observed = compiler._native_bid_root_model(exact, prefix, visibility="actor_information_set")
    fix_auction(observed)
    fixed = observed.model.clone()
    for variable, card in zip(observed.card_at_position, canonical, strict=True):
        fixed.add(variable == card)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(fixed) == cp_model.OPTIMAL
    assert tuple(solver.value(owner) for owner in observed.owner_of_card) == tuple(map(int, witness.owner_of_card))
    assert solver.value(observed.buyer) == int(caller)
    assert solver.value(observed.purchase_recipient) == int(recipient)

    # The old recipient-invariant ordering admits this cosmetic permutation.
    # The new guarded ordering must remove it, not remove its canonical world.
    invariant = layout.purchase_invariant_undealt_groups(rules.structural_domain)
    changed = None
    for packet in final_layout.final_packets:
        for left, right in zip(packet.positions, packet.positions[1:]):
            proposal = canonical.copy()
            proposal[left], proposal[right] = proposal[right], proposal[left]
            if all([proposal[index] for index in group] == sorted(proposal[index] for index in group) for group in invariant):
                changed = proposal
                break
        if changed is not None:
            break
    assert changed is not None
    duplicate = observed.model.clone()
    for variable, card in zip(observed.card_at_position, changed, strict=True):
        duplicate.add(variable == card)
    assert solver.solve(duplicate) == cp_model.INFEASIBLE
    for visibility, fixed_world in (("complete_state", False), ("actor_information_set", True)):
        manual = compiler._native_bid_root_model(exact, prefix, visibility=visibility, fixed_observed_world=fixed_world)
        fix_auction(manual)
        assert manual.solve(maximum_seconds=5).deck_order == tuple(deck)


def test_symbolic_auction_completion_keeps_only_actor_cards_fixed(rules, monkeypatch):
    import baloot_platform.research.ortools_scenario_model as module

    def forbidden(*args, **kwargs):
        raise AssertionError("symbolic auction used the interpreted auction enumerator")
    monkeypatch.setattr(module, "_enumerate_auction", forbidden)
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST)
    visible = tuple(range(5))
    purchase = Card.parse("SA")
    decks = []
    observations = []
    for _ in range(2):
        model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST,
            canonicalize_deal_packets=True, excluded_deck_orders=decks)
        for position in layout.positions_for(Seat.NORTH, final=False):
            model.model.add_allowed_assignments([model.card_at_position[position]], [(card,) for card in visible])
        model.add_native_auction_fragment(required_purchase_card=purchase, prefix_length=0)
        result = model.solve(maximum_seconds=5)
        assert result.status == "satisfied"
        assert set(result.deck_order[position] for position in layout.positions_for(Seat.NORTH, final=False)) == set(visible)
        assert result.purchase_card == purchase
        assert result.auction.prefix_terminal is False
        assert result.owner_of_card[purchase.id] == result.auction.completion_purchase_recipient
        decks.append(result.deck_order)
        observations.append({key: value for key, value in result.auction.to_dict().items()
            if key != "completion" and not key.startswith("completion")})
    assert decks[0] != decks[1]
    assert observations[0] == observations[1]


@pytest.mark.parametrize("above_target", [0, 20])
def test_native_redeal_accepts_unfinished_tied_match(rules, above_target):
    from baloot_platform.core.scoring_ir import ScoringDomainIRV1

    score = ScoringDomainIRV1.from_rules(rules.data).match_target + above_target
    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    model.add_native_auction_fragment(prefix_length=0, allow_all_passed=True)
    for index, variable in enumerate(model.card_at_position):
        model.model.add(variable == index)
    scope = model.add_native_redeal_continuation(match_score=(score, score))
    assert scope["allPassedCovered"]
    result = model.find_native_round_margin_below(threshold=1, target=Team.NORTH_SOUTH,
        maximum_seconds=5, maximum_conflicts=100_000)
    assert result["status"] == "witness_found"
    assert result["candidate"]["awardedPoints"] == [0, 0]
    assert result["candidate"]["utility"] == 0


@pytest.mark.parametrize("allow_all_passed", (False, True))
@pytest.mark.parametrize("shorten_counterexamples", (False, True))
def test_native_redeal_query_preserves_current_deal_boundary(root, rules, allow_all_passed, shorten_counterexamples, monkeypatch):
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1

    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    model.add_native_auction_fragment(prefix_length=0, allow_all_passed=allow_all_passed)
    for index, variable in enumerate(model.card_at_position):
        model.model.add(variable == index)
    scope = model.add_native_redeal_continuation(match_score=(0, 0))
    assert scope["allPassedCovered"] and not scope["nextDealAnalyzed"]
    assert not scope["strategicCoverage"]
    arguments = dict(target=Team.NORTH_SOUTH, maximum_seconds=5, maximum_conflicts=100_000)
    before = str(model.model.proto)
    result = model.find_native_round_margin_below(threshold=1, **arguments)
    assert result["status"] == ("witness_found" if allow_all_passed else "no_matching_outcome")
    if allow_all_passed:
        candidate = result["candidate"]
        assert candidate["allPassed"] and candidate["awardedPoints"] == [0, 0]
        assert candidate["utility"] == 0 and len(candidate["auctionActions"]) == 8
        assert candidate["playObservations"] == candidate["projectObservations"] == []
        assert "play" not in candidate and "projects" not in candidate
        compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
        proof_options = {}
        if shorten_counterexamples:
            import yaml
            from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1
            settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
            proof_options = dict(shorten_counterexamples=True,
                _play_fallback_id="observer_safe_partner_feed_secure_control_v1",
                _play_contexts={},
                minimax_budget=ScenarioProofBudgetV1.build("actor_information_set_exact_v1",
                    settings["proofBudgets"]["profiles"]["actor_information_set_exact_v1"]))
            def no_play_prefix(*args, **kwargs):
                pytest.fail("all-pass boundary has no play prefix to shorten")
            monkeypatch.setattr(compiler, "_bound_native_counterexample_prefix", no_play_prefix)
        refinement = compiler._refine_symbolic_round_policy(model, Team.NORTH_SOUTH, {},
            threshold=1, maximum_queries=4, maximum_observations=32,
            maximum_conflicts=100_000, maximum_seconds=5, **proof_options)
        assert refinement["status"] == "policy_extension_required"
        assert refinement["candidate"]["allPassed"] and refinement["queriesUsed"] == 1
        assert "prefixBoundQueryHash" not in refinement
        bound = compiler._solve_symbolic_round_policy_bounds([model], Team.NORTH_SOUTH,
            maximum_iterations=2, maximum_queries=2, maximum_observations=32,
            maximum_branches=1, maximum_conflicts=100_000, maximum_seconds=5)
        assert bound["status"] == "scoped_bound_converged" and bound["upperBound"] == 0
        assert not bound["strategicCoverage"]
        from baloot_platform.research.result_authority import (
            validate_native_incumbent_lower_bound_v2, validate_native_incumbent_threshold_floor_v1)
        policies = _native_ranked_play_policies(Team.NORTH_SOUTH)
        owner = dict(target_team="north_south", play_policies=policies,
            rules_hash=rules.sha256, source_model_hashes=(model._model_hash(search_branching="automatic"),))
        for threshold in (None, 0):
            ranked = compiler._solve_symbolic_round_policy_bounds([model], Team.NORTH_SOUTH,
                maximum_iterations=2, maximum_queries=2, maximum_observations=32,
                maximum_branches=1, maximum_conflicts=100_000, maximum_seconds=5,
                incumbent_play_policies=policies, incumbent_only=True,
                **({"incumbent_threshold": threshold} if threshold is not None else {}))
            validate = (validate_native_incumbent_lower_bound_v2 if threshold is None
                else validate_native_incumbent_threshold_floor_v1)
            assert validate(ranked, **owner) == 0
            policy = ranked["queries"][0]["receipt"]["policy"]
            assert "playPolicyModelHash" not in policy
            assert policy["playPolicies"]
    assert model.find_native_round_margin_below(threshold=0, **arguments)["status"] == "no_matching_outcome"
    assert model.find_native_round_margin_below(threshold=1, cancel_check=lambda: True, **arguments)["status"] == "incomplete"
    with pytest.raises(ValueError):
        model.add_native_redeal_continuation(match_score=(0, 0))
    assert str(model.model.proto) == before


@pytest.mark.parametrize("allow_all_passed", (False, True))
@pytest.mark.parametrize("purchase", (False, True))
def test_native_auction_all_pass_completion_has_no_fictitious_final_hands(rules, allow_all_passed, purchase):
    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST, canonicalize_deal_packets=True)
    model.add_native_auction_fragment(
        required_actions=[{"action": "sun"}] if purchase else [{"action": "pass"}] * 8, required_actions_complete=True,
        required_purchase_card="H7", allow_all_passed=allow_all_passed,
    )
    result = model.solve(maximum_seconds=5)
    if purchase:
        assert result.status == "satisfied"
        assert result.buyer == Seat.NORTH
        assert result.auction.completion_contract == Contract.SUN
        assert result.owner_of_card[result.purchase_card.id] == result.purchase_recipient
        return
    if not allow_all_passed:
        assert result.status == "infeasible"
        return
    assert result.status == "satisfied"
    assert result.buyer is result.purchase_recipient is None
    assert result.owner_of_card is result.owner_at_position is None
    assert len(result.deck_order) == 32
    assert result.auction.prefix_terminal is True
    assert len(result.auction.actions) == 8
    assert result.auction.legal_next_choices == ()
    assert result.auction.completion_contract is result.auction.completion_buyer is None
    assert result.auction.completion_purchase_round is None
    document = result.to_dict()
    assert document["ownerOfCard"] is None
    assert document["auction"]["completion"]["contract"] is None


@pytest.mark.parametrize("change", ("none", "hand", "purchase", "actor", "terminal"))
def test_native_auction_observation_literal_is_reified(rules, change):
    from ortools.sat.python import cp_model

    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    for index, variable in enumerate(model.card_at_position):
        model.model.add(variable == index)
    model.add_native_auction_fragment(
        required_actions=[{"action": "sun"}] if change == "terminal" else [],
        prefix_length=1 if change == "terminal" else 0, allow_all_passed=True,
    )
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST)
    hand = list(layout.positions_for(Seat.NORTH, final=False))
    if change == "hand":
        hand[0] = layout.positions_for(Seat.WEST, final=False)[0]
    purchase = layout.purchase_position if change != "purchase" else layout.undealt_positions[0]
    literal = model.native_auction_observation_literal(
        actor=Seat.WEST if change == "actor" else Seat.NORTH,
        visible_cards=hand, purchase_card=purchase,
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(model.model) == cp_model.OPTIMAL
    expected = change == "none"
    assert solver.boolean_value(literal) == expected
    model.model.add(literal != int(expected))
    assert solver.solve(model.model) == cp_model.INFEASIBLE


def test_native_auction_observation_rejects_solver_selected_public_history(rules):
    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    model.add_native_auction_fragment(prefix_length=1, allow_all_passed=True)
    with pytest.raises(ValueError, match="fixed public auction prefix"):
        model.native_auction_observation_literal(actor=Seat.WEST,
            visible_cards=(0, 1, 2, 3, 4), purchase_card=20)


@pytest.mark.parametrize("case", ("matching", "different_hand", "illegal", "unknown", "cancelled"))
@pytest.mark.parametrize("prefix_legality", (False, True))
def test_native_auction_policy_binding_is_observation_local_and_fail_closed(rules, monkeypatch, case, prefix_legality):
    from ortools.sat.python import cp_model
    import baloot_platform.research.ortools_scenario_model as module

    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    fragment = model.add_native_auction_fragment(prefix_length=0, allow_all_passed=True)
    arguments = dict(actor=Seat.NORTH, visible_cards=(0, 1, 2, 3, 4), purchase_card="C7",
        public_choices=(),
        action="hokum:hearts" if case == "illegal" else "pass",
        maximum_seconds=5, maximum_conflicts=10000, _prefix_legality=prefix_legality)
    if case == "cancelled":
        arguments["cancel_check"] = lambda: True
    with monkeypatch.context() as patch:
        def duplicate_legality(*args, **kwargs):
            raise AssertionError("binding must reuse the native auction path legality")
        if not prefix_legality:
            patch.setattr(module, "attach_auction_legal_actions_v1", duplicate_legality)
        if case == "unknown":
            patch.setattr(module, "_solve_cp_sat_with_cancel_v1", lambda *_args: cp_model.UNKNOWN)
        receipt = model.bind_native_auction_observed_choice(**arguments)
    assert receipt["status"] == ("incomplete" if case in {"unknown", "cancelled"} or case == "illegal" and prefix_legality
        else "invalid" if case == "illegal" else "bound")
    if prefix_legality:
        assert receipt["legalityScope"] == "actor_visible_auction_prefix_relaxation_v1"
    if receipt["status"] == "bound":
        bound_model = str(model.model.proto)
        assert model.bind_native_auction_observed_choice(**arguments) == receipt
        assert str(model.model.proto) == bound_model
        assert model.bind_native_auction_observed_choice(**{
            **arguments, "visible_cards": tuple(reversed(arguments["visible_cards"]))}) == receipt
        with pytest.raises(ValueError, match="conflicting actions"):
            model.bind_native_auction_observed_choice(**{**arguments, "action": "sun"})
        with pytest.raises(ValueError, match="explicit public auction history"):
            model.bind_native_auction_observed_choice(**{**arguments, "public_choices": ""})
        assert str(model.model.proto) == bound_model
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST)
    cards = (0, 1, 2, 3, 5) if case == "different_hand" else arguments["visible_cards"]
    for position, card in zip(layout.positions_for(Seat.NORTH, final=False), cards, strict=True):
        model.model.add(model.card_at_position[position] == card)
    model.model.add(model.purchase_card == Card.parse("C7").id)
    model.model.add(fragment.choices[0]["sun"] == 1)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(model.model) == (cp_model.INFEASIBLE if case == "matching" else cp_model.OPTIMAL)


def test_native_auction_prefix_legality_preserves_full_round_owner_with_smaller_probe(root, monkeypatch, record_property):
    import time
    import baloot_platform.research.ortools_scenario_model as module

    _, source, _, _ = _native_hokum_round_branch(root, "normal")
    candidate = source.find_native_round_margin_below(target=Team.NORTH_SOUTH, threshold=1000,
        maximum_seconds=15, maximum_conflicts=250000)["candidate"]
    row = candidate["auctionObservations"][0]
    original = str(source.model.proto)
    solve, sizes, receipts, owners, elapsed = module._solve_cp_sat_with_cancel_v1, [], [], [], []

    def measured(solver, model, cancel_check):
        sizes.append(len(model.proto.constraints))
        return solve(solver, model, cancel_check)

    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", measured)
    for prefix_legality in (False, True):
        owned = copy.copy(source)
        owned.model, owned._observed_auction_bindings = source.model.clone(), {}
        started = time.monotonic()
        receipt = owned.bind_native_auction_observed_choice(actor=Seat.parse(row["actor"]),
            visible_cards=row["visibleCards"], purchase_card=row["purchaseCard"],
            public_choices=row["publicChoices"], action=row["selectedAction"],
            maximum_seconds=10, maximum_conflicts=250000, _prefix_legality=prefix_legality)
        elapsed.append(time.monotonic() - started)
        assert receipt["status"] == "bound" and receipt["solverStatus"] == "INFEASIBLE"
        receipts.append(receipt)
        owners.append(str(owned.model.proto))
    assert len(sizes) == 2 and sizes[1] < sizes[0] / 10
    assert owners[0] == owners[1] and receipts[0]["modelHash"] == receipts[1]["modelHash"]
    assert receipts[0]["legalityModelHash"] != receipts[1]["legalityModelHash"]
    assert str(source.model.proto) == original and not source._observed_auction_bindings
    for name, value in zip(("fullProbeConstraints", "prefixProbeConstraints", "fullBindingSeconds", "prefixBindingSeconds"),
                           (*sizes, *elapsed), strict=True):
        record_property(name, value)


@pytest.mark.real_clock
def test_native_policy_legality_cache_rebuilds_guards_and_keeps_readback_fresh(rules, monkeypatch):
    import copy
    import time
    import baloot_platform.research.ortools_scenario_model as module
    from baloot_platform.research.scenario_contracts import canonical_sha256

    source = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    match, selected, legal = [source.model.new_bool_var(name) for name in ("match", "selected", "legal")]
    source.model.add(legal == 1)
    cache, calls = {}, []
    solve = module._solve_cp_sat_with_cancel_v1

    def counted(*args):
        calls.append(1)
        return solve(*args)

    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", counted)

    def bind(*, cached=True, changed=False, cancelled=False, expired=False):
        owner = copy.copy(source)
        owner.model = source.model.clone()
        owner._native_policy_legality_cache = cache if cached else None
        if changed:
            owner.model.add(match == 1)
        bindings = {}
        receipt = owner._bind_native_policy_choice(observation_hash="a" * 64,
            action="pass", choices=((match, selected, legal),), kind="auction", bindings=bindings,
            deadline=time.monotonic() + (-1 if expired else 5), maximum_conflicts=10000,
            cancel_check=(lambda: True) if cancelled else None)
        return owner, receipt, bindings

    first, receipt, _ = bind()
    second, repeated, bindings = bind()
    assert len(calls) == len(cache) == 1
    assert repeated == receipt and str(first.model.proto) == str(second.model.proto)
    assert len(first.model.proto.constraints) == len(source.model.proto.constraints) + 1
    assert bindings["a" * 64] == receipt
    # The guarded constraint is actually installed in both independent owners.
    for owner in (first, second):
        owner.model.add(match == 1)
        owner.model.add(selected == 0)
        assert cp_model.CpSolver().solve(owner.model) == cp_model.INFEASIBLE
    bind(changed=True)
    assert len(calls) == len(cache) == 2
    bind(cached=False)  # Fresh readers never inherit generation's legality cache.
    assert len(calls) == 3
    for options in ({"cancelled": True}, {"expired": True}):
        owner, stopped, bindings = bind(**options)
        assert stopped["status"] == "incomplete" and not bindings
        assert str(owner.model.proto) == str(source.model.proto) and len(cache) == 2
    key = next(iter(cache))
    cache[key]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="cache identity"):
        bind()
    cache.clear()
    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", lambda *args: cp_model.UNKNOWN)
    assert bind()[1]["status"] == "incomplete" and not cache
    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", counted)
    # Changed action legality cannot be certified by another probe's cached answer.
    source = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    match, selected, legal = [source.model.new_bool_var(name) for name in ("match", "selected", "legal")]
    source.model.add(legal == 0)
    invalid = bind()[1]
    assert invalid["status"] == "invalid" and not cache
    assert invalid["sha256"] == canonical_sha256({k: v for k, v in invalid.items() if k != "sha256"})


@pytest.mark.parametrize("different_hand", (False, True))
@pytest.mark.parametrize("prefix_legality", (False, True))
def test_native_auction_policy_binds_later_public_observation(rules, different_hand, prefix_legality):
    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    fragment = model.add_native_auction_fragment(prefix_length=0, allow_all_passed=True)
    receipt = model.bind_native_auction_observed_choice(actor=Seat.SOUTH,
        visible_cards=(0, 1, 2, 3, 4), purchase_card="C7", public_choices=("pass", "pass"),
        action="pass", maximum_seconds=5, maximum_conflicts=10000, _prefix_legality=prefix_legality)
    assert receipt["status"] == "bound"
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST)
    cards = (0, 1, 2, 3, 5) if different_hand else (0, 1, 2, 3, 4)
    for position, card in zip(layout.positions_for(Seat.SOUTH, final=False), cards, strict=True):
        model.model.add(model.card_at_position[position] == card)
    model.model.add(model.purchase_card == Card.parse("C7").id)
    for index, action in enumerate(("pass", "pass", "sun")):
        model.model.add(fragment.choices[index][action] == 1)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(model.model) == (cp_model.OPTIMAL if different_hand else cp_model.INFEASIBLE)


@pytest.mark.parametrize("history", ("pass", ("closed",), ("unknown",), (1,), ("pass",) * 9))
def test_native_auction_observation_rejects_invalid_explicit_history(rules, history):
    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    model.add_native_auction_fragment(prefix_length=0, allow_all_passed=True)
    before = str(model.model.proto)
    with pytest.raises(ValueError, match="explicit public auction history"):
        model.native_auction_observation_literal(actor=Seat.NORTH,
            visible_cards=(0, 1, 2, 3, 4), purchase_card="C7", public_choices=history)
    assert str(model.model.proto) == before


def test_deck_no_good_enforces_governed_hamming_distance(rules):
    first = OrToolsScenarioModelV1(
        rules=rules,
        dealer=Seat.NORTH,
        required_buyer=Seat.SOUTH,
    ).solve(maximum_seconds=5)
    assert first.deck_order is not None

    second = OrToolsScenarioModelV1(
        rules=rules,
        dealer=Seat.NORTH,
        required_buyer=Seat.SOUTH,
        excluded_deck_orders=[first.deck_order],
        minimum_deck_hamming_distance=4,
    ).solve(maximum_seconds=5)

    assert second.status == "satisfied"
    assert second.deck_order is not None
    assert (
        sum(
            left != right
            for left, right in zip(first.deck_order, second.deck_order, strict=True)
        )
        >= 4
    )
    assert second.minimum_deck_hamming_distance == 4
    assert second.to_dict()["minimumDeckHammingDistance"] == 4


def _auction_semantics(plan):
    return (
        tuple(
            (
                action.round_index,
                action.actor,
                action.action,
                action.trump,
            )
            for action in plan.actions
        ),
        plan.buyer,
        plan.contract,
        plan.trump,
        plan.purchase_round,
    )


@pytest.mark.parametrize("official", [False, True])
@pytest.mark.parametrize("actions", [(), ("sun",), ("pass", "pass", "hokum", "pass"),
                                   ("pass", "pass", "ashkal", "pass", "pass")])
def test_native_integrated_auction_observation_and_ownership_match_legacy(root, rules, monkeypatch, official, actions):
    import baloot_platform.research.ortools_scenario_model as module
    if official:
        rules = RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    kwargs = dict(required_actions=[{"action": value} for value in actions], required_purchase_card="HA")
    legacy = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    legacy.add_auction_fragment(**kwargs)
    expected = legacy.solve(seed=19, maximum_seconds=5)
    assert expected.status == "satisfied"
    def forbidden(*args, **kwargs):
        raise AssertionError("native extraction reached interpreted auction authority")
    for name in ("_enumerate_auction", "_advance_auction", "_auction_legal_choices"):
        monkeypatch.setattr(module, name, forbidden)
    native = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    native.add_native_auction_fragment(**kwargs)
    actual = native.solve(seed=19, maximum_seconds=5)
    assert actual.status == "satisfied"
    def prefix_only(observation):
        return {key: value for key, value in observation.to_dict().items()
                if key != "completion" and not key.startswith("completion")}
    assert prefix_only(actual.auction) == prefix_only(expected.auction)
    assert actual.buyer == actual.auction.completion_buyer
    assert actual.owner_of_card[actual.purchase_card.id] == actual.auction.completion_purchase_recipient
    if actual.auction.prefix_terminal:
        assert actual.auction == expected.auction
    repeated = native.solve(seed=19, maximum_seconds=5)
    assert repeated == actual


@pytest.mark.parametrize("invalid", ["actor", "rank_overcall", "null_trump", "nonterminal", "extra_action"])
def test_native_integrated_auction_rejects_illegal_prefix(root, invalid):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    prefix = [{"action": action} for action in ("pass", "pass", "hokum", "pass")]
    if invalid == "actor":
        prefix[0]["actor"] = "east"
    elif invalid == "rank_overcall":
        prefix.append({"action": "sun"})
    elif invalid == "null_trump":
        prefix[2]["trumpSuit"] = None
    elif invalid == "extra_action":
        prefix = [{"action": "sun"}, {"action": "pass"}]
    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    model.add_native_auction_fragment(required_actions=prefix, required_purchase_card="HA",
                                     required_actions_complete=invalid == "nonterminal")
    assert model.solve(maximum_seconds=5).status == "infeasible"


@pytest.mark.parametrize("dealer", tuple(Seat))
@pytest.mark.parametrize("purchase_suit", tuple(Suit))
def test_auction_fragment_table_matches_domain_reference(
    rules,
    dealer: Seat,
    purchase_suit: Suit,
):
    model = OrToolsScenarioModelV1(rules=rules, dealer=dealer)
    fragment = model.add_auction_fragment()
    actual = {
        _auction_semantics(plan)
        for plan in fragment.plans
        if plan.purchase_suit == purchase_suit
    }
    expected = {
        _auction_semantics(plan)
        for plan in OrToolsCpSatAuctionCompilerV1._legal_auctions(
            dealer=dealer,
            purchase_suit=purchase_suit,
            rules=rules,
        )
    }
    assert actual == expected


def test_auction_fragment_connects_buyer_purchase_and_ownership(rules):
    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    fragment = model.add_auction_fragment()
    result = model.solve(seed=9, maximum_seconds=5)

    assert result.status == "satisfied"
    assert result.auction is not None
    assert result.auction.buyer is None
    assert result.auction.contract is None
    assert result.auction.current_actor == Seat.EAST.next()
    assert result.auction.current_round == 1
    assert result.auction.completion_buyer == result.buyer
    assert result.purchase_suit is not None
    assert result.owner_of_card is not None
    assert result.purchase_card is not None
    assert (
        result.owner_of_card[result.purchase_card.id] == result.auction.completion_buyer
    )
    assert result.auction.completion_contract in set(Contract)
    assert fragment.domain_hash == AuctionDomainIRV1.from_rules(rules.data).sha256


def test_auction_fragment_prunes_to_a_fixed_purchase_card(rules):
    purchase_card = Card.parse("HA")
    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    fragment = model.add_auction_fragment(required_purchase_card=purchase_card)

    assert fragment.plans
    assert {plan.purchase_suit for plan in fragment.plans} == {purchase_card.suit}
    assert {plan.purchase_rank for plan in fragment.plans} == {purchase_card.rank}
    result = model.solve(seed=9, maximum_seconds=5)
    assert result.status == "satisfied"
    assert result.purchase_card == purchase_card


def test_auction_fragment_prunes_plans_for_a_required_buyer(rules):
    model = OrToolsScenarioModelV1(
        rules=rules,
        dealer=Seat.EAST,
        required_buyer=Seat.NORTH,
    )
    fragment = model.add_auction_fragment(allowed_contracts=(Contract.SUN,))

    assert fragment.plans
    assert {plan.buyer for plan in fragment.plans} == {Seat.NORTH}
    result = model.solve(maximum_seconds=5)
    assert result.status == "satisfied"
    assert result.auction is not None
    assert result.auction.completion_buyer == Seat.NORTH


def test_auction_fragment_can_keep_one_trace_per_completion(rules):
    complete = OrToolsScenarioModelV1(
        rules=rules,
        dealer=Seat.EAST,
        required_buyer=Seat.NORTH,
    ).add_auction_fragment(allowed_contracts=(Contract.SUN,))
    canonical_model = OrToolsScenarioModelV1(
        rules=rules,
        dealer=Seat.EAST,
        required_buyer=Seat.NORTH,
    )
    canonical = canonical_model.add_auction_fragment(
        allowed_contracts=(Contract.SUN,),
        canonical_completion_only=True,
    )

    def outcome(plan):
        return (
            plan.purchase_suit,
            plan.purchase_rank,
            plan.caller,
            plan.buyer,
            plan.purchase_recipient,
            plan.declarer,
            plan.contract,
            plan.trump,
            plan.purchase_round,
        )

    assert len(canonical.plans) < len(complete.plans)
    assert {outcome(plan) for plan in canonical.plans} == {
        outcome(plan) for plan in complete.plans
    }
    result = canonical_model.solve(maximum_seconds=5)
    assert result.status == "satisfied"
    assert result.auction is not None
    assert result.auction.completion_buyer == Seat.NORTH
    assert result.auction.completion_contract == Contract.SUN


def test_auction_fragment_filters_contracts_and_reports_empty_domain(rules):
    sun_model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    sun_model.add_auction_fragment(allowed_contracts=(Contract.SUN,))
    sun = sun_model.solve(maximum_seconds=5)

    assert sun.status == "satisfied"
    assert sun.auction is not None
    assert sun.auction.completion_contract == Contract.SUN

    impossible = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    impossible.add_auction_fragment(allowed_contracts=())
    assert impossible.solve(maximum_seconds=5).status == "infeasible"


def test_auction_fragment_supports_partial_and_exact_prefixes(rules):
    reference_plans = OrToolsCpSatAuctionCompilerV1._legal_auctions(
        dealer=Seat.EAST,
        purchase_suit=Suit.HEARTS,
        rules=rules,
    )
    target = max(reference_plans, key=lambda plan: len(plan.actions))
    partial_prefix = tuple(action.to_dict() for action in target.actions[:2])

    partial_model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    partial_model.model.add(partial_model.purchase_suit == int(Suit.HEARTS))
    partial_model.add_auction_fragment(required_actions=partial_prefix)
    partial = partial_model.solve(maximum_seconds=5)
    assert partial.auction is not None
    assert partial.auction.prefix_terminal is False
    assert partial.auction.buyer is None
    assert partial.auction.contract is None
    assert partial.auction.purchase_round is None
    assert len(partial.auction.actions) == 2
    assert len(partial.auction.completion_actions) >= len(partial.auction.actions)
    assert tuple(
        (action.actor, action.action, action.trump)
        for action in partial.auction.actions[:2]
    ) == tuple(
        (action.actor, action.action, action.trump) for action in target.actions[:2]
    )

    exact_model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    exact_model.model.add(exact_model.purchase_suit == int(Suit.HEARTS))
    exact_model.add_auction_fragment(
        required_actions=tuple(action.to_dict() for action in target.actions),
        required_actions_complete=True,
    )
    exact = exact_model.solve(maximum_seconds=5)
    assert exact.auction is not None
    assert exact.auction.prefix_terminal is True
    assert exact.auction.legal_next_actions == ()
    assert exact.auction.legal_next_choices == ()
    assert _auction_semantics(exact.auction) == _auction_semantics(target)


def test_auction_fragment_supports_unfixed_decision_prefix_length(rules):
    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    model.add_auction_fragment(prefix_length=4)

    result = model.solve(maximum_seconds=5)

    assert result.status == "satisfied"
    assert result.auction is not None
    assert len(result.auction.actions) == 4
    assert result.auction.prefix_terminal is False
    assert result.auction.current_actor is not None
    assert result.auction.legal_next_actions


def test_auction_prefix_actor_is_derived_when_omitted_and_checked_when_given(rules):
    derived = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    derived.add_auction_fragment(required_actions=({"action": "pass"},))
    derived_result = derived.solve(maximum_seconds=5)

    assert derived_result.status == "satisfied"
    assert derived_result.auction is not None
    assert derived_result.auction.actions[0].actor == Seat.EAST.next()
    assert derived_result.auction.current_actor == Seat.EAST.next().next()

    mismatched = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    mismatched.add_auction_fragment(
        required_actions=({"actor": "east", "action": "pass"},)
    )
    assert mismatched.solve(maximum_seconds=5).status == "infeasible"


def test_second_round_prefix_exposes_active_hokum_suit_choices(rules):
    dealer = Seat.EAST
    order = (dealer.next(), dealer.next().next(), dealer.next().next().next(), dealer)
    prefix = tuple(
        {"roundIndex": 1, "actor": actor.name.lower(), "action": "pass"}
        for actor in order
    )
    model = OrToolsScenarioModelV1(rules=rules, dealer=dealer)
    model.add_auction_fragment(required_actions=prefix)
    result = model.solve(maximum_seconds=5)

    assert result.auction is not None
    domain = AuctionDomainIRV1.from_rules(rules.data)
    expected_actions = tuple(sorted(domain.second_round_allowed))
    assert result.auction.legal_next_actions == expected_actions
    assert result.purchase_suit is not None
    expected_hokum = {
        f"hokum:{suit.name.lower()}" for suit in Suit if suit != result.purchase_suit
    }
    assert expected_hokum <= set(result.auction.legal_next_choices)
    assert f"hokum:{result.purchase_suit.name.lower()}" not in (
        result.auction.legal_next_choices
    )


@pytest.mark.parametrize("native_cp_sat", (False, True))
def test_exact_nonterminal_auction_prefix_is_infeasible(rules, monkeypatch, native_cp_sat):
    dealer = Seat.EAST
    order = (dealer.next(), dealer.next().next(), dealer.next().next().next(), dealer)
    prefix = tuple(
        {"actor": actor.name.lower(), "action": "pass"} for actor in order
    )
    if native_cp_sat:
        def forbidden_legacy_table(*_args, **_kwargs):
            pytest.fail("native auction proof must not read legacy auction tables")

        monkeypatch.setattr(OrToolsCpSatAuctionCompilerV1, "_initial_auction_table", forbidden_legacy_table)
    model = OrToolsScenarioModelV1(rules=rules, dealer=dealer)
    add_auction = model.add_native_auction_fragment if native_cp_sat else model.add_auction_fragment
    add_auction(
        required_actions=prefix,
        required_actions_complete=True,
    )
    solver = cp_model.CpSolver()
    result = model.solve(maximum_seconds=5, solver=solver)
    assert solver.response_proto.status == cp_model.INFEASIBLE
    assert result.status == "infeasible"
    assert result.auction is None
    if native_cp_sat:
        reachable = OrToolsScenarioModelV1(rules=rules, dealer=dealer)
        reachable.add_native_auction_fragment(required_actions=prefix)
        continuation = reachable.solve(maximum_seconds=5)
        assert continuation.status == "satisfied"
        assert continuation.auction is not None
        assert continuation.auction.prefix_terminal is False
        assert len(continuation.auction.actions) == len(prefix)


@pytest.mark.parametrize(
    "raw,values,expected",
    [
        (
            {"op": "all", "children": [_leaf("PRED-AAA"), _leaf("PRED-BBB")]},
            {"PRED-AAA": 1, "PRED-BBB": 0},
            0,
        ),
        (
            {"op": "any", "children": [_leaf("PRED-AAA"), _leaf("PRED-BBB")]},
            {"PRED-AAA": 1, "PRED-BBB": 0},
            1,
        ),
        (
            {"op": "not", "children": [_leaf("PRED-AAA")]},
            {"PRED-AAA": 0},
            1,
        ),
        (
            {
                "op": "count",
                "minimum": 2,
                "maximum": 2,
                "children": [
                    _leaf("PRED-AAA"),
                    _leaf("PRED-BBB"),
                    _leaf("PRED-CCC"),
                ],
            },
            {"PRED-AAA": 1, "PRED-BBB": 0, "PRED-CCC": 1},
            1,
        ),
        (
            {
                "op": "all",
                "children": [
                    {
                        "op": "any",
                        "children": [_leaf("PRED-AAA"), _leaf("PRED-BBB")],
                    },
                    {"op": "not", "children": [_leaf("PRED-CCC")]},
                ],
            },
            {"PRED-AAA": 0, "PRED-BBB": 1, "PRED-CCC": 0},
            1,
        ),
    ],
)
def test_reified_expression_is_an_exact_equivalence(raw, values, expected):
    expression = ScenarioExpressionV1.build(raw)
    model = cp_model.CpModel()
    literals = {
        predicate_id: model.new_bool_var(f"predicate_{predicate_id}")
        for predicate_id in values
    }
    for predicate_id, value in values.items():
        model.add(literals[predicate_id] == value)
    root = reify_scenario_expression(model, expression, literals)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    assert solver.value(root) == expected


def test_reified_expression_refuses_missing_child_without_mutating_model():
    expression = ScenarioExpressionV1.build(
        {
            "op": "all",
            "children": [_leaf("PRED-AAA"), _leaf("PRED-MISSING")],
        }
    )
    model = cp_model.CpModel()
    literals = {"PRED-AAA": model.new_bool_var("predicate_A")}
    constraints_before = len(model.proto.constraints)

    with pytest.raises(ValueError, match="PRED-MISSING"):
        reify_scenario_expression(model, expression, literals)

    assert len(model.proto.constraints) == constraints_before


@pytest.mark.parametrize(
    "deck",
    [tuple(range(31)), (*range(31), 30), (*range(31), "31")],
)
def test_no_good_decks_fail_closed(rules, deck):
    error = TypeError if any(type(item) is not int for item in deck) else ValueError
    with pytest.raises(error):
        OrToolsScenarioModelV1(
            rules=rules,
            dealer=Seat.EAST,
            excluded_deck_orders=[deck],
        )
