from __future__ import annotations

from pathlib import Path
import copy
import time

import pytest

from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.project_ir import ProjectDomainIRV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_exact_state import (
    ExactNormalizationStopped,
    OrToolsExactStateNormalizerV1,
)
from baloot_platform.research.ortools_complete_state_graph import OrToolsCompleteStateGraphCompilerV1
from baloot_platform.research.ortools_nonplay_complete_graph import (
    compile_nonplay_complete_graph_v1,
    compile_auction_hand_quality_graph_v1,
)
from baloot_platform.research.ortools_nonplay_graphs import (
    NonPlayDecisionObservationV1,
    NonPlayGraphCompilationStoppedV1,
    NonPlayGraphBudgetV1,
    compile_escalation_decision_graph_v1,
)
from baloot_platform.research.ortools_nonplay_play_graft import (
    OrToolsNonPlayPlayGraphGraftV1,
)
from baloot_platform.research.ortools_nonplay_observation import build_nonplay_root_observer_v1
from baloot_platform.research.scenario_contracts import (
    ExactTracePrefixV1,
    ScenarioProofRequestV1,
    ScenarioPlayingQualityPolicyV1,
    canonical_sha256,
)


@pytest.mark.parametrize("codes,expected", [
    (("S7", "H7", "C7", "D7", "S8"), "pass"),
    (("SJ", "S9", "SA", "S10", "SK"), "hokum:spades"),
])
@pytest.mark.parametrize("passes", [0, 4])
def test_normalized_auction_hand_graph_is_observer_local(rules, codes, expected, passes):
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST)
    deck = [None] * 32
    for position, code in zip(layout.initial_positions[Seat.NORTH], codes, strict=True):
        deck[position] = Card.parse(code).id
    deck[layout.purchase_position] = Card.parse("SQ").id
    remaining = iter(index for index in range(32) if index not in deck)
    deck = [next(remaining) if card is None else card for card in deck]
    raw = _escalation_prefix().to_dict()
    raw.pop("sha256")
    raw.update(deckOrder=deck, auctionActions=[{"action": "pass"} for _ in range(passes)])
    if passes == 4:
        expected = "pass"
    prefix = ExactTracePrefixV1.build(raw)
    receipts = []
    result = compile_auction_hand_quality_graph_v1(
        rules, prefix, maximum_nodes=100, maximum_depth=2, maximum_seconds=30,
        native_model_receipts=receipts,
    )
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=30,
    )
    reused = compile_auction_hand_quality_graph_v1(
        rules, prefix, maximum_nodes=100, maximum_depth=2, maximum_seconds=30,
        native_model_receipts=[], normalized=normalized,
        base_observer_factory=lambda exact, source: build_nonplay_root_observer_v1(exact, source, "bid"),
    )
    assert reused.graph.to_dict() == result.graph.to_dict()
    assert set(reused.decision_observation_ids) == {reused.graph.root_node_id}
    nodes = {node.node_id: node for node in result.graph.nodes}
    root = nodes[result.graph.root_node_id]
    best = max(nodes[edge.child_node_id].utility for edge in root.edges)
    assert [edge.action_code for edge in root.edges if nodes[edge.child_node_id].utility == best] == [expected]
    assert result.graph.scope == "bounded" and result.graph.boundary_evaluator == "auction_hand_margin_v1"
    assert receipts and all(node.leaf_kind == "boundary" for node in result.graph.nodes if node.is_leaf)
    # Change only hidden cards. Neither action values nor graph identity may change.
    other = copy.deepcopy(raw)
    left = layout.initial_positions[Seat.WEST][0]
    right = layout.initial_positions[Seat.SOUTH][0]
    other["deckOrder"][left], other["deckOrder"][right] = other["deckOrder"][right], other["deckOrder"][left]
    repeated = compile_auction_hand_quality_graph_v1(
        rules, ExactTracePrefixV1.build(other), maximum_nodes=100, maximum_depth=2,
        maximum_seconds=30, native_model_receipts=[],
    )
    assert repeated.graph.to_dict() == result.graph.to_dict()


def test_auction_hand_quality_rejects_terminal_auction(rules):
    raw = _escalation_prefix().to_dict()
    raw.pop("sha256")
    raw["auctionActions"] = [{"action": "pass"} for _ in range(8)]
    with pytest.raises(ValueError, match="live bid"):
        compile_auction_hand_quality_graph_v1(
            rules, ExactTracePrefixV1.build(raw), maximum_nodes=100, maximum_depth=2,
            maximum_seconds=30, native_model_receipts=[],
        )


def _request(
    mode: str, *, horizon: int = 1, unit: str = "plies"
) -> ScenarioProofRequestV1:
    value = {
        "schemaVersion": 1,
        "mode": mode,
        "visibility": "complete_state",
        "claim": "outcome_reachable",
        "targetTeam": "north_south",
        "utility": "awarded_point_margin",
        "successThreshold": -100_000,
        "budgetProfile": "test",
    }
    if mode == "bounded":
        value.update(
            {
                "horizon": {"unit": unit, "value": horizon},
                "boundaryEvaluator": "current_raw_point_margin_v1",
            }
        )
    return ScenarioProofRequestV1.build(value)


def _escalation_prefix() -> ExactTracePrefixV1:
    return ExactTracePrefixV1.build(
        {
            "schemaVersion": 1,
            "deckOrder": list(range(32)),
            "dealer": "east",
            "roundNumber": 1,
            "matchScore": [0, 0],
            "auctionActions": [
                {"action": action}
                for action in ("hokum", "pass", "pass", "pass")
            ],
            "escalationActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "playActions": [],
            "stateAssertions": {},
        }
    )


def test_bounded_bid_graph_skips_forced_escalation_closure(rules: RuleBook, monkeypatch) -> None:
    import baloot_platform.research.ortools_nonplay_graphs as graphs
    def forbidden(*args, **kwargs):
        pytest.fail("native complete graph used legacy auction transitions")
    for name in ("_initial_auction_state", "_advance_auction", "_auction_actor", "_auction_legal_choices"):
        monkeypatch.setattr(graphs, name, forbidden)
    monkeypatch.setattr(EscalationDomainIRV1, "legal_decisions", forbidden)
    prefix = ExactTracePrefixV1.build(
        {
            "schemaVersion": 1,
            "deckOrder": list(range(32)),
            "dealer": "east",
            "roundNumber": 1,
            "matchScore": [0, 0],
            "auctionActions": [
                {"action": "pass"}
                for _ in range(7)
            ],
            "escalationActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "playActions": [],
            "stateAssertions": {},
        }
    )
    decision = {"kind": "bid", "selector": {"actionIndex": 7}}
    normalized = OrToolsExactStateNormalizerV1(
        rules, native_cp_sat=True
    ).normalize(prefix, decision_point=decision, maximum_seconds=15)

    compiled = compile_nonplay_complete_graph_v1(
        rules,
        normalized,
        prefix,
        decision,
        _request("bounded", horizon=2),
        maximum_nodes=1_000,
        maximum_depth=4,
        maximum_seconds=15,
        native_cp_sat=True,
    )
    root = next(
        node
        for node in compiled.graph.nodes
        if node.node_id == compiled.graph.root_node_id
    )

    assert not root.is_leaf
    assert len(root.edges) == len(normalized.legal_actions) == 5


def test_native_bid_handoff_keeps_buyer_and_declarer_distinct(rules, monkeypatch) -> None:
    from types import SimpleNamespace
    import baloot_platform.research.ortools_nonplay_complete_graph as complete
    from baloot_platform.research.ortools_nonplay_graphs import AuctionGraphTerminalV1

    raw = _escalation_prefix().to_dict()
    raw.pop("sha256")
    raw["auctionActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    decision = {"kind": "bid"}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point=decision, maximum_seconds=15,
    )
    seen = []

    class NativeOpening:
        def __init__(self, _domain, *, declarer, **_kwargs):
            seen.append(("opening", declarer))

        def solve(self, **_kwargs):
            return SimpleNamespace(status="satisfied", observation=SimpleNamespace(legal_next_choices=("decline",)))

    class HandoffReached(Exception):
        pass

    def escalation_graph(_domain, *, declarer, **_kwargs):
        seen.append(("graph", declarer))
        raise HandoffReached

    def auction_graph(_domain, *, terminal_utility, **_kwargs):
        terminal_utility(AuctionGraphTerminalV1((), Seat.EAST, Seat.WEST, Contract.SUN, None, 1, False))

    monkeypatch.setattr(complete, "OrToolsEscalationStageV1", NativeOpening)
    monkeypatch.setattr(complete, "compile_escalation_decision_graph_v1", escalation_graph)
    monkeypatch.setattr(complete, "compile_auction_decision_graph_v1", auction_graph)
    with pytest.raises(HandoffReached):
        compile_nonplay_complete_graph_v1(
            rules, normalized, prefix, decision, _request("bounded", horizon=2),
            maximum_nodes=100, maximum_depth=4, maximum_seconds=15, native_cp_sat=True,
        )
    assert seen == [("opening", Seat.WEST), ("graph", Seat.WEST)]


def test_bid_to_escalation_normalization_receives_cancellation(rules, monkeypatch):
    raw = _escalation_prefix().to_dict()
    raw.pop("sha256")
    raw["auctionActions"] = [{"action": "pass"} for _ in range(7)]
    prefix = ExactTracePrefixV1.build(raw)
    decision = {"kind": "bid", "selector": {"actionIndex": 7}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix, decision_point=decision)
    cancelled = False
    reached = []
    original = OrToolsExactStateNormalizerV1.normalize

    def cancel_check():
        return cancelled

    def stop_at_escalation(*args, **kwargs):
        nonlocal cancelled
        assert kwargs["decision_point"]["kind"] == "escalation"
        assert kwargs.get("cancel_check") is cancel_check
        reached.append("escalation")
        cancelled = True
        return original(*args, **kwargs)

    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize", stop_at_escalation)

    def enter_branch_observer(_domain, **kwargs):
        # Wiring-only interruption probe: no graph or qualification is returned.
        assert kwargs.get("cancel_check") is cancel_check
        kwargs["decision_observer"](NonPlayDecisionObservationV1(
            "cancel-probe", "escalation", kwargs["declarer"].next(), {},
        ))
        raise AssertionError("branch normalization ignored cancellation")

    monkeypatch.setattr(
        "baloot_platform.research.ortools_nonplay_complete_graph.compile_escalation_decision_graph_v1",
        enter_branch_observer,
    )
    with pytest.raises(ExactNormalizationStopped) as stopped:
        compile_nonplay_complete_graph_v1(
            rules, normalized, prefix, decision, _request("bounded", horizon=2),
            maximum_nodes=1000, maximum_depth=4, maximum_seconds=15, native_cp_sat=True,
            base_observer_factory=_base_observer_factory, play_observer_factory=_play_observer_factory,
            cancel_check=cancel_check,
        )
    assert reached == ["escalation"]
    assert stopped.value.partial_receipt["exhaustionReason"] == "cancelled"


def test_observed_bid_uses_frozen_order_for_downstream_escalation(rules):
    from baloot_platform.research.ortools_research_executor import compile_observed_decision_v1

    raw = _escalation_prefix().to_dict()
    raw.pop("sha256")
    raw["auctionActions"] = raw["auctionActions"][:1]
    prefix = ExactTracePrefixV1.build(raw)
    decision = {"kind": "bid", "selector": {"actionIndex": 1}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point=decision, maximum_seconds=15,
    )
    request = _request("bounded", horizon=3).to_dict()
    request.pop("sha256")
    request.update(visibility="actor_information_set", strategyClass="pure_deterministic",
                   targetTeam=Seat.parse(normalized.next_actor).team.name.lower())
    compiled = compile_observed_decision_v1(
        rules, normalized, prefix, decision, ScenarioProofRequestV1.build(request),
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=15, native_cp_sat=True,
        opportunity_order=("west", "east", "south", "north"),
    )
    assert compiled.graph.root_actor == normalized.next_actor


@pytest.mark.parametrize("kind", ["bid", "escalation", "project"])
def test_stage_builder_receives_cancellation(root, rules, monkeypatch, kind):
    from baloot_platform.research import ortools_nonplay_complete_graph as graph_module

    if kind == "project":
        rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
        prefix = _optional_project_prefix()
        decision = {"kind": kind, "selector": {"owner": "east"}}
    else:
        raw = _escalation_prefix().to_dict()
        raw.pop("sha256")
        if kind == "bid":
            raw["auctionActions"] = [{"action": "pass"} for _ in range(7)]
        prefix = ExactTracePrefixV1.build(raw)
        decision = {"kind": kind, "selector": {"actionIndex": 7} if kind == "bid" else {"actingSeat": "west"}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix, decision_point=decision)
    name = {"bid": "compile_auction_decision_graph_v1", "escalation": "compile_escalation_decision_graph_v1",
            "project": "compile_project_declaration_graph_v1"}[kind]
    original = getattr(graph_module, name)
    cancelled = False

    def cancel_check():
        return cancelled

    def stop_in_builder(*args, **kwargs):
        nonlocal cancelled
        assert kwargs.get("cancel_check") is cancel_check
        cancelled = True
        return original(*args, **kwargs)

    monkeypatch.setattr(graph_module, name, stop_in_builder)
    with pytest.raises(NonPlayGraphCompilationStoppedV1) as stopped:
        compile_nonplay_complete_graph_v1(
            rules, normalized, prefix, decision, _request("bounded"),
            maximum_nodes=1000, maximum_depth=4, maximum_seconds=15,
            native_cp_sat=kind != "project", cancel_check=cancel_check,
        )
    assert stopped.value.reason == "cancelled"
    assert stopped.value.nodes == 0


@pytest.mark.real_clock
@pytest.mark.parametrize("checkpoint", ["project_root", "completed_graph"])
def test_native_project_deadline_preserves_structured_stop(root, monkeypatch, checkpoint):
    from types import SimpleNamespace
    from baloot_platform.research import ortools_nonplay_complete_graph as module

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = _optional_project_prefix()
    decision = {"kind": "project", "selector": {"owner": "east"}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point=decision,
    )
    clock = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    owner = module if checkpoint == "project_root" else OrToolsNonPlayPlayGraphGraftV1
    name = "project_graph_observation_rows_v1" if checkpoint == "project_root" else "graft"
    original = getattr(owner, name)

    def expire_after(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 16.0
        return result

    monkeypatch.setattr(owner, name, expire_after)
    with pytest.raises(NonPlayGraphCompilationStoppedV1) as stopped:
        compile_nonplay_complete_graph_v1(
            rules, normalized, prefix, decision, _request("bounded"),
            maximum_nodes=1000, maximum_depth=4, maximum_seconds=15,
            native_cp_sat=True,
        )
    assert stopped.value.reason == "maximum_seconds"
    assert stopped.value.nodes == 0 if checkpoint == "project_root" else stopped.value.nodes > 0


@pytest.mark.parametrize("unit", ("tricks", "round_end"))
def test_bid_play_horizon_reaches_play_without_spending_tricks_on_bids(
    rules: RuleBook, monkeypatch: pytest.MonkeyPatch, unit: str,
) -> None:
    raw = _escalation_prefix().to_dict()
    raw.pop("sha256")
    raw["auctionActions"] = [{"action": "pass"} for _ in range(7)]
    prefix = ExactTracePrefixV1.build(raw)
    decision = {"kind": "bid", "selector": {"actionIndex": 7}}
    normalized = OrToolsExactStateNormalizerV1(
        rules, native_cp_sat=True,
    ).normalize(prefix, decision_point=decision, maximum_seconds=15)
    reached = []

    def capture_play_request(graft, *, nonplay_depth, prefix_factory):
        reached.append(graft._adjusted_request(nonplay_depth).horizon)
        return 0

    # Stage-composition check: native play values are tested by the project
    # continuation fixture, not by substituting this callback as evidence.
    monkeypatch.setattr(
        OrToolsNonPlayPlayGraphGraftV1, "terminal_utility", capture_play_request,
    )
    compiled = compile_nonplay_complete_graph_v1(
        rules, normalized, prefix, decision, _request("bounded", unit=unit),
        maximum_nodes=1_000, maximum_depth=8, maximum_seconds=15,
        native_cp_sat=True,
    )
    assert reached and all(item == {"unit": unit, "value": 1} for item in reached)
    assert compiled.graph.scope == "bounded"
    assert compiled.graph.boundary_evaluator == "current_raw_point_margin_v1"


@pytest.mark.parametrize(
    "mode,unit",
    (("exact", "plies"), ("bounded", "plies"),
     ("bounded", "tricks"), ("bounded", "round_end")),
)
@pytest.mark.parametrize("native,opportunities", ((False, False), (True, False), (True, True)))
def test_shared_escalation_graph_matches_stage_compiler_and_defaults_unobserved(
    rules: RuleBook,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    unit: str,
    opportunities: bool,
    native: bool,
) -> None:
    prefix = _escalation_prefix()
    decision = {"kind": "escalation", "selector": {"actingSeat": "west"}}
    normalized = OrToolsExactStateNormalizerV1(
        rules, native_cp_sat=True
    ).normalize(prefix, decision_point=decision, maximum_seconds=15)
    request = _request(mode, unit=unit)
    orders = {canonical_sha256({"actions": []}): (Seat.WEST, Seat.EAST)} if opportunities else None
    depths = []

    # This test owns stage composition, not the exhaustive play tail. Returning
    # a plain terminal value leaves the shared graph at the same boundary as
    # the existing escalation compiler.
    monkeypatch.setattr(
        OrToolsNonPlayPlayGraphGraftV1,
        "terminal_utility",
        lambda _self, **kwargs: depths.append(kwargs["nonplay_depth"]) or 0,
    )
    actual = compile_nonplay_complete_graph_v1(
        rules,
        normalized,
        prefix,
        decision,
        request,
        maximum_nodes=1_000,
        maximum_depth=8,
        maximum_seconds=15,
        native_cp_sat=native,
        opportunity_orders=orders,
    )
    stage_options = (
        {
            "scope": "bounded",
            "horizon": 1,
            "boundary_evaluator": "current_raw_point_margin_v1",
            "boundary_utility": lambda _state: 0,
        }
        if mode == "bounded" and unit == "plies"
        else {"scope": "exact"}
    )
    expected = compile_escalation_decision_graph_v1(
        EscalationDomainIRV1.from_rules(rules.data),
        declarer=Seat.NORTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        acting_seat=Seat.WEST,
        terminal_utility=lambda _terminal: 0,
        budget=NonPlayGraphBudgetV1(1_000, 8),
        native_cp_sat=native,
        opportunity_orders=orders,
        **stage_options,
    )

    assert actual.graph.nodes == expected.nodes
    assert actual.graph.root_node_id == expected.root_node_id
    assert actual.graph.scope == mode
    assert actual.graph.boundary_evaluator == request.boundary_evaluator
    assert actual.decision_observation_ids == {}
    if mode == "exact":
        assert depths[0] == (2 if opportunities else 1)


@pytest.mark.parametrize("world_limit", [1, 2])
def test_observed_native_escalation_binds_generated_order_coverage(rules, world_limit, monkeypatch):
    from dataclasses import replace
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.ortools_nonplay_observation import build_nonplay_root_observer_v1
    from baloot_platform.research.ortools_research_executor import compile_observed_decision_v1
    def forbidden(*args, **kwargs):
        pytest.fail("native order coverage used interpreted escalation truth")
    for name in ("transition", "legal_decisions", "can_close"):
        monkeypatch.setattr(EscalationDomainIRV1, name, forbidden)
    prefix = _escalation_prefix()
    decision = {"kind": "escalation", "selector": {"actingSeat": "west"}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point=decision, maximum_seconds=15)
    raw = _request("bounded", unit="plies").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    request = ScenarioProofRequestV1.build(raw)
    if world_limit == 1:
        with pytest.raises(InformationSetCompilationStopped, match="maximum_order_worlds") as caught:
            compile_observed_decision_v1(rules, normalized, prefix, decision, request,
                maximum_nodes=1000, maximum_depth=8, maximum_worlds=world_limit,
                maximum_seconds=15, native_cp_sat=True)
        assert caught.value.partial_receipt["evidence"]["worlds"] == []
        return
    compiled = compile_observed_decision_v1(rules, normalized, prefix, decision, request,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=world_limit,
        maximum_seconds=15, native_cp_sat=True)
    coverage = compiled.graph.model_document["escalationOrderCoverage"]
    assert coverage["complete"] is True
    assert coverage["actingSeat"] == compiled.graph.root_actor == "west"
    assert coverage["orderWorldCount"] == 2
    assert coverage["rootHistoryScope"] == "actor_visible_private_histories"
    assert {tuple(row["priorNonAnnouncingSeats"]) for row in coverage["worlds"]} == {(), ("east",)}
    assert coverage["prefixHash"] == canonical_sha256({"actions": []})
    assert compiled.graph.analysis_scope["worldCoverageComplete"] is False
    assert len(compiled.graph.problem.worlds) == 2
    for history in ([], [{"action": "wrong", "seat": "north"}] * 4):
        damaged = replace(normalized, state={**normalized.state, "biddingHistory": history})
        with pytest.raises(ValueError, match="normalized auction history"):
            build_nonplay_root_observer_v1(damaged, prefix, "escalation")


@pytest.mark.real_clock
@pytest.mark.parametrize("reason", ["maximum_nodes", "maximum_depth", "cancelled", "native_transition_incomplete", "normalization"])
def test_observed_escalation_preserves_graph_stop_evidence(rules, monkeypatch, reason):
    import baloot_platform.research.ortools_research_executor as executor
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.ortools_nonplay_graphs import NonPlayGraphCompilationStoppedV1

    prefix = _escalation_prefix()
    decision = {"kind": "escalation", "selector": {"actingSeat": "west"}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point=decision, maximum_seconds=15)
    raw = _request("bounded", unit="plies").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    request = ScenarioProofRequestV1.build(raw)
    original = (ExactNormalizationStopped("maximum_seconds") if reason == "normalization"
                else NonPlayGraphCompilationStoppedV1(reason, nodes=7, depth=3))
    def stopped(*args, **kwargs):
        raise original
    monkeypatch.setattr(executor, "compile_nonplay_complete_graph_v1", stopped)
    with pytest.raises(InformationSetCompilationStopped) as caught:
        executor.compile_observed_decision_v1(rules, normalized, prefix, decision, request,
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=2,
            maximum_seconds=15, native_cp_sat=True)
    receipt = caught.value.partial_receipt
    assert caught.value.__cause__ is original
    assert receipt["status"] == "incomplete"
    assert receipt["exhaustionReason"] == ("maximum_seconds" if reason == "normalization" else reason)
    evidence = receipt["evidence"]
    assert evidence["requestHash"] == request.sha256
    assert evidence["decisionKind"] == "escalation"
    assert evidence["exactHash"] and evidence["prefixHash"]
    assert evidence["orderCoverage"]["complete"] is True
    assert evidence["orderWorldId"] == evidence["orderCoverage"]["worlds"][0]["worldId"]
    if reason == "normalization":
        assert evidence["normalizationReceipt"] == original.partial_receipt
    else:
        assert (evidence["nodes"], evidence["depth"]) == (7, 3)
    assert receipt["sha256"] == canonical_sha256({key: value for key, value in receipt.items() if key != "sha256"})


@pytest.mark.parametrize("selected", [None, "double:open"])
@pytest.mark.parametrize("use_budget", [False, True])
@pytest.mark.parametrize("prior", [(), ({"actor": "east", "actionCode": "decline"},)])
@pytest.mark.parametrize("visibility", ["actor_information_set", "complete_state"])
def test_native_quality_proof_binds_the_recorded_escalation_transition(rules, selected, root, use_budget, prior, visibility):
    import yaml
    from baloot_platform.research.ortools_research_executor import resolve_quality_escalation_opportunity_v1
    prefix = _escalation_prefix()
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility=visibility, strategyClass="pure_deterministic", targetTeam="west_east")
    if selected is not None:
        raw["selectedAction"] = selected
    request = ScenarioProofRequestV1.build(raw)
    profiles = None
    if use_budget:
        authored = yaml.safe_load((root / "configs/challenge/progressive_mining_ortools_v3.yaml").read_text(encoding="utf-8"))
        profiles = {"test": {**authored["scenarioResearchV4"]["proofBudgets"]["profiles"]["late_game_exact_v1"],
            "maximumNodes": 500, "maximumHiddenWorlds": 2}}
    result = resolve_quality_escalation_opportunity_v1(rules, prefix,
        {"kind": "escalation", "selector": {"actingSeat": "west"}}, request,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=3 if use_budget else 2, maximum_seconds=15,
        proof_budget_profiles=profiles, prior_private_events=prior,
        opportunity_order=("east", "west", "south", "north") if prior else ("west", "east", "south", "north"))
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    normalizer.normalize(prefix, decision_point={"kind": "escalation", "selector": {"actingSeat": "west"}})
    certificate = normalizer._deal_prefix_certificate
    repeated = resolve_quality_escalation_opportunity_v1(rules, prefix,
        {"kind": "escalation", "selector": {"actingSeat": "west"}}, request,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=3 if use_budget else 2, maximum_seconds=15,
        proof_budget_profiles=profiles, prior_private_events=prior, _normalizer=normalizer,
        opportunity_order=("east", "west", "south", "north") if prior else ("west", "east", "south", "north"))
    assert normalizer._deal_prefix_certificate is certificate
    assert repeated["sha256"] == result["sha256"]
    assert result["status"] == "recorded"
    assert result["scope"] == ("current_window_opportunity_only" if prior else "opening_opportunity_only")
    if prior:
        assert result["privateHistoryReplay"]["completedEvents"] == 1
        assert result["privateHistoryReplay"]["finalState"]["observation"]["actions"] == []
        assert result["transition"]["before"]["opportunityState"]["nonAnnouncingSeats"] == ["east"]
    assert result["sourcePrefixHash"] == prefix.sha256
    if use_budget:
        assert result["proofBudgetHash"]
        assert result["limits"]["maximumNodes"] == 500
        assert result["limits"]["maximumWorlds"] == 2
        profiles["test"]["maximumNodes"] = 1
        assert result["proofBudgetProfiles"]["test"]["maximumNodes"] == 500
    else:
        assert "proofBudgetHash" not in result
    quality = result["proofModel"]["actionQuality"]
    assert quality["scope"] == "compiled_" + visibility
    event = result["transition"]["event"]
    assert event["actor"] == "west"
    assert event["actionCode"] == quality["actualAction"] == (selected or "decline")
    assert quality["actualActionOptimal"] is True
    assert result["transition"]["status"] == "satisfied"
    assert result["proof"]["completionStatus"] == "bounded_complete"
    if selected is None:
        assert event["visibility"] == "private"
        assert result["nextPublicPrefix"] == prefix.to_dict()
    else:
        assert event["visibility"] == "public"
        assert len(result["nextPublicPrefix"]["escalationActions"]) == 1
    assert result["sha256"] == canonical_sha256({key: value for key, value in result.items() if key != "sha256"})


def test_escalation_opportunity_rejects_foreign_or_corrupted_normalizer(rules):
    from baloot_platform.research.ortools_research_executor import resolve_quality_escalation_opportunity_v1
    data = copy.deepcopy(rules.data)
    data["escalation"]["sun_score_threshold"] += 1
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    prefix = _escalation_prefix()
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic", targetTeam="west_east")
    request = ScenarioProofRequestV1.build(raw)
    def invoke(owned):
        return resolve_quality_escalation_opportunity_v1(rules, prefix,
            {"kind": "escalation", "selector": {"actingSeat": "west"}}, request,
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=2, maximum_seconds=15, _normalizer=owned)
    for foreign in (object(), OrToolsExactStateNormalizerV1(rules),
                    OrToolsExactStateNormalizerV1(RuleBook(data), native_cp_sat=True)):
        with pytest.raises(ValueError, match="owned native normalizer"):
            invoke(foreign)
    normalizer.normalize(prefix, decision_point={"kind": "escalation", "selector": {"actingSeat": "west"}})
    key, values, _digest = normalizer._deal_prefix_certificate
    normalizer._deal_prefix_certificate = (key, values, "0" * 64)
    with pytest.raises(ValueError, match="native deal certificate integrity mismatch"):
        invoke(normalizer)


def test_complete_state_escalation_quality_exact_exhaustion_is_incomplete(rules):
    from baloot_platform.research.ortools_research_executor import resolve_quality_escalation_continuation_v1
    result = resolve_quality_escalation_continuation_v1(
        rules, _escalation_prefix(), {"kind": "escalation", "selector": {"actingSeat": "west"}},
        _request("exact"), opportunity_order=("west", "east", "south", "north"),
        maximum_nodes=2, maximum_depth=1, maximum_worlds=1,
        maximum_seconds=15, maximum_decision_seconds=10, maximum_decisions=20,
        maximum_evidence_bytes=10_000_000,
    )
    assert result["status"] == "incomplete"
    assert result["partialReceipt"]["exhaustionReason"] in {"maximum_nodes", "maximum_depth"}
    assert result["partialReceipt"]["stage"] == "complete_state_quality"
    assert result["decisions"] == result["events"] == []
    assert result["artifacts"] == {}


def test_escalation_order_enumeration_has_explicit_bounded_scope(rules):
    from baloot_platform.research.ortools_nonplay_graphs import enumerate_escalation_opportunity_orders_v1
    kwargs = dict(declarer="north", contract="hokum", match_score=(0, 0), acting_seat="west",
                  maximum_worlds=2, maximum_seconds=15, budget=NonPlayGraphBudgetV1(1000, 8),
                  include_private_root_histories=True)
    domain = EscalationDomainIRV1.from_rules(rules.data)
    bounded = enumerate_escalation_opportunity_orders_v1(domain, maximum_announcement_depth=0, **kwargs)
    assert bounded["complete"] is True
    assert bounded["announcementDepthScope"] == 0
    assert bounded["orderWorldCount"] == 2
    assert len(bounded["windows"]) == 1
    assert {tuple(world["priorNonAnnouncingSeats"]) for world in bounded["worlds"]} == {(), ("east",)}
    full = enumerate_escalation_opportunity_orders_v1(domain, **kwargs)
    assert "announcementDepthScope" not in full
    assert full["complete"] is True and full["orderWorldCount"] == 2
    assert len(full["windows"]) > len(bounded["windows"])


def test_actual_private_history_does_not_condition_observer_proof(rules):
    from baloot_platform.research.ortools_research_executor import resolve_quality_escalation_opportunity_v1
    prefix = _escalation_prefix()
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic", targetTeam="west_east")
    request = ScenarioProofRequestV1.build(raw)
    outcomes = [resolve_quality_escalation_opportunity_v1(rules, prefix,
        {"kind": "escalation", "selector": {"actingSeat": "west"}}, request,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=2, maximum_seconds=15,
        prior_private_events=events) for events in ((), ({"actor": "east", "actionCode": "decline"},))]
    assert all(row["status"] == "recorded" for row in outcomes)
    assert outcomes[0]["graph"] == outcomes[1]["graph"]
    assert outcomes[0]["proofModel"] == outcomes[1]["proofModel"]
    assert outcomes[0]["transition"]["after"]["opportunityState"]["windowClosed"] is False
    assert outcomes[1]["transition"]["after"]["opportunityState"]["windowClosed"] is True


@pytest.mark.parametrize("recorded", [None, ({"actor": "west", "actionCode": "double:open"},
                                           {"actor": "north", "actionCode": "decline"}),
    ({"actor": "west", "actionCode": "decline"}, {"actor": "east", "actionCode": "double:open"},
     {"actor": "north", "actionCode": "decline"})])
@pytest.mark.parametrize("automatic_actor", [False, True])
@pytest.mark.parametrize("visibility", ["actor_information_set", "complete_state"])
def test_quality_escalation_driver_generates_and_audits_complete_suffix(rules, monkeypatch, recorded, automatic_actor, visibility):
    from baloot_platform.research.ortools_research_executor import resolve_quality_escalation_continuation_v1
    import baloot_platform.research.ortools_research_executor as executor
    opportunity, normalizers = executor.resolve_quality_escalation_opportunity_v1, []
    normalize_deal, deal_calls = OrToolsExactStateNormalizerV1._normalize_deal_auction, []
    def observed_deal(normalizer, *args, **kwargs):
        deal_calls.append((normalizer, normalizer._deal_prefix_certificate is None))
        return normalize_deal(normalizer, *args, **kwargs)
    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "_normalize_deal_auction", observed_deal)
    def observed(*args, **kwargs):
        normalizers.append(kwargs.get("_normalizer"))
        return opportunity(*args, **kwargs)
    monkeypatch.setattr(executor, "resolve_quality_escalation_opportunity_v1", observed)
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility=visibility, strategyClass="pure_deterministic")
    prefix, request = _escalation_prefix(), ScenarioProofRequestV1.build(raw)
    kwargs = dict(opportunity_order=("west", "east", "south", "north"), maximum_nodes=1000,
        maximum_depth=8, maximum_worlds=2, maximum_seconds=30, maximum_decision_seconds=15,
        maximum_decisions=20, maximum_evidence_bytes=10_000_000, recorded_events=recorded)
    if automatic_actor:
        kwargs["opportunity_order"] = ("north", "south", "west", "east")
    def run(**overrides):
        return resolve_quality_escalation_continuation_v1(rules, prefix,
            {"kind": "escalation", "selector": {} if automatic_actor else {"actingSeat": "west"}}, request, **{**kwargs, **overrides})
    checkpoints = []
    result = run(checkpoint_callback=checkpoints.append)
    assert result["status"] == "complete", result["partialReceipt"]
    assert normalizers and all(normalizer is normalizers[0] and normalizer is not None for normalizer in normalizers)
    own_calls = [cold for normalizer, cold in deal_calls if normalizer is normalizers[0]]
    assert len(own_calls) == len(normalizers) + 2  # Driver preflight and project/play handoff.
    assert sum(own_calls) == 1
    assert len(checkpoints) == len(result["decisions"])
    for checkpoint, decision in zip(checkpoints, result["decisions"], strict=True):
        assert checkpoint["stage"] == "escalation"
        assert checkpoint["decision"] == decision
        assert checkpoint["artifact"] == result["artifacts"][decision["artifactHash"]]
    from baloot_platform.research.result_authority import validate_native_escalation_quality_evidence_v1
    changed = copy.deepcopy(result)
    changed["privateRecall"] = {}
    if not result["privateRecall"]:
        changed["privateRecall"] = {"north": ["forged-window"]}
    changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
    with pytest.raises(ValueError, match="private recall differs"):
        validate_native_escalation_quality_evidence_v1(changed, source_prefix=prefix,
            rules_hash=rules.sha256, request=request)
    for field, replacement in (("actor", "south"), ("sourcePrefixHash", "0" * 64),
                               ("artifactHash", "0" * 64)):
        changed = copy.deepcopy(result)
        changed["decisions"][0][field] = replacement
        changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
        with pytest.raises(ValueError, match="escalation quality"):
            validate_native_escalation_quality_evidence_v1(changed, source_prefix=prefix,
                rules_hash=rules.sha256, request=request)
    assert result["scope"] == "escalation_suffix_only"
    if visibility == "complete_state":
        substituted = copy.deepcopy(result)
        first = substituted["decisions"][0]
        compact = substituted["artifacts"].pop(first["artifactHash"])
        compact["proofModel"]["kind"] = "native_escalation_selected_action_quality_model_v1"
        compact["proof"]["identities"]["modelHash"] = canonical_sha256(compact["proofModel"])
        compact["sha256"] = canonical_sha256({k: v for k, v in compact.items() if k != "sha256"})
        first["artifactHash"] = compact["sha256"]
        substituted["artifacts"][compact["sha256"]] = compact
        substituted["sha256"] = canonical_sha256({k: v for k, v in substituted.items() if k != "sha256"})
        with pytest.raises(ValueError, match="caller-owned frozen rules and search limits"):
            validate_native_escalation_quality_evidence_v1(substituted, source_prefix=prefix,
                rules_hash=rules.sha256, request=request)
        changed = copy.deepcopy(result)
        row = changed["decisions"][0]
        artifact = changed["artifacts"].pop(row["artifactHash"])
        allocation = artifact["allocationModel"]
        allocation["opportunityOrder"] = list(reversed(allocation["opportunityOrder"]))
        allocation["sha256"] = canonical_sha256({k: v for k, v in allocation.items() if k != "sha256"})
        artifact["sha256"] = canonical_sha256({k: v for k, v in artifact.items() if k != "sha256"})
        row["artifactHash"] = artifact["sha256"]
        changed["artifacts"][row["artifactHash"]] = artifact
        changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "sha256"})
        with pytest.raises(ValueError, match="complete-state context mismatch"):
            validate_native_escalation_quality_evidence_v1(changed, source_prefix=prefix,
                rules_hash=rules.sha256, request=request)
    if automatic_actor:
        assert result["initialActorSelection"] == {"kind": "native_eligible_order_v1", "actor": "west"}
        assert set(result["initialState"]["opportunityState"]["remainingEligibleSeats"]) == {"east", "west"}
    assert result["finalState"]["observation"]["terminal"] is True
    assert result["projectPlayHandoff"]["exactState"]["decisionKind"] == "play"
    assert result["projectPlayHandoff"]["normalizationReceipt"]["backendId"] == "ortools_cp_sat_v3"
    assert len(result["decisions"]) == len(result["events"]) == (2 if recorded is None else len(recorded))
    assert result["sha256"] == run()["sha256"]
    assert result["sha256"] == canonical_sha256({key: value for key, value in result.items() if key != "sha256"})
    audit = run(recorded_events=result["events"])
    assert audit["status"] == "complete"
    assert audit["finalPrefix"] == result["finalPrefix"]
    assert audit["privateRecall"] == result["privateRecall"]
    expected_recall = ({seat: [canonical_sha256({"actions": []})] for seat in ("west", "east")}
                       if recorded is None else {"north": [canonical_sha256({"actions": [[
                           "east" if len(recorded) == 3 else "west", "double", "open"]]})]})
    if recorded is not None and len(recorded) == 3:
        prior_recall = {"west": [canonical_sha256({"actions": []})]}
        expected_recall.update(prior_recall)
        evidence = [result["artifacts"][row["artifactHash"]]["allocationModel"] for row in result["decisions"]]
        assert all("privateRecallHash" not in item for item in evidence[:2])
        if visibility == "actor_information_set":
            coverage = evidence[2]["priorHistoryCoverage"]
            assert coverage["complete"] is True and coverage["worldCount"] == 2
            assert coverage["observerSeat"] == "north" and coverage["ownDeclinedWindows"] == []
        else:
            assert evidence[2]["priorWindowRecall"] == prior_recall
    assert result["privateRecall"] == expected_recall
    for decision in result["decisions"]:
        artifact = result["artifacts"][decision["artifactHash"]]
        assert artifact["proof"]["completionStatus"] == "bounded_complete"
        assert artifact["proofModel"]["actionQuality"]["actualActionOptimal"] is True
        assert artifact["request"]["targetTeam"] == Seat.parse(decision["actor"]).team.name.lower()
    if recorded is None:
        assert result["finalPrefix"] == prefix.to_dict()
        assert [row["actor"] for row in result["events"]] == ["west", "east"]
    else:
        assert len(result["finalPrefix"]["escalationActions"]) == 1
        assert result["events"] == list(recorded)


@pytest.mark.parametrize("assertions", [{}, {"currentTrick": None}])
def test_automatic_escalation_actor_is_absent_for_closed_sun(rules, assertions):
    from baloot_platform.research.ortools_research_executor import resolve_quality_escalation_continuation_v1

    raw = _escalation_prefix().to_dict()
    raw.pop("sha256")
    raw["auctionActions"] = [{"action": "sun"}]
    raw["stateAssertions"] = assertions
    request = _request("bounded").to_dict()
    request.pop("sha256")
    request.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    result = resolve_quality_escalation_continuation_v1(
        rules, ExactTracePrefixV1.build(raw), {"kind": "escalation"}, ScenarioProofRequestV1.build(request),
        opportunity_order=("north", "south", "west", "east"), maximum_nodes=1000,
        maximum_depth=8, maximum_worlds=1, maximum_seconds=30, maximum_decision_seconds=15,
        maximum_decisions=20, maximum_evidence_bytes=10_000_000)
    assert result["status"] == "complete", result["partialReceipt"]
    assert result["initialActorSelection"] == {"kind": "native_eligible_order_v1", "actor": None}
    assert result["decisions"] == result["events"] == []
    assert result["projectPlayHandoff"] is not None


@pytest.mark.real_clock
@pytest.mark.parametrize("limit,reason,count", [("maximum_decisions", "maximum_decisions", 1),
    ("maximum_evidence_bytes", "maximum_evidence_bytes", 0), ("cancel", "cancelled", 0),
    ("global_time", "maximum_seconds", 0), ("runtime", "maximum_seconds", 0)])
def test_escalation_driver_preserves_incomplete_prefix(rules, limit, reason, count):
    from baloot_platform.research.ortools_research_executor import resolve_quality_escalation_continuation_v1
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    kwargs = dict(maximum_decisions=20, maximum_evidence_bytes=10_000_000)
    if limit == "runtime":
        kwargs["runtime_maximum_seconds"] = 1e-12
    elif limit != "global_time":
        kwargs.update({"cancel_check": lambda: True} if limit == "cancel" else {limit: 1})
    prefix = _escalation_prefix()
    result = resolve_quality_escalation_continuation_v1(rules, prefix,
        {"kind": "escalation", "selector": {"actingSeat": "west"}}, ScenarioProofRequestV1.build(raw),
        opportunity_order=("west", "east", "south", "north"), maximum_nodes=1000,
        maximum_depth=8, maximum_worlds=2, maximum_seconds=1e-12 if limit == "global_time" else 30,
        maximum_decision_seconds=15, **kwargs)
    assert result["status"] == "incomplete"
    assert result["partialReceipt"]["exhaustionReason"] == reason
    assert len(result["events"]) == count
    assert result["finalPrefix"] == prefix.to_dict()


@pytest.mark.parametrize("events", [(), ({"actor": "south", "actionCode": "decline"},),
    ({"actor": "west", "actionCode": "decline", "visibility": "public"},)])
def test_escalation_driver_never_repairs_recorded_events(rules, events):
    from baloot_platform.research.ortools_research_executor import resolve_quality_escalation_continuation_v1
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    def run():
        return resolve_quality_escalation_continuation_v1(rules, _escalation_prefix(),
            {"kind": "escalation", "selector": {"actingSeat": "west"}}, ScenarioProofRequestV1.build(raw),
            opportunity_order=("west", "east", "south", "north"), maximum_nodes=1000,
            maximum_depth=8, maximum_worlds=2, maximum_seconds=30, maximum_decision_seconds=15,
            maximum_decisions=20, maximum_evidence_bytes=10_000_000, recorded_events=events)
    if not events:
        result = run()
        assert result["status"] == "incomplete"
        assert result["partialReceipt"]["exhaustionReason"] == "recorded_events_exhausted"
        assert result["events"] == []
    else:
        with pytest.raises(ValueError):
            run()


@pytest.mark.parametrize("failure", ["budget", "model", "readback"])
def test_escalation_driver_retains_prior_proof_when_next_decision_fails(rules, monkeypatch, failure):
    import baloot_platform.research.ortools_research_executor as executor
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    original, calls = executor.resolve_quality_escalation_opportunity_v1, []
    def second_fails(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            if failure in {"budget", "readback"}:
                raise InformationSetCompilationStopped("maximum_worlds", stage="test", evidence={"queriesUsed": 7})
            raise RuntimeError("test model failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(executor, "resolve_quality_escalation_opportunity_v1", second_fails)
    if failure == "readback":
        from baloot_platform.research import result_authority
        def stopped_readback(*args, **kwargs):
            raise InformationSetCompilationStopped("cancelled", stage="quality_readback")
        monkeypatch.setattr(result_authority, "_validate_native_escalation_quality_evidence_v1", stopped_readback)
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    result = executor.resolve_quality_escalation_continuation_v1(rules, _escalation_prefix(),
        {"kind": "escalation", "selector": {"actingSeat": "west"}}, ScenarioProofRequestV1.build(raw),
        opportunity_order=("west", "east", "south", "north"), maximum_nodes=1000,
        maximum_depth=8, maximum_worlds=2, maximum_seconds=30, maximum_decision_seconds=15,
        maximum_decisions=20, maximum_evidence_bytes=10_000_000)
    assert len(result["events"]) == len(result["artifacts"]) == 1
    assert result["status"] == "incomplete"
    if failure == "model":
        assert result["partialReceipt"]["failureClass"] == "runtime_error"
    if failure == "readback":
        receipt = result["partialReceipt"]
        assert receipt["exhaustionReason"] == "cancelled"
        assert receipt["evidence"]["sourceReceipt"]["exhaustionReason"] == "maximum_worlds"
        assert receipt["evidence"]["sourceReceipt"]["evidence"] == {"queriesUsed": 7}
        assert receipt["evidence"]["readbackReceipt"]["stage"] == "quality_readback"
    assert result["partialReceipt"]
    assert result["finalState"]["opportunityState"]["windowClosed"] is False


@pytest.mark.parametrize("project_actions", [False, True])
def test_closed_sun_escalation_hands_off_without_fabricating_decisions(rules, monkeypatch, project_actions):
    import baloot_platform.research.ortools_research_executor as executor
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    value = _escalation_prefix().to_dict()
    value.pop("sha256")
    value["auctionActions"] = [{"action": "sun"}]
    if project_actions:
        value["projectActions"] = []
    else:
        value["stateAssertions"] = {"currentTrick": None}
    prefix = ExactTracePrefixV1.build(value)
    def forbidden(*args, **kwargs):
        raise AssertionError("closed stage must not create a decision proof")
    monkeypatch.setattr(executor, "resolve_quality_escalation_opportunity_v1", forbidden)
    result = executor.resolve_quality_escalation_continuation_v1(rules, prefix,
        {"kind": "escalation", "selector": {"actingSeat": "west"}}, ScenarioProofRequestV1.build(raw),
        opportunity_order=("west", "east", "south", "north"), maximum_nodes=1000,
        maximum_depth=8, maximum_worlds=2, maximum_seconds=30, maximum_decision_seconds=15,
        maximum_decisions=20, maximum_evidence_bytes=10_000_000)
    assert result["status"] == "complete", result["partialReceipt"]
    assert result["initialState"]["observation"]["terminal"] is True
    assert result["events"] == result["decisions"] == []
    assert result["artifacts"] == {}
    expected = prefix.to_dict()
    expected.pop("sha256")
    expected["stateAssertions"] = {}
    assert result["finalPrefix"] == ExactTracePrefixV1.build(expected).to_dict()
    assert result["initialNormalization"]["backendId"] == "ortools_cp_sat_v3"
    handoff = result["projectPlayHandoff"]["exactState"]
    assert handoff["decisionKind"] == ("project" if project_actions else "play")
    assert handoff["exactStateId"] and handoff["legalActions"]


@pytest.mark.real_clock
def test_escalation_handoff_exhaustion_retains_completed_proofs(rules, monkeypatch):
    import baloot_platform.research.ortools_research_executor as executor
    def exhausted(*args, **kwargs):
        raise ExactNormalizationStopped("maximum_seconds")
    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize_play_or_project", exhausted)
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    result = executor.resolve_quality_escalation_continuation_v1(rules, _escalation_prefix(),
        {"kind": "escalation", "selector": {"actingSeat": "west"}}, ScenarioProofRequestV1.build(raw),
        opportunity_order=("west", "east", "south", "north"), maximum_nodes=1000,
        maximum_depth=8, maximum_worlds=2, maximum_seconds=30, maximum_decision_seconds=15,
        maximum_decisions=20, maximum_evidence_bytes=10_000_000)
    assert result["status"] == "incomplete"
    assert result["finalState"]["observation"]["terminal"] is True
    assert len(result["events"]) == len(result["artifacts"]) == 2
    assert result["projectPlayHandoff"] is None
    assert result["partialReceipt"]["exhaustionReason"] == "maximum_seconds"


@pytest.mark.parametrize("events", [({"actor": "east", "actionCode": "double:open"},),
    ({"actor": "west", "actionCode": "decline"},), ({"actor": "east", "actionCode": "decline"},) * 2])
def test_quality_private_replay_rejects_public_repeated_or_spent_opportunities(rules, events):
    from baloot_platform.research.ortools_research_executor import resolve_quality_escalation_opportunity_v1
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic", targetTeam="west_east")
    with pytest.raises(ValueError):
        resolve_quality_escalation_opportunity_v1(rules, _escalation_prefix(),
            {"kind": "escalation", "selector": {"actingSeat": "west"}}, ScenarioProofRequestV1.build(raw),
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=2, maximum_seconds=15,
            prior_private_events=events)


@pytest.mark.parametrize("field", ["request_hash", "graph_hash", "selected_action"])
def test_escalation_quality_rejects_substituted_proof_before_transition(rules, monkeypatch, field):
    from dataclasses import replace
    import baloot_platform.research.ortools_research_executor as executor
    prefix = _escalation_prefix()
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic", targetTeam="west_east")
    request = ScenarioProofRequestV1.build(raw)
    original = executor.prove_actor_information_set_v1
    def substituted(*args, **kwargs):
        return replace(original(*args, **kwargs), **{field: "substituted"})
    def forbidden(*args, **kwargs):
        raise AssertionError("mismatched proof must not record a transition")
    monkeypatch.setattr(executor, "prove_actor_information_set_v1", substituted)
    monkeypatch.setattr(executor.OrToolsEscalationStageV1, "transition_opportunity", forbidden)
    with pytest.raises(ValueError, match="action-quality evidence"):
        executor.resolve_quality_escalation_opportunity_v1(rules, prefix,
            {"kind": "escalation", "selector": {"actingSeat": "west"}}, request,
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=2, maximum_seconds=15)


@pytest.mark.parametrize("outcome", ["false_claim", "model_error", "cancel_after_proof"])
def test_escalation_quality_retains_proof_without_recording_on_failure(rules, monkeypatch, outcome):
    from dataclasses import replace
    import baloot_platform.research.ortools_research_executor as executor
    prefix = _escalation_prefix()
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic", targetTeam="west_east")
    if outcome == "false_claim":
        raw["successThreshold"] = 100_000
    request = ScenarioProofRequestV1.build(raw)
    cancelled = [False]
    original = executor.prove_actor_information_set_v1
    def prove(*args, **kwargs):
        result = original(*args, **kwargs)
        if outcome == "cancel_after_proof":
            cancelled[0] = True
        return replace(result, completion_status="model_error") if outcome == "model_error" else result
    def forbidden(*args, **kwargs):
        raise AssertionError("unproven or cancelled quality must not record a transition")
    monkeypatch.setattr(executor, "prove_actor_information_set_v1", prove)
    monkeypatch.setattr(executor.OrToolsEscalationStageV1, "transition_opportunity", forbidden)
    result = executor.resolve_quality_escalation_opportunity_v1(rules, prefix,
        {"kind": "escalation", "selector": {"actingSeat": "west"}}, request,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=2, maximum_seconds=15,
        cancel_check=lambda: cancelled[0])
    assert result["status"] == {"false_claim": "proof_failed", "model_error": "model_error",
                               "cancel_after_proof": "incomplete"}[outcome]
    assert result["proof"] and result["proofModel"]
    assert result["transition"] is None and result["nextPublicPrefix"] is None
    if outcome == "cancel_after_proof":
        assert result["proof"]["completionStatus"] == "bounded_complete"
        assert result["partialReceipt"]["exhaustionReason"] == "cancelled"


@pytest.mark.real_clock
def test_escalation_quality_cpu_cap_is_not_mislabeled_user_cancellation(rules, root, monkeypatch):
    import yaml
    import baloot_platform.research.ortools_research_executor as executor
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    authored = yaml.safe_load((root / "configs/challenge/progressive_mining_ortools_v3.yaml").read_text(encoding="utf-8"))
    profiles = {"test": {**authored["scenarioResearchV4"]["proofBudgets"]["profiles"]["late_game_exact_v1"],
        "maximumCpuSeconds": 1}}
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic", targetTeam="west_east")
    clock = [0.0]
    monkeypatch.setattr(executor.time, "process_time", lambda: clock[0])
    def exhausted(*args, **kwargs):
        assert kwargs["cancel_check"]() is False
        clock[0] = 2.0
        assert kwargs["cancel_check"]() is True
        raise InformationSetCompilationStopped("cancelled", stage="fixture_cp_solve")
    monkeypatch.setattr(executor, "compile_observed_decision_v1", exhausted)
    with pytest.raises(InformationSetCompilationStopped) as caught:
        executor.resolve_quality_escalation_opportunity_v1(rules, _escalation_prefix(),
            {"kind": "escalation", "selector": {"actingSeat": "west"}}, ScenarioProofRequestV1.build(raw),
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=2, maximum_seconds=15,
            proof_budget_profiles=profiles)
    assert caught.value.reason == "maximum_cpu_seconds"
    assert caught.value.partial_receipt["evidence"]["proofBudgetHash"]
    assert caught.value.partial_receipt["evidence"]["sourceReceipt"]["exhaustionReason"] == "cancelled"


def _optional_project_prefix() -> ExactTracePrefixV1:
    return ExactTracePrefixV1.build(
        {
            "schemaVersion": 1,
            "deckOrder": list(range(32)),
            "dealer": "east",
            "roundNumber": 1,
            "matchScore": [0, 0],
            "auctionActions": [{"actor": "north", "action": "sun"}],
            "escalationActions": [],
            "projectActions": [
                {"actor": actor, "action": "pass"}
                for actor in ("north", "west", "south")
            ],
            "projectDeclarations": [],
            "projectAssertions": [],
            "playActions": [
                {"actor": "north", "card": "S7", "ekka": False},
                {"actor": "west", "card": "S10", "ekka": False},
                {"actor": "south", "card": "SK", "ekka": False},
            ],
            "stateAssertions": {},
        }
    )


@pytest.mark.parametrize("leader", [Seat.NORTH, Seat.WEST])
def test_native_pending_project_seed_binds_award_to_first_trick(root, leader):
    from ortools.sat.python import cp_model
    from baloot_platform.core.scoring_ir import ScoringDomainIRV1
    from baloot_platform.research.ortools_complete_state_graph import (
        CompleteStateGraphSeedV1, OrToolsCompleteStateGraphCompilerV1,
    )
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")
    hands = ('S7 S8 S9 HA SQ D10 C10 DA', 'H7 H8 H9 SA SK DJ CJ CA',
             'D7 D8 D9 HK HQ DQ CQ H10', 'C7 C8 C9 SJ HJ DK CK S10')
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST, buyer=Seat.NORTH)
    deck = [None] * 32
    for seat in Seat:
        for position, card in zip(layout.final_hand_positions[seat], hands[seat].split(), strict=True):
            deck[position] = Card.parse(card).id
    prefix = ExactTracePrefixV1.build({'schemaVersion': 1, 'deckOrder': deck, 'dealer': 'east',
        'auctionActions': [{'actor': 'north', 'action': 'sun'}],
        'escalationActions': [], 'projectDeclarations': [], 'playActions': []})
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={'kind': 'play'}, maximum_seconds=30)
    graft = OrToolsNonPlayPlayGraphGraftV1(rules, _request('bounded'),
        maximum_nodes=100, maximum_depth=4, maximum_seconds=30, native_cp_sat=True)
    start, seed = graft.native_project_play_seed(normalized, prefix)
    assert start.project_evidence is None
    assert seed.awarded_project_points is None
    assert seed.pending_projects['declarations']
    assert CompleteStateGraphSeedV1.from_dict(seed.to_dict()) == seed
    changed = copy.deepcopy(seed.to_dict())
    changed['pendingProjects']['declarations'][0][1] += 1
    with pytest.raises(ValueError, match='noncanonical proof seed'):
        CompleteStateGraphSeedV1.from_dict(changed)
    compiler = OrToolsCompleteStateGraphCompilerV1(play_domain=rules.play_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data), structural_domain=rules.structural_domain,
        project_domain=rules.project_domain, native_conversion=True)
    parsed = compiler._parse_seed_start(seed, expected_rules_hash=rules.sha256)
    oracle = compiler._path_oracle(parsed, target=leader.team, utility='awarded_point_margin',
        play_step_count=32)
    model = oracle.build.model
    model.add(oracle.build._extract.winner_seats[0] == int(leader))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 30
    assert solver.solve(model) == cp_model.OPTIMAL
    evidence = oracle.terminal_project_fragment.extract(solver)
    assert evidence.winning_team == leader.team
    awarded = [project for project in evidence.projects if project.awarded]
    assert {project.owner for project in awarded} == {leader, leader.partner}
    assert evidence.combined_awarded_value[int(leader.team)] == sum(project.value for project in awarded) > 0
    assert evidence.combined_awarded_value[int(leader.team.other)] == 0
    assert oracle.scoring_fragment is not None
    assert tuple(solver.value(value) for value in oracle.scoring_fragment.inputs.project_points) == evidence.combined_awarded_value


@pytest.mark.real_clock
@pytest.mark.parametrize("stop", ["cancelled", "expired", "construction", "unknown"])
def test_project_lifecycle_graft_budget_receipt(root, monkeypatch, stop):
    from baloot_platform.research import ortools_nonplay_play_graft as module
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    prefix = _optional_project_prefix()
    normalized = normalizer.normalize(prefix, decision_point={"kind": "project", "selector": {"owner": "east"}})
    prefix = normalizer.append_normalized_action(prefix, normalized, "pass")
    normalized = normalizer.normalize(prefix, decision_point={"kind": "play"})
    clock = [0.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    graft = OrToolsNonPlayPlayGraphGraftV1(
        rules, _request("bounded"), maximum_nodes=100, maximum_depth=4,
        maximum_seconds=10, native_cp_sat=True, cancel_check=lambda: stop == "cancelled")
    if stop == "expired":
        clock[0] = 10
    builds, solves = [], []
    original = module.attach_project_lifecycle_fragment_v1
    def build(*args, **kwargs):
        builds.append(1)
        result = original(*args, **kwargs)
        if stop == "construction":
            clock[0] = 10
        return result
    def solve(*args):
        solves.append(1)
        return module.cp_model.UNKNOWN
    monkeypatch.setattr(module, "attach_project_lifecycle_fragment_v1", build)
    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", solve)
    with pytest.raises(ExactNormalizationStopped) as caught:
        graft._native_project_lifecycle_evidence(normalized, prefix)
    receipt = caught.value.partial_receipt
    assert receipt["exhaustionReason"] == (stop if stop in {"cancelled", "unknown"} else "maximum_seconds")
    assert receipt["evidence"]["sourceStage"] == "nonplay_project_lifecycle"
    assert builds == ([] if stop in {"cancelled", "expired"} else [1])
    assert solves == ([1] if stop == "unknown" else [])
    if stop == "unknown":
        assert len(receipt["evidence"]["modelHash"]) == 64
        assert receipt["evidence"]["prefixHash"] == prefix.sha256
    assert receipt["sha256"] == canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})


@pytest.mark.parametrize("declare", [False, True])
def test_native_graft_compaction_preserves_project_lifecycle(root, monkeypatch, declare):
    from baloot_platform.research import ortools_nonplay_play_graft as module
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    prefix = _optional_project_prefix()
    normalized = normalizer.normalize_play_or_project(prefix)
    action = next(row["code"] for row in normalized.legal_actions
                  if (row["code"] != "pass") == declare)
    prefix = normalizer.append_normalized_action(prefix, normalized, action)
    normalized = normalizer.normalize_play_or_project(prefix)
    original = module.attach_project_lifecycle_fragment_v1
    sizes = []

    def measured(model, **kwargs):
        fragment = original(model, **kwargs)
        sizes.append((len(model.proto.variables), len(model.proto.constraints)))
        return fragment

    monkeypatch.setattr(module, "attach_project_lifecycle_fragment_v1", measured)
    evidence = []
    for native in (False, True):
        graft = OrToolsNonPlayPlayGraphGraftV1(rules, _request("bounded"),
            maximum_nodes=100, maximum_depth=4, maximum_seconds=10, native_cp_sat=native)
        document = graft._native_project_lifecycle_evidence(normalized, prefix).to_dict()
        evidence.append({key: value for key, value in document.items()
                         if key not in {"fragmentHash", "evidenceHash"}})
    assert evidence[0] == evidence[1]
    assert all(compact < reference for compact, reference in zip(sizes[1], sizes[0], strict=True))


@pytest.mark.real_clock
@pytest.mark.parametrize("reason", ["maximum_seconds", "maximum_nodes", "maximum_depth", "cancelled", None])
def test_nonplay_continuation_budget_preserves_reason(rules, reason):
    request = _request("bounded", horizon=2)
    graft = OrToolsNonPlayPlayGraphGraftV1(
        rules, request, maximum_nodes=10, maximum_depth=4, maximum_seconds=30,
        native_cp_sat=True, cancel_check=lambda: reason == "cancelled",
    )
    budget = dict(seconds=0 if reason == "maximum_seconds" else 1,
                  nodes=0 if reason == "maximum_nodes" else 1,
                  depth=1 if reason == "maximum_depth" else 2,
                  required_depth=2, stage="project_continuation")
    if reason is None:
        graft._check_budget(**budget)
        return
    with pytest.raises(ExactNormalizationStopped) as stopped:
        graft._check_budget(**budget)
    receipt = stopped.value.partial_receipt
    assert receipt["exhaustionReason"] == reason
    assert receipt["evidence"]["sourceStage"] == "project_continuation"
    assert receipt["evidence"]["requestHash"] == request.sha256
    assert receipt["evidence"]["remainingNodes"] == budget["nodes"]
    assert receipt["evidence"]["remainingDepth"] == budget["depth"]
    assert receipt["sha256"] == canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})


@pytest.mark.parametrize("stage", ["entry", "normalization", "play", "exhaustion"])
def test_native_project_graph_propagates_cancellation(root, monkeypatch, stage):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = _optional_project_prefix()
    decision = {"kind": "project", "selector": {"owner": "east"}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix, decision_point=decision)
    cancelled = stage == "entry"
    visited = []

    def cancel_check():
        return cancelled

    owner = OrToolsExactStateNormalizerV1 if stage == "normalization" else OrToolsCompleteStateGraphCompilerV1
    method = "normalize" if stage == "normalization" else "compile"
    original = getattr(owner, method)

    def stop_at_stage(*args, **kwargs):
        nonlocal cancelled
        assert kwargs.get("cancel_check" if stage == "normalization" else "cancelled") is cancel_check
        visited.append(stage)
        if stage == "exhaustion":
            kwargs["maximum_nodes"] = 1
        else:
            cancelled = True
        return original(*args, **kwargs)

    if stage != "entry":
        monkeypatch.setattr(owner, method, stop_at_stage)
    if stage == "normalization":
        # Keep cancellation coverage for the generic fallback; the final-play
        # parent-evidence route deliberately performs no normalization.
        monkeypatch.setattr(OrToolsNonPlayPlayGraphGraftV1, "_final_project_play_graph",
                            lambda *args, **kwargs: None)
    with pytest.raises(ExactNormalizationStopped) as stopped:
        compile_nonplay_complete_graph_v1(
            rules, normalized, prefix, decision, _request("bounded", horizon=2),
            maximum_nodes=1000, maximum_depth=4, maximum_seconds=30,
            native_cp_sat=True, cancel_check=cancel_check,
        )
    if stage == "exhaustion":
        receipt = stopped.value.partial_receipt
        assert receipt["evidence"]["sourceStage"] == "final_project_play"
        cause = receipt["evidence"]["causeReceipt"]
        assert cause["status"] == "incomplete"
        assert receipt["exhaustionReason"] == cause["exhaustionReason"] == "maximum_nodes"
        assert cause["sha256"] == canonical_sha256({key: value for key, value in cause.items() if key != "sha256"})
    else:
        assert stopped.value.partial_receipt["exhaustionReason"] == "cancelled"
    assert visited == ([] if stage == "entry" else [stage])


@pytest.mark.real_clock
@pytest.mark.parametrize("played,forced", [
    *((played, forced) for played in (1, 2, 3) for forced in (False, True)),
    (1, "segment"),
])
@pytest.mark.parametrize("compact", [False, True])
def test_forced_frontier_matches_generic_native_graph(root, monkeypatch, forced, played, compact):
    production_observed = forced == "segment" or (played == 1 and forced is False) or played in (2, 3)
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix().to_dict()
    raw.pop("sha256")
    if forced:
        raw["deckOrder"][1], raw["deckOrder"][9] = raw["deckOrder"][9], raw["deckOrder"][1]
        if played < 3 and forced != "segment":
            raw["deckOrder"][13], raw["deckOrder"][29] = raw["deckOrder"][29], raw["deckOrder"][13]
        if played == 1:
            raw["deckOrder"][12], raw["deckOrder"][27] = raw["deckOrder"][27], raw["deckOrder"][12]
    raw["projectActions"] = raw["projectActions"][:played]
    raw["playActions"] = raw["playActions"][:played]
    prefix = ExactTracePrefixV1.build(raw)
    decision = {"kind": "project", "selector": {"owner": ("north", "west", "south", "east")[played]}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point=decision)
    method = ("_third_project_parent_rows" if played == 2 else
              "_forced_project_sequence_graph" if played < 3 else "_final_project_play_graph")
    original = getattr(OrToolsNonPlayPlayGraphGraftV1, method)
    hits = []
    hit_calls = []
    normalizations = []
    original_normalize = OrToolsExactStateNormalizerV1.normalize

    def counted_normalize(self, *args, **kwargs):
        normalizations.append(1)
        return original_normalize(self, *args, **kwargs)

    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize", counted_normalize)
    original_count = OrToolsNonPlayPlayGraphGraftV1._replacement_node_count
    count_checks = []

    def checked_count(self, base, replacements):
        count = original_count(self, base, replacements)
        assembled = self._replace_leaves(base, replacements)
        assert count == self._node_count(assembled)
        count_checks.append(count)
        return count

    monkeypatch.setattr(OrToolsNonPlayPlayGraphGraftV1, "_replacement_node_count", checked_count)

    def measured(self, *args, **kwargs):
        graph = original(self, *args, **kwargs)
        if graph is not None:
            hits.append(graph)
            hit_calls.append((self, args, kwargs))
        return graph

    def compile_graph():
        from baloot_platform.research.ortools_nonplay_observation import (
            build_nonplay_root_observer_v1, build_nonplay_decision_observer_v1,
        )
        base_factory = _base_observer_factory if played == 3 else None
        play_factory = _play_observer_factory if played == 3 else None
        if production_observed:
            def base_factory(value, exact):
                return build_nonplay_root_observer_v1(value, exact, "project")

            def play_factory(value, start):
                return build_nonplay_decision_observer_v1(
                    rules=rules, normalized_exact=value, exact_prefix=start.prefix, decision_kind="project",
                    projects=start.projects, native_cp_sat=True, project_evidence=start.project_evidence)
        return compile_nonplay_complete_graph_v1(
            rules, normalized, prefix, decision, _request("bounded", horizon=2 * (4 - played)),
            maximum_nodes=1000, maximum_depth=6, maximum_seconds=30,
            native_cp_sat=True, _compact_forced=compact,
            base_observer_factory=base_factory,
            play_observer_factory=play_factory)

    monkeypatch.setattr(OrToolsNonPlayPlayGraphGraftV1, method, measured)
    fast = compile_graph()
    assert hits
    fast_normalizations = len(normalizations)
    normalizations.clear()
    monkeypatch.setattr(OrToolsNonPlayPlayGraphGraftV1, method, lambda *args, **kwargs: None)
    reference = compile_graph()
    if hits:
        assert fast_normalizations < len(normalizations)

    def paths(compiled):
        graph = compiled.graph
        nodes = {node.node_id: node for node in graph.nodes}
        def walk(node_id):
            node = nodes[node_id]
            if node.is_leaf:
                return (node.leaf_kind, node.utility)
            return (node.actor_seat,
                    compiled.decision_observation_ids.get(node_id)
                    if production_observed else None, tuple(
                ({key: value for key, value in edge.to_dict().items() if key != "childNodeId"},
                 walk(edge.child_node_id)) for edge in node.edges))
        if played == 3 or forced == "segment" or (played == 1 and forced is False):
            assert set(compiled.decision_observation_ids) == {
                node.node_id for node in graph.nodes if not node.is_leaf}
        return graph.scope, graph.boundary_evaluator, graph.utility_scale, walk(graph.root_node_id)

    assert paths(fast) == paths(reference)
    assert count_checks
    if played == 1 and forced is True:
        from baloot_platform.research.ortools_proof import OrToolsScenarioGraphProofV1
        counts = []
        for compiled in (fast, reference):
            nodes = {node.node_id: node for node in compiled.graph.nodes}
            selected = nodes[compiled.graph.root_node_id].edges[0]
            raw_request = _request("bounded", horizon=6).to_dict()
            raw_request.pop("sha256")
            raw_request.update(lineUniverse="all_legal", selectedAction=selected.action_code)
            proof = OrToolsScenarioGraphProofV1().prove(compiled.graph, ScenarioProofRequestV1.build(raw_request))
            def count_paths(node_id):
                node = nodes[node_id]
                return 1 if node.is_leaf else sum(count_paths(edge.child_node_id) for edge in node.edges)
            expected = count_paths(selected.child_node_id)
            assert proof.continuation_lines.complete
            assert proof.continuation_lines.exact_count == str(expected)
            counts.append(expected)
        assert counts[0] == counts[1]
        owner, args, kwargs = hit_calls[0]
        graph = hits[0][0]
        values = frozenset(node.utility for node in graph.nodes if node.is_leaf)
        limited = (*args[:4], owner._node_count(graph) - len(values), *args[5:])
        # Reuse the real compiled branch: only already retained matching leaves
        # may offset the available node slots, not arbitrary claimed overlap.
        owner.deadline = time.perf_counter() + 30
        shared = original(owner, *limited, **{**kwargs, "shared_boundary_values": values})
        assert shared[0].sha256 == graph.sha256
        owner.deadline = time.perf_counter() + 30
        with pytest.raises(ExactNormalizationStopped):
            original(owner, *limited, **{**kwargs, "shared_boundary_values": frozenset({999999})})


def test_complete_state_pending_projects_prepare_every_terminal_branch(root, monkeypatch):
    from baloot_platform.config import build_bundle
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1)
    from baloot_platform.research.ortools_research_executor import _complete_state_quality_artifact_v1
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix().to_dict()
    raw.pop("sha256")
    raw["playActions"] = raw["playActions"][:2]
    raw["deckOrder"][1], raw["deckOrder"][9] = raw["deckOrder"][9], raw["deckOrder"][1]
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    profile = copy.deepcopy(build_bundle(root, bundle_path="configs/bundle_intelligent_research.yaml").component(
        "challenge")["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"])
    profile["nativeObserved"]["maximumIterations"] = 1
    budget = ScenarioProofBudgetV1.build("test", profile)
    request = _request("exact")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    worlds, contexts, preparation = compiler._prepare_complete_state_play(exact, prefix, request,
        budget=budget, maximum_seconds=30)
    assert len(worlds) == len(contexts) == len(preparation["branches"]) == 2
    assert preparation["remainingActions"] == 31  # 30 cards plus one declaration.
    assert preparation["coverage"] == "exact_fixed_deal_terminal_tail"
    coverage = preparation["projectCoverage"]
    assert coverage["queries"][-1]["status"] == "infeasible"
    assert {sequence[0]["action"] for sequence in coverage["sequences"]} == {"pass", "declare"}
    assert prefix.project_actions == tuple(raw["projectActions"])
    for state, oracle, recall in worlds:
        assert state == exact.state and recall == {}
        assert len(oracle.build.selected_play_indexes) == 32
        assert oracle.boundary_evaluator is None
        assert oracle._query_utility() is oracle.terminal_utility_variable
        assert oracle.scoring_fragment.inputs.project_points == oracle.project_lifecycle.combined_awarded_value
    with pytest.raises(ValueError, match="every legal declaration"):
        compiler._solve_observed_worlds(worlds[:1], worlds[0][1].target,
            maximum_iterations=1, maximum_queries=100, maximum_prefixes=1000,
            maximum_seconds=5, project_contexts={worlds[0][1].model_hash: contexts[worlds[0][1].model_hash]})
    with pytest.raises(InformationSetCompilationStopped) as depth:
        _complete_state_quality_artifact_v1(rules, exact, prefix, {"kind": "play"}, request,
            maximum_nodes=100_000, maximum_depth=30, maximum_seconds=30, cancel_check=None, budget=budget)
    assert depth.value.reason == "maximum_depth"
    # Production must use these owned branches, not enumerate the full-round DAG.
    with monkeypatch.context() as patch:
        patch.setattr(OrToolsActorInformationSetCompilerV1, "_prepare_complete_state_play",
                      lambda *args, **kwargs: (worlds, contexts, preparation))
        patch.setattr("baloot_platform.research.ortools_research_executor.compile_nonplay_complete_graph_v1",
                      lambda *args, **kwargs: pytest.fail("pending projects routed to full-tree enumeration"))
        with pytest.raises(InformationSetCompilationStopped) as incomplete:
            _complete_state_quality_artifact_v1(rules, exact, prefix, {"kind": "play"}, request,
                maximum_nodes=100_000, maximum_depth=31, maximum_seconds=30, cancel_check=None, budget=budget)
    assert incomplete.value.reason == "maximum_iterations"
    assert incomplete.value.partial_receipt["evidence"]["preparation"] == preparation


@pytest.mark.real_clock
@pytest.mark.parametrize("reason", ["cancelled", "maximum_seconds"])
def test_observed_play_preparation_stops_during_model_construction(root, monkeypatch, reason):
    from types import SimpleNamespace
    import baloot_platform.research.ortools_information_set_compiler as module

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix().to_dict()
    raw.pop("sha256")
    raw["playActions"] = raw["playActions"][:1]
    raw["projectActions"] = raw["projectActions"][:2]
    prefix = ExactTracePrefixV1.build(raw)
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    request = _request("bounded", unit="tricks").to_dict()
    request.pop("sha256")
    request.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    compiler = module.OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    clock, cancelled = [0.0], [False]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def build(*args, **kwargs):
        cancelled[0] = reason == "cancelled"
        clock[0] = 2.0 if reason == "maximum_seconds" else 0.0
        assert callable(kwargs.get("cancel_check")), "model construction lacks the owned budget callback"
        kwargs["cancel_check"]()
        pytest.fail("model construction continued after its budget stopped")

    monkeypatch.setattr(compiler.graph_compiler.play_stage, "build_model", build)
    with pytest.raises(module.InformationSetCompilationStopped) as caught:
        compiler._prepare_observed_play_worlds(normalized, ScenarioProofRequestV1.build(request),
            maximum_worlds=1, maximum_attempts=100, maximum_model_bytes=30_000_000,
            maximum_seconds=1, cancel_check=lambda: cancelled[0], project_prefix=prefix,
            project_contexts={}, _world_inputs=((normalized, prefix),))
    assert caught.value.reason == reason
    assert caught.value.partial_receipt["stage"] == "native_play_preparation"
    assert caught.value.partial_receipt["evidence"] == {
        "sourceStateHash": normalized.state_hash, "preparedModels": [], "modelBytes": 0,
    }


@pytest.mark.parametrize("declare", [False, True])
def test_interleaved_terminal_project_score_matches_exact_normalization(root, declare, record_property):
    from ortools.sat.python import cp_model
    from baloot_platform.core.enums import Team
    from baloot_platform.core.project_ir import ProjectDeclarationActionV1
    from baloot_platform.core.scoring_ir import ScoringDomainIRV1
    from baloot_platform.research.ortools_scoring_stage import extract_scoring_evidence_v1

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix().to_dict()
    raw.pop("sha256")
    raw["playActions"] = raw["playActions"][:2]
    raw["deckOrder"][1], raw["deckOrder"][9] = raw["deckOrder"][9], raw["deckOrder"][1]
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    normalized = normalizer.normalize_play_or_project(prefix)
    coverage = normalizer._enumerate_project_continuations(normalized, prefix,
        maximum_sequences=2, maximum_seconds=10)
    action = next(ProjectDeclarationActionV1.build(sequence[0]) for sequence in coverage["sequences"]
                  if (sequence[0]["action"] == "declare") == declare)
    domain = ScoringDomainIRV1.from_rules(rules.data)
    compiler = OrToolsCompleteStateGraphCompilerV1(play_domain=rules.play_domain,
        scoring_domain=domain, structural_domain=rules.structural_domain,
        project_domain=rules.project_domain, native_conversion=True)
    start = compiler._parse_start(normalized, expected_rules_hash=rules.sha256, request=_request("bounded"))
    oracle = compiler._path_oracle(start, target=Team.NORTH_SOUTH, utility="awarded_point_margin",
        play_step_count=32, project_prefix=prefix, project_continuation_actions=(action,),
        project_counterfactual=True)
    # Alwaleed green p99 is 62.9s; 3x margin rounded up to 240s.
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    witness = oracle.build.solve(maximum_time_seconds=240, solver=solver)
    score = extract_scoring_evidence_v1(solver, oracle.scoring_fragment, domain, contract=start.contract)
    raw["projectActions"].append(action.to_dict())
    raw["playActions"] = [play.to_dict() for play in witness.plays]
    complete = normalizer.normalize_play_or_project(ExactTracePrefixV1.build(raw))
    assert complete.phase == "scored"
    assert score == complete.normalization_receipt.to_dict()["derivedScoring"]["score"]
    assert bool(sum(score["projectPoints"])) == declare
    assert solver.value(oracle.terminal_utility_variable) == score["gamePoints"][0] - score["gamePoints"][1]
    # On this optional-project fixture, all-pass declarations independently reproduce
    # the scoring intervention without changing any card or play action.
    counter_raw = copy.deepcopy(raw)
    counter_raw["projectActions"] = [{"actor": item["actor"], "action": "pass"}
                                     for item in raw["projectActions"]]
    counter_complete = normalizer.normalize_play_or_project(ExactTracePrefixV1.build(counter_raw))
    counter_score = counter_complete.normalization_receipt.to_dict()["derivedScoring"]["score"]
    assert counter_score["projectPoints"] == [0, 0]
    assert solver.value(oracle.project_counterfactual_utility_variable) == (
        counter_score["gamePoints"][0] - counter_score["gamePoints"][1])
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1,
    )
    request_raw = _request("bounded", horizon=8, unit="tricks").to_dict()
    request_raw.pop("sha256")
    request_raw["visibility"] = "actor_information_set"
    request_raw["strategyClass"] = "pure_deterministic"
    request = ScenarioProofRequestV1.build(request_raw)
    observed = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    preparation_arguments = dict(maximum_worlds=1, maximum_attempts=2,
        maximum_model_bytes=30_000_000, maximum_seconds=240, project_prefix=prefix,
        _world_inputs=((normalized, prefix),))
    with pytest.raises(InformationSetCompilationStopped) as refused:
        observed._prepare_observed_play_worlds(normalized, request,
            **preparation_arguments, project_contexts={}, maximum_horizon_actions=30)
    assert refused.value.reason == "maximum_horizon_actions"
    contexts = {}
    worlds, preparation = observed._prepare_observed_play_worlds(normalized, request,
        **preparation_arguments, project_contexts=contexts, maximum_horizon_actions=31, maximum_depth=31)
    assert len(worlds) == len(contexts) == 2
    assert preparation["coverage"] == "supplied_worlds_only"
    for _, branch, _ in worlds:
        assert branch._query_utility() is branch.terminal_utility_variable
        assert branch.scoring_fragment.inputs.project_points == branch.project_lifecycle.combined_awarded_value
        assert len(branch.build.selected_play_indexes) == 32
    if not declare:
        with pytest.raises(InformationSetCompilationStopped) as incomplete:
            observed._solve_observed_worlds(worlds, Team.NORTH_SOUTH,
                maximum_iterations=1, maximum_queries=100, maximum_prefixes=1000,
                maximum_seconds=240, project_contexts=contexts)
        assert incomplete.value.reason == "maximum_iterations"
        evidence = incomplete.value.partial_receipt["evidence"]
        assert evidence["scope"]["utilityScope"]["playStepCount"] == 32
        assert evidence["scope"]["coverage"] == "supplied_worlds_only"
        assert evidence["resources"]["usage"]["counterexampleQueries"] >= len(worlds)
        assert all(evidence["resources"]["usage"]["cachedPrefixesByBranch"].values())
        assert evidence["resources"]["usage"]["observations"] > 0
        started = time.monotonic()
        try:
            diagnostic = observed._solve_observed_worlds(worlds, Team.NORTH_SOUTH,
                maximum_iterations=1000, maximum_queries=1000, maximum_prefixes=1000,
                maximum_seconds=240, project_contexts=contexts)
            status = diagnostic["status"]
            assert status == "converged"
        except InformationSetCompilationStopped as stopped:
            assert stopped.reason in {"maximum_seconds", "maximum_iterations", "maximum_queries",
                                      "maximum_prefixes", "maximum_observations", "unknown"}
            diagnostic = stopped.partial_receipt["evidence"]
            status = "incomplete:" + stopped.reason
        import json
        record_property("terminalStrategyDiagnostic", json.dumps({
            "status": status, "seconds": time.monotonic() - started,
            "scope": diagnostic["scope"], "resources": diagnostic["resources"],
            "value": diagnostic.get("value"), "diagnosticOnly": True,
        }, sort_keys=True))


@pytest.mark.parametrize("forced", [False, True])
def test_final_parent_play_rejects_unowned_or_illegal_declaration(root, forced, monkeypatch):
    from baloot_platform.core.scoring_ir import ScoringDomainIRV1
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix().to_dict()
    raw.pop("sha256")
    if forced:
        raw["deckOrder"][1], raw["deckOrder"][9] = raw["deckOrder"][9], raw["deckOrder"][1]
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    parent = normalizer.normalize_play_or_project(prefix)
    child = normalizer.append_normalized_action(prefix, parent, "pass")
    request = _request("bounded")
    graft = OrToolsNonPlayPlayGraphGraftV1(rules, request, maximum_nodes=1000,
        maximum_depth=4, maximum_seconds=30, native_cp_sat=True, root_decision_kind="project")
    compiler = OrToolsCompleteStateGraphCompilerV1(play_domain=rules.play_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        structural_domain=rules.structural_domain, project_domain=rules.project_domain,
        native_conversion=True)
    kwargs = dict(compiler=compiler, maximum_nodes=1000, maximum_depth=4, graph_cache={})
    graph = graft._final_project_play_graph(parent, child, request, **kwargs)
    assert graph is not None
    root_node = next(node for node in graph.nodes if node.node_id == graph.root_node_id)
    assert (len(root_node.edges) == 1) is forced
    if not forced:
        assert len(parent.legal_actions) > 1
        monkeypatch.setattr(compiler, "compile", lambda *args, **kwargs:
                            pytest.fail("same parent raw frontier was compiled twice"))
        for action in parent.legal_actions:
            alternative = normalizer.append_normalized_action(prefix, parent, action["code"])
            reused = graft._final_project_play_graph(parent, alternative, request, **kwargs)
            assert reused.sha256 == graph.sha256
        with pytest.raises(ExactNormalizationStopped) as exhausted:
            graft._final_project_play_graph(parent, child, request, **{**kwargs, "maximum_nodes": 1})
        assert exhausted.value.partial_receipt["exhaustionReason"] == "maximum_nodes"
    illegal = child.to_dict()
    illegal.pop("sha256")
    illegal["projectActions"][-1] = {"actor": "east", "action": "declare", "projectId": "PROJECT-0000000000000000"}
    with pytest.raises(ValueError, match="uncertified declaration"):
        graft._final_project_play_graph(parent, ExactTracePrefixV1.build(illegal), request, **kwargs)
    unowned = child.to_dict()
    unowned.pop("sha256")
    unowned["matchScore"] = [1, 0]
    with pytest.raises(ValueError, match="parent differs from declaration prefix"):
        graft._final_project_play_graph(parent, ExactTracePrefixV1.build(unowned), request, **kwargs)
    assert graft._final_project_play_graph(parent, child, _request("bounded", horizon=2), **kwargs) is None


@pytest.mark.parametrize("forced", [False, True])
def test_native_path_oracle_attaches_only_forced_project_checkpoints(root, forced):
    from baloot_platform.core.enums import Team
    from baloot_platform.core.scoring_ir import ScoringDomainIRV1
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix().to_dict()
    raw.pop("sha256")
    raw["projectActions"] = raw["projectActions"][:2]
    raw["playActions"] = raw["playActions"][:2]
    raw["deckOrder"][1], raw["deckOrder"][9] = raw["deckOrder"][9], raw["deckOrder"][1]
    if forced:
        raw["deckOrder"][13], raw["deckOrder"][29] = raw["deckOrder"][29], raw["deckOrder"][13]
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    project = normalizer.normalize_play_or_project(prefix)
    prefix = normalizer.append_normalized_action(prefix, project, "pass")
    normalized = normalizer.normalize_play_or_project(prefix)
    request = _request("bounded")
    compiler = OrToolsCompleteStateGraphCompilerV1(play_domain=rules.play_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data), structural_domain=rules.structural_domain,
        project_domain=rules.project_domain, native_conversion=True)
    start = compiler._parse_start(normalized, expected_rules_hash=rules.sha256, request=request)
    arguments = dict(target=Team.NORTH_SOUTH, utility=request.utility,
                     play_step_count=4, project_prefix=prefix)
    if not forced:
        with pytest.raises(ValueError, match="optional declarations"):
            compiler._path_oracle(start, **arguments)
        from baloot_platform.core.project_ir import ProjectDeclarationActionV1
        from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
        advanced = normalizer.append_normalized_action(prefix, normalized, normalized.legal_actions[0]["code"])
        pending = normalizer.normalize_play_or_project(advanced)
        assert pending.decision_kind == "project" and len(pending.legal_actions) == 2
        coverage = normalizer._enumerate_project_continuations(normalized, prefix,
            maximum_sequences=2, maximum_seconds=10)
        assert coverage["status"] == "complete" and coverage["queries"][-1]["status"] == "infeasible"
        assert coverage["sha256"] == canonical_sha256(
            {key: value for key, value in coverage.items() if key != "sha256"})
        assert {ProjectDeclarationActionV1.build(sequence[0]).code for sequence in coverage["sequences"]} == {
            row["code"] for row in pending.legal_actions}
        assert coverage == normalizer._enumerate_project_continuations(normalized, prefix,
            maximum_sequences=2, maximum_seconds=10)
        with pytest.raises(ExactNormalizationStopped) as limited:
            normalizer._enumerate_project_continuations(normalized, prefix, maximum_sequences=1, maximum_seconds=10)
        assert limited.value.partial_receipt["exhaustionReason"] == "maximum_sequences"
        assert len(limited.value.partial_receipt["evidence"]["sequences"]) == 1
        with pytest.raises(ExactNormalizationStopped) as cancelled:
            normalizer._enumerate_project_continuations(normalized, prefix, maximum_sequences=2,
                maximum_seconds=10, cancel_check=lambda: True)
        assert cancelled.value.partial_receipt["exhaustionReason"] == "cancelled"
        project_id = ProjectDeclarationActionV1.build(
            next(row["code"] for row in pending.legal_actions if row["code"] != "pass"),
            actor=pending.next_actor).project_id
        model_hashes = set()
        observation_hashes = set()
        policy_worlds, project_contexts = [], {}
        for sequence in coverage["sequences"]:
            action = ProjectDeclarationActionV1.build(sequence[0], actor=pending.next_actor)
            oracle = compiler._path_oracle(start, **arguments,
                boundary_evaluator="current_raw_point_margin_v1", project_continuation_actions=(action,))
            model_hashes.add(oracle.model_hash)
            _, upper = oracle.query_utility_bounds()
            excluded, utility, excluded_hash = oracle.find_path_below(upper + 1,
                conditional_actions={(): None}, maximum_seconds=10, cancelled=None)
            assert excluded is None and utility is None and excluded_hash != oracle.model_hash
            from ortools.sat.python import cp_model
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            witness = oracle.build.solve(maximum_time_seconds=10, solver=solver)
            variable = next(project.declared for project in oracle.project_checkpoints[3].fragment.projects
                            if project.project_id == project_id)
            assert solver.boolean_value(variable) == (action.action == "declare")
            from baloot_platform.research.ortools_nonplay_observation import _build_native_project_checkpoint_observer
            from baloot_platform.research.ortools_complete_state_graph import CompleteStateDecisionObservationV1
            from baloot_platform.research.ortools_play_tail import TailPrefixActionV1
            from dataclasses import replace
            played = advanced.play_actions[-1]
            common_path = (TailPrefixActionV1.build(played["actor"], played["card"], ekka=played.get("ekka", False)),)
            derived, native_prefix = oracle.solve(common_path, maximum_seconds=10, cancelled=None)
            observe = _build_native_project_checkpoint_observer(normalized, prefix, path_oracle=oracle)
            value = CompleteStateDecisionObservationV1(derived.next_actor, common_path, native_prefix,
                derived.remaining_hands, derived.current_trick, derived.trick_winners,
                derived.trick_raw_points, (0, 0), (0, 0), derived.project_checkpoint, derived.native_play)
            observation_hashes.add(observe(value))
            with pytest.raises(ValueError, match="play evidence mismatch"):
                observe(replace(value, native_play=replace(derived.native_play, model_hash="0" * 64),
                    project_checkpoint=replace(derived.project_checkpoint, model_hash="0" * 64)))
            from baloot_platform.research.ortools_exact_state import _render_native_play_prefix
            from baloot_platform.core.exact_state_view import canonical_observer_information_hash_v1
            projected = copy.deepcopy(dict(normalized.state))
            marker = next(index for index, event in enumerate(projected["events"])
                          if event["type"] == "project_holdings_detected")
            projected["events"] = projected["events"][:marker]
            projected["trumpBroken"] = False
            _render_native_play_prefix(projected, play=derived.native_play,
                projection=derived.project_checkpoint, trump=None, project_actions=prefix.project_actions)
            for seat in Seat:
                assert canonical_observer_information_hash_v1(complete_state=projected,
                    acting_seat=seat.name.lower()) == canonical_observer_information_hash_v1(
                    complete_state=pending.state, acting_seat=seat.name.lower())
            with pytest.raises(ValueError, match="state identity mismatch"):
                _build_native_project_checkpoint_observer(replace(normalized, state_hash="0" * 64),
                    prefix, path_oracle=oracle)
            raw = prefix.to_dict()
            raw.pop("sha256")
            raw["projectActions"].append(action.to_dict())
            raw["playActions"] = [play.to_dict() for play in witness.plays]
            complete = normalizer.normalize_play_or_project(ExactTracePrefixV1.build(raw))
            assert len(complete.state["completedTricks"]) == 1
            lifecycle = complete.normalization_receipt.to_dict()["derivedProjectPrefix"]["lifecycleById"]
            assert lifecycle[project_id]["declared"] == (action.action == "declare")
            policy_oracle = compiler._path_oracle(start, **{**arguments, "target": action.actor.team},
                boundary_evaluator="current_raw_point_margin_v1", project_continuation_actions=(action,))
            policy_compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
            deeper = compiler._path_oracle(start, **{**arguments, "play_step_count": 8},
                boundary_evaluator="current_raw_point_margin_v1", project_continuation_actions=(action,))
            deeper_witness = deeper.build.solve(maximum_time_seconds=10, solver=cp_model.CpSolver())
            later_observe = _build_native_project_checkpoint_observer(normalized, prefix, path_oracle=deeper)
            for length in range(4, 8):
                path = tuple(deeper_witness.plays[deeper.base_length:length])
                later, later_prefix = deeper.solve(path, maximum_seconds=10, cancelled=None)
                later_value = CompleteStateDecisionObservationV1(later.next_actor, path, later_prefix,
                    later.remaining_hands, later.current_trick, later.trick_winners, later.trick_raw_points,
                    (0, 0), (0, 0), later.project_checkpoint, later.native_play)
                reference_raw = prefix.to_dict()
                reference_raw.pop("sha256")
                reference_raw["projectActions"].append(action.to_dict())
                reference_raw["playActions"] = [play.to_dict() for play in deeper_witness.plays[:length]]
                reference = normalizer.normalize_play_or_project(ExactTracePrefixV1.build(reference_raw))
                assert later_observe(later_value) == canonical_observer_information_hash_v1(
                    complete_state=reference.state, acting_seat=reference.next_actor)
            cached_observations = {}
            first_bindings = policy_compiler._observed_policy_bindings(normalized.state, deeper, deeper_witness,
                action.actor.team, {}, {}, known_domains={}, project_context=(normalized, prefix, coverage),
                maximum_seconds=10, _prefix_observations=cached_observations)
            assert any(row[0] is not None and row[2] is not None
                       for row in cached_observations.values())
            def forbidden_projection(_value):
                raise AssertionError("cached project observation was reprojected")
            assert policy_compiler._observed_policy_bindings(normalized.state, deeper, deeper_witness,
                action.actor.team, {}, {}, known_domains=first_bindings[1],
                project_context=(normalized, prefix, coverage), maximum_seconds=10,
                _known_prefix_observations=cached_observations,
                _project_observer=forbidden_projection) == first_bindings
            project_observation, choice = next((row[0], row[1])
                for row in cached_observations.values() if row[0] is not None and row[2] is not None)
            other = next(code for code in first_bindings[1][project_observation] if code != choice)
            opposing_policy = {project_observation: other}
            uncached_opposition = policy_compiler._observed_policy_bindings(
                normalized.state, deeper, deeper_witness, action.actor.team,
                opposing_policy, {}, known_domains=first_bindings[1],
                project_context=(normalized, prefix, coverage), maximum_seconds=10)
            assert policy_compiler._observed_policy_bindings(
                normalized.state, deeper, deeper_witness, action.actor.team,
                opposing_policy, {}, known_domains=first_bindings[1],
                project_context=(normalized, prefix, coverage), maximum_seconds=10,
                _known_prefix_observations=cached_observations,
                _project_observer=forbidden_projection) == uncached_opposition
            policy_worlds.append((normalized.state, policy_oracle, {}))
            project_contexts[policy_oracle.model_hash] = (normalized, prefix, coverage)
            _, ceiling = policy_oracle.query_utility_bounds()
            for reject_branch in (False, True):
                policy, domains = {}, {}
                for _ in range(30):
                    result = policy_compiler._find_observed_policy_counterexample(normalized.state,
                        policy_oracle, action.actor.team, policy, {}, threshold=ceiling + 1,
                        known_domains=domains, maximum_queries=100, maximum_prefixes=1000, maximum_seconds=10,
                        project_context=(normalized, prefix, coverage))
                    if result["status"] != "policy_extension_required":
                        break
                    domains = result["domains"]
                    for observation, options in domains.items():
                        policy[observation] = (next(code for code in options if code != action.code)
                            if reject_branch else action.code) if "pass" in options else options[0]
                assert result["status"] == ("no_counterexample" if reject_branch else "counterexample")
                assert all(query["projectCoverageHash"] == coverage["sha256"] for query in result["queries"])
                if reject_branch:
                    from baloot_platform.research.ortools_complete_state_graph import iter_native_path_query_guards_v1
                    assert any(binding is None for _, guards in iter_native_path_query_guards_v1(
                        result["queries"], maximum_bindings=1000) for binding in guards.values())
                else:
                    assert action.code in {code for _, code in result["targetActions"]}
            with pytest.raises(ValueError, match="owned complete declaration coverage"):
                policy_compiler._find_observed_policy_counterexample(normalized.state, policy_oracle,
                    action.actor.team, {}, {}, threshold=-10000, known_domains={}, maximum_queries=1,
                    maximum_prefixes=1000, maximum_seconds=10,
                    project_context=(normalized, prefix, {**coverage, "status": "incomplete"}))
            with pytest.raises(ValueError, match="declaration coverage context"):
                policy_compiler._find_observed_policy_counterexample(normalized.state, policy_oracle,
                    action.actor.team, {}, {}, threshold=-10000, known_domains={}, maximum_queries=1,
                    maximum_prefixes=1000, maximum_seconds=10)
            with pytest.raises(ValueError, match="strategic policy coverage"):
                OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._solve_observed_worlds(
                    [(normalized.state, oracle, {})], Team.NORTH_SOUTH, maximum_iterations=10,
                    maximum_queries=10, maximum_prefixes=1000, maximum_seconds=10)
        assert len(model_hashes) == 2
        assert len(observation_hashes) == 1  # Future pass/declare is not yet observable.
        proof_arguments = dict(maximum_iterations=100, maximum_queries=500, maximum_prefixes=1000,
            maximum_seconds=30, project_contexts=project_contexts)
        observer_cache = {}
        for world_state, branch_oracle, _ in policy_worlds:
            context = project_contexts[branch_oracle.model_hash]
            observer = policy_compiler._project_policy_observer(world_state, branch_oracle, context,
                cache=observer_cache)
            assert policy_compiler._project_policy_observer(world_state, branch_oracle, context,
                cache=observer_cache) is observer
            # A cache hit must not bypass fresh declaration-coverage ownership.
            with pytest.raises(ValueError, match="owned complete declaration coverage"):
                policy_compiler._project_policy_observer(world_state, branch_oracle,
                    (*context[:2], {**context[2], "status": "incomplete"}), cache=observer_cache)
        assert len(observer_cache) == len(project_contexts)
        solved = policy_compiler._solve_observed_worlds(policy_worlds, action.actor.team, **proof_arguments)
        assert solved["status"] == "converged" and solved["worldCount"] == 1 and solved["branchCount"] == 2
        assert len(solved["scope"]["worlds"]) == 1 and len(solved["scope"]["branches"]) == 2
        assert policy_compiler._solve_observed_worlds(list(reversed(policy_worlds)), action.actor.team,
            **proof_arguments) == solved
        for _, branch_oracle, _ in policy_worlds:
            assert branch_oracle.enumerate_paths(maximum_nodes=1000, maximum_seconds=10, cancelled=None) is None
        root_values = {}
        for root_action in start.expected_legal_actions:
            # One opponent play, then target declaration and target play.
            root_values[root_action.card.code] = max(max(state.boundary_utility
                for path, state in branch_oracle.results.items() if len(path) == 2 and path[0] == root_action)
                for _, branch_oracle, _ in policy_worlds)
        assert solved["value"] == min(root_values.values())
        for root_code, value in root_values.items():
            fixed = policy_compiler._solve_observed_worlds(policy_worlds, action.actor.team,
                selected_root_action=root_code, **proof_arguments)
            assert fixed["value"] == value
        with pytest.raises(ValueError, match="every legal declaration sequence"):
            policy_compiler._solve_observed_worlds(policy_worlds[:1], action.actor.team,
                **{**proof_arguments, "project_contexts": {policy_worlds[0][1].model_hash:
                    project_contexts[policy_worlds[0][1].model_hash]}})
        recalled = (*policy_worlds, (normalized.state, policy_worlds[0][1], {action.actor.name.lower(): ("0" * 64,)}))
        with pytest.raises(ValueError, match="every legal declaration sequence"):
            policy_compiler._solve_observed_worlds(recalled, action.actor.team, **proof_arguments)
        with pytest.raises(ValueError, match="pending actor order"):
            compiler._path_oracle(start, **arguments, project_continuation_actions=())
        return
    oracle = compiler._path_oracle(start, **arguments)
    from dataclasses import replace
    from baloot_platform.core.enums import Suit
    from baloot_platform.research.ortools_exact_state import (
        OrToolsExactNormalizationReceiptV1, _native_forced_pass_certificate,
    )
    certified_arguments = {**arguments, "project_normalization": normalized}
    certificate_copy = _native_forced_pass_certificate(normalized)
    original_certificate = copy.deepcopy(certificate_copy)
    certificate_copy["seats"].clear()
    assert _native_forced_pass_certificate(normalized) == original_certificate
    certified = compiler._path_oracle(start, **certified_arguments)
    assert certified.model_hash != oracle.model_hash
    for altered in (replace(start, rules_hash="0" * 64),
                    replace(start, contract=Contract.HOKUM, trump=Suit.CLUBS)):
        with pytest.raises(ValueError, match="project normalization differs"):
            compiler._path_oracle(altered, **certified_arguments)
    owners = dict(start.ownership)
    north = next(card for card, seat in owners.items() if seat == Seat.NORTH)
    east = next(card for card, seat in owners.items() if seat == Seat.EAST)
    owners[north], owners[east] = owners[east], owners[north]
    with pytest.raises(ValueError, match="project normalization differs from graph ownership"):
        compiler._path_oracle(replace(start, ownership=owners), **certified_arguments)
    for field, value, message in (
        ("keepAllFeasibleSolutionsInPresolve", False, "feasibility preservation"),
        ("feasibilityOnly", False, "feasibility preservation"),
        ("modelHash", "0" * 64, "ownership mismatch"),
    ):
        receipt = normalized.normalization_receipt.to_dict()
        receipt["nativePlayNormalization"]["conditionalForcedProjectPasses"][field] = value
        receipt.pop("sha256")
        receipt["sha256"] = canonical_sha256(receipt)
        with pytest.raises(ValueError, match=message):
            _native_forced_pass_certificate(replace(normalized,
                normalization_receipt=OrToolsExactNormalizationReceiptV1(receipt)))
    assert set(oracle.project_checkpoints) == {3}
    binding = oracle.project_checkpoints[3]
    assert binding.project_phase_open and len(binding.auto_events) == 3
    from ortools.sat.python import cp_model
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    witness = oracle.build.solve(maximum_time_seconds=10, solver=solver)
    assert [solver.value(value) for value in binding.auto_events] == [0, 0, 0]
    assert [solver.value(value) for value in binding.declared_events] == [0, 0, 0]
    derived, _ = oracle.solve(tuple(witness.plays[2:3]), maximum_seconds=10, cancelled=None)
    assert derived.project_checkpoint is not None and derived.native_play is not None
    assert derived.project_checkpoint.uniqueness_model_hash is None
    assert derived.project_checkpoint.uniqueness_solver_status == "not_requested"
    from baloot_platform.research import ortools_exact_state as exact_module
    from baloot_platform.core.exact_state_view import canonical_observer_information_hash_v1
    state = copy.deepcopy(dict(normalized.state))
    marker = next(index for index, event in enumerate(state["events"])
                  if event["type"] == "project_holdings_detected")
    state["events"] = state["events"][:marker]
    state["trumpBroken"] = False
    exact_module._render_native_play_prefix(state, play=derived.native_play,
        projection=derived.project_checkpoint, trump=None, project_actions=prefix.project_actions)
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["playActions"] = [{"actor": play.actor.name.lower(), "card": play.card.id, "ekka": play.ekka}
                          for play in derived.native_play.plays[:3]]
    reference = normalizer.normalize_play_or_project(ExactTracePrefixV1.build(raw))
    for seat in Seat:
        assert canonical_observer_information_hash_v1(complete_state=state, acting_seat=seat.name.lower()) == canonical_observer_information_hash_v1(
            complete_state=reference.state, acting_seat=seat.name.lower())
    from baloot_platform.research.ortools_nonplay_play_graft import selected_projects_from_exact_receipt_v1
    graft = OrToolsNonPlayPlayGraphGraftV1(rules, request, maximum_nodes=1000,
        maximum_depth=4, maximum_seconds=30, native_cp_sat=True)
    seed = graft._seed(normalized, selected_projects_from_exact_receipt_v1(normalized),
        awarded_project_points=tuple(normalized.normalization_receipt.to_dict()["derivedProjects"]["combinedAwardedValue"]),
        project_evidence_hash=normalized.normalization_receipt.sha256)
    observed = []

    def observe(value):
        if len(value.prefix) == 3:
            assert value.project_checkpoint is not None
            assert value.native_play is not None and value.native_play.prefix_length == 3
            assert value.project_checkpoint.model_hash == value.native_play.model_hash
            observed.append(value)
        return canonical_sha256([action.to_dict() for action in value.prefix])

    compiled = compiler.compile(seed, _request("bounded", horizon=2), expected_rules_hash=rules.sha256,
        maximum_nodes=1000, maximum_depth=2, maximum_seconds=30, decision_observer=observe,
        _project_prefix=prefix)
    assert compiled.complete and observed


@pytest.mark.real_clock
def test_project_continuation_normalization_deadline_is_incomplete(root, monkeypatch):
    from baloot_platform.research import ortools_nonplay_complete_graph as module
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix().to_dict()
    raw.pop("sha256")
    raw["projectActions"] = raw["projectActions"][:2]
    raw["playActions"] = raw["playActions"][:2]
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    normalized = normalizer.normalize_play_or_project(prefix)
    clock = module.time.perf_counter
    offset = [0]
    original = OrToolsExactStateNormalizerV1.normalize_play_or_project

    def finish_after_deadline(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        offset[0] = 31
        return result

    monkeypatch.setattr(module.time, "perf_counter", lambda: clock() + offset[0])
    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize_play_or_project", finish_after_deadline)
    with pytest.raises(ExactNormalizationStopped) as stopped:
        compile_nonplay_complete_graph_v1(
            rules, normalized, prefix, {"kind": "project", "selector": {"owner": "south"}},
            _request("bounded", horizon=4), maximum_nodes=1000, maximum_depth=6,
            maximum_seconds=30, native_cp_sat=True)
    receipt = stopped.value.partial_receipt
    assert receipt["status"] == "incomplete"
    assert receipt["exhaustionReason"] == "maximum_seconds"
    assert receipt["evidence"]["sourceStage"] == "project_continuation_normalization"
    assert receipt["evidence"]["remainingSeconds"] <= 0
    assert receipt["sha256"] == canonical_sha256({key: value for key, value in receipt.items() if key != "sha256"})


def _base_observer_factory(_normalized, _prefix):
    def observe(value) -> str:
        return canonical_sha256(
            {
                "stage": value.stage,
                "actor": value.actor.name.lower(),
                "publicState": value.public_state,
            }
        )

    return observe


def _play_observer_factory(_normalized, _start):
    def observe(value) -> str:
        return canonical_sha256(
            {
                "actor": value.actor.name.lower(),
                "path": [action.to_dict() for action in value.path],
                "prefix": [action.to_dict() for action in value.prefix],
                "remainingHands": [
                    [card.id for card in hand] for hand in value.remaining_hands
                ],
            }
        )

    return observe


@pytest.mark.parametrize("unit,horizon", (("plies", 2), ("tricks", 1)))
def test_optional_project_root_and_observed_graft_have_complete_namespaced_map(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    unit: str,
    horizon: int,
) -> None:
    rules = RuleBook.load(
        root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
    )
    prefix = _optional_project_prefix()
    decision = {
        "kind": "project",
        "selector": {"owner": "east", "projectIndex": 0},
    }

    def forbidden_python_project_authority(*_args, **_kwargs):
        raise AssertionError("Python project detection/scoring is not proof authority")

    monkeypatch.setattr(
        ProjectDomainIRV1,
        "detect",
        forbidden_python_project_authority,
    )
    monkeypatch.setattr(
        ProjectDomainIRV1,
        "score_team",
        forbidden_python_project_authority,
    )
    for method in ("declaration_state", "apply_declaration_action", "legal_declaration_actions"):
        monkeypatch.setattr(ProjectDomainIRV1, method, forbidden_python_project_authority)
    from baloot_platform.research import ortools_nonplay_complete_graph as complete_module
    monkeypatch.setattr(complete_module, "compile_project_declaration_graph_v1", forbidden_python_project_authority)
    normalized = OrToolsExactStateNormalizerV1(
        rules, native_cp_sat=True
    ).normalize(prefix, decision_point=decision, maximum_seconds=15)

    compiled = compile_nonplay_complete_graph_v1(
        rules,
        normalized,
        prefix,
        decision,
        _request("bounded", horizon=horizon, unit=unit),
        maximum_nodes=1_000,
        maximum_depth=4,
        maximum_seconds=15,
        native_cp_sat=True,
        base_observer_factory=_base_observer_factory,
        play_observer_factory=_play_observer_factory,
    )
    graph = compiled.graph
    assert graph.scope == "bounded"
    assert any(node.is_leaf and node.utility != 0 for node in graph.nodes)
    root_node = next(
        node for node in graph.nodes if node.node_id == graph.root_node_id
    )
    decision_nodes = {node.node_id for node in graph.nodes if not node.is_leaf}

    assert [edge.action_code.split(":", 1)[0] for edge in root_node.edges] == [
        "pass",
        "declare",
    ]
    assert set(compiled.decision_observation_ids) == decision_nodes
    assert graph.root_node_id in compiled.decision_observation_ids
    assert any(
        node_id.startswith("graft-")
        for node_id in compiled.decision_observation_ids
    )
    assert all(
        len(observation_id) == 64
        for observation_id in compiled.decision_observation_ids.values()
    )


@pytest.mark.parametrize("declaration_index", [0, 1, 2, 3])
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_optional_project_single_decision_horizon_keeps_strategic_root(
    root: Path, declaration_index: int, horizon: int, monkeypatch,
) -> None:
    rules = RuleBook.load(
        root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
    )
    raw = _optional_project_prefix().to_dict()
    raw.pop("sha256")
    raw["projectActions"] = raw["projectActions"][:declaration_index]
    raw["playActions"] = raw["playActions"][:declaration_index]
    prefix = ExactTracePrefixV1.build(raw)
    decision = {
        "kind": "project",
        "selector": {"owner": ("north", "west", "south", "east")[declaration_index], "projectIndex": 0},
    }
    normalized = OrToolsExactStateNormalizerV1(
        rules, native_cp_sat=True
    ).normalize(prefix, decision_point=decision, maximum_seconds=15)

    if horizon == 3 and declaration_index < 3:
        def redundant_compile(*args, **kwargs):
            pytest.fail("certified play edge was solved again before project normalization")
        monkeypatch.setattr(
            "baloot_platform.research.ortools_nonplay_play_graft.OrToolsCompleteStateGraphCompilerV1.compile",
            redundant_compile,
        )
    try:
        compiled = compile_nonplay_complete_graph_v1(
            rules,
            normalized,
            prefix,
            decision,
            _request("bounded", horizon=horizon),
            maximum_nodes=100,
            maximum_depth=horizon,
            maximum_seconds=15,
            native_cp_sat=True,
            base_observer_factory=_base_observer_factory,
            play_observer_factory=_play_observer_factory,
        )
    except ExactNormalizationStopped as stopped:
        # The newly exercised post-first-trick observed tree exceeds this
        # fixture's 100-node budget. Preserve refusal rather than raising it.
        assert (horizon, declaration_index) == (3, 3)
        assert stopped.partial_receipt["exhaustionReason"] in {"maximum_nodes", "maximum_seconds", "unknown"}
        assert stopped.partial_receipt["evidence"]["causeReceipt"]["graphHash"] is None
        return
    root_node = next(
        node
        for node in compiled.graph.nodes
        if node.node_id == compiled.graph.root_node_id
    )

    assert not root_node.is_leaf
    assert [edge.action_code for edge in root_node.edges] == [row["code"] for row in normalized.legal_actions]
    assert all(node.leaf_kind == "boundary" for node in compiled.graph.nodes if node.is_leaf)
    assert set(compiled.decision_observation_ids) == {
        node.node_id for node in compiled.graph.nodes if not node.is_leaf
    }
    if declaration_index < 3:
        with pytest.raises(ExactNormalizationStopped) as exact_stopped:
            compile_nonplay_complete_graph_v1(
                rules, normalized, prefix, decision, _request("exact"),
                maximum_nodes=100, maximum_depth=2, maximum_seconds=15,
                native_cp_sat=True,
            )
        receipt = exact_stopped.value.partial_receipt
        assert receipt["exhaustionReason"] == "maximum_depth"
        assert receipt["evidence"]["sourceStage"] == "project_continuation"
        assert receipt["evidence"]["remainingDepth"] < receipt["evidence"]["requiredDepth"]
        assert receipt["evidence"]["requestHash"] == _request("exact").sha256


@pytest.mark.parametrize("maximum_nodes", [1, 2])
def test_graft_replacement_reuses_sentinel_node_capacity(rules, maximum_nodes):
    from baloot_platform.research.ortools_proof import (
        ProofGraphEdgeV1, ProofGraphNodeV1, ProofGraphV1,
    )

    graft = OrToolsNonPlayPlayGraphGraftV1(
        rules, _request("exact"), maximum_nodes=maximum_nodes,
        maximum_depth=8, maximum_seconds=15,
    )

    def admitted_prefix():
        raise LookupError("replacement admitted")

    sentinel = graft.terminal_utility(nonplay_depth=1, prefix_factory=admitted_prefix)
    base = ProofGraphV1.build(scope="exact", root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", "north_south",
            (ProofGraphEdgeV1.build(0, "pass", "leaf"),), actor_seat="north"),
        ProofGraphNodeV1.leaf("leaf", sentinel),
    ))
    replacement = ProofGraphV1.build(scope="exact", root_node_id="replacement", nodes=(
        ProofGraphNodeV1.decision("replacement", "north_south",
            (ProofGraphEdgeV1.build(0, "pass", "terminal"),), actor_seat="north"),
        ProofGraphNodeV1.leaf("terminal", 0),
    ))
    assert len(graft._replace_leaves(base, {"leaf": replacement}).nodes) == 3
    if maximum_nodes == 2:
        with pytest.raises(LookupError, match="replacement admitted"):
            graft.graft(base)
    else:
        with pytest.raises(ExactNormalizationStopped) as stopped:
            graft.graft(base)
        assert stopped.value.partial_receipt["exhaustionReason"] == "maximum_nodes"


@pytest.mark.parametrize("missing_companion", [False, True])
@pytest.mark.parametrize("declaration_index", [2, 3])
def test_native_project_compiler_carries_play_companion(root, monkeypatch, missing_companion, declaration_index):
    """Real declaration normalization; tiny play stub checks the handoff, not optimality."""
    from types import SimpleNamespace
    from baloot_platform.research.ortools_proof import ProofGraphV1, ProofGraphNodeV1, ProofGraphEdgeV1
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix().to_dict()
    raw.pop("sha256")
    raw["projectActions"] = raw["projectActions"][:declaration_index]
    raw["playActions"] = raw["playActions"][:declaration_index]
    prefix = ExactTracePrefixV1.build(raw)
    decision = {"kind": "project", "selector": {"owner": "south" if declaration_index == 2 else "east"}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point=decision, maximum_seconds=15)
    calls = []
    def compile_play(self, seed, request, **kwargs):
        assert kwargs["project_counterfactual"] is True
        calls.append(seed)
        root_node = ProofGraphNodeV1.decision("play", "west_east",
            (ProofGraphEdgeV1.build(0, "S7", "terminal"),), actor_seat="east")
        actual = ProofGraphV1.build(scope="exact", root_node_id="play",
            nodes=(root_node, ProofGraphNodeV1.leaf("terminal", 3)))
        counter = ProofGraphV1.build(scope="exact", root_node_id="play",
            nodes=(root_node, ProofGraphNodeV1.leaf("terminal", -4)))
        model = {"projectCounterfactualGraphHash": counter.sha256}
        receipt = {"modelHash": canonical_sha256(model), "elapsedSeconds": 0.1, "sha256": "stub"}
        return SimpleNamespace(complete=True, graph=actual, decision_observation_ids={},
            model_document=model, receipt=SimpleNamespace(model_hash=receipt["modelHash"],
                to_dict=lambda: dict(receipt)),
            project_counterfactual_graph=None if missing_companion else counter)
    monkeypatch.setattr(OrToolsCompleteStateGraphCompilerV1, "compile", compile_play)
    def run():
        return compile_nonplay_complete_graph_v1(rules, normalized, prefix, decision,
            _request("exact"), maximum_nodes=100, maximum_depth=40, maximum_seconds=20,
            native_cp_sat=True, project_counterfactual=True)
    if missing_companion:
        with pytest.raises(ValueError, match="omitted project counterfactual"):
            run()
    else:
        result = run()
        result.graph.require_same_legal_graph(result.project_counterfactual_graph)
        assert result.project_compilation_evidence
        for item in result.project_compilation_evidence:
            assert "elapsedSeconds" not in item["receipt"]
            assert canonical_sha256(item["model"]) == item["receipt"]["modelHash"]
            assert item["seed"]["schemaVersion"] == 1
            retained_prefix = ExactTracePrefixV1.build(item["exactTracePrefix"])
            assert retained_prefix.deck_order == prefix.deck_order
            assert retained_prefix.play_actions[:len(prefix.play_actions)] == prefix.play_actions
        assert result.graph.tactic_context["baseTricks"] == [0, 0]
        owners = result.graph.tactic_context["ownerByCard"]
        assert all(owner in range(4) for owner in owners)
        for action in prefix.play_actions:
            assert owners[Card.parse(action["card"]).id] == int(Seat.parse(action["actor"]))
        from baloot_platform.research.ortools_proof import OrToolsScenarioGraphProofV1
        proof = OrToolsScenarioGraphProofV1().prove(result.graph, _request("exact"),
            native_root_only=True, project_counterfactual_graph=result.project_counterfactual_graph,
            tactic_configure=lambda bindings: {"PAIRED": SimpleNamespace(
                literal=bindings.root_winning_variables[0],
                extract=lambda solver: {"deltas": [solver.value(value)
                    for value in bindings.root_project_consequence_variables]})})
        assert set(proof.native_qualification["PAIRED"]["deltas"]) == {7}
        assert {node.utility for node in result.graph.nodes if node.is_leaf} == {3}
        assert {node.utility for node in result.project_counterfactual_graph.nodes if node.is_leaf} == {-4}
        assert len(calls) >= len(normalized.legal_actions)
        if declaration_index == 3:
            from baloot_platform.research.result_authority import _verify_project_parent_paths
            frontiers = {}
            for item in result.project_compilation_evidence:
                for node in result.graph.nodes:
                    if node.node_id.endswith(":" + item["graph"]["rootNodeId"]):
                        frontiers.setdefault(node.node_id, []).append(ExactTracePrefixV1.build(item["exactTracePrefix"]))
            _verify_project_parent_paths(result.graph, rules=rules, source_prefix=prefix,
                frontiers=frontiers, maximum_nodes=100, maximum_seconds=15)
            with pytest.raises(ValueError, match="missing its declaration root"):
                _verify_project_parent_paths(result.graph, rules=rules, source_prefix=prefix,
                    frontiers={result.graph.root_node_id: [prefix]}, maximum_nodes=100, maximum_seconds=15)
            from dataclasses import replace
            changed = ProofGraphV1.build(scope=result.graph.scope, root_node_id=result.graph.root_node_id,
                tactic_context=result.graph.tactic_context,
                nodes=[replace(node, edges=(replace(node.edges[0], action_code="invented"), *node.edges[1:]))
                       if node.node_id == result.graph.root_node_id else node for node in result.graph.nodes])
            with pytest.raises(ValueError, match="legal actions"):
                _verify_project_parent_paths(changed, rules=rules, source_prefix=prefix,
                    frontiers=frontiers, maximum_nodes=100, maximum_seconds=15)


def test_ordinary_native_project_graph_has_owned_root_proof_context(root):
    from types import SimpleNamespace
    from baloot_platform.research.ortools_proof import OrToolsScenarioGraphProofV1
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = _optional_project_prefix()
    decision = {"kind": "project", "selector": {"owner": "east"}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point=decision, maximum_seconds=15)
    request = _request("bounded", horizon=2)
    result = compile_nonplay_complete_graph_v1(rules, normalized, prefix, decision, request,
        maximum_nodes=100, maximum_depth=3, maximum_seconds=15, native_cp_sat=True)
    assert result.project_counterfactual_graph is None
    owners = result.graph.tactic_context["ownerByCard"]
    assert [owners.count(seat) for seat in range(4)] == [8, 8, 8, 8]
    outcome = OrToolsScenarioGraphProofV1().prove(result.graph, request, native_root_only=True,
        tactic_configure=lambda bindings: {"ROOT": SimpleNamespace(
            literal=bindings.root_winning_variables[0],
            extract=lambda solver: {"winning": [solver.boolean_value(item)
                for item in bindings.root_winning_variables]})})
    assert outcome.native_qualification["ROOT"]["winning"] == [row.satisfies for row in outcome.root_actions]


def test_paired_terminal_graft_preserves_project_scores_and_rejects_missing_branch(rules):
    from dataclasses import replace
    from baloot_platform.research.ortools_proof import (
        ProofGraphEdgeV1, ProofGraphNodeV1, ProofGraphV1, OrToolsScenarioGraphProofV1,
    )
    graft = OrToolsNonPlayPlayGraphGraftV1(
        rules, _request("exact"), maximum_nodes=20, maximum_depth=8,
        maximum_seconds=15, native_cp_sat=True,
    )
    sentinel = graft.terminal_utility(nonplay_depth=1, prefix_factory=lambda: None)
    context = {"baseTricks": [2, 3], "ownerByCard": [0] * 32, "trumpSuit": None}
    base = ProofGraphV1.build(scope="exact", root_node_id="project", tactic_context=context, nodes=(
        ProofGraphNodeV1.decision("project", "north_south", (
            ProofGraphEdgeV1.build(0, "declare", "pending"),
            ProofGraphEdgeV1.build(1, "pass", "zero")), actor_seat="north"),
        ProofGraphNodeV1.leaf("pending", sentinel), ProofGraphNodeV1.leaf("zero", 0),
    ))
    actual = ProofGraphV1.build(scope="exact", root_node_id="play", nodes=(
        ProofGraphNodeV1.decision("play", "west_east", (
            ProofGraphEdgeV1.build(0, "S7", "low"),
            ProofGraphEdgeV1.build(1, "S8", "high")), actor_seat="west"),
        ProofGraphNodeV1.leaf("low", 3), ProofGraphNodeV1.leaf("high", 5),
    ))
    counter = ProofGraphV1.build(scope="exact", root_node_id="play", nodes=[
        replace(node, utility=-4 if node.node_id == "high" else 0) if node.is_leaf else node
        for node in actual.nodes],
    )
    combined, intervention = graft.graft_terminal_graph_pairs(base, {sentinel: (actual, counter)})
    combined.require_same_legal_graph(intervention)
    prover = OrToolsScenarioGraphProofV1()
    assert [row.value for row in prover.prove(combined, _request("exact")).root_actions] == [3, 0]
    assert [row.value for row in prover.prove(intervention, _request("exact")).root_actions] == [-4, 0]
    # Exercise the production recursive replacement path, not only the pair helper.
    graft.project_counterfactual = True
    with pytest.raises(ValueError, match="omitted its counterfactual"):
        graft._namespace_paired_graph(actual, "project-branch")
    graft.project_counterfactual_graphs[actual.sha256] = counter
    namespaced = graft._namespace_paired_graph(actual, "project-branch")
    automatic = graft._replace_leaves(base, {"pending": namespaced})
    automatic_counter = graft.project_counterfactual_graphs[automatic.sha256]
    automatic.require_same_legal_graph(automatic_counter)
    assert [row.value for row in prover.prove(automatic, _request("exact")).root_actions] == [3, 0]
    assert [row.value for row in prover.prove(automatic_counter, _request("exact")).root_actions] == [-4, 0]
    graft.project_counterfactual = False
    with pytest.raises(ValueError, match="every registered continuation"):
        graft.graft_terminal_graph_pairs(base, {})
    with pytest.raises(ValueError, match="complete legal graph"):
        graft.graft_terminal_graph_pairs(base, {sentinel: (actual, base)})
    graft.cancel_check = lambda: True
    with pytest.raises(ExactNormalizationStopped) as stopped:
        graft.graft_terminal_graph_pairs(base, {sentinel: (actual, counter)})
    assert stopped.value.partial_receipt["exhaustionReason"] == "cancelled"


def test_graph_rekeying_preserves_trick_evidence(rules):
    from baloot_platform.research.ortools_proof import (
        ProofGraphEdgeV1, ProofGraphNodeV1, ProofGraphV1,
    )
    edge = ProofGraphEdgeV1.build(0, "S7", "leaf", play_trick_index=5,
        completed_trick_index=5, completed_trick_winner_seat="north",
        completed_trick_winning_card_id=0)
    context = {"baseTricks": [2, 3], "ownerByCard": [0] * 32, "trumpSuit": None}
    base = ProofGraphV1.build(scope="exact", root_node_id="root", utility_scale=2,
        tactic_context=context, nodes=(
            ProofGraphNodeV1.decision("root", "north_south", (edge,), actor_seat="north"),
            ProofGraphNodeV1.leaf("leaf", 2),
        ))
    graft = OrToolsNonPlayPlayGraphGraftV1(rules, _request("exact"),
        maximum_nodes=10, maximum_depth=8, maximum_seconds=15)
    child = graft._namespace_graph(base, "child")
    combined = graft._replace_leaves(base, {"leaf": child})
    for graph in (child, combined):
        copied = next(node for node in graph.nodes if node.node_id == graph.root_node_id).edges[0]
        assert {key: value for key, value in copied.to_dict().items() if key != "childNodeId"} == {
            key: value for key, value in edge.to_dict().items() if key != "childNodeId"}
        assert graph.utility_scale == base.utility_scale
        assert graph.tactic_context == base.tactic_context
        assert graph.sha256 != base.sha256
    assert combined.nodes != base.nodes
    incompatible = ProofGraphV1.build(scope=child.scope, root_node_id=child.root_node_id,
        nodes=child.nodes, utility_scale=1, tactic_context=child.tactic_context)
    with pytest.raises(ValueError, match="identical utility scales"):
        graft._replace_leaves(base, {"leaf": incompatible})


@pytest.mark.parametrize("compact", [False, True])
def test_interleaved_play_root_shares_equal_boundary_nodes(root, compact):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix().to_dict()
    raw.pop("sha256")
    raw["projectActions"] = raw["projectActions"][:1]
    raw["playActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    assert normalized.decision_kind == "play" and len(normalized.legal_actions) > 1

    compiled = compile_nonplay_complete_graph_v1(
        rules, normalized, prefix, {"kind": "play"}, _request("bounded", horizon=2),
        maximum_nodes=1_000, maximum_depth=4, maximum_seconds=15,
        native_cp_sat=True, _compact_forced=compact,
    )
    graph = compiled.graph
    by_id = {node.node_id: node for node in graph.nodes}
    assert len(by_id) == len(graph.nodes)
    edges = by_id[graph.root_node_id].edges
    assert [edge.action_code for edge in edges] == [action["code"] for action in normalized.legal_actions]
    assert len({edge.child_node_id for edge in edges}) == len(edges)
    assert sum(node.is_leaf for node in graph.nodes) == 1
    assert all(node.leaf_kind == "boundary" and node.utility == 0
               for node in graph.nodes if node.is_leaf)


@pytest.mark.parametrize("compact", [False, True])
def test_sibling_boundary_sharing_keeps_decisions_and_edges(rules, compact):
    from baloot_platform.research.ortools_proof import (
        ProofGraphEdgeV1, ProofGraphNodeV1, ProofGraphV1,
    )
    def graph(root, leaves, utility=2):
        return ProofGraphV1.build(scope="bounded", root_node_id=root,
            boundary_evaluator="current_raw_point_margin_v1", nodes=(
                ProofGraphNodeV1.decision(root, "north_south", tuple(
                    ProofGraphEdgeV1.build(i, str(i), leaf) for i, leaf in enumerate(leaves)),
                    actor_seat="north"),
                *(ProofGraphNodeV1.leaf(leaf, utility, kind="boundary") for leaf in leaves)))
    graft = OrToolsNonPlayPlayGraphGraftV1(rules, _request("bounded"),
        maximum_nodes=4, maximum_depth=8, maximum_seconds=15,
        native_cp_sat=compact, _compact_forced=compact)
    child = graph("child", ("outcome",))
    first = graft._namespace_graph(child, "first")
    second = graft._namespace_graph(child, "second")
    assert first.root_node_id != second.root_node_id
    combined = graft._replace_leaves(graph("root", ("left", "right")),
                                    {"left": first, "right": second})
    assert len(combined.nodes) == 4
    assert graft._replacement_node_count(graph("root", ("left", "right")),
                                        {"left": first, "right": second}) == (2 if compact else 4)
    assert graft._node_count(combined) == (2 if compact else 4)
    assert sum(node.is_leaf for node in combined.nodes) == 1
    assert sum(len(node.edges) for node in combined.nodes) == 4
    other = graft._namespace_graph(graph("child", ("outcome",), utility=3), "second")
    distinct = graft._replace_leaves(graph("root", ("left", "right")),
                                    {"left": first, "right": other})
    assert len(distinct.nodes) == 5
    assert graft._replacement_node_count(graph("root", ("left", "right")),
                                        {"left": first, "right": other}) == (3 if compact else 5)
    assert graft._node_count(distinct) == (3 if compact else 5)
    for scope, evaluator in (("exact", None), ("bounded", "auction_hand_margin_v1")):
        incompatible = ProofGraphV1.build(scope=scope, root_node_id="incompatible",
            boundary_evaluator=evaluator, nodes=(
                ProofGraphNodeV1.decision("incompatible", "north_south",
                    (ProofGraphEdgeV1.build(0, "pass", "incompatible-end"),), actor_seat="north"),
                ProofGraphNodeV1.leaf("incompatible-end", 2)))
        with pytest.raises(ValueError, match="identical proof boundaries"):
            graft._replace_leaves(graph("root", ("left", "right")),
                                  {"left": first, "right": incompatible})
    limits = []
    def continue_project(prefix, request, nodes, depth, seconds):
        limits.append(nodes)
        return child, {}
    graft.project_continuation = continue_project
    assert graft._continue_project(None, _request("bounded"), 1, 8, 15,
                                   retained={"first": first}) == (child, {})
    assert limits == [2]
    child = graph("child", ("same-outcome-a", "same-outcome-b"))
    assert graft._continue_project(None, _request("bounded"), 1, 8, 15,
                                   retained={"first": first}) == (child, {})
    with pytest.raises(ExactNormalizationStopped) as stopped:
        graft._continue_project(None, _request("bounded"), 1, 8, 15,
                                retained={"other": other})
    assert stopped.value.partial_receipt["exhaustionReason"] == "maximum_nodes"

    calls_before_cancel = len(limits)
    graft.cancel_check = lambda: True
    with pytest.raises(ExactNormalizationStopped) as cancelled:
        graft._continue_project(None, _request("bounded"), 1, 8, 15,
                                retained={"first": first})
    assert cancelled.value.partial_receipt["exhaustionReason"] == "cancelled"
    assert len(limits) == calls_before_cancel


def test_native_project_root_exhaustion_retains_budget_evidence(root):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = _optional_project_prefix()
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    request = _request("bounded")
    with pytest.raises(ExactNormalizationStopped) as stopped:
        compile_nonplay_complete_graph_v1(rules, normalized, prefix,
            {"kind": "project", "selector": {"owner": normalized.next_actor}}, request,
            maximum_nodes=2, maximum_depth=8, maximum_seconds=15, native_cp_sat=True)
    receipt = stopped.value.partial_receipt
    assert receipt["exhaustionReason"] == "maximum_nodes"
    assert receipt["evidence"]["availableNodes"] == 2
    assert receipt["evidence"]["requiredNodes"] == len(normalized.legal_actions) + 1
    assert receipt["evidence"]["prefixHash"] == prefix.sha256


@pytest.mark.parametrize(
    "mode,unit,horizon",
    (("exact", "plies", 1), ("bounded", "plies", 2),
     ("bounded", "tricks", 1), ("bounded", "round_end", 1)),
)
def test_nonplay_depth_exhaustion_does_not_become_a_zero_frontier(
    rules: RuleBook, mode: str, unit: str, horizon: int,
) -> None:
    from baloot_platform.research.ortools_proof import (
        ProofGraphEdgeV1, ProofGraphNodeV1, ProofGraphV1,
    )

    graft = OrToolsNonPlayPlayGraphGraftV1(
        rules, _request(mode, horizon=horizon, unit=unit),
        maximum_nodes=100, maximum_depth=1, maximum_seconds=15,
        native_cp_sat=True, root_decision_kind="project",
    )

    def forbidden_prefix():
        raise AssertionError("exhausted proof must not build another model")

    sentinel = graft.terminal_utility(nonplay_depth=1, prefix_factory=forbidden_prefix)
    base = ProofGraphV1.build(
        scope="exact", root_node_id="root",
        nodes=(
            ProofGraphNodeV1.decision(
                "root", "north_south",
                (ProofGraphEdgeV1.build(0, "pass", "leaf"),),
                actor_seat="north",
            ),
            ProofGraphNodeV1.leaf("leaf", sentinel),
        ),
    )
    with pytest.raises(ExactNormalizationStopped) as stopped:
        graft.graft(base)
    receipt = stopped.value.partial_receipt
    assert receipt["status"] == "incomplete"
    assert receipt["exhaustionReason"] == "maximum_depth"
    assert receipt["evidence"]["sourceStage"] == "combined_nonplay_frontier"
    assert receipt["evidence"]["remainingDepth"] == 0
    assert receipt["evidence"]["requiredDepth"] == 1
    assert receipt["evidence"]["requestHash"] == graft.request.sha256
    assert receipt["evidence"]["rulesHash"] == rules.sha256
    assert receipt["sha256"] == canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})


@pytest.mark.parametrize("mode", ["exact", "bounded"])
def test_empty_recall_does_not_certify_prior_private_history_coverage(rules, mode):
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1,
    )
    from baloot_platform.research.ortools_research_executor import compile_observed_decision_v1
    raw = _escalation_prefix().to_dict()
    raw.pop("sha256")
    raw["escalationActions"] = [{"actor": "east", "action": "double", "mode": "open"}]
    prefix = ExactTracePrefixV1.build(raw)
    decision = {"kind": "escalation", "selector": {"actingSeat": "north"}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point=decision, maximum_seconds=15)
    value = _request(mode).to_dict()
    value.pop("sha256")
    value.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    request = ScenarioProofRequestV1.build(value)
    def run():
        return compile_observed_decision_v1(rules, normalized, prefix, decision, request,
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=2, maximum_seconds=20, native_cp_sat=True)
    if mode == "exact":
        with pytest.raises(InformationSetCompilationStopped) as caught:
            run()
        assert caught.value.reason == "private_history_coverage_unavailable"
        play = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
            prefix, maximum_seconds=15)
        with pytest.raises(InformationSetCompilationStopped) as direct:
            OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True).compile(play, request,
                maximum_nodes=1000, maximum_depth=8, maximum_worlds=2, maximum_seconds=20)
        assert direct.value.reason == "private_history_coverage_unavailable"
    else:
        result = run()
        assert result.graph.hidden_worlds["complete"] is False
        assert result.graph.analysis_scope["privateHistoryCoverage"] == "declared_history_only"


def test_escalation_proof_enumerates_prior_histories_without_opponent_oracle(rules):
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.ortools_research_executor import compile_observed_decision_v1
    raw = _escalation_prefix().to_dict()
    raw.pop("sha256")
    raw["escalationActions"] = [{"actor": "east", "action": "double", "mode": "open"}]
    prefix = ExactTracePrefixV1.build(raw)
    decision = {"kind": "escalation", "selector": {"actingSeat": "north"}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point=decision, maximum_seconds=15)
    value = _request("bounded").to_dict()
    value.pop("sha256")
    value.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    request = ScenarioProofRequestV1.build(value)
    def run(recall, worlds=2):
        return compile_observed_decision_v1(rules, normalized, prefix, decision, request,
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=worlds, maximum_seconds=20,
            native_cp_sat=True, private_recall=recall)
    omitted = run({})
    recorded = run({"west": [canonical_sha256({"actions": []})]})
    assert omitted.graph.to_dict() == recorded.graph.to_dict()
    assert omitted.graph.model_document == recorded.graph.model_document
    assert omitted.graph.model_document["priorHistoryCoverage"]["worldCount"] == 2
    assert len(omitted.graph.problem.worlds) == 2
    with pytest.raises(InformationSetCompilationStopped) as caught:
        run({}, worlds=1)
    assert caught.value.reason == "maximum_worlds"
    assert caught.value.partial_receipt["stage"] == "escalation_history_preflight"


def test_observed_proof_consumes_normalized_escalation_history(rules):
    from dataclasses import replace
    from baloot_platform.research.ortools_exact_state import OrToolsExactNormalizationReceiptV1
    from baloot_platform.research.ortools_research_executor import compile_observed_decision_v1
    prefix = _escalation_prefix()
    decision = {"kind": "escalation", "selector": {"actingSeat": "east"}}
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    normalized = normalizer.normalize(prefix, decision_point=decision,
        recorded_escalation_opportunities=[{"actor": "west", "actionCode": "decline"}],
        opportunity_order=("west", "east", "south", "north"))
    complete = compile_nonplay_complete_graph_v1(rules, normalized, prefix, decision,
        _request("bounded"), maximum_nodes=1000, maximum_depth=8, maximum_seconds=20,
        native_cp_sat=True)
    root = next(node for node in complete.graph.nodes if node.node_id == complete.graph.root_node_id)
    assert root.actor_seat == normalized.next_actor
    assert {edge.action_code for edge in root.edges} == {row["code"] for row in normalized.legal_actions}
    with pytest.raises(ValueError, match="cannot be overridden"):
        compile_nonplay_complete_graph_v1(rules, normalized, prefix, decision,
            _request("bounded"), maximum_nodes=1000, maximum_depth=8, maximum_seconds=20,
            native_cp_sat=True, opportunity_order=("east", "west", "south", "north"))
    public = normalizer.normalize(prefix, decision_point=decision)
    value = _request("bounded").to_dict()
    value.pop("sha256")
    value.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    request = ScenarioProofRequestV1.build(value)
    def run(state, **kwargs):
        return compile_observed_decision_v1(rules, state, prefix, decision, request,
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=2, maximum_seconds=20,
            native_cp_sat=True, **kwargs)
    result = run(normalized)
    expected = run(public, private_recall={})
    assert result.graph.to_dict() == expected.graph.to_dict()
    assert result.graph.model_document == expected.graph.model_document
    automatic = compile_observed_decision_v1(rules, normalized, prefix, {"kind": "escalation", "selector": {}}, request,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=2, maximum_seconds=20, native_cp_sat=True)
    assert automatic.graph.to_dict() == expected.graph.to_dict()
    wrong_prefix = prefix.to_dict()
    wrong_prefix.pop("sha256")
    wrong_prefix["roundNumber"] += 1
    for source, point in ((ExactTracePrefixV1.build(wrong_prefix), decision),
                          (prefix, {"kind": "escalation", "selector": {"actingSeat": "west"}})):
        with pytest.raises(ValueError, match="recorded escalation source context"):
            compile_observed_decision_v1(rules, normalized, source, point, request,
                maximum_nodes=1000, maximum_depth=8, maximum_worlds=2, maximum_seconds=20,
                native_cp_sat=True)
    with pytest.raises(ValueError, match="recall disagrees"):
        run(normalized, private_recall={})
    for mutation in ("missing", "actor", "digest", "order"):
        payload = normalized.normalization_receipt.to_dict()
        history = payload["derivedEscalationOpportunities"]
        if mutation == "missing":
            del payload["derivedEscalationOpportunities"]
        elif mutation == "actor":
            history["nextActor"] = "west"
        elif mutation == "digest":
            history["sha256"] = "0" * 64
        else:
            history["opportunityOrder"] = ["east", "west", "south", "north"]
        if mutation in {"actor", "order"}:
            history["sha256"] = canonical_sha256({key: value for key, value in history.items() if key != "sha256"})
        payload["sha256"] = canonical_sha256({key: value for key, value in payload.items() if key != "sha256"})
        substituted = replace(normalized, normalization_receipt=OrToolsExactNormalizationReceiptV1(payload))
        with pytest.raises(ValueError, match="recorded escalation proof history"):
            run(substituted)
        with pytest.raises(ValueError, match="recorded escalation proof history"):
            compile_nonplay_complete_graph_v1(rules, substituted, prefix, decision,
                _request("bounded"), maximum_nodes=1000, maximum_depth=8, maximum_seconds=20,
                native_cp_sat=True)


def test_recorded_escalation_mismatch_precedes_exhausted_handoff(rules, monkeypatch):
    import baloot_platform.research.ortools_research_executor as executor
    raw = _escalation_prefix().to_dict()
    raw.pop("sha256")
    raw["projectActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    recorded = {**raw, "escalationActions": [{"actor": "west", "action": "double", "requestedMode": "open"}],
                "playActions": [{"actor": seat, "card": card} for card, seat in
                    zip(range(32), ("north", "east", "south", "west") * 8)]}
    recorded = ExactTracePrefixV1.build(recorded)
    request = _request("bounded").to_dict()
    analysis = {key: request[key] for key in ("utility", "budgetProfile", "horizon", "boundaryEvaluator")}
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: analysis for kind in ("bid", "escalation", "project", "play")}})
    def completed(*args, **kwargs):
        return {"status": "complete", "partialReceipt": None, "finalPrefix": prefix.to_dict(),
                "initialNormalization": None, "initialState": None,
                "projectPlayHandoff": {}, "artifacts": {}}
    def forbidden(*args, **kwargs):
        raise AssertionError("contradictory public history must not enter play analysis")
    monkeypatch.setattr(executor, "resolve_quality_escalation_continuation_v1", completed)
    monkeypatch.setattr(executor, "resolve_quality_play_continuation_v1", forbidden)
    with pytest.raises(ValueError, match="changes the source context"):
        executor.resolve_quality_post_auction_continuation_v1(rules, prefix,
            {"kind": "escalation", "selector": {}}, policy,
            opportunity_order=("west", "east", "south", "north"), maximum_nodes=1000,
            maximum_depth=8, maximum_worlds=2, maximum_seconds=60, maximum_decision_seconds=20,
            maximum_escalation_decisions=20, maximum_evidence_bytes=1,
            recorded_trace=recorded, recorded_events=[])


def test_partial_escalation_preserves_private_decline_then_generates(rules):
    from baloot_platform.research.ortools_research_executor import resolve_quality_escalation_continuation_v1
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    request = ScenarioProofRequestV1.build(raw)
    prefix = _escalation_prefix()
    recorded = [{"actor": "west", "actionCode": "decline"}]
    result = resolve_quality_escalation_continuation_v1(rules, prefix,
        {"kind": "escalation", "selector": {"actingSeat": "east"}}, request,
        opportunity_order=("west", "east", "south", "north"), maximum_nodes=1000,
        maximum_depth=8, maximum_worlds=2, maximum_seconds=30, maximum_decision_seconds=15,
        maximum_decisions=20, maximum_evidence_bytes=10_000_000,
        recorded_events=recorded, continue_recorded_prefix=True)
    assert result["status"] == "complete", result["partialReceipt"]
    assert result["operation"] == "audit_then_generate"
    assert result["recordedEvents"] == recorded
    assert result["events"] == [*recorded, {"actor": "east", "actionCode": "decline"}]
    assert result["finalPrefix"] == prefix.to_dict()
    artifacts = [result["artifacts"][row["artifactHash"]] for row in result["decisions"]]
    assert artifacts[0]["request"]["selectedAction"] == "decline"
    assert artifacts[1]["request"].get("selectedAction") is None
    assert set(result["privateRecall"]) == {"west", "east"}
    with pytest.raises(ValueError, match="suffix misses requested actor"):
        resolve_quality_escalation_continuation_v1(rules, prefix,
            {"kind": "escalation", "selector": {"actingSeat": "west"}}, request,
            opportunity_order=("west", "east", "south", "north"), maximum_nodes=1000,
            maximum_depth=8, maximum_worlds=2, maximum_seconds=30, maximum_decision_seconds=15,
            maximum_decisions=20, maximum_evidence_bytes=10_000_000,
            recorded_events=recorded, continue_recorded_prefix=True)


def test_empty_recorded_escalation_prefix_generates_and_audits_declines(rules):
    from baloot_platform.research.ortools_research_executor import resolve_quality_escalation_continuation_v1
    raw = _request("bounded").to_dict()
    raw.pop("sha256")
    raw.update(visibility="actor_information_set", strategyClass="pure_deterministic")
    request = ScenarioProofRequestV1.build(raw)
    prefix = _escalation_prefix()
    result = resolve_quality_escalation_continuation_v1(rules, prefix,
        {"kind": "escalation", "selector": {"actingSeat": "west"}}, request,
        opportunity_order=("west", "east", "south", "north"), maximum_nodes=1000,
        maximum_depth=8, maximum_worlds=2, maximum_seconds=30, maximum_decision_seconds=15,
        maximum_decisions=20, maximum_evidence_bytes=10_000_000,
        recorded_events=[], continue_recorded_prefix=True)
    assert result["status"] == "complete", result["partialReceipt"]
    assert result["finalPrefix"] == prefix.to_dict()
    assert [row["actionCode"] for row in result["events"]] == ["decline", "decline"]
    assert len(result["decisions"]) == 2


@pytest.mark.parametrize("recorded_decline,through_round,finish_round", [
    (False, False, False), (True, False, False), (True, True, False), (True, True, True)])
def test_post_auction_real_escalation_retains_proofs_at_project_cancellation(rules, monkeypatch, recorded_decline, through_round, finish_round):
    import baloot_platform.research.ortools_research_executor as executor
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    rules = RuleBook.load(Path(__file__).resolve().parents[1] / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    if finish_round:
        entry = executor.OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
            bundle_revision_id="CBR-PARTIAL-ESCALATION-TEST", bundle_hash="a" * 64)
    raw = _escalation_prefix().to_dict()
    raw.pop("sha256")
    raw["projectActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    request = _request("bounded").to_dict()
    analysis = {key: request[key] for key in ("utility", "budgetProfile", "horizon", "boundaryEvaluator")}
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: analysis for kind in ("bid", "escalation", "project", "play")}})
    original, kinds = executor.compile_observed_decision_v1, []
    def stop_at_project(*args, **kwargs):
        kinds.append(args[3]["kind"])
        if kinds[-1] == "project":
            assert kwargs["private_recall"] == {
                seat: (canonical_sha256({"actions": []}),) for seat in ("west", "east")}
            if not finish_round:
                raise InformationSetCompilationStopped("cancelled", stage="action_analysis")
        return original(*args, **kwargs)
    monkeypatch.setattr(executor, "compile_observed_decision_v1", stop_at_project)
    source = ExactTracePrefixV1.build({**raw, "auctionActions": []}) if through_round else prefix
    resolver = executor.resolve_quality_round_continuation_v1 if through_round else executor.resolve_quality_post_auction_continuation_v1
    decision = {"kind": "escalation", "selector": {"actingSeat": "east" if recorded_decline else "west"}}
    result = resolver(rules, source,
        decision, policy,
        recorded_trace=prefix if recorded_decline else None,
        recorded_events=[{"actor": "west", "actionCode": "decline"}] if recorded_decline else None,
        continue_recorded_prefix=recorded_decline,
        opportunity_order=("west", "east", "south", "north"), maximum_nodes=1000,
        maximum_depth=8, maximum_worlds=2, maximum_seconds=180 if finish_round else 60, maximum_decision_seconds=20,
        maximum_escalation_decisions=20, maximum_evidence_bytes=10_000_000)
    execution_receipt = result["executionReceipt"]
    if through_round:
        from baloot_platform.research.result_authority import validate_native_round_quality_evidence_v1
        validate_native_round_quality_evidence_v1(result, source_prefix=source,
            rules_hash=rules.sha256, policy=policy,
            decision_point=decision)
        if finish_round:
            assert result["status"] == "complete", result["partialReceipt"]
            assert len(result["finalPrefix"]["playActions"]) == 32
            trace = {"completeExactTracePrefix": result["finalPrefix"], "playingQualityEvidence": {
                "sourcePrefix": source.to_dict(),
                "decisionPoint": decision,
                "receipt": {key: value for key, value in result.items() if key != "executionReceipt"}}}
            candidate = entry.candidate_from_quality_trace(trace, maximum_seconds=30)
            assert candidate.origin == "exact_trace_prefix"
            assert candidate.trace["completeExactTracePrefix"] == result["finalPrefix"]
            return
        for mutation in ("public", "private"):
            changed = copy.deepcopy(result)
            post = changed["postAuction"]
            if mutation == "public":
                altered = prefix.to_dict()
                altered.pop("sha256")
                altered["escalationActions"] = [{"actor": "west", "action": "double"}]
                altered = ExactTracePrefixV1.build(altered)
                for envelope in (changed, post):
                    envelope["submittedPrefix"] = altered.to_dict()
                    envelope["recordedPrefixHash"] = altered.sha256
            else:
                escalation = post["escalation"]
                escalation["recordedEvents"][0]["actor"] = "east"
                escalation["sha256"] = canonical_sha256({key: value for key, value in escalation.items() if key != "sha256"})
            for envelope in (post, changed):
                envelope["sha256"] = canonical_sha256({key: value for key, value in envelope.items()
                    if key not in {"sha256", "executionReceipt"}})
            with pytest.raises(ValueError, match="recorded.*(boundary|prefix)"):
                validate_native_round_quality_evidence_v1(changed, source_prefix=source,
                    rules_hash=rules.sha256, policy=policy,
                    decision_point=decision)
        truncated = copy.deepcopy(result)
        post = truncated["postAuction"]
        post["projectPlay"] = None
        for envelope in (post, truncated):
            envelope["status"] = "complete"
            envelope["partialReceipt"] = None
            envelope["finalPrefix"] = post["escalation"]["finalPrefix"]
            envelope["sha256"] = canonical_sha256({key: value for key, value in envelope.items()
                if key not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="round quality false completion"):
            validate_native_round_quality_evidence_v1(truncated, source_prefix=source,
                rules_hash=rules.sha256, policy=policy,
                decision_point=decision)
        result = result["postAuction"]
    assert [kind for kind in kinds if kind != "bid"] == ["escalation", "escalation", "project"], result["partialReceipt"]
    assert result["status"] == "incomplete"
    assert result["partialReceipt"]["exhaustionReason"] == "cancelled"
    stage = result["escalation"]
    assert stage["status"] == "complete"
    assert stage["projectPlayHandoff"]["exactState"]["decisionKind"] == "project"
    assert len(stage["events"]) == len(stage["artifacts"]) == 2
    assert result["projectPlay"]["decisions"] == []
    assert result["finalPrefix"] == prefix.to_dict()
    assert execution_receipt["evidenceBytes"] <= 10_000_000
    if recorded_decline:
        assert stage["operation"] == "audit_then_generate"
        assert stage["events"][0] == {"actor": "west", "actionCode": "decline"}
        assert result["projectPlay"]["operation"] == "generate"
        assert result["submittedPrefix"] == prefix.to_dict()
