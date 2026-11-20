from __future__ import annotations

import ast
import base64
import inspect
import json
import zlib
from pathlib import Path

import pytest

from deterministic_work import set_monotonic

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Rank, Seat, Suit
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_complete_state_graph import (
    OrToolsCompleteStateGraphCompilerV1,
)
from baloot_platform.research.ortools_nonplay_graphs import (
    AuctionGraphChoiceV1,
    EscalationGraphActionV1,
    NonPlayGraphBudgetV1,
    NonPlayGraphCompilationStoppedV1,
    compile_auction_decision_graph_v1,
    compile_escalation_decision_graph_v1,
    compile_project_declaration_graph_v1,
    enumerate_escalation_opportunity_orders_v1,
    encode_native_auction_model_receipt_v1,
    decode_native_auction_model_receipt_v1,
)
from baloot_platform.research.ortools_nonplay_play_graft import (
    OrToolsNonPlayPlayGraphGraftV1,
    build_nonplay_play_start_prefix_v1,
)
from baloot_platform.research.ortools_nonplay_policy_continuation import (
    composed_team_legal_action_order_v1,
)
from baloot_platform.research.ortools_proof import (
    DeclarativeActionPolicyV1,
    OrToolsScenarioGraphProofV1,
)
from baloot_platform.research.scenario_contracts import (
    ExactTracePrefixV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)


def _request(
    *,
    target_team: str,
    claim: str,
    selected_action: str | None = None,
    mode: str = "exact",
) -> ScenarioProofRequestV1:
    value = {
        "schemaVersion": 1,
        "mode": mode,
        "visibility": "complete_state",
        "claim": claim,
        "targetTeam": target_team,
        "utility": "awarded_point_margin",
        "successThreshold": 1,
        "budgetProfile": "test",
    }
    if selected_action is not None:
        value["selectedAction"] = selected_action
    if mode == "bounded":
        value["horizon"] = {"unit": "plies", "value": 1}
        value["boundaryEvaluator"] = "current_raw_point_margin_v1"
    return ScenarioProofRequestV1.build(value)


def _auction_domain(rules) -> AuctionDomainIRV1:
    return AuctionDomainIRV1.from_rules(rules.data)


def _root(graph):
    return next(item for item in graph.nodes if item.node_id == graph.root_node_id)


@pytest.mark.parametrize("official", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_native_auction_graph_matches_tree_without_interpreter(root, rules, monkeypatch, official, bounded):
    import baloot_platform.research.ortools_nonplay_graphs as graphs
    if official:
        rules = RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    options = dict(dealer=Seat.EAST, purchase_suit=Suit.HEARTS, purchase_rank=Rank.ACE,
                   prefix=tuple(AuctionGraphChoiceV1.build("pass") for _ in range(0 if bounded else 6)),
                   terminal_utility=lambda row: -1 if row.all_passed else int(row.buyer.team))
    if bounded:
        options.update(scope="bounded", horizon=1, boundary_evaluator="test_round",
                       boundary_utility=lambda row: row.round_index)
    domain = _auction_domain(rules)

    def tree(graph):
        nodes = {node.node_id: node for node in graph.nodes}
        def visit(node_id):
            node = nodes[node_id]
            return (node.actor_team, node.actor_seat, node.leaf_kind, node.utility,
                    tuple((edge.action_id, edge.action_code, visit(edge.child_node_id)) for edge in node.edges))
        return visit(graph.root_node_id)

    expected = tree(compile_auction_decision_graph_v1(domain, **options))
    def forbidden(*args, **kwargs):
        pytest.fail("native auction graph used Python transition authority")
    for name in ("_initial_auction_state", "_advance_auction", "_auction_actor", "_auction_legal_choices"):
        monkeypatch.setattr(graphs, name, forbidden)
    graph = compile_auction_decision_graph_v1(domain, **options, native_cp_sat=True, maximum_seconds=10)
    assert tree(graph) == expected
    assert graph.to_dict() == compile_auction_decision_graph_v1(
        domain, **options, native_cp_sat=True, maximum_seconds=10).to_dict()


@pytest.mark.parametrize("codes,expected", [
    (("S7", "H7", "C7", "D7", "S8"), "pass"),
    (("SJ", "S9", "SA", "S10", "SK"), "hokum:spades"),
])
def test_native_auction_graph_retains_native_hand_frontier(rules, codes, expected):
    from ortools.sat.python import cp_model
    from baloot_platform.research.ortools_scenario_model import attach_auction_action_values_v1

    cards = {Card.parse(code).id for code in codes}
    purchase = Card.parse("SQ")
    def native_value(model, current, choices):
        hand, values = attach_auction_action_values_v1(
            model, rules, visible_cards=[model.new_constant(int(i in cards)) for i in range(32)],
            purchase_card=purchase, round_index=model.new_constant(1),
        )
        result = model.new_int_var(-1000 * hand.scale, 1000 * hand.scale,
                                   f"hand_margin_{hand.profile_hash}_scale_{hand.scale}")
        for code, choice in choices[0].items():
            if code != "closed":
                model.add(result == values[code]).only_enforce_if(choice)
        return result

    def forbidden_terminal(_):
        pytest.fail("auction frontier was mislabeled as a terminal round payoff")

    options = dict(dealer=Seat.EAST, purchase_suit=purchase.suit, purchase_rank=purchase.rank,
                   scope="bounded", horizon=1, boundary_evaluator="auction_hand_margin_test",
                   native_cp_sat=True, maximum_seconds=30, native_boundary_value=native_value,
                   terminal_utility=forbidden_terminal)
    receipts = []
    graph = compile_auction_decision_graph_v1(_auction_domain(rules), **options, native_model_receipts=receipts)
    nodes = {node.node_id: node for node in graph.nodes}
    edges = nodes[graph.root_node_id].edges
    assert all(nodes[edge.child_node_id].leaf_kind == "boundary" for edge in edges)
    best = max(nodes[edge.child_node_id].utility for edge in edges)
    assert [edge.action_code for edge in edges if nodes[edge.child_node_id].utility == best] == [expected]
    assert graph.to_dict() == compile_auction_decision_graph_v1(_auction_domain(rules), **options).to_dict()
    assert len(receipts) == len(graph.nodes)
    for receipt in receipts:
        encoded = encode_native_auction_model_receipt_v1(receipt)
        assert decode_native_auction_model_receipt_v1(encoded, maximum_model_bytes=10_000_000) == receipt
        assert len(json.dumps(encoded)) < len(json.dumps(receipt))
        assert encoded == encode_native_auction_model_receipt_v1(receipt)
        legacy = {**encoded, "modelEncoding": "cp_sat_proto_base64_v1",
                  "modelIdentity": {**encoded["modelIdentity"]}}
        for key in ("model", "boundaryUniquenessModel"):
            if key in legacy["modelIdentity"]:
                from google.protobuf import text_format
                from ortools.sat import cp_model_pb2
                proto = cp_model_pb2.CpModelProto()
                text_format.Parse(receipt["modelIdentity"][key], proto)
                legacy["modelIdentity"][key] = base64.b64encode(
                    proto.SerializeToString(deterministic=True)).decode("ascii")
        legacy["sha256"] = canonical_sha256({key: value for key, value in legacy.items() if key != "sha256"})
        assert len(json.dumps(encoded)) < len(json.dumps(legacy))
        assert decode_native_auction_model_receipt_v1(legacy, maximum_model_bytes=10_000_000) == receipt
        compressed_legacy = {**legacy, "modelEncoding": "cp_sat_proto_zlib_base64_v1",
                             "modelIdentity": {**legacy["modelIdentity"]}}
        for key in ("model", "boundaryUniquenessModel"):
            if key in compressed_legacy["modelIdentity"]:
                compressed_legacy["modelIdentity"][key] = base64.b64encode(zlib.compress(
                    base64.b64decode(legacy["modelIdentity"][key]))).decode("ascii")
        compressed_legacy["sha256"] = canonical_sha256({key: value for key, value in compressed_legacy.items() if key != "sha256"})
        assert decode_native_auction_model_receipt_v1(compressed_legacy, maximum_model_bytes=10_000_000) == receipt
        for payload, error in ((zlib.compress(b"x" * 1001), "budget"),
                               (zlib.compress(b"x")[:-1], "invalid"),
                               (zlib.compress(b"x") + b"trailing", "invalid")):
            invalid = {**encoded, "modelIdentity": {**encoded["modelIdentity"],
                       "model": base64.b64encode(payload).decode("ascii")}}
            invalid["sha256"] = canonical_sha256({key: value for key, value in invalid.items() if key != "sha256"})
            with pytest.raises(ValueError, match=error):
                decode_native_auction_model_receipt_v1(invalid, maximum_model_bytes=1000)
        with pytest.raises(ValueError, match="budget"):
            decode_native_auction_model_receipt_v1(encoded, maximum_model_bytes=1)
        changed = dict(encoded, sourceReceiptHash="0" * 64)
        changed["sha256"] = canonical_sha256({key: value for key, value in changed.items() if key != "sha256"})
        with pytest.raises(ValueError, match="identity mismatch"):
            decode_native_auction_model_receipt_v1(changed, maximum_model_bytes=10_000_000)
        malformed = {**encoded, "modelIdentity": {**encoded["modelIdentity"], "model": "AA=="}}
        malformed["sha256"] = canonical_sha256({key: value for key, value in malformed.items() if key != "sha256"})
        with pytest.raises(ValueError, match="protobuf is invalid"):
            decode_native_auction_model_receipt_v1(malformed, maximum_model_bytes=10_000_000)
        assert receipt["sha256"] == canonical_sha256({key: value for key, value in receipt.items() if key != "sha256"})
        assert receipt["modelHash"] == canonical_sha256(receipt["modelIdentity"])
        restored = cp_model.CpModel()
        assert restored.proto.parse_text_format(receipt["modelIdentity"]["model"])
        verifier = cp_model.CpSolver()
        verifier.parameters.num_search_workers = 1
        assert verifier.solve(restored) == cp_model.OPTIMAL
        for key, index in receipt["variableBindings"].items():
            assert verifier.value(restored.get_int_var_from_proto_index(index)) == receipt["derivedValues"][key]
        if "boundaryValue" in receipt["derivedValues"]:
            assert receipt["boundaryUniquenessStatus"] == "infeasible"
            uniqueness = cp_model.CpModel()
            assert uniqueness.proto.parse_text_format(receipt["modelIdentity"]["boundaryUniquenessModel"])
            assert verifier.solve(uniqueness) == cp_model.INFEASIBLE
    assert any(receipt["actions"] and receipt["actions"][-1]["trumpSuit"] == "spades" for receipt in receipts)
    repeated = []
    compile_auction_decision_graph_v1(_auction_domain(rules), **options, native_model_receipts=repeated)
    assert repeated == receipts


@pytest.mark.parametrize("overrides", [
    {"native_cp_sat": False}, {"boundary_utility": lambda _: 0}, {"scope": "exact"},
    {"native_boundary_value": lambda *args: 1},
    {"native_boundary_value": lambda model, *args: model.new_int_var(0, 1, "unconstrained")},
])
def test_native_auction_boundary_rejects_mixed_or_non_native_authority(rules, overrides):
    options = dict(dealer=Seat.EAST, purchase_suit=Suit.SPADES, purchase_rank=Rank.QUEEN,
                   scope="bounded", horizon=1, boundary_evaluator="native_test",
                   native_cp_sat=True, maximum_seconds=10,
                   native_boundary_value=lambda model, *args: model.new_constant(0),
                   terminal_utility=lambda _: 0)
    options.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        compile_auction_decision_graph_v1(_auction_domain(rules), **options)


@pytest.mark.parametrize("stop", ["unknown", "model_invalid", "cancelled"])
def test_native_auction_boundary_uniqueness_exhaustion_is_incomplete(rules, monkeypatch, stop):
    from ortools.sat.python import cp_model
    import baloot_platform.research.ortools_nonplay_graphs as graphs

    original = graphs._solve_cp_sat_with_cancel_v1
    cancelled = False
    receipts = []
    def solve(solver, model, cancel_check):
        nonlocal cancelled
        if any(row.name == "auction_boundary_value_uniqueness" for row in model.proto.constraints):
            cancelled = stop == "cancelled"
            return cp_model.MODEL_INVALID if stop == "model_invalid" else cp_model.UNKNOWN
        return original(solver, model, cancel_check)
    monkeypatch.setattr(graphs, "_solve_cp_sat_with_cancel_v1", solve)
    with pytest.raises(NonPlayGraphCompilationStoppedV1) as result:
        compile_auction_decision_graph_v1(
            _auction_domain(rules), dealer=Seat.EAST, purchase_suit=Suit.SPADES, purchase_rank=Rank.QUEEN,
            scope="bounded", horizon=1, boundary_evaluator="native_test", native_cp_sat=True,
            maximum_seconds=10, native_boundary_value=lambda model, *args: model.new_constant(0),
            terminal_utility=lambda _: 0, cancel_check=lambda: cancelled,
            native_model_receipts=receipts,
        )
    assert result.value.reason == ("cancelled" if cancelled else "native_boundary_incomplete")
    assert receipts[-1]["constructionStatus"] == "optimal"
    assert receipts[-1]["boundaryUniquenessStatus"] == ("model_invalid" if stop == "model_invalid" else "unknown")
    assert "boundaryUniquenessModel" in receipts[-1]["modelIdentity"]


def test_native_auction_graph_budget_and_cancellation_fail_closed(rules, monkeypatch):
    from baloot_platform.research import ortools_nonplay_graphs as module

    options = dict(dealer=Seat.EAST, purchase_suit=Suit.HEARTS,
                   terminal_utility=lambda _: 0, native_cp_sat=True)
    domain = _auction_domain(rules)
    for invalid in (None, True, 0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="time budget"):
            compile_auction_decision_graph_v1(domain, **options, maximum_seconds=invalid)
    with pytest.raises(NonPlayGraphCompilationStoppedV1, match="cancelled"):
        compile_auction_decision_graph_v1(domain, **options, maximum_seconds=10, cancel_check=lambda: True)
    with monkeypatch.context() as patch, pytest.raises(NonPlayGraphCompilationStoppedV1, match="maximum_seconds"):
        set_monotonic(patch, module, 0, 2)
        compile_auction_decision_graph_v1(domain, **options, maximum_seconds=1)
    with pytest.raises(ValueError, match="one solver worker"):
        compile_auction_decision_graph_v1(domain, **options, maximum_seconds=10, root_worker_count=2)


@pytest.mark.parametrize("official", [False, True])
@pytest.mark.parametrize("rank", [Rank.ACE, Rank.KING])
def test_auction_graph_preserves_exposed_rank(root, rules, official, rank):
    from baloot_platform.core.bidding_machine import BiddingMachine
    if official:
        rules = RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml")
    domain = _auction_domain(rules)
    actions = ("pass", "pass", "hokum", "pass")
    prefix = tuple(AuctionGraphChoiceV1.build(action, Suit.HEARTS if action == "hokum" else None)
                   for action in actions)
    options = dict(dealer=Seat.EAST, purchase_suit=Suit.HEARTS,
                   prefix=prefix, terminal_utility=lambda _: 0)
    graph = compile_auction_decision_graph_v1(domain, purchase_rank=rank, **options)
    machine = BiddingMachine(Seat.EAST, Suit.HEARTS, domain, rank)
    for action in actions:
        machine.submit(action)
    assert tuple(edge.action_code for edge in _root(graph).edges) == tuple(sorted(machine.legal_actions()))
    if official and rank == Rank.ACE:
        with pytest.raises(ValueError, match="illegal action"):
            compile_auction_decision_graph_v1(domain, purchase_rank=rank,
                **{**options, "prefix": (*prefix, AuctionGraphChoiceV1.build("sun"))})
    if not official:
        assert graph.to_dict() == compile_auction_decision_graph_v1(domain, **options).to_dict()


@pytest.mark.parametrize("rank", [None, True])
def test_rank_aware_auction_graph_rejects_missing_or_coercive_rank(root, rank):
    domain = _auction_domain(RuleBook.load(root / "configs/rules/saudi_federation_2023.yaml"))
    with pytest.raises((ValueError, TypeError), match="purchase_rank"):
        compile_auction_decision_graph_v1(domain, dealer=Seat.EAST, purchase_suit=Suit.HEARTS,
                                        purchase_rank=rank, terminal_utility=lambda _: 0)


def _policy_templates() -> dict:
    result = {}
    for seed, seat in enumerate(("north", "west", "south", "east"), start=1):
        core = {
            "schemaVersion": 1,
            "seat": seat,
            "policyId": "fixed_action_map_v1",
            "version": "1.0.0",
            "seed": seed,
            "choices": [],
            "materialization": "seeded_legal_action_order_v1",
        }
        result[seat] = {key: value for key, value in core.items() if key != "seat"}
        result[seat]["profileHash"] = canonical_sha256(core)
    return result


@pytest.mark.parametrize(
    ("claim", "utility_kind", "selected_action", "line_universe"),
    (
        ("outcome_reachable", "sun_only", "sun", None),
        ("outcome_forced", "all", "sun", None),
        ("winning_root_action_count", "sun_only", "sun", None),
        ("selected_action_is_unique_winner", "sun_only", "sun", None),
        ("target_team_has_no_salvage", "none", "pass", None),
        (
            "winning_continuation_line_count",
            "sun_only",
            "sun",
            "policy_conditioned",
        ),
    ),
)
def test_policy_conditioned_auction_graph_supports_all_six_claims(
    rules,
    claim: str,
    utility_kind: str,
    selected_action: str,
    line_universe: str | None,
) -> None:
    prefix = tuple(AuctionGraphChoiceV1.build("pass") for _ in range(7))

    def utility(terminal) -> int:
        if utility_kind == "all":
            return 1
        if utility_kind == "none":
            return 0
        return int(terminal.contract == Contract.SUN)

    graph = compile_auction_decision_graph_v1(
        _auction_domain(rules),
        dealer=Seat.EAST,
        purchase_suit=Suit.DIAMONDS,
        prefix=prefix,
        terminal_utility=utility,
    )
    request = {
        "schemaVersion": 1,
        "mode": "policy_conditioned",
        "visibility": "complete_state",
        "strategyClass": "frozen_policy",
        "claim": claim,
        "targetTeam": "west_east",
        "utility": "awarded_point_margin",
        "successThreshold": 1,
        "selectedAction": selected_action,
        "policies": _policy_templates(),
        "budgetProfile": "test",
    }
    if line_universe is not None:
        request["lineUniverse"] = line_universe

    proof = OrToolsScenarioGraphProofV1().prove(
        graph,
        ScenarioProofRequestV1.build(request),
    )

    assert proof.completion_status == "policy_complete"
    assert proof.claim_satisfied is True
    assert len(proof.root_actions) == 5
    if line_universe is not None:
        assert proof.continuation_lines is not None
        assert proof.continuation_lines.exact_count == "1"


@pytest.mark.parametrize("winning_count", [0, 1, 2])
def test_auction_graph_proves_zero_one_and_multiple_root_actions(
    rules,
    winning_count: int,
) -> None:
    domain = _auction_domain(rules)
    baseline = compile_auction_decision_graph_v1(
        domain,
        dealer=Seat.SOUTH,
        purchase_suit=Suit.HEARTS,
        terminal_utility=lambda _terminal: 0,
    )
    root_codes = tuple(edge.action_code for edge in _root(baseline).edges)
    winning = frozenset(root_codes[:winning_count])
    selected = root_codes[0] if winning_count == 1 else None
    claim = {
        0: "target_team_has_no_salvage",
        1: "selected_action_is_unique_winner",
        2: "outcome_reachable",
    }[winning_count]
    graph = compile_auction_decision_graph_v1(
        domain,
        dealer=Seat.SOUTH,
        purchase_suit=Suit.HEARTS,
        terminal_utility=lambda terminal: int(
            terminal.actions[0].choice_code in winning
        ),
    )

    proof = OrToolsScenarioGraphProofV1().prove(
        graph,
        _request(
            target_team="west_east",
            claim=claim,
            selected_action=selected,
        ),
    )

    assert proof.completion_status == "exact_complete"
    assert proof.winning_root_action_count == winning_count
    assert proof.claim_satisfied is True
    assert proof.principal_variation[0] in root_codes


def test_policy_auction_root_shards_are_semantically_deterministic(rules) -> None:
    def continuation(
        _actor: Seat,
        _node_id: str,
        actions: tuple[tuple[int, str], ...],
    ) -> int:
        return actions[0][0]

    options = dict(
        dealer=Seat.SOUTH,
        purchase_suit=Suit.HEARTS,
        terminal_utility=lambda terminal: int(terminal.contract == Contract.SUN),
        continuation_action_selector=continuation,
    )
    serial = compile_auction_decision_graph_v1(
        _auction_domain(rules),
        root_worker_count=1,
        **options,
    )
    parallel = compile_auction_decision_graph_v1(
        _auction_domain(rules),
        root_worker_count=4,
        **options,
    )

    assert parallel == serial


def test_auction_bounded_graph_uses_only_the_declared_frontier(rules) -> None:
    domain = _auction_domain(rules)
    baseline = compile_auction_decision_graph_v1(
        domain,
        dealer=Seat.SOUTH,
        purchase_suit=Suit.HEARTS,
        terminal_utility=lambda _terminal: 0,
    )
    selected = _root(baseline).edges[0].action_code

    def frontier(state) -> int:
        return int(state.actions[-1].choice_code == selected)

    first = compile_auction_decision_graph_v1(
        domain,
        dealer=Seat.SOUTH,
        purchase_suit=Suit.HEARTS,
        terminal_utility=lambda _terminal: 0,
        scope="bounded",
        horizon=1,
        boundary_evaluator="current_raw_point_margin_v1",
        boundary_utility=frontier,
    )
    second = compile_auction_decision_graph_v1(
        domain,
        dealer=Seat.SOUTH,
        purchase_suit=Suit.HEARTS,
        terminal_utility=lambda _terminal: 0,
        scope="bounded",
        horizon=1,
        boundary_evaluator="current_raw_point_margin_v1",
        boundary_utility=frontier,
    )

    assert first.sha256 == second.sha256
    assert {item.leaf_kind for item in first.nodes if item.is_leaf} == {
        "boundary",
        "terminal",
    }
    proof = OrToolsScenarioGraphProofV1().prove(
        first,
        _request(
            target_team="west_east",
            claim="selected_action_is_unique_winner",
            selected_action=selected,
            mode="bounded",
        ),
    )
    assert proof.completion_status == "bounded_complete"
    assert proof.winning_root_action_count == 1
    assert proof.principal_variation == (selected,)


def test_escalation_graph_uses_domain_transitions_and_stable_action_ids(rules) -> None:
    domain = EscalationDomainIRV1.from_rules(rules.data)
    kwargs = dict(
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        acting_seat=Seat.WEST,
        terminal_utility=lambda _terminal: 1,
    )
    first = compile_escalation_decision_graph_v1(domain, **kwargs)
    second = compile_escalation_decision_graph_v1(domain, **kwargs)
    root = _root(first)

    assert first.sha256 == second.sha256
    assert [(edge.action_id, edge.action_code) for edge in root.edges] == [
        (0, "decline"),
        (1, "double:locked"),
        (2, "double:open"),
    ]
    proof = OrToolsScenarioGraphProofV1().prove(
        first,
        _request(target_team="west_east", claim="outcome_forced"),
    )
    assert proof.completion_status == "exact_complete"
    assert proof.winning_root_action_count == 3
    assert proof.principal_variation[0] == "decline"


@pytest.mark.parametrize("contract,score", [(Contract.HOKUM, (0, 0)), (Contract.SUN, (101, 50))])
def test_native_escalation_graph_matches_reference_action_tree(rules, contract, score):
    domain = EscalationDomainIRV1.from_rules(rules.data)
    kwargs = dict(declarer=Seat.SOUTH, acting_seat=Seat.WEST, contract=contract,
                  match_score=score, terminal_utility=lambda terminal: int(terminal.state.level == "double"))
    legacy = compile_escalation_decision_graph_v1(domain, **kwargs)
    native = compile_escalation_decision_graph_v1(domain, **kwargs, native_cp_sat=True)

    def tree(graph):
        nodes = {node.node_id: node for node in graph.nodes}
        def visit(node_id):
            node = nodes[node_id]
            if node.is_leaf:
                return node.to_dict()
            return (node.actor_seat, tuple((edge.action_code, visit(edge.child_node_id)) for edge in node.edges))
        # Node identities deliberately differ because native graph nodes bind CP evidence.
        def strip(value):
            if isinstance(value, dict):
                return {key: item for key, item in value.items() if key != "nodeId"}
            if isinstance(value, tuple):
                return tuple(strip(item) for item in value)
            return value
        return strip(visit(graph.root_node_id))

    assert tree(native) == tree(legacy)
    assert native.sha256 != legacy.sha256
    assert native.sha256 == compile_escalation_decision_graph_v1(domain, **kwargs, native_cp_sat=True).sha256


def test_native_escalation_graph_derives_response_actor_and_skips_legacy_choices(rules, monkeypatch):
    import baloot_platform.research.ortools_nonplay_graphs as graphs

    def forbidden(*args, **kwargs):
        raise AssertionError("native graph called historical choice enumeration")

    monkeypatch.setattr(graphs, "_escalation_choices", forbidden)
    for name in ("initial_state", "can_close", "transition", "legal_decisions"):
        monkeypatch.setattr(EscalationDomainIRV1, name, forbidden)
    seen = []
    graph = compile_escalation_decision_graph_v1(
        EscalationDomainIRV1.from_rules(rules.data), declarer=Seat.SOUTH,
        contract=Contract.HOKUM, match_score=(0, 0), native_cp_sat=True,
        prefix=(EscalationGraphActionV1(Seat.WEST, "double", "open"),),
        terminal_utility=lambda terminal: int(terminal.state.level == "gahwa"),
        decision_observer=lambda observation: seen.append(observation) or "observer",
    )
    assert _root(graph).actor_seat == "south"
    assert {edge.action_code for edge in _root(graph).edges} == {"decline", "triple"}
    assert seen and all(item.actor is not None for item in seen)
    assert all("nativeTransitionEvidence" in item.public_state for item in seen)


def test_native_escalation_graph_refuses_parallel_roots(rules):
    with pytest.raises(ValueError, match="one root worker"):
        compile_escalation_decision_graph_v1(
            EscalationDomainIRV1.from_rules(rules.data), declarer=Seat.SOUTH,
            contract=Contract.HOKUM, match_score=(0, 0), native_cp_sat=True,
            root_worker_count=2, terminal_utility=lambda terminal: 0,
        )


@pytest.mark.parametrize("prior", [(), ("west",)])
def test_frozen_escalation_order_matches_explicit_window_graph(rules, prior):
    domain = EscalationDomainIRV1.from_rules(rules.data)
    kwargs = dict(declarer=Seat.SOUTH, contract=Contract.HOKUM, match_score=(0, 0),
        native_cp_sat=True, terminal_utility=lambda terminal: int(terminal.state.level == "double"),
        prior_non_announcing_seats=prior, acting_seat="east" if prior else "west")
    explicit = compile_escalation_decision_graph_v1(domain, **kwargs,
        opportunity_orders={canonical_sha256({"actions": []}): (Seat.WEST, Seat.EAST)})
    frozen = compile_escalation_decision_graph_v1(domain, **kwargs,
        opportunity_order=("west", "east", "south", "north"))
    assert frozen.to_dict() == explicit.to_dict()


@pytest.mark.parametrize("order", [(), ("west",), ("west",) * 4,
    (True, "east", "south", "north"), "west"])
def test_frozen_escalation_order_rejects_invalid_seats(rules, order):
    with pytest.raises(ValueError, match="every seat exactly once"):
        compile_escalation_decision_graph_v1(EscalationDomainIRV1.from_rules(rules.data),
            declarer=Seat.SOUTH, contract=Contract.HOKUM, match_score=(0, 0),
            native_cp_sat=True, terminal_utility=lambda _terminal: 0, opportunity_order=order)


@pytest.mark.parametrize("maximum_worlds", [1, 2])
def test_private_root_histories_cover_prior_declines_without_observer_leak(rules, maximum_worlds):
    domain = EscalationDomainIRV1.from_rules(rules.data)
    kwargs = dict(declarer=Seat.SOUTH, contract=Contract.HOKUM, match_score=(0, 0), acting_seat=Seat.WEST)
    coverage = enumerate_escalation_opportunity_orders_v1(domain, **kwargs,
        maximum_worlds=maximum_worlds, maximum_seconds=5, budget=NonPlayGraphBudgetV1(100, 8),
        include_private_root_histories=True)
    assert coverage["rootHistoryScope"] == "actor_visible_private_histories"
    if maximum_worlds == 1:
        assert coverage["complete"] is False
        assert coverage["stopReason"] == "maximum_order_worlds"
        assert coverage["worlds"] == []
        return
    assert coverage["complete"] is True
    assert coverage["orderWorldCount"] == 2
    assert {tuple(world["priorNonAnnouncingSeats"]) for world in coverage["worlds"]} == {(), ("east",)}
    roots, views = [], []
    for world in coverage["worlds"]:
        observed = []
        graph = compile_escalation_decision_graph_v1(domain, **kwargs, native_cp_sat=True,
            opportunity_orders=world["opportunityOrders"],
            prior_non_announcing_seats=world["priorNonAnnouncingSeats"],
            terminal_utility=lambda terminal: 0,
            decision_observer=lambda value: observed.append(value) or "observer")
        root = _root(graph)
        assert root.actor_seat == "west"
        roots.append(root)
        views.append(next(row.public_state for row in observed if row.node_id == root.node_id))
    assert roots[0].node_id != roots[1].node_id
    assert views[0] == views[1]
    assert [edge.action_code for edge in roots[0].edges] == [edge.action_code for edge in roots[1].edges]


@pytest.mark.parametrize("prior", [(Seat.WEST,), (Seat.EAST, Seat.EAST), (Seat.NORTH,), (Seat.EAST, Seat.WEST)])
def test_private_root_history_rejects_impossible_order_or_closed_window(rules, prior):
    with pytest.raises(ValueError):
        compile_escalation_decision_graph_v1(EscalationDomainIRV1.from_rules(rules.data),
            declarer=Seat.SOUTH, contract=Contract.HOKUM, match_score=(0, 0), acting_seat=Seat.WEST,
            native_cp_sat=True, opportunity_orders={canonical_sha256({"actions": []}): (Seat.EAST, Seat.WEST)},
            prior_non_announcing_seats=prior, terminal_utility=lambda terminal: 0)


@pytest.mark.parametrize("observer,own,count", [("south", False, 2), ("west", False, 1),
    ("west", True, 1), ("east", False, 2), ("east", True, 0)])
def test_native_prior_history_enumeration_conditions_only_on_own_recall(rules, observer, own, count, monkeypatch):
    from baloot_platform.research.ortools_nonplay_graphs import enumerate_escalation_prior_histories_v1
    def forbidden(*args, **kwargs):
        pytest.fail("native history coverage used interpreted escalation truth")
    for name in ("transition", "legal_decisions", "can_close"):
        monkeypatch.setattr(EscalationDomainIRV1, name, forbidden)
    key = canonical_sha256({"actions": []})
    kwargs = dict(declarer=Seat.SOUTH, contract=Contract.HOKUM, match_score=(0, 0),
        observer_seat=observer, prefix=(EscalationGraphActionV1.build("east", "double", "open"),),
        own_declined_windows=(key,) if own else (), maximum_worlds=4, maximum_seconds=15,
        budget=NonPlayGraphBudgetV1(maximum_nodes=30, maximum_depth=8))
    domain = EscalationDomainIRV1.from_rules(rules.data)
    result = enumerate_escalation_prior_histories_v1(domain, **kwargs)
    assert result["complete"] is True
    assert result["worldCount"] == count
    assert result["status"] == ("satisfied" if count else "infeasible")
    assert result["sha256"] == enumerate_escalation_prior_histories_v1(domain, **kwargs)["sha256"]
    if count == 2:
        assert {canonical_sha256(row["privateRecall"]) for row in result["worlds"]} == {
            canonical_sha256({}), canonical_sha256({"west": [key]})}
    elif count == 1:
        assert result["worlds"][0]["privateRecall"] == ({"west": [key]} if own else {})
    assert result["evidence"]
    assert all(row["status"] in {"satisfied", "infeasible"} for row in result["evidence"].values())


@pytest.mark.parametrize("own,expected", [(False, "infeasible"), (True, "satisfied")])
def test_native_prior_history_closed_window_is_solver_constrained(rules, own, expected):
    from baloot_platform.research.ortools_nonplay_graphs import enumerate_escalation_prior_histories_v1
    earlier = canonical_sha256({"actions": []})
    current = canonical_sha256({"actions": [["east", "double", "open"]]})
    result = enumerate_escalation_prior_histories_v1(EscalationDomainIRV1.from_rules(rules.data),
        declarer=Seat.SOUTH, contract=Contract.HOKUM, match_score=(0, 0), observer_seat="south",
        prefix=(EscalationGraphActionV1.build("east", "double", "open"),), closed=True,
        own_declined_windows=(current,) if own else (), maximum_worlds=4, maximum_seconds=15,
        budget=NonPlayGraphBudgetV1(maximum_nodes=30, maximum_depth=8))
    assert result["complete"] is True
    assert result["status"] == expected
    assert result["worldCount"] == (2 if own else 0)
    for row in result["worlds"]:
        assert row["privateRecall"]["south"] == [current]
        assert row["privateRecall"].get("west", []) in ([], [earlier])


@pytest.mark.parametrize("limit,reason", [("worlds", "maximum_worlds"), ("nodes", "maximum_nodes"),
    ("depth", "maximum_depth"), ("time", "maximum_seconds"), ("cancel", "cancelled")])
def test_native_prior_history_exhaustion_retains_evidence_not_worlds(rules, limit, reason, monkeypatch):
    from baloot_platform.research import ortools_nonplay_graphs as module

    if limit == "time":
        set_monotonic(monkeypatch, module, 0, 2)
    result = module.enumerate_escalation_prior_histories_v1(EscalationDomainIRV1.from_rules(rules.data),
        declarer=Seat.SOUTH, contract=Contract.HOKUM, match_score=(0, 0), observer_seat="south",
        prefix=(EscalationGraphActionV1.build("east", "double", "open"),),
        maximum_worlds=1 if limit == "worlds" else 4,
        maximum_seconds=1 if limit == "time" else 15,
        budget=NonPlayGraphBudgetV1(maximum_nodes=2 if limit == "nodes" else 30,
                                   maximum_depth=1 if limit == "depth" else 8),
        cancel_check=(lambda: True) if limit == "cancel" else None)
    assert result["complete"] is False
    assert result["status"] == "incomplete"
    assert result["stopReason"] == reason
    assert result["worldCount"] is None and result["worlds"] == []
    if limit in {"worlds", "nodes", "depth"}:
        assert result["evidence"]


def test_prior_window_recall_survives_next_native_escalation_window(rules):
    earlier = canonical_sha256({"actions": []})
    current = canonical_sha256({"actions": [["east", "double", "open"]]})
    observed, terminals = [], []
    graph = compile_escalation_decision_graph_v1(EscalationDomainIRV1.from_rules(rules.data),
        declarer=Seat.SOUTH, contract=Contract.HOKUM, match_score=(0, 0), native_cp_sat=True,
        prefix=(EscalationGraphActionV1.build("east", "double", "open"),),
        opportunity_orders={current: (Seat.SOUTH,)}, prior_window_recall={"west": [earlier]},
        decision_observer=lambda value: observed.append(value) or "observer",
        terminal_utility=lambda value: terminals.append(value) or 0)
    assert _root(graph).actor_seat == "south"
    root = next(value for value in observed if value.node_id == graph.root_node_id)
    assert root.own_private_windows == ()
    assert "privateHistory" not in root.public_state
    assert any(value.private_history == (("west", earlier), ("south", current)) for value in terminals)


@pytest.mark.parametrize("bad", ["current", "unknown", "reversed"])
def test_prior_window_recall_rejects_unreachable_history(rules, bad):
    actions = (("east", "double", "open"), ("south", "triple", None), ("east", "four", "open"))
    windows = [canonical_sha256({"actions": [list(action) for action in actions[:index]]}) for index in range(4)]
    recall = {"west": [windows[-1]] if bad == "current" else ["f" * 64] if bad == "unknown" else windows[1::-1]}
    with pytest.raises(ValueError, match="earlier public announcement window|chronological"):
        compile_escalation_decision_graph_v1(EscalationDomainIRV1.from_rules(rules.data),
            declarer=Seat.SOUTH, contract=Contract.HOKUM, match_score=(0, 0), native_cp_sat=True,
            prefix=tuple(EscalationGraphActionV1.build(*action) for action in actions),
            opportunity_orders={windows[-1]: (Seat.SOUTH,)}, prior_window_recall=recall,
            terminal_utility=lambda value: 0)


def test_native_opportunity_decline_reaches_teammate_without_public_leak(rules):
    key = canonical_sha256({"actions": []})
    observed = []
    terminals = []
    def utility(terminal):
        terminals.append(terminal)
        return int(terminal.state.level == "double")
    kwargs = dict(declarer=Seat.SOUTH, contract=Contract.HOKUM, match_score=(0, 0),
                  native_cp_sat=True, terminal_utility=utility)
    graph = compile_escalation_decision_graph_v1(
        EscalationDomainIRV1.from_rules(rules.data), **kwargs,
        opportunity_orders={key: (Seat.WEST, Seat.EAST)},
        decision_observer=lambda value: observed.append(value) or "observer",
    )
    nodes = {node.node_id: node for node in graph.nodes}
    root = _root(graph)
    assert root.actor_seat == "west"
    east = nodes[next(edge.child_node_id for edge in root.edges if edge.action_code == "decline")]
    assert east.actor_seat == "east"
    closed = nodes[next(edge.child_node_id for edge in east.edges if edge.action_code == "decline")]
    assert closed.is_leaf
    west_view = next(value for value in observed if value.node_id == root.node_id)
    east_view = next(value for value in observed if value.node_id == east.node_id)
    assert west_view.public_state == east_view.public_state
    assert west_view.public_state["actions"] == []
    assert "nonAnnouncingSeats" not in east_view.public_state
    assert "opportunityOrder" not in east_view.public_state
    assert "opportunityEvidence" not in east_view.public_state
    assert east_view.own_private_windows == ()
    assert any(terminal.private_history == (("west", key), ("east", key)) for terminal in terminals)
    from baloot_platform.research.ortools_nonplay_play_graft import OrToolsNonPlayPlayGraphGraftV1
    observations = {node.node_id: "public-observation" for node in graph.nodes if not node.is_leaf}
    public = OrToolsNonPlayPlayGraphGraftV1._namespace_observations(graph, "test", observations)
    recalled = OrToolsNonPlayPlayGraphGraftV1._namespace_observations(graph, "test", observations,
        private_history=(("west", key),))
    for node in graph.nodes:
        if not node.is_leaf:
            namespaced = next(name for name in public if name.endswith(node.node_id))
            assert (recalled[namespaced] != public[namespaced]) is (node.actor_seat == "west")
    reverse_views = []
    reverse = compile_escalation_decision_graph_v1(
        EscalationDomainIRV1.from_rules(rules.data), **kwargs,
        opportunity_orders={key: (Seat.EAST, Seat.WEST)},
        decision_observer=lambda value: reverse_views.append(value) or "observer",
    )
    assert reverse.sha256 != graph.sha256
    assert _root(reverse).actor_seat == "east"
    assert next(value for value in reverse_views if value.node_id == reverse.root_node_id).public_state == east_view.public_state
    assert graph.sha256 == compile_escalation_decision_graph_v1(
        EscalationDomainIRV1.from_rules(rules.data), **kwargs,
        opportunity_orders={key: (Seat.WEST, Seat.EAST)},
    ).sha256


@pytest.mark.parametrize("order", [(), (Seat.WEST,), (Seat.WEST, Seat.WEST), (Seat.NORTH, Seat.SOUTH)])
def test_native_opportunity_order_cannot_omit_or_invent_actors(rules, order):
    with pytest.raises(ValueError):
        compile_escalation_decision_graph_v1(
            EscalationDomainIRV1.from_rules(rules.data), declarer=Seat.SOUTH,
            contract=Contract.HOKUM, match_score=(0, 0), native_cp_sat=True,
            opportunity_orders={canonical_sha256({"actions": []}): order},
            terminal_utility=lambda terminal: 0,
        )


def test_native_opportunity_proof_can_choose_teammates_announcement(rules):
    # A synthetic utility oracle isolates graph behavior; not a full-round strength claim.
    graph = compile_escalation_decision_graph_v1(
        EscalationDomainIRV1.from_rules(rules.data), declarer=Seat.SOUTH,
        contract=Contract.SUN, match_score=(101, 50), native_cp_sat=True,
        opportunity_orders={canonical_sha256({"actions": []}): (Seat.WEST, Seat.EAST)},
        terminal_utility=lambda terminal: int(bool(terminal.actions) and terminal.actions[-1].actor == Seat.EAST),
    )
    proof = OrToolsScenarioGraphProofV1().prove(
        graph, _request(target_team="west_east", claim="selected_action_is_unique_winner",
                        selected_action="decline")
    )
    assert proof.completion_status == "exact_complete"
    assert proof.winning_root_action_count == 1
    assert proof.principal_variation[0] == "decline"
    assert len(proof.principal_variation) == 2
    forced = OrToolsScenarioGraphProofV1().prove(
        graph, _request(target_team="west_east", claim="outcome_forced")
    )
    assert forced.completion_status == "proof_failed"


@pytest.mark.parametrize("contract,score", [(Contract.SUN, (101, 50)), (Contract.HOKUM, (0, 0))])
def test_native_order_enumeration_is_complete_and_directly_compilable(rules, contract, score):
    domain = EscalationDomainIRV1.from_rules(rules.data)
    kwargs = dict(declarer=Seat.SOUTH, contract=contract, match_score=score,
                  maximum_worlds=2, maximum_seconds=10, budget=NonPlayGraphBudgetV1(100, 8))
    coverage = enumerate_escalation_opportunity_orders_v1(domain, **kwargs)
    assert coverage["complete"] is True
    assert coverage["orderWorldCount"] == len(coverage["worlds"]) == 2
    assert coverage["sha256"] == enumerate_escalation_opportunity_orders_v1(domain, **kwargs)["sha256"]
    roots = set()
    for world in coverage["worlds"]:
        graph = compile_escalation_decision_graph_v1(domain,
            declarer=Seat.SOUTH, contract=contract, match_score=score, native_cp_sat=True,
            opportunity_orders=world["opportunityOrders"], terminal_utility=lambda value: 0)
        roots.add(_root(graph).actor_seat)
    assert roots == {"west", "east"}
    assert all(window["solverStatus"] == "optimal" for window in coverage["windows"].values())


@pytest.mark.parametrize("limit,reason", [("worlds", "maximum_order_worlds"), ("nodes", "maximum_nodes"), ("depth", "maximum_depth"), ("cancel", "cancelled")])
def test_native_order_enumeration_preserves_partial_receipts(rules, limit, reason):
    result = enumerate_escalation_opportunity_orders_v1(
        EscalationDomainIRV1.from_rules(rules.data), declarer=Seat.SOUTH,
        contract=Contract.HOKUM, match_score=(0, 0), maximum_worlds=1 if limit == "worlds" else 2,
        maximum_seconds=10, budget=NonPlayGraphBudgetV1(2 if limit == "nodes" else 100, 1 if limit == "depth" else 8),
        cancel_check=(lambda: True) if limit == "cancel" else None,
    )
    assert result["complete"] is False
    assert result["stopReason"] == reason
    assert result["orderWorldCount"] is None
    assert result["worlds"] == []
    assert bool(result["windows"]) == (limit != "cancel")
    assert result["sha256"] == canonical_sha256({key: value for key, value in result.items() if key != "sha256"})


@pytest.mark.parametrize("reason", ["maximum_seconds", "native_transition_incomplete"])
def test_native_order_enumeration_time_and_solver_exhaustion(rules, monkeypatch, reason):
    import itertools
    from types import SimpleNamespace
    import baloot_platform.research.ortools_nonplay_graphs as graphs

    calls = []
    def incomplete(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status="incomplete", to_dict=lambda: {"status": "incomplete"})
    monkeypatch.setattr(graphs.OrToolsEscalationStageV1, "solve", incomplete)
    if reason == "maximum_seconds":
        clock = itertools.chain((0,), itertools.repeat(2))
        monkeypatch.setattr(graphs.time, "monotonic", lambda: next(clock))
    result = enumerate_escalation_opportunity_orders_v1(
        EscalationDomainIRV1.from_rules(rules.data), declarer=Seat.SOUTH,
        contract=Contract.HOKUM, match_score=(0, 0), maximum_worlds=2,
        maximum_seconds=1, budget=NonPlayGraphBudgetV1(100, 8),
    )
    assert result["complete"] is False
    assert result["stopReason"] == reason
    assert result["worlds"] == []
    assert bool(calls) == (reason == "native_transition_incomplete")


def test_native_order_enumeration_counts_independent_response_windows(rules):
    import copy
    data = copy.deepcopy(rules.data)
    data["escalation"]["response_actor_policy"] = "team"
    result = enumerate_escalation_opportunity_orders_v1(
        EscalationDomainIRV1.from_rules(data), declarer=Seat.SOUTH,
        contract=Contract.HOKUM, match_score=(0, 0), maximum_worlds=2,
        maximum_seconds=5, budget=NonPlayGraphBudgetV1(100, 8),
    )
    assert result["complete"] is False
    assert result["stopReason"] == "maximum_order_worlds"
    assert result["orderWorldCountLowerBound"] == 4
    assert result["worlds"] == []


@pytest.mark.parametrize("hidden,cap,complete,reason", [(3, 6, True, None), (3, 5, False, "maximum_order_worlds"), (7, 6, False, "maximum_combined_worlds")])
def test_order_and_hidden_worlds_share_one_budget(rules, hidden, cap, complete, reason):
    result = enumerate_escalation_opportunity_orders_v1(
        EscalationDomainIRV1.from_rules(rules.data), declarer=Seat.SOUTH,
        contract=Contract.SUN, match_score=(101, 50), hidden_world_count=hidden,
        maximum_worlds=cap, maximum_seconds=5, budget=NonPlayGraphBudgetV1(100, 8),
    )
    assert result["complete"] is complete
    assert result["stopReason"] == reason
    assert result["hiddenWorldCount"] == hidden
    assert result["combinedWorldCount"] == (6 if complete else None)
    assert len(result["worlds"]) == (2 if complete else 0)
    if hidden > cap:
        assert result["windows"] == {}


@pytest.mark.parametrize("hidden", [True, 0, -1, 1.0, "1"])
def test_combined_world_count_rejects_coercion(rules, hidden):
    with pytest.raises(ValueError, match="hidden_world_count"):
        enumerate_escalation_opportunity_orders_v1(
            EscalationDomainIRV1.from_rules(rules.data), declarer=Seat.SOUTH,
            contract=Contract.SUN, match_score=(101, 50), hidden_world_count=hidden,
            maximum_worlds=6, maximum_seconds=5, budget=NonPlayGraphBudgetV1(100, 8),
        )


@pytest.mark.parametrize("actor,complete", [(Seat.WEST, True), (Seat.EAST, True), (Seat.NORTH, False)])
def test_order_enumeration_preserves_the_known_root_actor(rules, actor, complete):
    domain = EscalationDomainIRV1.from_rules(rules.data)
    result = enumerate_escalation_opportunity_orders_v1(domain, declarer=Seat.SOUTH,
        contract=Contract.SUN, match_score=(101, 50), acting_seat=actor,
        maximum_worlds=2, maximum_seconds=5, budget=NonPlayGraphBudgetV1(100, 8))
    assert result["complete"] is complete
    assert result["actingSeat"] == actor.name.lower()
    if complete:
        assert result["orderWorldCount"] == 1
        graph = compile_escalation_decision_graph_v1(domain, declarer=Seat.SOUTH,
            contract=Contract.SUN, match_score=(101, 50), acting_seat=actor,
            native_cp_sat=True, opportunity_orders=result["worlds"][0]["opportunityOrders"],
            terminal_utility=lambda terminal: 0)
        assert _root(graph).actor_seat == actor.name.lower()
    else:
        assert result["stopReason"] == "ineligible_root_actor"
        assert result["worlds"] == []


@pytest.mark.real_clock
@pytest.mark.parametrize("reason", ["cancelled", "native_transition_incomplete"])
def test_native_escalation_graph_preserves_incomplete_status(rules, monkeypatch, reason):
    from types import SimpleNamespace
    import baloot_platform.research.ortools_nonplay_graphs as graphs

    def stopped(*args, **kwargs):
        if reason == "cancelled":
            raise InterruptedError("cancelled")
        return SimpleNamespace(status="incomplete")

    monkeypatch.setattr(graphs.OrToolsEscalationStageV1, "solve", stopped)
    with pytest.raises(NonPlayGraphCompilationStoppedV1) as exc:
        compile_escalation_decision_graph_v1(
            EscalationDomainIRV1.from_rules(rules.data), declarer=Seat.SOUTH,
            contract=Contract.HOKUM, match_score=(0, 0), native_cp_sat=True,
            terminal_utility=lambda terminal: 0,
        )
    assert exc.value.reason == reason


def test_native_forced_escalation_retains_its_legal_root_action(rules) -> None:
    domain = EscalationDomainIRV1.from_rules(rules.data)
    graph = compile_escalation_decision_graph_v1(
        domain, declarer=Seat.NORTH, contract=Contract.SUN,
        match_score=(0, 0), acting_seat=Seat.WEST,
        terminal_utility=lambda _terminal: 0, native_cp_sat=True,
    )
    assert tuple(edge.action_code for edge in _root(graph).edges) == (domain.closure_action,)
    assert len(graph.nodes) == 2


def test_forced_escalation_closure_is_not_a_strategic_root(rules) -> None:
    with pytest.raises(
        ValueError,
        match="escalation decision root requires at least two legal actions",
    ):
        compile_escalation_decision_graph_v1(
            EscalationDomainIRV1.from_rules(rules.data),
            declarer=Seat.NORTH,
            contract=Contract.SUN,
            match_score=(0, 0),
            acting_seat=Seat.WEST,
            terminal_utility=lambda _terminal: 0,
        )


def test_policy_escalation_root_shards_are_semantically_deterministic(rules) -> None:
    def continuation(
        _actor: Seat,
        _node_id: str,
        actions: tuple[tuple[int, str], ...],
    ) -> int:
        return actions[0][0]

    options = dict(
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        acting_seat=Seat.WEST,
        terminal_utility=lambda terminal: int(terminal.state.level != "normal"),
        continuation_action_selector=continuation,
    )
    serial = compile_escalation_decision_graph_v1(
        EscalationDomainIRV1.from_rules(rules.data),
        root_worker_count=1,
        **options,
    )
    parallel = compile_escalation_decision_graph_v1(
        EscalationDomainIRV1.from_rules(rules.data),
        root_worker_count=4,
        **options,
    )

    assert parallel == serial


def test_escalation_graph_has_one_team_closure_and_both_opening_seats(rules) -> None:
    graph = compile_escalation_decision_graph_v1(
        EscalationDomainIRV1.from_rules(rules.data),
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        terminal_utility=lambda _terminal: 0,
    )
    root = _root(graph)

    assert root.actor_seat is None
    assert [edge.action_code for edge in root.edges] == [
        "decline",
        "east:double:locked",
        "east:double:open",
        "west:double:locked",
        "west:double:open",
    ]


def test_composed_team_escalation_policy_is_public_deterministic_and_explicit(
    rules,
) -> None:
    domain = EscalationDomainIRV1.from_rules(rules.data)
    state = domain.initial_state()
    decisions = domain.legal_decisions(
        state,
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        include_closure=True,
    )
    policies = {
        Seat.parse(seat): DeclarativeActionPolicyV1.build(seat, value)
        for seat, value in _policy_templates().items()
    }

    first, first_receipt = composed_team_legal_action_order_v1(
        domain=domain,
        team=Seat.SOUTH.team.other,
        node_id="public-escalation-opening",
        decisions=decisions,
        policies=policies,
    )
    second, second_receipt = composed_team_legal_action_order_v1(
        domain=domain,
        team=Seat.SOUTH.team.other,
        node_id="public-escalation-opening",
        decisions=tuple(reversed(decisions)),
        policies=dict(reversed(tuple(policies.items()))),
    )

    assert first.code == second.code
    assert first_receipt == second_receipt
    assert first_receipt["strategyId"] == "composed_team_legal_action_order_v1"
    assert first_receipt["domainIrHash"] == domain.sha256
    assert {item["seat"] for item in first_receipt["members"]} == {"west", "east"}
    assert {item["code"] for item in first_receipt["legalDecisions"]} == {
        "decline",
        "east:double:locked",
        "east:double:open",
        "west:double:locked",
        "west:double:open",
    }
    assert first_receipt["selectedDecision"]["code"] == first.code
    with pytest.raises(ValueError, match="requires both team members"):
        composed_team_legal_action_order_v1(
            domain=domain,
            team=Seat.SOUTH.team.other,
            node_id="public-escalation-opening",
            decisions=decisions,
            policies={Seat.WEST: policies[Seat.WEST]},
        )


def test_escalation_prefix_derives_the_exact_response_actor(rules) -> None:
    graph = compile_escalation_decision_graph_v1(
        EscalationDomainIRV1.from_rules(rules.data),
        declarer=Seat.SOUTH,
        contract=Contract.HOKUM,
        match_score=(0, 0),
        acting_seat=Seat.SOUTH,
        prefix=(EscalationGraphActionV1.build(Seat.WEST, "double", "open"),),
        terminal_utility=lambda terminal: int(terminal.state.level != "normal"),
    )
    root = _root(graph)

    assert root.actor_seat == "south"
    assert [edge.action_code for edge in root.edges] == ["decline", "triple"]


def test_project_graph_models_mandatory_declaration_as_forced_transition(rules) -> None:
    hand = (
        Card.of(Suit.HEARTS, Rank.SEVEN),
        Card.of(Suit.HEARTS, Rank.EIGHT),
        Card.of(Suit.HEARTS, Rank.NINE),
        Card.of(Suit.SPADES, Rank.ACE),
        Card.of(Suit.SPADES, Rank.KING),
        Card.of(Suit.CLUBS, Rank.TEN),
        Card.of(Suit.DIAMONDS, Rank.JACK),
        Card.of(Suit.CLUBS, Rank.QUEEN),
    )
    domain = rules.project_domain
    projects = domain.select(domain.candidates(Seat.NORTH, hand, Contract.SUN))
    assert projects
    graph = compile_project_declaration_graph_v1(
        domain,
        projects=projects,
        actor=Seat.NORTH,
        dealer=Seat.NORTH,
        prefix=(
            {"actor": "west", "action": "pass"},
            {"actor": "south", "action": "pass"},
            {"actor": "east", "action": "pass"},
        ),
        trump_suit=None,
        terminal_utility=lambda terminal: int(
            terminal.announced is not None and terminal.announced.value > 0
        ),
    )
    root = _root(graph)
    assert len(root.edges) == 1
    assert root.edges[0].action_code.startswith("declare:")
    terminal = next(
        node for node in graph.nodes if node.node_id == root.edges[0].child_node_id
    )
    assert terminal.is_leaf
    assert terminal.utility == 1


def test_project_graph_uses_pass_when_no_declaration_is_legal(rules) -> None:
    graph = compile_project_declaration_graph_v1(
        rules.project_domain,
        projects=(),
        actor=Seat.NORTH,
        dealer=Seat.NORTH,
        prefix=(
            {"actor": "west", "action": "pass"},
            {"actor": "south", "action": "pass"},
            {"actor": "east", "action": "pass"},
        ),
        trump_suit=None,
        terminal_utility=lambda terminal: int(terminal.announced is None),
    )

    root = _root(graph)
    assert [edge.action_code for edge in root.edges] == ["pass"]
    terminal = next(
        node for node in graph.nodes if node.node_id == root.edges[0].child_node_id
    )
    assert terminal.is_leaf
    assert terminal.utility == 1


@pytest.mark.parametrize("mode", ("exact", "bounded", "policy_conditioned"))
def test_optional_project_graph_proves_real_pass_declare_root(
    mode: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    rules = RuleBook.load(
        root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
    )
    hand = (
        Card.of(Suit.HEARTS, Rank.SEVEN),
        Card.of(Suit.HEARTS, Rank.EIGHT),
        Card.of(Suit.HEARTS, Rank.NINE),
        Card.of(Suit.SPADES, Rank.ACE),
        Card.of(Suit.SPADES, Rank.KING),
        Card.of(Suit.CLUBS, Rank.TEN),
        Card.of(Suit.DIAMONDS, Rank.JACK),
        Card.of(Suit.CLUBS, Rank.QUEEN),
    )
    projects = rules.project_domain.select(
        rules.project_domain.candidates(Seat.NORTH, hand, Contract.SUN)
    )
    graph = compile_project_declaration_graph_v1(
        rules.project_domain,
        projects=projects,
        actor=Seat.NORTH,
        dealer=Seat.NORTH,
        prefix=(
            {"actor": "west", "action": "pass"},
            {"actor": "south", "action": "pass"},
            {"actor": "east", "action": "pass"},
        ),
        trump_suit=None,
        terminal_utility=lambda terminal: int(terminal.announced is not None),
        scope="bounded" if mode == "bounded" else "exact",
        horizon=1 if mode == "bounded" else None,
        boundary_evaluator=(
            "current_raw_point_margin_v1" if mode == "bounded" else None
        ),
        boundary_utility=(lambda _terminal: 0) if mode == "bounded" else None,
    )
    root_node = _root(graph)
    assert [edge.action_code.split(":", 1)[0] for edge in root_node.edges] == [
        "pass",
        "declare",
    ]
    request = {
        "schemaVersion": 1,
        "mode": mode,
        "visibility": "complete_state",
        "claim": "outcome_reachable",
        "targetTeam": "north_south",
        "utility": "awarded_point_margin",
        "successThreshold": 1,
        "budgetProfile": "test",
    }
    if mode == "bounded":
        request.update(
            {
                "horizon": {"unit": "plies", "value": 1},
                "boundaryEvaluator": "current_raw_point_margin_v1",
            }
        )
    elif mode == "policy_conditioned":
        request.update(
            {
                "strategyClass": "frozen_policy",
                "policies": _policy_templates(),
            }
        )
    proof = OrToolsScenarioGraphProofV1().prove(
        graph,
        ScenarioProofRequestV1.build(request),
    )
    assert proof.completion_status == {
        "exact": "exact_complete",
        "bounded": "proof_failed",
        "policy_conditioned": "policy_complete",
    }[mode]
    if mode == "bounded":
        assert {item.value for item in proof.root_actions} == {0}
        assert all(node.leaf_kind == "boundary" for node in graph.nodes if node.is_leaf)
    if mode == "exact":
        line_proof = OrToolsScenarioGraphProofV1().prove(
            graph,
            ScenarioProofRequestV1.build(
                {
                    "schemaVersion": 1,
                    "mode": "exact",
                    "visibility": "complete_state",
                    "claim": "winning_continuation_line_count",
                    "targetTeam": "north_south",
                    "utility": "awarded_point_margin",
                    "successThreshold": 1,
                    "lineUniverse": "all_legal",
                    "budgetProfile": "test",
                }
            ),
        )
        assert line_proof.continuation_lines is not None
        assert line_proof.continuation_lines.exact_count == "1"


def test_optional_project_graft_preserves_interleaved_first_trick_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    rules = RuleBook.load(
        root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
    )
    prefix = ExactTracePrefixV1.build(
        {
            "schemaVersion": 1,
            "deckOrder": list(range(32)),
            "dealer": "east",
            "roundNumber": 1,
            "matchScore": [0, 0],
            "auctionActions": [
                {"action": action}
                for action in ("pass", "pass", "sun", "pass", "sun")
            ],
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
    layout = DealLayoutV1.build(
        rules.structural_domain,
        dealer=Seat.EAST,
        buyer=Seat.NORTH,
    )
    assert layout.final_hand_positions is not None
    hands = {
        seat: tuple(
            Card.parse(prefix.deck_order[position])
            for position in layout.final_hand_positions[int(seat)]
        )
        for seat in Seat
    }
    projects = rules.project_domain.detect(hands, Contract.SUN, Seat.EAST)
    request = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "bounded",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": 1,
            "horizon": {"unit": "plies", "value": 2},
            "boundaryEvaluator": "current_raw_point_margin_v1",
            "budgetProfile": "test",
        }
    )
    graft = OrToolsNonPlayPlayGraphGraftV1(
        rules,
        request,
        maximum_nodes=1_000,
        maximum_depth=4,
        maximum_seconds=15,
        native_cp_sat=True,
    )
    compile_calls = 0
    original_compile = OrToolsCompleteStateGraphCompilerV1.compile

    def counted_compile(self, *args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(self, *args, **kwargs)

    monkeypatch.setattr(
        OrToolsCompleteStateGraphCompilerV1,
        "compile",
        counted_compile,
    )

    def terminal_utility(terminal) -> int:
        return graft.terminal_utility(
            nonplay_depth=1,
            prefix_factory=lambda: build_nonplay_play_start_prefix_v1(
                rules,
                prefix,
                buyer=Seat.NORTH,
                contract=Contract.SUN,
                trump=None,
                escalation_actions=(),
                projects=terminal.projects,
                project_actions=terminal.actions,
            ),
        )

    base = compile_project_declaration_graph_v1(
        rules.project_domain,
        projects=projects,
        actor=Seat.EAST,
        dealer=Seat.EAST,
        prefix=prefix.project_actions,
        trump_suit=None,
        terminal_utility=terminal_utility,
        scope="bounded",
        horizon=2,
        boundary_evaluator="current_raw_point_margin_v1",
        boundary_utility=lambda _terminal: 0,
    )
    combined = graft.graft(base)

    assert [
        edge.action_code.split(":", 1)[0] for edge in _root(combined).edges
    ] == ["pass", "declare"]
    assert compile_calls == 1
    assert any(node.leaf_kind == "boundary" for node in combined.nodes if node.is_leaf)


def test_nonplay_play_start_is_syntactic_only(rules) -> None:
    fixture = json.loads(
        Path("examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )["exactTracePrefix"]
    fixture.update(
        {
            "projectDeclarations": [],
            "projectAssertions": [],
            "playActions": [],
            "stateAssertions": {},
        }
    )
    start = build_nonplay_play_start_prefix_v1(
        rules,
        ExactTracePrefixV1.build(fixture),
        buyer=Seat.NORTH,
        contract=Contract.SUN,
        trump=None,
        escalation_actions=(),
    )

    assert start.projects == ()
    assert start.prefix.project_declarations == ()


def test_exact_nonplay_graft_accepts_action_edge_line_count_request(rules) -> None:
    request = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "winning_continuation_line_count",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": 1,
            "lineUniverse": "all_legal",
            "budgetProfile": "test",
        }
    )

    graft = OrToolsNonPlayPlayGraphGraftV1(
        rules,
        request,
        maximum_nodes=100,
        maximum_depth=8,
        maximum_seconds=1,
    )
    assert graft.request is request


def test_v3_nonplay_graft_selects_native_exact_normalization(rules) -> None:
    request = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": 1,
            "budgetProfile": "test",
        }
    )

    graft = OrToolsNonPlayPlayGraphGraftV1(
        rules,
        request,
        maximum_nodes=100,
        maximum_depth=8,
        maximum_seconds=1,
        native_cp_sat=True,
    )

    assert graft.native_cp_sat is True


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (NonPlayGraphBudgetV1(maximum_nodes=2), "maximum_nodes"),
        (NonPlayGraphBudgetV1(maximum_depth=1), "maximum_depth"),
    ],
)
def test_graph_budget_stops_before_emitting_partial_exact_evidence(
    rules,
    budget: NonPlayGraphBudgetV1,
    expected: str,
) -> None:
    with pytest.raises(NonPlayGraphCompilationStoppedV1) as captured:
        compile_auction_decision_graph_v1(
            _auction_domain(rules),
            dealer=Seat.SOUTH,
            purchase_suit=Suit.HEARTS,
            terminal_utility=lambda _terminal: 0,
            budget=budget,
        )
    assert captured.value.reason == expected


def test_graph_compilation_honors_cancellation(rules) -> None:
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    with pytest.raises(NonPlayGraphCompilationStoppedV1) as captured:
        compile_auction_decision_graph_v1(
            _auction_domain(rules),
            dealer=Seat.SOUTH,
            purchase_suit=Suit.HEARTS,
            terminal_utility=lambda _terminal: 0,
            cancel_check=cancelled,
        )
    assert captured.value.reason == "cancelled"
    assert checks == 2


def test_nonplay_graph_module_has_no_runtime_or_legacy_imports() -> None:
    import baloot_platform.research.ortools_nonplay_graphs as module

    tree = ast.parse(inspect.getsource(module))
    imports = {
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for name in (
            *((node.module or "",) if isinstance(node, ast.ImportFrom) else ()),
            *(alias.name for alias in node.names),
        )
    }
    forbidden = (
        "BalootEngine",
        "HeuristicBot",
        "core.engine",
        "proof_service",
        "scenario_evidence",
        "best_response_proof",
    )
    assert not any(token in item for token in forbidden for item in imports)


def test_contracts_reject_coercion_and_undeclared_boundaries(rules) -> None:
    with pytest.raises(ValueError, match="maximum_nodes"):
        NonPlayGraphBudgetV1(maximum_nodes=True)
    with pytest.raises(ValueError, match="requires its evaluator"):
        compile_auction_decision_graph_v1(
            _auction_domain(rules),
            dealer=Seat.SOUTH,
            purchase_suit=Suit.HEARTS,
            terminal_utility=lambda _terminal: 0,
            scope="bounded",
            horizon=1,
        )
    with pytest.raises(TypeError, match="return an integer"):
        compile_auction_decision_graph_v1(
            _auction_domain(rules),
            dealer=Seat.SOUTH,
            purchase_suit=Suit.HEARTS,
            terminal_utility=lambda _terminal: True,
        )
