"""Shared exact non-play graph construction for complete and information proof."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.exact_state_view import canonical_observer_information_hash_v1
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.rules import RuleBook

from .ortools_complete_state_graph import CompleteStateDecisionObserverV1
from .ortools_exact_state import (
    ExactNormalizationStopped,
    OrToolsExactStateNormalizerV1,
    OrToolsNormalizedExactStateV1,
    _effective_prefix_hash,
)
from .ortools_nonplay_graphs import (
    AuctionGraphChoiceV1,
    EscalationGraphActionV1,
    EscalationGraphTerminalV1,
    NonPlayDecisionObserverV1,
    NonPlayDecisionObservationV1,
    NonPlayGraphBudgetV1,
    NonPlayGraphCompilationStoppedV1,
    compile_auction_decision_graph_v1,
    compile_escalation_decision_graph_v1,
    compile_project_declaration_graph_v1,
    project_graph_observation_rows_v1,
    _escalation_window_key,
)
from .ortools_nonplay_play_graft import (
    NonPlayPlayStartV1,
    OrToolsNonPlayPlayGraphGraftV1,
    build_nonplay_play_start_prefix_v1,
    selected_projects_from_exact_receipt_v1,
)
from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1
from .ortools_escalation_stage import OrToolsEscalationStageV1
from .ortools_nonplay_observation import _normalized_auction_rows
from .ortools_proof import ProofGraphEdgeV1, ProofGraphNodeV1, ProofGraphV1
from .ortools_scenario_model import attach_auction_action_values_v1
from .scenario_contracts import ExactTracePrefixV1, ScenarioProofRequestV1, canonical_sha256, validate_proof_boundary_scope_v1


@dataclass(frozen=True, slots=True)
class NonPlayCompleteGraphCompilationV1:
    graph: ProofGraphV1
    decision_observation_ids: Mapping[str, str]
    boundary_evaluation: Mapping[str, Any] | None = None
    project_counterfactual_graph: ProofGraphV1 | None = None
    project_compilation_evidence: tuple[Mapping[str, Any], ...] = ()


BaseObserverFactoryV1 = Callable[
    [OrToolsNormalizedExactStateV1, ExactTracePrefixV1],
    NonPlayDecisionObserverV1,
]
PlayObserverFactoryV1 = Callable[
    [OrToolsNormalizedExactStateV1, NonPlayPlayStartV1],
    CompleteStateDecisionObserverV1,
]


def compile_auction_hand_quality_graph_v1(
    rules: RuleBook, exact_prefix: ExactTracePrefixV1, *,
    maximum_nodes: int, maximum_depth: int, maximum_seconds: float,
    native_model_receipts: list[dict[str, Any]],
    normalized: OrToolsNormalizedExactStateV1 | None = None,
    base_observer_factory: BaseObserverFactoryV1 | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> NonPlayCompleteGraphCompilationV1:
    """One-decision heuristic graph, not a round-outcome or strength certificate."""
    if not isinstance(native_model_receipts, list):
        raise TypeError("auction hand quality requires a model receipt list")
    started = time.perf_counter()
    budget = NonPlayGraphBudgetV1(maximum_nodes, maximum_depth)
    if normalized is None:
        normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
            exact_prefix, decision_point={"kind": "bid"}, maximum_seconds=maximum_seconds,
            cancel_check=cancel_check,
        )
    elif normalized.prefix_hash != _effective_prefix_hash(exact_prefix) or normalized.rules_hash != rules.sha256:
        raise ValueError("auction hand quality normalized state has the wrong authority")
    normalized.require_integrity()
    if normalized.decision_kind != "bid" or normalized.next_actor is None or not normalized.legal_actions:
        raise ValueError("auction hand quality requires a live bid decision")
    actor = Seat.parse(normalized.next_actor)
    own = {Card.parse(card).id for card in normalized.state["hands"][actor.name.lower()]}
    purchase = Card.parse(normalized.state["purchaseCard"])
    legal_codes = tuple(sorted(str(item["code"]) for item in normalized.legal_actions))
    if len(legal_codes) + 1 > budget.maximum_nodes or budget.maximum_depth < 1:
        raise NonPlayGraphCompilationStoppedV1("maximum_nodes", nodes=0, depth=0)
    model = cp_model.CpModel()
    hand, margins = attach_auction_action_values_v1(
        model, rules, visible_cards=[model.new_constant(int(index in own)) for index in range(32)],
        purchase_card=purchase, round_index=model.new_constant(normalized.state["bidding"]["round"]),
    )
    if any(code not in margins for code in legal_codes):
        raise ValueError("auction hand evaluator lacks a normalized legal action")
    deadline = started + maximum_seconds
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = max(0.001, deadline - time.perf_counter())
    status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise NonPlayGraphCompilationStoppedV1(
            "cancelled" if cancel_check is not None and cancel_check() else "native_boundary_incomplete",
            nodes=0, depth=0,
        )
    values = {code: solver.value(margins[code]) for code in legal_codes}
    uniqueness = model.clone()
    differences = []
    for index, code in enumerate(legal_codes):
        variable = uniqueness.get_int_var_from_proto_index(margins[code].index)
        different = uniqueness.new_bool_var(f"auction_hand_value_diff_{index}")
        uniqueness.add(variable != values[code]).only_enforce_if(different)
        uniqueness.add(variable == values[code]).only_enforce_if(different.Not())
        differences.append(different)
    uniqueness.add_bool_or(differences)
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise NonPlayGraphCompilationStoppedV1("maximum_seconds", nodes=0, depth=0)
    checker = cp_model.CpSolver()
    checker.parameters.num_search_workers = 1
    checker.parameters.max_time_in_seconds = remaining
    uniqueness_status = _solve_cp_sat_with_cancel_v1(checker, uniqueness, cancel_check)
    identity = {
        "domain": AuctionDomainIRV1.from_rules(rules.data).sha256,
        "kind": "auction_hand_quality_vector_v1",
        "model": str(model.proto),
        "boundaryUniquenessModel": str(uniqueness.proto),
    }
    receipt = {
        "schemaVersion": 1, "kind": "native_auction_model_receipt_v1",
        "modelIdentity": identity, "modelHash": canonical_sha256(identity),
        "dealer": Seat.parse(exact_prefix.dealer).name.lower(),
        "purchaseSuit": purchase.suit.name.lower(), "purchaseRank": purchase.rank.name.lower(),
        "actions": [dict(item) for item in exact_prefix.auction_actions],
        "variableBindings": {code: margins[code].index for code in legal_codes},
        "derivedValues": values, "constructionStatus": solver.status_name(status).lower(),
        "boundaryUniquenessStatus": checker.status_name(uniqueness_status).lower(),
    }
    native_model_receipts.append({**receipt, "sha256": canonical_sha256(receipt)})
    if uniqueness_status != cp_model.INFEASIBLE:
        raise NonPlayGraphCompilationStoppedV1(
            "cancelled" if cancel_check is not None and cancel_check() else "native_boundary_incomplete",
            nodes=0, depth=0,
        )
    evaluation = {"evaluatorId": "auction_hand_margin_v1", "kind": "heuristic",
                  "scale": hand.scale, "profileHash": hand.profile_hash}
    graph_identity = {
        "kind": "auction_hand_quality_graph_v1", "rulesHash": rules.sha256,
        "actor": actor.name.lower(), "purchaseCard": purchase.code,
        "visibleCards": sorted(own), "legalActions": list(legal_codes),
        "auctionActions": [dict(item) for item in exact_prefix.auction_actions],
        "modelHash": receipt["modelHash"],
    }
    root_id = canonical_sha256({**graph_identity, "node": "root"})
    leaves = []
    edges = []
    for action_id, code in enumerate(legal_codes):
        leaf_id = canonical_sha256({**graph_identity, "node": "boundary", "action": code})
        leaves.append(ProofGraphNodeV1.leaf(leaf_id, values[code], kind="boundary"))
        edges.append(ProofGraphEdgeV1.build(action_id, code, leaf_id))
    root = ProofGraphNodeV1.decision(
        root_id, actor.team.name.lower(), edges, actor_seat=actor.name.lower(),
    )
    graph = ProofGraphV1.build(
        scope="bounded", root_node_id=root_id, nodes=[*leaves, root],
        boundary_evaluator="auction_hand_margin_v1",
    )
    observations = {}
    if base_observer_factory is not None:
        records = [
            {"actor": row["actor"], "action": row["action"], "trump": row.get("trumpSuit")}
            for row in _normalized_auction_rows(normalized, exact_prefix)
        ]
        public_state = {"stage": "auction", "state": {"records": records}}
        observations[root_id] = base_observer_factory(normalized, exact_prefix)(
            NonPlayDecisionObservationV1(root_id, "bid", actor, public_state)
        )
    return NonPlayCompleteGraphCompilationV1(graph, observations, evaluation)


def public_nonplay_history_v1(
    exact_prefix: ExactTracePrefixV1,
) -> tuple[Mapping[str, Any], ...]:
    """Return the ordered public auction/escalation/project action memory."""

    rows: list[dict[str, Any]] = []

    def append(stage: str, item: Mapping[str, Any], action: str) -> None:
        row: dict[str, Any] = {
            "ordinal": len(rows),
            "stage": stage,
            "actor": item["actor"],
            "action": action,
        }
        if item.get("trumpSuit") is not None:
            row["trumpSuit"] = item["trumpSuit"]
        if item.get("mode", item.get("requestedMode")) is not None:
            row["mode"] = item.get("mode", item.get("requestedMode"))
        rows.append(row)

    for item in exact_prefix.auction_actions:
        append("auction", item, str(item["action"]))
    for item in exact_prefix.escalation_actions:
        append("escalation", item, str(item["action"]))
    for item in exact_prefix.project_actions:
        action = str(item["action"])
        if item.get("projectId") is not None:
            action += ":" + str(item["projectId"])
        append("project", item, action)
    return tuple(rows)


def _collector(
    observer: NonPlayDecisionObserverV1 | None,
) -> tuple[NonPlayDecisionObserverV1 | None, dict[str, str]]:
    values: dict[str, str] = {}
    if observer is None:
        return None, values

    def collect(context):
        observation_id = observer(context)
        existing = values.get(context.node_id)
        if existing is not None and existing != observation_id:
            raise AssertionError("non-play observation identity changed")
        values[context.node_id] = observation_id
        return observation_id

    return collect, values


def compile_nonplay_complete_graph_v1(
    rules: RuleBook,
    normalized: OrToolsNormalizedExactStateV1,
    exact_prefix: ExactTracePrefixV1,
    decision_point: Mapping[str, Any],
    request: ScenarioProofRequestV1,
    *,
    maximum_nodes: int,
    maximum_depth: int,
    maximum_seconds: float,
    native_cp_sat: bool,
    base_observer_factory: BaseObserverFactoryV1 | None = None,
    play_observer_factory: PlayObserverFactoryV1 | None = None,
    cancel_check: Callable[[], bool] | None = None,
    opportunity_orders: Mapping[str, Sequence[Seat | str | int]] | None = None,
    opportunity_order: Sequence[Seat | str | int] | None = None,
    prior_non_announcing_seats: Sequence[Seat | str | int] = (),
    prior_window_recall: Mapping[str, Sequence[str]] | None = None,
    native_model_receipts: list[dict[str, Any]] | None = None,
    _normalizer: OrToolsExactStateNormalizerV1 | None = None,
    _compact_forced: bool = False,
    project_counterfactual: bool = False,
) -> NonPlayCompleteGraphCompilationV1:
    """Compile a non-play root through its requested legal play continuation."""

    if type(project_counterfactual) is not bool:
        raise TypeError("project_counterfactual must be boolean")
    if project_counterfactual and (not native_cp_sat or request.mode != "exact"
            or request.utility != "awarded_point_margin" or decision_point.get("kind") != "project"
            or base_observer_factory is not None or play_observer_factory is not None):
        raise ValueError("project counterfactual requires complete-state exact native project proof")

    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("cancel_check must be callable")
    if type(_compact_forced) is not bool or (_compact_forced and not native_cp_sat):
        raise ValueError("compact non-play construction requires native CP-SAT")
    validate_proof_boundary_scope_v1(request, str(decision_point["kind"]))
    if request.boundary_evaluator == "auction_hand_margin_v1":
        if not native_cp_sat or request.target_team != Seat.parse(normalized.next_actor).team.name.lower():
            raise ValueError("auction hand margin requires native actor-team evaluation")
        return compile_auction_hand_quality_graph_v1(
            rules, exact_prefix, maximum_nodes=maximum_nodes, maximum_depth=maximum_depth,
            maximum_seconds=maximum_seconds, native_model_receipts=native_model_receipts,
            normalized=normalized,
            base_observer_factory=base_observer_factory, cancel_check=cancel_check,
        )
    if native_model_receipts is not None and (
        not native_cp_sat or decision_point.get("kind") != "bid" or not isinstance(native_model_receipts, list)
    ):
        raise ValueError("native auction receipts require a native bid compilation")
    recorded_history = normalized.recorded_escalation_history()
    if recorded_history is not None:
        if not native_cp_sat or decision_point.get("kind") != "escalation":
            raise ValueError("recorded escalation proof requires a native escalation root")
        if opportunity_orders is not None or opportunity_order is not None or prior_non_announcing_seats or prior_window_recall:
            raise ValueError("recorded escalation proof history cannot be overridden")
        opportunity_order = recorded_history["opportunityOrder"]
        prior_non_announcing_seats = recorded_history["finalState"]["opportunityState"]["nonAnnouncingSeats"]
        current_actions = recorded_history["finalState"]["observation"]["actions"]
        prior_window_recall = {}
        for transition in recorded_history["transitions"]:
            actions = transition["before"]["observation"]["actions"]
            if transition["event"]["visibility"] == "private" and actions != current_actions:
                window = _escalation_window_key(tuple(EscalationGraphActionV1.build(
                    row["actor"], row["action"], row["requestedMode"]) for row in actions))
                prior_window_recall.setdefault(transition["event"]["actor"], []).append(window)
    ordered = opportunity_orders is not None or opportunity_order is not None
    if opportunity_orders is not None and opportunity_order is not None:
        raise ValueError("select frozen or per-window opportunity orders, not both")
    if ordered and (not native_cp_sat or decision_point.get("kind") not in {"bid", "escalation"}):
        raise ValueError("opportunity orders require a native bid or escalation root")
    if prior_non_announcing_seats and not ordered:
        raise ValueError("private escalation history requires native opportunity orders")
    if prior_window_recall and not ordered:
        raise ValueError("prior-window recall requires native opportunity orders")

    def check_cancel():
        if cancel_check is not None and cancel_check():
            raise ExactNormalizationStopped("cancelled")

    check_cancel()

    observed = base_observer_factory is not None or play_observer_factory is not None
    if observed and (base_observer_factory is None or play_observer_factory is None):
        raise ValueError("observed graph requires both observer factories")
    if request.visibility != "complete_state" or request.mode not in {
        "exact",
        "bounded",
    }:
        raise ValueError("non-play complete graph requires exact/bounded complete state")
    started = time.perf_counter()
    kind = str(decision_point["kind"])
    selector = dict(decision_point.get("selector") or {})
    if (
        native_cp_sat and kind in {"project", "play"} and request.mode == "bounded"
        and request.horizon is not None and request.horizon["unit"] == "tricks"
        and request.boundary_evaluator == "current_raw_point_margin_v1"
        and exact_prefix.project_actions_present
    ):
        declaration_count = len(rules.project_domain.declaration_seats(Seat.parse(exact_prefix.dealer)))
        remaining_declarations = declaration_count - len(exact_prefix.project_actions)
        # Declarations consume decisions, not tricks. After the first trick
        # every remaining action is a play, so this is an exact horizon mapping.
        decisions = (
            int(request.horizon["value"]) * 4 - len(exact_prefix.play_actions) % 4
            + remaining_declarations
        )
        translated = request.to_dict()
        translated.pop("sha256", None)
        translated["horizon"] = {"unit": "decisions", "value": decisions}
        request = ScenarioProofRequestV1.build(translated)
    budget = NonPlayGraphBudgetV1(maximum_nodes, maximum_depth)

    normalizer = _normalizer or OrToolsExactStateNormalizerV1(rules, native_cp_sat=native_cp_sat)
    if normalizer.rules.sha256 != rules.sha256 or normalizer.native_cp_sat != native_cp_sat:
        raise ValueError("shared normalizer differs from graph authority")

    def project_continuation(prefix, child_request, nodes, depth, seconds):
        child_started = time.perf_counter()
        child_normalized = normalizer.normalize_play_or_project(
            prefix, maximum_seconds=seconds, cancel_check=cancel_check,
        )
        if child_normalized.decision_kind != "project":
            raise ValueError("interleaved project continuation did not reach a declaration")
        remaining = seconds - (time.perf_counter() - child_started)
        if remaining <= 0:
            raise ExactNormalizationStopped("maximum_seconds", evidence={
                "sourceStage": "project_continuation_normalization",
                "prefixHash": prefix.sha256, "requestHash": child_request.sha256,
                "remainingSeconds": remaining,
            })
        child = compile_nonplay_complete_graph_v1(
            rules, child_normalized, prefix,
            {"kind": "project", "selector": {"owner": child_normalized.next_actor}}, child_request,
            maximum_nodes=nodes, maximum_depth=depth, maximum_seconds=remaining, native_cp_sat=True,
            base_observer_factory=base_observer_factory, play_observer_factory=play_observer_factory,
            cancel_check=cancel_check,
            _normalizer=normalizer,
            _compact_forced=_compact_forced,
            **({"project_counterfactual": True} if project_counterfactual else {}),
        )
        if project_counterfactual:
            if child.project_counterfactual_graph is None:
                raise ValueError("project continuation omitted its counterfactual")
            graft.project_compilation_evidence.update({canonical_sha256(item): item
                for item in child.project_compilation_evidence})
            return child.graph, child.decision_observation_ids, child.project_counterfactual_graph
        return child.graph, child.decision_observation_ids

    graft = OrToolsNonPlayPlayGraphGraftV1(
        rules,
        request,
        maximum_nodes=maximum_nodes,
        maximum_depth=maximum_depth,
        maximum_seconds=maximum_seconds,
        native_cp_sat=native_cp_sat,
        root_decision_kind=kind,
        project_continuation=project_continuation if native_cp_sat else None,
        cancel_check=cancel_check,
        _normalizer=normalizer,
        _compact_forced=_compact_forced,
        project_counterfactual=project_counterfactual,
    )
    base_options: dict[str, Any]
    decision_horizon = (
        int(request.horizon["value"])
        if request.mode == "bounded"
        and request.horizon is not None
        and request.horizon["unit"] in {"plies", "decisions"}
        else None
    )
    # Trick/round horizons begin in play; bidding and declarations do not
    # consume tricks. The graft retains the original bounded proof scope.
    if decision_horizon is not None:
        base_options = {
            "scope": "bounded",
            "horizon": decision_horizon,
            "boundary_evaluator": request.boundary_evaluator,
            "boundary_utility": lambda _state: 0,
        }
    else:
        base_options = {"scope": "exact"}

    if kind == "play":
        if (
            not native_cp_sat
            or normalized.decision_kind != "play" or normalized.phase != "playing"
            or (request.mode == "bounded" and (
                decision_horizon is None or request.boundary_evaluator != "current_raw_point_margin_v1"))
            or not exact_prefix.project_actions_present
            or len(exact_prefix.project_actions) >= len(rules.project_domain.declaration_seats(Seat.parse(exact_prefix.dealer)))
        ):
            raise ValueError("interleaved play root requires native pending declarations")
        if decision_horizon is not None and decision_horizon < 2:
            raise ValueError("single-play horizons use the native play compiler")
        child_request = graft._adjusted_request(1)
        assert child_request is not None
        actor = Seat.parse(normalized.next_actor)
        root_id = canonical_sha256({"kind": "native_interleaved_play_root_v1", "state": normalized.sha256})
        nodes, edges, observations = {}, [], {}
        retained_nodes = 0
        if observed:
            observations[root_id] = canonical_observer_information_hash_v1(
                complete_state=normalized.state, acting_seat=actor.name.lower(),
            )
        for action_id, action in enumerate(normalized.legal_actions):
            check_cancel()
            remaining = maximum_seconds - (time.perf_counter() - started)
            nodes_left = maximum_nodes - (retained_nodes if _compact_forced else len(nodes)) - 1
            if remaining <= 0 or nodes_left < 1 or maximum_depth < 2:
                reason = "maximum_seconds" if remaining <= 0 else "maximum_nodes" if nodes_left < 1 else "maximum_depth"
                raise NonPlayGraphCompilationStoppedV1(reason, nodes=retained_nodes, depth=maximum_depth)
            advanced = normalizer.append_normalized_action(exact_prefix, normalized, str(action["code"]))
            child, child_observations = project_continuation(
                advanced, child_request, nodes_left, maximum_depth - 1, remaining,
            )
            namespace = canonical_sha256({"root": root_id, "action": action["code"]})
            namespaced = graft._namespace_graph(child, namespace)
            retained_nodes += graft._node_count(namespaced)
            # Equal boundary outcomes are shared across sibling continuations.
            for node in namespaced.nodes:
                existing = nodes.get(node.node_id)
                if existing is not None and existing != node:
                    raise AssertionError("combined graph node identity collision")
                nodes[node.node_id] = node
            edges.append(ProofGraphEdgeV1.build(action_id, str(action["code"]), namespaced.root_node_id))
            if observed:
                observations.update(graft._namespace_observations(child, namespace, child_observations))
        nodes[root_id] = ProofGraphNodeV1.decision(root_id, actor.team.name.lower(), edges, actor_seat=actor.name.lower())
        graph = ProofGraphV1.build(
            scope="exact" if request.mode == "exact" else "bounded",
            root_node_id=root_id, nodes=tuple(nodes.values()), boundary_evaluator=request.boundary_evaluator,
        )
        if time.perf_counter() - started > maximum_seconds:
            raise NonPlayGraphCompilationStoppedV1("maximum_seconds", nodes=len(nodes), depth=maximum_depth)
        return NonPlayCompleteGraphCompilationV1(graph, observations)

    base_observer = (
        base_observer_factory(normalized, exact_prefix)
        if base_observer_factory is not None
        else None
    )

    if kind == "bid":
        layout = DealLayoutV1.build(
            rules.structural_domain,
            dealer=Seat.parse(exact_prefix.dealer),
        )
        purchase_suit = Card.parse(
            exact_prefix.deck_order[layout.purchase_position]
        ).suit
        auction_prefix = tuple(
            AuctionGraphChoiceV1.build(item["action"], item.get("trumpSuit"))
            for item in exact_prefix.auction_actions
        )
        terminal_graphs: dict[
            int, tuple[ProofGraphV1, Mapping[str, str]]
        ] = {}

        def auction_terminal(terminal: Any) -> int:
            if terminal.all_passed:
                return 0
            if terminal.buyer is None or terminal.declarer is None or terminal.contract is None:
                raise ValueError("purchased auction terminal is incomplete")
            branch = exact_prefix.to_dict()
            branch.pop("sha256", None)
            branch.update(
                {
                    "auctionActions": [
                        {
                            "actor": item.actor.name.lower(),
                            "action": item.action,
                            **(
                                {"trumpSuit": item.trump.name.lower()}
                                if item.trump is not None
                                else {}
                            ),
                        }
                        for item in terminal.actions
                    ],
                    "escalationActions": [],
                    "projectDeclarations": [],
                    "projectAssertions": [],
                    "playActions": [],
                    "stateAssertions": {},
                }
            )
            branch_prefix = ExactTracePrefixV1.build(branch)
            auction_depth = len(terminal.actions) - len(auction_prefix)
            if (
                decision_horizon is not None
                and auction_depth >= decision_horizon
            ):
                return 0

            def escalation_terminal(value: Any) -> int:
                depth = auction_depth + len(value.actions) + int(value.declined)
                if (
                    decision_horizon is not None
                    and depth >= decision_horizon
                ):
                    return 0
                return graft.terminal_utility(
                    nonplay_depth=depth,
                    prefix_factory=lambda value=value: (
                        build_nonplay_play_start_prefix_v1(
                            rules,
                            branch_prefix,
                            buyer=terminal.buyer,
                            contract=terminal.contract,
                            trump=terminal.trump,
                            escalation_actions=value.actions,
                        )
                    ),
                )

            remaining_horizon = None
            if decision_horizon is not None:
                remaining_horizon = max(1, decision_horizon - auction_depth)
            escalation_domain = EscalationDomainIRV1.from_rules(rules.data)
            if native_cp_sat:
                opening_seconds = maximum_seconds - (time.perf_counter() - started)
                if opening_seconds <= 0:
                    raise NonPlayGraphCompilationStoppedV1(
                        "maximum_seconds", nodes=0, depth=auction_depth
                    )
                opening = OrToolsEscalationStageV1(
                    escalation_domain, declarer=terminal.declarer,
                    contract=terminal.contract, match_score=exact_prefix.match_score,
                    native_cp_sat=True,
                ).solve(
                    required_actions_complete=True,
                    maximum_seconds=opening_seconds,
                    cancel_check=cancel_check,
                )
                if opening.status == "incomplete":
                    raise NonPlayGraphCompilationStoppedV1(
                        "native_transition_incomplete", nodes=0, depth=auction_depth
                    )
                if opening.status != "satisfied" or opening.observation is None:
                    raise ValueError("native escalation opening is infeasible")
                forced_closure = not opening.observation.legal_next_choices
            else:
                opening_decisions = escalation_domain.legal_decisions(
                    escalation_domain.initial_state(),
                    declarer=terminal.declarer,
                    contract=terminal.contract,
                    match_score=exact_prefix.match_score,
                )
                forced_closure = (
                    len(opening_decisions) == 1
                    and opening_decisions[0].actor is None
                )
            if forced_closure:
                return escalation_terminal(
                    EscalationGraphTerminalV1(
                        (), escalation_domain.initial_state(), True
                    )
                )
            branch_observers: dict[Seat, NonPlayDecisionObserverV1] = {}

            def branch_observer(context):
                observer = branch_observers.get(context.actor)
                if observer is None:
                    remaining = maximum_seconds - (time.perf_counter() - started)
                    if remaining <= 0:
                        raise NonPlayGraphCompilationStoppedV1(
                            "maximum_seconds", nodes=0, depth=0
                        )
                    branch_normalized = OrToolsExactStateNormalizerV1(
                        rules,
                        native_cp_sat=native_cp_sat,
                    ).normalize(
                        branch_prefix,
                        decision_point={
                            "kind": "escalation",
                            "selector": {
                                "actingSeat": context.actor.name.lower(),
                            },
                        },
                        maximum_seconds=remaining,
                        cancel_check=cancel_check,
                    )
                    assert base_observer_factory is not None
                    observer = base_observer_factory(
                        branch_normalized, branch_prefix
                    )
                    branch_observers[context.actor] = observer
                return observer(context)

            child_observer, child_ids = _collector(
                branch_observer if base_observer_factory is not None else None
            )
            escalation = compile_escalation_decision_graph_v1(
                escalation_domain,
                native_cp_sat=native_cp_sat,
                declarer=terminal.declarer,
                contract=terminal.contract,
                match_score=exact_prefix.match_score,
                terminal_utility=escalation_terminal,
                budget=NonPlayGraphBudgetV1(
                    maximum_nodes=maximum_nodes,
                    maximum_depth=max(1, maximum_depth - auction_depth),
                ),
                decision_observer=child_observer,
                cancel_check=cancel_check,
                opportunity_order=opportunity_order,
                **(
                    {
                        "scope": "bounded",
                        "horizon": remaining_horizon,
                        "boundary_evaluator": request.boundary_evaluator,
                        "boundary_utility": lambda _state: 0,
                    }
                    if decision_horizon is not None
                    else {"scope": "exact"}
                ),
            )
            sentinel = -1_000_000_000 - len(terminal_graphs)
            if observed:
                assert play_observer_factory is not None
                terminal_graphs[sentinel] = graft.graft_observed(
                    escalation,
                    base_observation_ids=child_ids,
                    decision_observer_factory=play_observer_factory,
                )
            else:
                terminal_graphs[sentinel] = (graft.graft(escalation), {})
            return sentinel

        auction_observer, auction_ids = _collector(base_observer)
        remaining = maximum_seconds - (time.perf_counter() - started)
        if native_cp_sat and remaining <= 0:
            raise NonPlayGraphCompilationStoppedV1("maximum_seconds", nodes=0, depth=0)
        auction = compile_auction_decision_graph_v1(
            AuctionDomainIRV1.from_rules(rules.data),
            dealer=exact_prefix.dealer,
            purchase_suit=purchase_suit,
            purchase_rank=Card.parse(exact_prefix.deck_order[layout.purchase_position]).rank,
            terminal_utility=auction_terminal,
            prefix=auction_prefix,
            budget=budget,
            decision_observer=auction_observer,
            cancel_check=cancel_check,
            native_cp_sat=native_cp_sat,
            maximum_seconds=remaining if native_cp_sat else None,
            native_model_receipts=native_model_receipts,
            **base_options,
        )
        if observed:
            graph, observations = graft.graft_terminal_graphs_observed(
                auction,
                terminal_graphs,
                base_observation_ids=auction_ids,
            )
        else:
            graph = graft.graft_terminal_graphs(
                auction,
                {key: value[0] for key, value in terminal_graphs.items()},
            )
            observations = {}
        return NonPlayCompleteGraphCompilationV1(graph, observations)

    buyer_raw = normalized.state.get("buyer")
    contract_raw = normalized.state.get("contract")
    if buyer_raw is None or contract_raw is None:
        raise ValueError("non-play proof state is incomplete")
    buyer = Seat.parse(buyer_raw)
    contract = Contract.parse(contract_raw)
    trump_raw = normalized.state.get("trumpSuit")
    trump = Suit.parse(trump_raw) if trump_raw else None
    escalation_prefix = tuple(
        EscalationGraphActionV1.build(
            item["actor"],
            item["action"],
            item.get("mode", item.get("requestedMode")),
        )
        for item in exact_prefix.escalation_actions
    )

    if kind == "escalation":
        actor = selector.get("actingSeat")
        if actor is None:
            raise ValueError("escalation proof actor is missing")

        def escalation_terminal(terminal: Any) -> int:
            depth = terminal.decision_depth if terminal.decision_depth is not None else (
                len(terminal.actions)
                - len(escalation_prefix)
                + int(terminal.declined)
            )
            return graft.terminal_utility(
                nonplay_depth=depth,
                private_history=terminal.private_history,
                prefix_factory=lambda terminal=terminal: (
                    build_nonplay_play_start_prefix_v1(
                        rules,
                        exact_prefix,
                        buyer=buyer,
                        contract=contract,
                        trump=trump,
                        escalation_actions=terminal.actions,
                    )
                ),
            )

        observer, observations = _collector(base_observer)
        base = compile_escalation_decision_graph_v1(
            EscalationDomainIRV1.from_rules(rules.data),
            declarer=Seat.parse(normalized.state["declarer"]) if ordered else buyer,
            contract=contract,
            match_score=exact_prefix.match_score,
            acting_seat=actor,
            terminal_utility=escalation_terminal,
            prefix=escalation_prefix,
            native_cp_sat=native_cp_sat or opportunity_orders is not None,
            opportunity_orders=opportunity_orders,
            opportunity_order=opportunity_order,
            prior_non_announcing_seats=prior_non_announcing_seats,
            prior_window_recall=prior_window_recall,
            budget=budget,
            decision_observer=observer,
            cancel_check=cancel_check,
            **base_options,
        )
    elif kind == "project":
        actor = selector.get("owner")
        if actor is None:
            raise ValueError("project proof actor is missing")
        projects = selected_projects_from_exact_receipt_v1(normalized)

        def project_terminal(terminal: Any) -> int:
            return graft.terminal_utility(
                nonplay_depth=1,
                prefix_factory=lambda terminal=terminal: (
                    build_nonplay_play_start_prefix_v1(
                        rules,
                        exact_prefix,
                        buyer=buyer,
                        contract=contract,
                        trump=trump,
                        escalation_actions=escalation_prefix,
                        project_actions=terminal.actions,
                    )
                ),
            )

        observer, observations = _collector(base_observer)
        if native_cp_sat:
            if normalized.decision_kind != "project" or normalized.next_actor != str(actor):
                raise ValueError("native project graph actor disagrees with normalized root")
            declaration_count = len(rules.project_domain.declaration_seats(Seat.parse(exact_prefix.dealer)))
            short_project_horizon = decision_horizon == 1 or (
                decision_horizon is not None and request.boundary_evaluator == "current_raw_point_margin_v1"
            )
            if (len(exact_prefix.project_actions) + 1 < declaration_count
                    and not short_project_horizon and request.mode != "exact"):
                raise ValueError("project graph requires later declaration/play interleaving support")
            if len(normalized.legal_actions) + 1 > maximum_nodes:
                raise ExactNormalizationStopped("maximum_nodes", evidence={
                    "sourceStage": "native_project_root", "prefixHash": exact_prefix.sha256,
                    "requestHash": request.sha256, "availableNodes": maximum_nodes,
                    "requiredNodes": len(normalized.legal_actions) + 1,
                })
            public_state = {
                "stage": "project", "prefix": [dict(item) for item in exact_prefix.project_actions],
                "projects": project_graph_observation_rows_v1(projects),
            }
            root_id = canonical_sha256({
                "kind": "native_project_root_v1", "exactStateId": normalized.exact_state_id,
                "normalizationReceiptHash": normalized.normalization_receipt.to_dict()["sha256"],
                "prefixHash": exact_prefix.sha256,
            })
            nodes, edges = [], []
            for action_id, action in enumerate(normalized.legal_actions):
                check_cancel()
                if time.perf_counter() - started > maximum_seconds:
                    raise NonPlayGraphCompilationStoppedV1(
                        "maximum_seconds", nodes=len(nodes), depth=1
                    )
                advanced = normalizer.append_normalized_action(exact_prefix, normalized, str(action["code"]))
                child_id = canonical_sha256({"root": root_id, "action": action["code"], "prefix": advanced.sha256})
                start = build_nonplay_play_start_prefix_v1(
                    rules, advanced, buyer=buyer, contract=contract, trump=trump,
                    escalation_actions=escalation_prefix,
                )
                start = replace(start, normalization_parent=normalized)
                value = graft.terminal_utility(nonplay_depth=1, prefix_factory=lambda start=start: start)
                nodes.append(ProofGraphNodeV1.leaf(child_id, value))
                edges.append(ProofGraphEdgeV1.build(action_id, str(action["code"]), child_id))
            parsed_actor = Seat.parse(actor)
            nodes.append(ProofGraphNodeV1.decision(
                root_id, parsed_actor.team.name.lower(), edges, actor_seat=parsed_actor.name.lower(),
            ))
            if observer is not None:
                observer(NonPlayDecisionObservationV1(root_id, "project", parsed_actor, public_state))
            base = ProofGraphV1.build(
                scope=str(base_options["scope"]), root_node_id=root_id, nodes=nodes,
                boundary_evaluator=base_options.get("boundary_evaluator"),
            )
        else:
            base = compile_project_declaration_graph_v1(
                rules.project_domain,
                projects=projects,
                actor=actor,
                dealer=exact_prefix.dealer,
                prefix=exact_prefix.project_actions,
                trump_suit=trump,
                terminal_utility=project_terminal,
                budget=budget,
                decision_observer=observer,
                cancel_check=cancel_check,
                **base_options,
            )
        root = next(node for node in base.nodes if node.node_id == base.root_node_id)
        expected_actions = tuple(str(item["code"]) for item in normalized.legal_actions)
        if tuple(edge.action_code for edge in root.edges) != expected_actions:
            raise AssertionError(
                "project graph actions disagree with native exact-normalization evidence"
            )
    else:
        raise ValueError("non-play graph decision kind is unsupported")

    if native_cp_sat and kind == "project" and not observed:
        owners = [-1] * 32
        for seat in Seat:
            for card in normalized.state["hands"][seat.name.lower()]:
                owners[int(card["id"])] = int(seat)
        for action in exact_prefix.play_actions:
            card = Card.parse(action["card"])
            if owners[card.id] != -1:
                raise ValueError("project proof context contains duplicate card ownership")
            owners[card.id] = int(Seat.parse(action["actor"]))
        if -1 in owners:
            raise ValueError("project proof context requires complete native card ownership")
        base_tricks = [0, 0]
        for trick in normalized.state["completedTricks"]:
            base_tricks[int(Seat.parse(trick["winner"]).team)] += 1
        base = ProofGraphV1.build(scope=base.scope, root_node_id=base.root_node_id,
            nodes=base.nodes, utility_scale=base.utility_scale, boundary_evaluator=base.boundary_evaluator,
            tactic_context={"baseTricks": base_tricks, "ownerByCard": owners,
                "trumpSuit": int(trump) if trump is not None else None})
    if observed:
        assert play_observer_factory is not None
        graph, observations = graft.graft_observed(
            base,
            base_observation_ids=observations,
            decision_observer_factory=play_observer_factory,
            project_observer_factory=getattr(base_observer, "native_checkpoint_factory", None),
        )
    else:
        graph = graft.graft(base)
        observations = {}
    if time.perf_counter() - started > maximum_seconds:
        raise NonPlayGraphCompilationStoppedV1(
            "maximum_seconds", nodes=len(graph.nodes), depth=maximum_depth
        )
    counterfactual = graft.project_counterfactual_graphs.get(graph.sha256) if project_counterfactual else None
    if project_counterfactual and counterfactual is None:
        raise ValueError("project graph omitted its counterfactual")
    return NonPlayCompleteGraphCompilationV1(graph, observations, project_counterfactual_graph=counterfactual,
        project_compilation_evidence=tuple(graft.project_compilation_evidence[key]
            for key in sorted(graft.project_compilation_evidence)))


__all__ = [
    "BaseObserverFactoryV1",
    "NonPlayCompleteGraphCompilationV1",
    "PlayObserverFactoryV1",
    "compile_nonplay_complete_graph_v1",
    "public_nonplay_history_v1",
]
