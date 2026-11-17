"""Exhaustive CP-SAT hidden-world and Domain-IR tree compilation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing
from multiprocessing.connection import wait
from multiprocessing.util import Finalize
import os
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_CEILING
from threading import Thread
from typing import Any, Callable, Mapping, Sequence, TypeAlias

from ortools.sat.python import cp_model

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.cards import Card, FULL_DECK
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.project_ir import ProjectDeclarationActionV1
from baloot_platform.core.project_visibility import project_rank_is_public
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.core.exact_state_view import (
    canonical_observer_information_hash_v1,
    build_canonical_observer_information_v1,
)

from .ortools_complete_state_graph import OrToolsCompleteStateGraphCompilerV1, CompleteStateDecisionObservationV1
from .ortools_escalation_stage import OrToolsEscalationStageV1
from .ortools_exact_state import (
    ExactNormalizationStopped,
    OrToolsExactStateNormalizerV1,
    OrToolsNormalizedExactStateV1,
    _effective_prefix_hash,
)
from .ortools_information_set_proof import (
    ActorInformationSetProofReceiptV1,
    FiniteInformationNodeV1,
    FiniteInformationSetProblemV1,
    FiniteInformationWorldV1,
    InformationSetProofBudgetV1,
    NativeRootValueCertificateV1,
    OrToolsActorInformationSetProofV1,
    _solve_cp_sat_with_cancel_v1,
    _Catalog, _CounterexampleCut,
    _FiniteProofBudgetExceeded,
    build_native_bid_dominance_model_v1,
    native_bid_branch_specs_v1,
)
from .ortools_play_tail import (PlayTailSolveStopped, TailPrefixActionV1,
    NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS, NATIVE_COMPACT_PARTNER_FEED_POLICY_ID,
    NATIVE_SECURE_CONTROL_POLICY_ID, NATIVE_GUARD_PROTECTION_POLICY_ID, NATIVE_CLOSED_TRICK_FEED_POLICY_ID)
from .ortools_proof import (
    ProofGraphSequencePackV1,
    OrToolsGraphProofOutcomeV1,
    ProofGraphV1,
    ProofRootActionV1,
)
from .ortools_project_fragment import attach_project_lifecycle_fragment_v1
from .ortools_nonplay_graphs import (
    bind_escalation_recall_v1, normalize_escalation_recall_v1,
    encode_native_model_receipt_v1,
)
from .ortools_project_model import OrToolsProjectLifecycleModelV1
from .ortools_play_state_predicates import CanonicalObserverScopeV1
from .ortools_scenario_model import OrToolsScenarioModelV1
from .scenario_contracts import (
    ExactTracePrefixV1,
    NativeObservedProofLimitsV1,
    ScenarioProofBudgetV1,
    ScenarioProofRequestV1,
    canonical_json,
    canonical_sha256,
)


_world_compile_cancel = None
_world_compile_cpu = None
_world_compile_slot = None


def _sample_world_compile_cpu():
    with _world_compile_cpu.get_lock():
        _world_compile_cpu[_world_compile_slot] = time.process_time()


def _world_compile_stopped():
    _sample_world_compile_cpu()
    return _world_compile_cancel.is_set()


def _initialize_world_compile_worker(cancel_event, cpu_slots, next_slot):
    parent = multiprocessing.parent_process()
    if parent is not None:
        def exit_with_parent():
            wait([parent.sentinel])
            # The candidate may die holding shared locks; do not run finalizers.
            os._exit(1)
        Thread(target=exit_with_parent, daemon=True).start()
    global _world_compile_cancel, _world_compile_cpu, _world_compile_slot
    _world_compile_cancel = cancel_event
    _world_compile_cpu = cpu_slots
    with next_slot.get_lock():
        _world_compile_slot = next_slot.value
        next_slot.value += 1
    _sample_world_compile_cpu()
    Finalize(None, _sample_world_compile_cpu, exitpriority=0)


def _compile_play_world_fragment(payload):
    rules, native, exact, request, world_state, limits = payload
    compiler = OrToolsActorInformationSetCompilerV1(RuleBook(rules), native_cp_sat=native)
    try:
        return compiler._compile_play_worlds(
            exact, request, (world_state,), cancel_check=_world_compile_stopped, **limits)
    finally:
        _sample_world_compile_cpu()


def _validate_native_opponent_ceiling_policy(policy, target):
    if policy is None:
        return
    if (not isinstance(policy, Mapping)
            or set(policy) not in ({"auction_action", "escalation_action"},
                {"auction_action", "escalation_action", "native_play_policy_id"})
            or any(value is not None and type(value) is not str for value in policy.values())):
        raise ValueError("native opponent ceiling requires a closed registered policy")
    if "native_play_policy_id" in policy:
        from .ortools_play_tail import native_observer_safe_policy_identity_v1
        native_observer_safe_policy_identity_v1(policy["native_play_policy_id"], target.other)


class InformationSetCompilationStopped(RuntimeError):
    """A governed incomplete compilation with immutable partial evidence."""

    def __init__(
        self,
        reason: str,
        *,
        stage: str = "information_set_compilation",
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason = str(reason)
        core = {
            "schemaVersion": 1,
            "status": "incomplete",
            "stage": str(stage),
            "exhaustionReason": self.reason,
            "evidence": dict(evidence or {}),
        }
        encoded = canonical_json(core)
        self.partial_receipt = json.loads(encoded)
        self.partial_receipt["sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        super().__init__(self.reason)


class _IncompatibleProjectWorld(ValueError):
    """A legal allocation that cannot reproduce the recorded public projects."""


def _integral_success_threshold(value: float) -> int:
    """Preserve ``utility >= threshold`` for the integer utility domain."""

    threshold = Decimal(str(value))
    if not threshold.is_finite():
        raise ValueError("actor-information-set success threshold must be finite")
    return int(threshold.to_integral_value(rounding=ROUND_CEILING))


def _remaining_native_path_bound_v1(oracle, selected_root_action, maximum_paths):
    """Forecast card orders, not legal paths or strategic truth."""
    played = {action.card for action in oracle.start.base_prefix}
    if selected_root_action is not None:
        played.add(next(action.card for action in oracle.start.expected_legal_actions
            if action.card.code + ("+ekka" if action.ekka else "") == selected_root_action))
    bound = 1
    for seat in Seat:
        count = sum(card not in played and oracle.start.ownership[card] == seat
            for card in oracle.start.cards)
        bound *= math.factorial(count)
        if bound > maximum_paths:
            return maximum_paths + 1
    # Seat card orders determine the interleaving; each future lead can also
    # choose Ekka. Legality only removes paths from this conservative bound.
    first = oracle.base_length + int(selected_root_action is not None)
    ekka_steps = {step for step in oracle.binary_ekka_steps if step >= first}
    bound *= 2 ** len(ekka_steps)
    return min(bound, maximum_paths + 1)


@dataclass(frozen=True, slots=True)
class ActorInformationSetGraphV1:
    problem: FiniteInformationSetProblemV1
    hidden_worlds: Mapping[str, Any]
    model_document: Mapping[str, Any]
    root_actor: str | None = None
    root_actor_team: str | None = None
    root_actor_relation: str = "target"
    decision_kind: str = "play"
    root_action_ids: Mapping[str, int] | None = None
    analysis_scope: Mapping[str, Any] | None = None
    observer_scope: CanonicalObserverScopeV1 | None = None

    def __post_init__(self) -> None:
        if self.root_actor_relation not in {"target", "opponent"}:
            raise ValueError("information-set root actor must be target or opponent")
        if self.decision_kind not in {"bid", "escalation", "project", "play"}:
            raise ValueError("information-set decision kind is unsupported")
        if self.root_action_ids is not None:
            actions = tuple(action for action, _node in self.problem.worlds[0].roots)
            if set(self.root_action_ids) != set(actions):
                raise ValueError(
                    "information-set root action IDs must cover the root domain"
                )
            values = tuple(self.root_action_ids.values())
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError(
                    "information-set root action IDs must be non-negative integers"
                )
            if len(set(values)) != len(values):
                raise ValueError("information-set root action IDs must be unique")

    def to_dict(self) -> dict[str, Any]:
        core = {
            "schemaVersion": 1,
            "kind": "actor_information_set_graph_v1",
            "problem": self.problem.to_dict(),
            "hiddenWorlds": copy.deepcopy(dict(self.hidden_worlds)),
            "allocationModelHash": canonical_sha256(self.model_document),
            "rootActor": {
                "seat": self.root_actor,
                "team": self.root_actor_team,
                "relation": self.root_actor_relation,
            },
        }
        if self.decision_kind != "play":
            core["decisionKind"] = self.decision_kind
        if self.root_action_ids is not None:
            core["rootActionIds"] = dict(sorted(self.root_action_ids.items()))
        if self.analysis_scope is not None:
            core["analysisScope"] = copy.deepcopy(dict(self.analysis_scope))
        return {**core, "sha256": canonical_sha256(core)}


@dataclass(frozen=True, slots=True)
class ActorInformationSetCompilationV1:
    graph: ActorInformationSetGraphV1
    node_count: int


_GRAPH_DECISION_KINDS = frozenset({"bid", "escalation", "project", "play"})


def hidden_allocation_upper_bound_v1(group_sizes: Sequence[int]) -> int:
    """Return the exact multinomial ownership upper bound for hidden cards."""

    normalized = tuple(group_sizes)
    if not normalized or any(type(size) is not int or size < 0 for size in normalized):
        raise ValueError("hidden-card group sizes must be non-negative integers")
    remaining = sum(normalized)
    worlds = 1
    for size in normalized[:-1]:
        worlds *= math.comb(remaining, size)
        remaining -= size
    return worlds


@dataclass(frozen=True, slots=True)
class ActorInformationGraphWorldV1:
    """One legal non-play world with compiler-certified observation identities."""

    world_id: str
    decision_kind: str
    graph: ProofGraphV1
    observation_ids: Mapping[str, str]

    @classmethod
    def build(
        cls,
        *,
        world_id: str,
        decision_kind: str,
        graph: ProofGraphV1,
        observation_ids: Mapping[str, str],
    ) -> "ActorInformationGraphWorldV1":
        identity = str(world_id).strip()
        kind = str(decision_kind).strip().lower()
        if not identity:
            raise ValueError("actor-information graph world id is required")
        if kind not in _GRAPH_DECISION_KINDS:
            raise ValueError("actor-information graph world kind is unsupported")
        if not isinstance(graph, ProofGraphV1) or graph.scope not in {"exact", "bounded"}:
            raise ValueError("actor-information graph world requires an exact/bounded graph")
        observations = {
            str(node_id): str(observation_id).strip()
            for node_id, observation_id in observation_ids.items()
        }
        decisions = tuple(node for node in graph.nodes if not node.is_leaf)
        missing = tuple(
            node.node_id for node in decisions if not observations.get(node.node_id)
        )
        if missing:
            raise ValueError(
                "actor-information graph world lacks observation identities"
            )
        unknown = set(observations) - {node.node_id for node in decisions}
        if unknown:
            raise ValueError(
                "actor-information graph world contains unknown observation nodes"
            )
        return cls(identity, kind, graph, observations)

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                "worldId": self.world_id,
                "decisionKind": self.decision_kind,
                "graphHash": self.graph.sha256,
                "observations": dict(sorted(self.observation_ids.items())),
            }
        )


ActorInformationGraphCompilerV1: TypeAlias = Callable[
    [
        OrToolsNormalizedExactStateV1,
        ExactTracePrefixV1,
        ScenarioProofRequestV1,
    ],
    ActorInformationGraphWorldV1 | Sequence[ActorInformationGraphWorldV1],
]


class OrToolsActorInformationSetCompilerV1:
    def __init__(self, rules: RuleBook, *, native_cp_sat: bool = False,
                 prune_own_project_pass_worlds: bool = False) -> None:
        if type(native_cp_sat) is not bool or type(prune_own_project_pass_worlds) is not bool:
            raise TypeError("native compiler options must be booleans")
        self.rules = rules
        self.native_cp_sat = native_cp_sat
        self.prune_own_project_pass_worlds = prune_own_project_pass_worlds
        self._native_bid_models = None
        self._native_bid_partial_upper = None
        self._native_nonplay_model = None
        self.graph_compiler = OrToolsCompleteStateGraphCompilerV1(
            play_domain=rules.play_domain,
            scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
            structural_domain=rules.structural_domain,
            project_domain=rules.project_domain,
            native_conversion=native_cp_sat,
        )

    def _bind_observer_scope(self, compilation, exact, request, world_states):
        if not self.native_cp_sat or request.mode != "exact":
            return compilation
        graph = compilation.graph
        if graph.hidden_worlds.get("complete") is not True or graph.analysis_scope is not None:
            return compilation
        scope = CanonicalObserverScopeV1.build(
            visibility_scope="actor_information_set", acting_observer=exact.next_actor,
            complete_states=tuple(world_states),
        )
        binding = {"scope": scope.to_safe_dict(), "sourceStateHash": exact.state_hash,
            "exactStateId": exact.exact_state_id, "rulesHash": self.rules.sha256,
            "graphWorldCount": len(graph.problem.worlds)}
        return replace(compilation, graph=replace(graph, observer_scope=scope,
            model_document={**graph.model_document, "observerScopeBinding": binding}))

    def compile_graph_worlds(
        self,
        graph_worlds: Sequence[ActorInformationGraphWorldV1],
        request: ScenarioProofRequestV1,
        *,
        maximum_nodes: int,
        maximum_depth: int,
        maximum_worlds: int,
        maximum_seconds: float,
        cancel_check: Callable[[], bool] | None = None,
        _compact_forced: bool = False,
    ) -> ActorInformationSetCompilationV1:
        """Compile non-play graph worlds into one nonanticipative game.

        The graph producer owns legal reachability and the terminal utility.  It
        must also bind every decision node to an observer-safe information ID;
        complete-state graph node IDs are deliberately not accepted as an
        implicit observation because they can contain hidden hands.
        """

        if type(_compact_forced) is not bool:
            raise TypeError("compact forced traversal must be bool")
        bounded = request.mode == "bounded"
        if request.visibility != "actor_information_set" or request.mode not in {"exact", "bounded"}:
            raise ValueError(
                "actor-information graph worlds require exact/bounded actor visibility"
            )
        if bounded and not self.native_cp_sat:
            raise ValueError("bounded non-play information graphs require native CP-SAT")
        if request.claim == "winning_continuation_line_count":
            raise ValueError(
                "actor-information-set continuation counts are unsupported"
            )
        limits = (maximum_nodes, maximum_depth, maximum_worlds)
        if any(type(value) is not int or value < 1 for value in limits):
            raise ValueError("actor-information graph limits must be positive integers")
        if (
            type(maximum_seconds) not in {int, float}
            or maximum_seconds <= 0
            or not math.isfinite(float(maximum_seconds))
        ):
            raise ValueError("actor-information graph time limit must be positive")
        if any(
            not isinstance(world, ActorInformationGraphWorldV1)
            for world in graph_worlds
        ):
            raise TypeError(
                "actor-information graph worlds must use ActorInformationGraphWorldV1"
            )
        ordered = tuple(sorted(graph_worlds, key=lambda item: item.world_id))
        if not ordered:
            raise ValueError(
                "actor-information proof requires at least one graph world"
            )
        if len({world.world_id for world in ordered}) != len(ordered):
            raise ValueError("actor-information graph world IDs must be unique")
        if len(ordered) > maximum_worlds:
            raise InformationSetCompilationStopped(
                "maximum_hidden_worlds",
                stage="information_graph_preflight",
                evidence={
                    "estimatedHiddenWorlds": len(ordered),
                    "maximumHiddenWorlds": maximum_worlds,
                },
            )
        kinds = {world.decision_kind for world in ordered}
        if len(kinds) != 1:
            raise ValueError(
                "actor-information graph worlds must share a decision kind"
            )
        decision_kind = next(iter(kinds))
        target = Team[str(request.target_team).upper()]
        target_name = target.name.lower()
        deadline = time.monotonic() + float(maximum_seconds)
        node_count = 0
        boundary_count = 0
        compiled: list[FiniteInformationWorldV1] = []
        root_actor: str | None = None
        root_actor_team: str | None = None
        root_actor_relation: str | None = None
        root_observation_id: str | None = None
        root_domain: tuple[tuple[int, str], ...] | None = None

        def stopped(reason: str) -> InformationSetCompilationStopped:
            return InformationSetCompilationStopped(
                reason,
                stage="information_graph_compilation",
                evidence={
                    "decisionKind": decision_kind,
                    "compiledWorlds": len(compiled),
                    "compatibleWorlds": len(ordered),
                    "nodeCount": node_count,
                    "maximumNodes": maximum_nodes,
                    "maximumDepth": maximum_depth,
                    "maximumHiddenWorlds": maximum_worlds,
                },
            )

        def check_budget() -> None:
            if cancel_check is not None and cancel_check():
                raise stopped("cancelled")
            if time.monotonic() >= deadline:
                raise stopped("maximum_seconds")

        for source in ordered:
            check_budget()
            source.graph.validate()
            if source.graph.scope != request.mode or (
                bounded and source.graph.boundary_evaluator != request.boundary_evaluator
            ):
                raise ValueError("non-play graph scope/evaluator disagrees with request")
            nodes = {node.node_id: node for node in source.graph.nodes}
            graph_root = nodes[source.graph.root_node_id]
            if graph_root.actor_seat is None or graph_root.actor_team is None:
                raise ValueError(
                    "actor-information graph root requires one acting seat"
                )
            relation = "target" if graph_root.actor_team == target_name else "opponent"
            observation_id = source.observation_ids[graph_root.node_id]
            domain = tuple(
                (edge.action_id, edge.action_code) for edge in graph_root.edges
            )
            if root_actor is None:
                root_actor = graph_root.actor_seat
                root_actor_team = graph_root.actor_team
                root_actor_relation = relation
                root_observation_id = observation_id
                root_domain = domain
            elif (
                graph_root.actor_seat != root_actor
                or graph_root.actor_team != root_actor_team
                or relation != root_actor_relation
                or observation_id != root_observation_id
                or domain != root_domain
            ):
                raise ValueError(
                    "hidden worlds disagree on the visible root information or actions"
                )

            node_count += 1  # The visible root is a certified graph node too.
            if node_count > maximum_nodes:
                raise stopped("maximum_nodes")
            compiled_nodes: dict[tuple[str, int, tuple], FiniteInformationNodeV1] = {}

            def visit(node_id: str, depth: int) -> FiniteInformationNodeV1:
                nonlocal node_count, boundary_count
                check_budget()
                node = nodes[node_id]
                forced = []
                while _compact_forced and not node.is_leaf and len(node.edges) == 1:
                    check_budget()
                    if depth >= maximum_depth:
                        raise stopped("maximum_depth")
                    if node.actor_seat is None or node.actor_team is None:
                        raise ValueError("actor-information decision requires one acting seat")
                    edge = node.edges[0]
                    forced.append(("target" if node.actor_team == target_name else "opponent",
                                   source.observation_ids[node.node_id], edge.action_code))
                    node = nodes[edge.child_node_id]
                    depth += 1
                if forced:
                    check_budget()
                # Share the actual endpoint only when its complete forced
                # history and depth agree; never merge different observations.
                key = (node.node_id, depth, tuple(forced))
                if key in compiled_nodes:
                    return compiled_nodes[key]
                node_count += 1
                if node_count > maximum_nodes:
                    raise stopped("maximum_nodes")
                if node.is_leaf:
                    if node.leaf_kind != "terminal" and not bounded:
                        raise ValueError(
                            "exact actor-information graph cannot contain a boundary"
                        )
                    boundary_count += int(node.leaf_kind == "boundary")
                    compiled_nodes[key] = FiniteInformationNodeV1.terminal(int(node.utility)).with_forced_prefix(forced)
                    return compiled_nodes[key]
                if depth >= maximum_depth:
                    raise stopped("maximum_depth")
                if node.actor_seat is None or node.actor_team is None:
                    raise ValueError(
                        "actor-information decision requires one acting seat"
                    )
                compiled_nodes[key] = FiniteInformationNodeV1.decision_observed(
                    ("target" if node.actor_team == target_name else "opponent"),
                    observation_id=source.observation_ids[node.node_id],
                    children={
                        edge.action_code: visit(edge.child_node_id, depth + 1)
                        for edge in node.edges
                    },
                ).with_forced_prefix(forced)
                return compiled_nodes[key]

            compiled.append(
                FiniteInformationWorldV1.build(
                    source.world_id,
                    {
                        edge.action_code: visit(edge.child_node_id, 1)
                        for edge in graph_root.edges
                    },
                )
            )

        assert (
            root_actor is not None
            and root_actor_team is not None
            and root_actor_relation is not None
            and root_observation_id is not None
            and root_domain is not None
        )
        problem = FiniteInformationSetProblemV1.build(
            problem_id=(
                "actor-information-"
                + canonical_sha256(
                    {
                        "request": request.sha256,
                        "decisionKind": decision_kind,
                        "worlds": [world.sha256 for world in ordered],
                    }
                )
            ),
            worlds=compiled,
            success_threshold=_integral_success_threshold(request.success_threshold),
        )
        graph_document = {
            "schemaVersion": 1,
            "kind": "actor_information_set_nonplay_graph_worlds_v1",
            "decisionKind": decision_kind,
            "rootObservationId": root_observation_id,
            "worlds": [
                {
                    "worldId": world.world_id,
                    "worldHash": world.sha256,
                    "graphHash": world.graph.sha256,
                    "observationBindingsHash": canonical_sha256(
                        dict(sorted(world.observation_ids.items()))
                    ),
                }
                for world in ordered
            ],
        }
        if _compact_forced:
            graph_document["representation"] = {
                "kind": "forced_prefix_nodes_v1", "retainedNodeCount": node_count,
                "nodeBudgetUnit": "retained_information_nodes", "depthUnit": "original_actions",
                "constructionNodeBudgetUnit": "retained_graph_nodes",
                "constructionNodeCount": sum(len(ProofGraphSequencePackV1.retained_node_ids(
                    world.graph.nodes, world.graph.root_node_id)) for world in ordered),
                "expandedActionNodeCount": sum(len(world.graph.nodes) for world in ordered)}
        coverage = {
            "count": len(compiled),
            "complete": not bounded,
            "coverage": {
                "method": (
                    "bounded_nonplay_observation_graphs_v1"
                    if bounded else "exact_nonplay_observation_graphs_v1"
                ),
                "scopeHash": canonical_sha256([world.world_id for world in problem.worlds]),
                "rootObservationId": root_observation_id,
            },
        }
        analysis_scope = None
        if bounded:
            # Supplied worlds establish only their declared finite scope. The
            # graph compiler cannot infer exhaustive hidden-deal enumeration.
            analysis_scope = {
                "mode": "bounded",
                "requestHash": request.sha256,
                "horizon": dict(request.horizon),
                "boundaryEvaluator": request.boundary_evaluator,
                "boundaryEvaluated": boundary_count > 0,
                "boundaryNodeCount": boundary_count,
                "continuationCoverage": "requested_horizon" if boundary_count else "round_end",
                "worldCoverageComplete": False,
                "worldScopeHash": coverage["coverage"]["scopeHash"],
                "worldLimit": maximum_worlds,
                "probabilityModel": None,
            }
        return ActorInformationSetCompilationV1(
            ActorInformationSetGraphV1(
                problem,
                coverage,
                graph_document,
                root_actor=root_actor,
                root_actor_team=root_actor_team,
                root_actor_relation=root_actor_relation,
                decision_kind=decision_kind,
                root_action_ids={code: action_id for action_id, code in root_domain},
                analysis_scope=analysis_scope,
            ),
            node_count,
        )

    @staticmethod
    def _nonplay_actor(exact: OrToolsNormalizedExactStateV1) -> Seat:
        if exact.next_actor is None:
            raise ValueError("actor-information non-play state lacks its next actor")
        return Seat.parse(exact.next_actor)

    def _public_fixed_ownership(
        self,
        exact: OrToolsNormalizedExactStateV1,
    ) -> dict[Card, Seat]:
        """Return only ownership visible at a final-deal non-play root."""

        state = exact.state
        actor = self._nonplay_actor(exact)
        initial = state.get("initialHands")
        if not isinstance(initial, Mapping):
            raise ValueError("non-play final-deal state lacks initial hands")
        fixed: dict[Card, Seat] = {}

        def bind(card: Card, seat: Seat) -> None:
            existing = fixed.get(card)
            if existing is not None and existing != seat:
                raise ValueError("public card ownership is contradictory")
            fixed[card] = seat

        actor_rows = initial.get(actor.name.lower())
        if not isinstance(actor_rows, list) or len(actor_rows) != 8:
            raise ValueError("non-play actor must observe one complete hand")
        for raw in actor_rows:
            bind(self._card(raw), actor)
        purchase = state.get("purchaseCard")
        recipient = state.get("purchaseRecipient")
        if purchase is not None and recipient is not None:
            bind(self._card(purchase), Seat.parse(recipient))
        for action in self._history(state):
            bind(action.card, action.actor)
        for project in state.get("projects") or ():
            if not isinstance(project, Mapping) or project.get("revealed") is not True:
                continue
            owner = Seat.parse(project.get("owner"))
            for raw in project.get("cards") or ():
                bind(self._card(raw), owner)
        return fixed

    def _observer_project_facts_fixed(self, exact, fixed, history) -> bool:
        """Own-hand declarations cannot constrain unknown opponent allocation."""
        if not self.native_cp_sat or history or exact.state.get("projectWinningTeam") is not None:
            return False
        actor = self._nonplay_actor(exact)
        own_cards = {card for card, owner in fixed.items() if owner == actor}
        if len(own_cards) != 8:
            return False
        public = [row for row in exact.state.get("projects", ())
                  if row.get("declarationOrder", 999) < 999]
        if (any(row.get("revealed") is True or row.get("awarded") is not None
                for row in exact.state.get("projects", ()))
                or any(row.get("owner") != actor.name.lower()
                             or any(self._card(card) not in own_cards for card in row.get("cards") or ())
                       for row in public)):
            return False
        events = [event for event in exact.state.get("events", ())
                  if str(event.get("type", "")).startswith("project")
                  and event.get("type") != "project_holdings_detected"]
        if not public:
            return (self.prune_own_project_pass_worlds
                    and len(events) == 1 and events[0].get("type") == "project_passed"
                    and events[0].get("owner") == actor.name.lower())
        return (len(events) == len(public)
                and all(event.get("type") == "project_declared"
                        and event.get("owner") == actor.name.lower() for event in events))

    def _bid_hidden_group_positions(
        self, layout: DealLayoutV1, actor: Seat,
    ) -> tuple[tuple[int, ...], ...]:
        initial = tuple(
            layout.positions_for(seat, final=False)
            for seat in Seat if seat != actor
        )
        if not self.native_cp_sat:
            return (*initial, *((position,) for position in layout.undealt_positions))
        # Cards within one group have identical owners for EVERY possible
        # purchase recipient. Never merge merely by the observed buyer.
        return (*initial, *layout.purchase_invariant_undealt_groups(self.rules.structural_domain))

    def measure_nonplay_hidden_worlds(
        self,
        exact: OrToolsNormalizedExactStateV1,
        exact_prefix: ExactTracePrefixV1,
        decision_kind: str,
    ) -> dict[str, Any]:
        """Measure the exhaustive canonical hidden-deck universe before launch."""

        kind = str(decision_kind).strip().lower()
        if kind not in _GRAPH_DECISION_KINDS or exact.decision_kind != kind:
            raise ValueError("non-play hidden-world decision kind is inconsistent")
        if kind == "play" and not self.native_cp_sat:
            raise ValueError("interleaved play graph worlds require native CP-SAT")
        actor = self._nonplay_actor(exact)
        dealer = Seat.parse(exact_prefix.dealer)
        if kind == "bid":
            layout = DealLayoutV1.build(
                self.rules.structural_domain,
                dealer=dealer,
            )
            hands = exact.state.get("hands")
            if not isinstance(hands, Mapping):
                raise ValueError("bid state lacks its initial hands")
            visible = hands.get(actor.name.lower())
            if not isinstance(visible, list) or len(visible) != 5:
                raise ValueError("bid actor must observe one five-card hand")
            purchase = self._card(exact.state.get("purchaseCard"))
            known = {self._card(raw) for raw in visible} | {purchase}
            if len(known) != 6:
                raise ValueError("bid observer cards are not unique")
            group_sizes = tuple(len(group) for group in self._bid_hidden_group_positions(layout, actor))
            hidden_cards = len(FULL_DECK) - len(known)
            method = (
                "initial_hands_and_purchase_invariant_undealt_groups_v1"
                if self.native_cp_sat else "initial_hands_and_exact_undealt_positions_v1"
            )
        else:
            fixed = self._public_fixed_ownership(exact)
            counts = {seat: 0 for seat in Seat}
            for seat in fixed.values():
                counts[seat] += 1
            group_sizes = tuple(8 - counts[seat] for seat in Seat)
            if any(size < 0 for size in group_sizes):
                raise ValueError("public ownership exceeds a Baloot hand")
            hidden_cards = len(FULL_DECK) - len(fixed)
            method = "final_hand_public_ownership_partitions_v1"
        if sum(group_sizes) != hidden_cards:
            raise ValueError("hidden-world group sizes do not conserve the deck")
        upper_bound = hidden_allocation_upper_bound_v1(group_sizes)
        core = {
            "schemaVersion": 1,
            "decisionKind": kind,
            "observer": actor.name.lower(),
            "method": method,
            "hiddenCards": hidden_cards,
            "groupSizes": list(group_sizes),
            "hiddenWorldUpperBound": upper_bound,
        }
        return {**core, "sha256": canonical_sha256(core)}

    def _native_post_auction_root_model(self, exact, prefix, *, opportunity_order,
            escalation_events, cancel_check=None, final_locked=None,
            visibility="complete_state", maximum_seconds=None, maximum_conflicts=None,
            compact_public_ownership=False):
        """Construct a continuation; observed play projects a native source witness."""
        if (not self.native_cp_sat or not isinstance(exact, OrToolsNormalizedExactStateV1)
                or not isinstance(prefix, ExactTracePrefixV1) or exact.decision_kind not in {"escalation", "project", "play"}
                or not prefix.project_actions_present or opportunity_order is None):
            raise ValueError("native post-auction root requires an explicit escalation/project/play prefix and order")
        active_escalation = exact.decision_kind == "escalation"
        if ((active_escalation and (type(final_locked) is not bool or prefix.project_actions or prefix.play_actions))
                or (not active_escalation and final_locked is not None)):
            raise ValueError("active escalation requires an explicit final lock partition and no later actions")
        exact.require_integrity()
        if exact.rules_hash != self.rules.sha256 or exact.prefix_hash != _effective_prefix_hash(prefix):
            raise ValueError("native post-auction root differs from frozen rules or prefix")
        if visibility not in {"complete_state", "actor_information_set"}:
            raise ValueError("unsupported native post-auction visibility")
        if type(compact_public_ownership) is not bool:
            raise ValueError("public ownership compaction must be boolean")
        observed = visibility == "actor_information_set"
        information = None
        if observed:
            if (type(maximum_seconds) not in {int, float}
                    or not math.isfinite(maximum_seconds) or maximum_seconds <= 0
                    or type(maximum_conflicts) is not int or maximum_conflicts < 1):
                raise ValueError("observed native root requires positive finite source-query budgets")
            deadline = time.monotonic() + maximum_seconds
            external_cancel = cancel_check
            def stopped():
                return bool(external_cancel and external_cancel()) or time.monotonic() >= deadline
            cancel_check = stopped
            source = self._native_post_auction_root_model(exact, prefix,
                opportunity_order=opportunity_order, escalation_events=escalation_events,
                final_locked=exact.state["locked"] if active_escalation else None,
                cancel_check=cancel_check, compact_public_ownership=compact_public_ownership)
            source_policy = None
            if exact.decision_kind == "play":
                # This chooses only a source witness. The returned observed
                # model still admits every legal continuation and root action.
                policy_id = "observer_safe_trick_control_v1"
                for team in Team:
                    source._native_round_fragment.play.constrain_native_observer_safe_policy(
                        team=team, policy_id=policy_id, start_step=len(prefix.play_actions))
                source_policy = {"policyId": policy_id, "scope": "source_observation_only"}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise InformationSetCompilationStopped("maximum_seconds", stage="native_observed_play_root")
            maximum = max(max(map(int, value.proto.domain)) for value in source._native_round_query_context()[1])
            witness = source.find_native_round_margin_below(target=Seat.parse(exact.next_actor).team,
                threshold=maximum + 1, maximum_seconds=remaining / 2 if source_policy else remaining,
                maximum_conflicts=maximum_conflicts, cancel_check=cancel_check, optimize_margin=False)
            if source_policy is not None:
                source_policy["probe"] = {key: witness.get(key) for key in
                    ("status", "solverStatus", "exhaustionReason", "queryModelHash", "sha256")}
                if witness["status"] != "witness_found":
                    # A witness policy cannot prove reachability impossible.
                    if time.monotonic() >= deadline or cancel_check is not None and cancel_check():
                        raise InformationSetCompilationStopped(
                            "maximum_seconds" if time.monotonic() >= deadline else "cancelled",
                            stage="native_observed_play_root",
                            evidence={"sourceObservationPolicy": source_policy})
                    source = self._native_post_auction_root_model(exact, prefix,
                        opportunity_order=opportunity_order, escalation_events=escalation_events,
                        cancel_check=cancel_check, compact_public_ownership=compact_public_ownership)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise InformationSetCompilationStopped("maximum_seconds", stage="native_observed_play_root",
                            evidence={"sourceObservationPolicy": source_policy})
                    maximum = max(max(map(int, value.proto.domain)) for value in source._native_round_query_context()[1])
                    witness = source.find_native_round_margin_below(target=Seat.parse(exact.next_actor).team,
                        threshold=maximum + 1, maximum_seconds=remaining,
                        maximum_conflicts=maximum_conflicts, cancel_check=cancel_check, optimize_margin=False)
            if witness["status"] == "no_matching_outcome":
                raise ValueError("native source observation disagrees with normalized reachability")
            if witness["status"] != "witness_found":
                raise InformationSetCompilationStopped(
                    "maximum_seconds" if time.monotonic() >= deadline else
                    witness.get("exhaustionReason") or "source_observation_unavailable",
                    stage="native_observed_play_root", evidence={"sourceObservationQuery": witness,
                        **({"sourceObservationPolicy": source_policy} if source_policy else {})})
            row = witness["candidate"][exact.decision_kind + "Observations"][
                len(escalation_events) if active_escalation else
                len(prefix.play_actions) if exact.decision_kind == "play" else len(prefix.project_actions)]
            if row["actor"] != exact.next_actor:
                raise ValueError("native source observation actor differs from normalized root")
            information = {key: value for key, value in row["information"].items()
                           if not key.startswith("complete")}
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="native_post_auction_root")
        scenario = OrToolsScenarioModelV1(rules=self.rules, dealer=Seat.parse(prefix.dealer))
        scenario.add_native_auction_fragment(required_actions=prefix.auction_actions,
            prefix_length=len(prefix.auction_actions), allow_all_passed=False)
        if not observed:
            for variable, card in zip(scenario.card_at_position, prefix.deck_order, strict=True):
                scenario.model.add(variable == card)
            scenario._complete_information_deck = tuple(prefix.deck_order)
        state = exact.state
        known_owners = None
        if compact_public_ownership:
            # Private opponent cards are permitted only in the source-witness model.
            fixed = self._public_fixed_ownership(exact) if observed else {
                self._card(card): Seat.parse(seat)
                for seat, cards in state["initialHands"].items() for card in cards}
            known_owners = tuple(fixed.get(card) for card in FULL_DECK)
        played = [row for trick in state["completedTricks"] for row in trick["plays"]]
        played.extend((state.get("currentTrick") or {}).get("plays", []))
        if len(played) != len(prefix.play_actions):
            raise ValueError("native post-auction normalized play history differs from prefix")
        branch = scenario.add_native_round_continuation(contract=state["contract"], trump=state["trumpSuit"],
            first_leader=state["firstLeader"], locked=final_locked if active_escalation else state["locked"], match_score=prefix.match_score,
            prefix=tuple(TailPrefixActionV1.build(derived["seat"], row["card"], ekka=row.get("ekka", False))
                          for row, derived in zip(prefix.play_actions, played, strict=True)),
            opportunity_order=None if observed and active_escalation else opportunity_order,
            cancel_check=cancel_check, fixed_card_owners=known_owners,
            all_project_actions_passed=(exact.decision_kind == "play"
                and len(prefix.project_actions) == len(self.rules.project_domain.declaration_seats(Seat.parse(prefix.dealer)))
                and all(ProjectDeclarationActionV1.build(row).action == "pass" for row in prefix.project_actions)))
        native = branch.escalation.native
        scenario.model.add(native["length"] >= len(prefix.escalation_actions) if active_escalation
                           else native["length"] == len(prefix.escalation_actions))
        for index, action in enumerate(prefix.escalation_actions):
            values = (int(Seat.parse(action["actor"])), native["actions"][action["action"]],
                      native["modes"][action.get("mode", action.get("requestedMode"))])
            for variable, value in zip(native["actionSlots"][index], values, strict=True):
                scenario.model.add(variable == value)
        if not observed:
            scenario.constrain_native_escalation_prefix(escalation_events, complete=not active_escalation,
                next_actor=exact.next_actor if active_escalation else None)
        if active_escalation:
            closure = branch.escalation.stage.domain.closure_action
            announcements = [(row["actor"], row["actionCode"]) for row in escalation_events
                             if row["actionCode"] != closure]
            expected = [(row["actor"], row["action"] + (
                ":" + row.get("mode", row.get("requestedMode")) if row.get("mode", row.get("requestedMode")) else ""))
                for row in prefix.escalation_actions]
            if announcements != expected:
                raise ValueError("escalation opportunity history differs from public prefix")
        if observed:
            if exact.decision_kind == "play":
                scenario.constrain_native_play_observation(information=information)
            elif active_escalation:
                scenario.constrain_native_escalation_observation(information=information)
            else:
                scenario.constrain_native_project_observation(information=information)
            scenario._native_play_source_query = witness
            scenario._native_play_source_policy = source_policy
            if time.monotonic() >= deadline:
                raise InformationSetCompilationStopped("maximum_seconds", stage="native_observed_play_root")
        else:
            scenario.constrain_native_project_prefix(tuple(ProjectDeclarationActionV1.build(row)
                                                           for row in prefix.project_actions))
            scenario._recorded_play_count = len(prefix.play_actions)
        return scenario

    def _solve_native_post_auction_policy_bounds(self, exact, prefix, target, *, selected_root_action,
            opportunity_order, escalation_events, maximum_iterations, maximum_queries,
            maximum_observations, maximum_model_bytes, maximum_conflicts, maximum_depth,
            maximum_seconds, cancel_check=None, escalation_receipt_version=2, visibility="complete_state",
            reuse_source_observation=True, observed_play_bounds_version=2, path_floor_version=2,
            progress_callback=None, opponent_ceiling_policy=None, compact_public_ownership=False):
        """Bound a post-auction root or recorded choice without path enumeration.

        Internal evidence only: production admission still requires an independent
        reconstruction of this root and its complete continuation coverage.
        """
        if (not isinstance(exact, OrToolsNormalizedExactStateV1) or not isinstance(target, Team)
                or exact.decision_kind not in {"escalation", "project", "play"}
                or Seat.parse(exact.next_actor).team != target
                or (selected_root_action is not None and (type(selected_root_action) is not str
                    or selected_root_action not in {row["code"] for row in exact.legal_actions}))
                or any(type(value) is not int or value < 1 for value in (maximum_iterations,
                    maximum_queries, maximum_observations, maximum_model_bytes, maximum_conflicts, maximum_depth))
                or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
                or maximum_seconds <= 0):
            raise ValueError("native post-auction bounds require an owned legal action and positive finite budgets")
        if type(escalation_receipt_version) is not int or escalation_receipt_version not in {1, 2}:
            raise ValueError("unsupported escalation receipt version")
        if type(reuse_source_observation) is not bool:
            raise ValueError("source observation reuse must be boolean")
        if type(observed_play_bounds_version) is not int or observed_play_bounds_version not in {1, 2}:
            raise ValueError("unsupported observed play bounds version")
        if type(path_floor_version) is not int or path_floor_version not in {1, 2}:
            raise ValueError("unsupported path floor version")
        if visibility not in {"complete_state", "actor_information_set"}:
            raise ValueError("unsupported post-auction bounds visibility")
        if (type(compact_public_ownership) is not bool
                or compact_public_ownership and visibility != "actor_information_set"):
            raise ValueError("public ownership compaction requires observed post-auction bounds")
        _validate_native_opponent_ceiling_policy(opponent_ceiling_policy, target)
        deadline = time.monotonic() + maximum_seconds
        preparation, strategy, queries_used = {}, None, 0
        if compact_public_ownership:
            preparation["ownershipCompaction"] = "public_fixed_v1"
        if visibility == "actor_information_set" and path_floor_version == 2:
            preparation["pathFloorVersion"] = 2
        if visibility == "actor_information_set" and exact.decision_kind == "play" and observed_play_bounds_version == 2:
            preparation["observedPlayBoundsVersion"] = 2

        def evidence(status, cause=None):
            core = {"schemaVersion": escalation_receipt_version if exact.decision_kind == "escalation" else 1,
                "kind": "native_post_auction_policy_bounds_v1",
                "status": status, "strategicCoverage": False, "visibility": visibility,
                "rulesHash": self.rules.sha256, "prefixHash": exact.prefix_hash,
                "sourcePrefixHash": prefix.sha256, "exactStateId": exact.exact_state_id,
                "stateHash": exact.state_hash, "decisionKind": exact.decision_kind,
                "utility": "awarded_point_margin", "continuationBoundary": "current_deal_completion",
                "targetTeam": target.name.lower(), "selectedRootAction": selected_root_action,
                "preparation": preparation, "strategy": strategy, "queriesUsed": queries_used,
                "limits": {"iterations": maximum_iterations, "queries": maximum_queries,
                    "observations": maximum_observations, "modelBytes": maximum_model_bytes,
                    "conflictsPerSolve": maximum_conflicts, "depth": maximum_depth, "seconds": maximum_seconds},
                **({"cause": cause} if cause is not None else {})}
            return {**core, "sha256": canonical_sha256(core)}

        def remaining():
            reason = "cancelled" if cancel_check is not None and cancel_check() else (
                "maximum_seconds" if time.monotonic() >= deadline else None)
            if reason:
                raise InformationSetCompilationStopped(reason, stage="native_post_auction_bounds")
            return deadline - time.monotonic()

        def reserve():
            nonlocal queries_used
            remaining()
            if queries_used >= maximum_queries:
                raise InformationSetCompilationStopped("maximum_queries", stage="native_post_auction_bounds")
            queries_used += 1

        try:
            kind = exact.decision_kind
            active_escalation = kind == "escalation"
            partitions = (False, True) if active_escalation else (None,)
            models, model_bytes, root_hash = [], 0, None
            upper_witnesses = []
            actions = tuple(sorted(row["code"] for row in exact.legal_actions))
            if active_escalation:
                preparation["partitions"] = []
            for locked in partitions:
                remaining()
                record = {"finalLocked": locked} if active_escalation else preparation
                if active_escalation:
                    preparation["partitions"].append(record)
                if visibility == "actor_information_set":
                    reserve()  # Native source-observation extraction is a real solver query.
                model = self._native_post_auction_root_model(exact, prefix,
                    opportunity_order=opportunity_order, escalation_events=escalation_events,
                    final_locked=locked, compact_public_ownership=compact_public_ownership,
                    **({"visibility": visibility, "maximum_seconds": remaining(),
                        "maximum_conflicts": maximum_conflicts} if visibility == "actor_information_set" else {}),
                    cancel_check=lambda: bool(cancel_check and cancel_check()) or time.monotonic() >= deadline)
                if visibility == "actor_information_set":
                    record["sourceObservationQuery"] = model._native_play_source_query
                    if model._native_play_source_policy is not None:
                        record["sourceObservationPolicy"] = model._native_play_source_policy
                remaining()
                proto = str(model.model.proto).encode("utf-8")
                size = len(proto)
                model_bytes += size
                record.update({"sourceModelHash": model._model_hash(search_branching="automatic",
                    proto_hash=hashlib.sha256(proto).hexdigest()),
                    "modelBytes": size, "escalationEvents": [dict(row) for row in escalation_events],
                    "opportunityOrder": [Seat.parse(seat).name.lower() for seat in opportunity_order]})
                if model_bytes > maximum_model_bytes:
                    raise InformationSetCompilationStopped("maximum_model_bytes", stage="native_post_auction_bounds")
                reserve()
                depth = model.find_native_continuation_depth_above(maximum_depth=maximum_depth,
                    maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
                    _source_proto=proto)
                del proto
                record["depthQuery"] = depth
                remaining()
                if depth["status"] != "depth_bound_proven":
                    raise InformationSetCompilationStopped("maximum_depth" if depth["status"] == "over_depth_witness"
                        else "depth_bound_incomplete", stage="native_post_auction_bounds")
                maximum = max(max(map(int, value.proto.domain)) for value in model._native_round_query_context()[1])
                closure = model._native_round_fragment.escalation.stage.domain.closure_action
                reuse_upper = (visibility == "complete_state" and active_escalation and escalation_receipt_version == 2
                    and closure in actions and selected_root_action in {None, closure})
                if visibility == "actor_information_set" and reuse_source_observation and not active_escalation:
                    # The source solve already owns this legal root. Project only
                    # its observation; never use its continuation as a strategy.
                    witness = model._native_play_source_query
                    record["rootObservationSource"] = "source_observation_projection"
                else:
                    reserve()
                    witness = model.find_native_round_margin_below(target=target.other if reuse_upper else target, threshold=maximum + 1,
                        maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
                        **({"optimize_margin": False} if visibility == "actor_information_set" else {}))
                record["rootQuery"] = witness
                remaining()
                if witness["status"] == "no_matching_outcome":
                    if not active_escalation:
                        raise ValueError("native post-auction root disagrees with normalized reachability")
                    if witness["solverStatus"] != "infeasible":
                        raise ValueError("native escalation partition lacks exact infeasibility")
                    record["retained"] = False
                    continue
                if witness["status"] != "witness_found":
                    raise InformationSetCompilationStopped(witness["exhaustionReason"] or "root_query_incomplete",
                        stage="native_post_auction_bounds")
                if visibility == "actor_information_set" and active_escalation:
                    roots = [row for row in witness["candidate"]["escalationObservations"]
                        if row["observationHash"] == model._native_escalation_root["observationHash"]
                        and not row.get("authoredPrefix", False)]
                    if len(roots) != 1:
                        raise ValueError("observed escalation lacks a unique native root")
                    root = roots[0]
                else:
                    index = len(escalation_events if active_escalation else
                                prefix.project_actions if kind == "project" else prefix.play_actions)
                    root = witness["candidate"][kind + "Observations"][index]
                if visibility == "actor_information_set" and reuse_source_observation and not active_escalation:
                    information = {key: value for key, value in root["information"].items()
                                   if not key.startswith("complete")}
                    root = {**root, "information": information,
                            "observationHash": model._round_observation_hash(kind, information)}
                if (root["actor"] != exact.next_actor or tuple(sorted(root["legalActions"])) != actions
                        or root.get("authoredPrefix", False)
                        or root_hash is not None and root_hash != root["observationHash"]):
                    raise ValueError("native post-auction root differs from normalized legal choices or partition observation")
                root_hash = root["observationHash"]
                record["rootObservationHash"] = root_hash
                if active_escalation:
                    record["retained"] = True
                models.append(model)
                upper_witnesses.append(witness if reuse_upper else None)
            if not models:
                raise ValueError("native post-auction root disagrees with normalized reachability")
            preparation["rootObservationHash"] = root_hash
            preparation["rootLegalActions"] = list(actions)
            if queries_used >= maximum_queries:
                raise InformationSetCompilationStopped("maximum_queries", stage="native_post_auction_bounds")
            closure = models[0]._native_round_fragment.escalation.stage.domain.closure_action
            if visibility == "complete_state" and active_escalation and closure in actions and selected_root_action in {None, closure}:
                probe = {"kind": "native_constant_escalation_bounds_v1", "selectedAction": closure,
                    "targetTeam": target.name.lower(), "partitions": [], "status": "incomplete"}
                preparation["constantPolicyProbe"] = probe
                lower_bounds, upper_bounds = [], []
                for model, upper_witness in zip(models, upper_witnesses, strict=True):
                    row = {"sourceModelHash": model._model_hash(search_branching="automatic")}
                    probe["partitions"].append(row)
                    maximum = max(max(map(int, value.proto.domain)) for value in model._native_round_query_context()[1])
                    if upper_witness is None:
                        reserve()
                        upper_query = model.find_native_round_margin_below(target=target.other, threshold=maximum + 1,
                            maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
                    else:
                        upper_query = upper_witness
                    row["upperQuery"] = upper_query
                    remaining()
                    if upper_query["status"] != "witness_found" or upper_query["solverStatus"] != "optimal":
                        raise InformationSetCompilationStopped("constant_upper_bound_incomplete", stage="native_post_auction_bounds")
                    upper_bounds.append(-upper_query["candidate"]["utility"])
                    # Reserve the legality solve and outcome solve together.
                    if maximum_queries - queries_used < 2:
                        raise InformationSetCompilationStopped("maximum_queries", stage="native_post_auction_bounds")
                    policy_query = model.find_native_round_with_constant_policy_below(target=target, policy_team=target,
                        auction_action=None, escalation_action=closure, threshold=maximum + 1,
                        maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
                    queries_used += len(policy_query["bindings"]) + int(policy_query["query"] is not None)
                    row["lowerQuery"] = policy_query
                    remaining()
                    query = policy_query["query"]
                    if policy_query["status"] != "queried" or query is None:
                        raise InformationSetCompilationStopped("constant_policy_bound_incomplete", stage="native_post_auction_bounds")
                    if query["status"] == "no_matching_outcome" and query["solverStatus"] == "infeasible":
                        continue  # This legal policy cannot reach this final-lock cell.
                    if query["status"] != "witness_found" or query["solverStatus"] != "optimal":
                        raise InformationSetCompilationStopped("constant_lower_bound_incomplete", stage="native_post_auction_bounds")
                    lower_bounds.append(query["candidate"]["utility"])
                if not lower_bounds:
                    raise ValueError("native constant policy has no continuation in the complete partition")
                lower, upper = min(lower_bounds), max(upper_bounds)
                if lower > upper:
                    raise ValueError("native constant policy bounds contradict legal-universe bounds")
                probe.update(status="bounds_coincide" if lower == upper else "bounds_differ",
                    lowerBound=lower, upperBound=upper)
                probe["sha256"] = canonical_sha256(probe)
                if lower == upper:
                    strategy = probe
                    return evidence("scoped_constant_bound_converged")
            observed_multi_partition = visibility == "actor_information_set" and active_escalation and len(models) > 1
            strategy = self._solve_symbolic_round_policy_bounds(models, target,
                maximum_iterations=maximum_iterations, maximum_queries=maximum_queries - queries_used,
                maximum_observations=maximum_observations, maximum_branches=len(partitions),
                maximum_conflicts=maximum_conflicts, maximum_seconds=remaining(), cancel_check=cancel_check,
                fixed_root=(root_hash, actions, selected_root_action)
                    if selected_root_action is not None else None,
                tighten_utility_ceiling=not observed_multi_partition,
                project_root_policy_probe=visibility == "actor_information_set" and (active_escalation or reuse_source_observation),
                observed_play_path_bounds=visibility == "actor_information_set" and observed_play_bounds_version == 2,
                retain_path_floor=visibility == "actor_information_set" and path_floor_version == 2,
                shorten_counterexamples=active_escalation, progress_callback=progress_callback,
                opponent_ceiling_policy=None if observed_multi_partition else opponent_ceiling_policy,
                _unavailable_opponent_ceiling_policy=opponent_ceiling_policy if observed_multi_partition else None)
            queries_used += strategy["queriesUsed"]
            remaining()
        except (InformationSetCompilationStopped, PlayTailSolveStopped) as exc:
            queries_used += exc.partial_receipt.get("evidence", {}).get("queriesUsed", 0)
            reason = exc.reason if isinstance(exc, InformationSetCompilationStopped) else exc.solver_status
            if time.monotonic() >= deadline and not (cancel_check and cancel_check()):
                reason = "maximum_seconds"
            raise InformationSetCompilationStopped(reason, stage="native_post_auction_bounds",
                evidence=evidence("incomplete", exc.partial_receipt)) from exc
        return evidence(strategy["status"])

    def _native_bid_root_model(self, exact, prefix, *, visibility, fixed_observed_world=False):
        """Build only legitimate root constraints, never candidate outcomes."""
        if (not self.native_cp_sat or not isinstance(exact, OrToolsNormalizedExactStateV1)
                or not isinstance(prefix, ExactTracePrefixV1) or exact.decision_kind != "bid"
                or visibility not in {"complete_state", "actor_information_set"}):
            raise ValueError("native bid root requires a normalized bid and explicit visibility")
        if (type(fixed_observed_world) is not bool
                or fixed_observed_world and visibility != "actor_information_set"):
            raise ValueError("fixed observed world requires actor-information visibility")
        exact.require_integrity()
        if exact.rules_hash != self.rules.sha256 or exact.prefix_hash != _effective_prefix_hash(prefix):
            raise ValueError("native bid root differs from frozen rules or prefix")
        actor = self._nonplay_actor(exact)
        scenario = OrToolsScenarioModelV1(rules=self.rules, dealer=Seat.parse(prefix.dealer),
            canonicalize_deal_packets=visibility == "actor_information_set" and not fixed_observed_world)
        scenario.add_native_auction_fragment(required_actions=prefix.auction_actions,
            required_purchase_card=self._card(exact.state["purchaseCard"]),
            prefix_length=len(prefix.auction_actions), allow_all_passed=True)
        if scenario.canonicalize_deal_packets:
            # Future packet order is unobserved; preserve its hand for the
            # solved card taker, which need not be the Ashkal caller/buyer.
            for recipient in Seat:
                selected = scenario.model.new_bool_var(f"native_packet_recipient_{int(recipient)}")
                scenario.model.add(scenario.purchase_recipient == int(recipient)).only_enforce_if(selected)
                scenario.model.add(scenario.purchase_recipient != int(recipient)).only_enforce_if(selected.Not())
                layout = DealLayoutV1.build(self.rules.structural_domain, dealer=Seat.parse(prefix.dealer), buyer=recipient)
                for packet in layout.final_packets:
                    for left, right in zip(packet.positions, packet.positions[1:]):
                        scenario.model.add(scenario.card_at_position[left] < scenario.card_at_position[right]).only_enforce_if(selected)
        scenario.model.add(scenario.native_auction_observation_literal(actor=actor,
            visible_cards=[self._card(card) for card in exact.state["hands"][actor.name.lower()]],
            purchase_card=self._card(exact.state["purchaseCard"])) == 1)
        if visibility == "complete_state" or fixed_observed_world:
            for variable, card in zip(scenario.card_at_position, prefix.deck_order, strict=True):
                scenario.model.add(variable == card)
                scenario.model.add_hint(variable, card)
        if visibility == "complete_state":
            scenario._complete_information_deck = tuple(prefix.deck_order)
        return scenario

    def _find_native_bid_compatible_world(self, exact, prefix, *, selected_action,
            excluded_decks=(), maximum_seconds, maximum_conflicts, maximum_model_bytes, maximum_worlds,
            cancel_check=None, world_ranking_id=None):
        """Find a candidate deal, never a strategic bound or impossibility proof.

        Exclusions may represent unfinished proofs. Exhausting these search
        candidates therefore cannot establish that the original claim is false.
        The caller must normalize the returned prefix and freshly prove its cap.
        """
        if (type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
                or maximum_seconds <= 0
                or any(type(value) is not int or value < 1 for value in (
                    maximum_conflicts, maximum_model_bytes, maximum_worlds))
                or not isinstance(exact, OrToolsNormalizedExactStateV1)
                or not isinstance(prefix, ExactTracePrefixV1)
                or type(selected_action) is not str
                or selected_action not in {row["code"] for row in exact.legal_actions}
                or not isinstance(excluded_decks, (tuple, list))):
            raise ValueError("native bid world search requires an owned action and finite budgets")
        cards = set(self.rules.structural_domain.card_ids)
        from .scenario_contracts import NATIVE_BID_WORLD_RANKING_IDS
        if world_ranking_id is not None and world_ranking_id not in NATIVE_BID_WORLD_RANKING_IDS:
            raise ValueError("native bid world ranking must name a registered ordering")
        if any(not isinstance(deck, (tuple, list)) or len(deck) != len(cards)
                or any(type(card) is not int for card in deck) or set(deck) != cards
                for deck in excluded_decks):
            raise ValueError("native bid world exclusions require complete unique decks")
        exclusions = sorted({tuple(deck) for deck in excluded_decks})
        deadline = time.monotonic() + maximum_seconds
        core = {"schemaVersion": 1, "kind": "native_bid_compatible_world_search_v1",
            "rulesHash": self.rules.sha256, "sourcePrefixHash": prefix.sha256,
            "selectedAction": selected_action, "strategicCoverage": False,
            "impossibilityProven": False, "excludedDeckHashes": [canonical_sha256(deck) for deck in exclusions],
            "queriesUsed": 0, "modelBytes": 0, "modelByteUnit": "utf8_cp_model_proto",
            "includesSolverWorkingMemory": False, "solverStatus": "not_started",
            "limits": {"seconds": maximum_seconds, "conflicts": maximum_conflicts,
                "modelBytes": maximum_model_bytes, "worlds": maximum_worlds, "workers": 1, "seed": 0}}

        def evidence(status):
            value = {**core, "status": status}
            return {**value, "sha256": canonical_sha256(value)}

        def check():
            reason = ("cancelled" if cancel_check is not None and cancel_check() else
                "maximum_seconds" if time.monotonic() >= deadline else None)
            if reason is not None:
                raise InformationSetCompilationStopped(reason, stage="native_bid_world_search",
                    evidence=evidence("incomplete"))

        check()
        if len(exclusions) >= maximum_worlds:
            raise InformationSetCompilationStopped("maximum_worlds", stage="native_bid_world_search",
                evidence=evidence("incomplete"))
        model = self._native_bid_root_model(exact, prefix, visibility="actor_information_set")
        actor = self._nonplay_actor(exact)
        core["observerRootHash"] = canonical_observer_information_hash_v1(
            complete_state=exact.state, acting_seat=actor.name.lower())
        model.constrain_native_auction_root_choice(selected_action, actor=actor)
        ranking_value = None
        if world_ranking_id is not None:
            from .ortools_scenario_model import attach_auction_hand_values_v1
            opponent, values = actor.team.other, []
            trump = Suit[selected_action.split(":", 1)[1].upper()] if selected_action.startswith("hokum:") else None
            default_order = []
            if trump is None:
                default_order = [copy.deepcopy(row) for row in model.model.proto.search_strategy]
                model.model.proto.search_strategy.clear()
            for seat in Seat:
                if seat.team != opponent:
                    continue
                membership = []
                for card, owner in enumerate(model.owner_of_card):
                    owned = model.model.new_bool_var(f"world_rank_{seat.name}_{card}")
                    model.model.add(owner == int(seat)).only_enforce_if(owned)
                    model.model.add(owner != int(seat)).only_enforce_if(owned.Not())
                    membership.append(owned)
                hand = attach_auction_hand_values_v1(model.model, self.rules,
                    visible_cards=membership, name=f"world_rank_{seat.name}")
                if trump is None:
                    model.model.add(sum(membership) == self.rules.structural_domain.cards_per_player)
                    weights = self.rules.data["bidding"]["heuristics"]["sun_rank_weights"]
                    order = sorted(self.rules.structural_domain.card_ids,
                        key=lambda card: (-weights[int(Card(card).rank)], card))
                    model.model.add_decision_strategy([membership[card] for card in order],
                        cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)
                values.append(hand.sun if trump is None else hand.hokum[trump])
            if trump is None:
                model.model.proto.search_strategy.extend(default_order)
            ranking_value = sum(values)
            model.model.maximize(ranking_value)
            core["ranking"] = {"id": world_ranking_id, "scope": "construction_preference_only",
                "stoppingRule": "first_feasible_solution" if trump is None else "ranking_optimum",
                "searchOrder": "native_sun_rank_then_deck_v1" if trump is None else "automatic",
                "team": opponent.name.lower(), "contract": "sun" if trump is None else "hokum",
                "trump": None if trump is None else trump.name.lower(), "profileHash": hand.profile_hash,
                "scale": hand.scale}
        if exclusions:
            model.model.add_forbidden_assignments(model.card_at_position, exclusions)
        proto = str(model.model.proto).encode("utf-8")
        core.update(modelBytes=len(proto), cpModelProtoHash=hashlib.sha256(proto).hexdigest())
        del proto
        check()
        if core["modelBytes"] > maximum_model_bytes:
            raise InformationSetCompilationStopped("maximum_model_bytes", stage="native_bid_world_search",
                evidence=evidence("incomplete"))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.stop_after_first_solution = ranking_value is not None and not selected_action.startswith("hokum:")
        if ranking_value is not None and not selected_action.startswith("hokum:"):
            solver.parameters.search_branching = cp_model.FIXED_SEARCH
        solver.parameters.max_number_of_conflicts = maximum_conflicts
        solver.parameters.max_time_in_seconds = deadline - time.monotonic()
        check()
        status = _solve_cp_sat_with_cancel_v1(solver, model.model, cancel_check)
        core.update(queriesUsed=1, solverStatus=solver.status_name(status).lower())
        check()
        if status == cp_model.MODEL_INVALID:
            raise ValueError("native bid world search model is invalid")
        if status not in {cp_model.FEASIBLE, cp_model.OPTIMAL}:
            return None, evidence("candidate_space_exhausted" if status == cp_model.INFEASIBLE
                else "not_found_within_budget")
        world = self._prefix_with_deck(prefix, [solver.value(card) for card in model.card_at_position])
        core.update(candidatePrefixHash=world.sha256, candidateDeckHash=canonical_sha256(world.deck_order))
        if ranking_value is not None:
            core["rankingValue"] = solver.value(ranking_value)
        check()
        return world, evidence("candidate_found")

    def _search_observed_bid_information_set_upper(self, exact, prefix, *, selected_action, upper_bound,
            maximum_seconds, maximum_queries, maximum_iterations, maximum_observations,
            maximum_branches, maximum_model_bytes, maximum_evidence_bytes, maximum_conflicts,
            maximum_depth, opportunity_order=None, cancel_check=None, progress_callback=None):
        """Refute a stronger all-world strategy; never restrict it to a fallback policy."""
        from .result_authority import validate_native_observed_bid_refuted_upper_v1

        if (type(upper_bound) is not int or not -(2**31) <= upper_bound < 2**31
                or type(maximum_evidence_bytes) is not int or maximum_evidence_bytes < 1):
            raise ValueError("information-set upper search requires typed bound and evidence budget")
        deadline = time.monotonic() + maximum_seconds
        generation, verification_queries, charged = None, 0, 0
        stage = "strategy_refutation"
        common = dict(maximum_branches=maximum_branches, maximum_model_bytes=maximum_model_bytes,
            maximum_conflicts=maximum_conflicts, maximum_depth=maximum_depth,
            opportunity_order=opportunity_order, cancel_check=cancel_check)

        def receipt(status, partial=None):
            core = {"schemaVersion": 1, "kind": "native_observed_bid_information_set_search_v1",
                "status": status, "stage": stage, "strategicCoverage": False, "impossibilityProven": False,
                "rulesHash": self.rules.sha256, "sourcePrefixHash": prefix.sha256,
                "action": selected_action, "upperBound": upper_bound, "queriesCharged": charged,
                "queryCountComplete": status != "incomplete",
                "generation": generation, "verificationQueriesUsed": verification_queries,
                "limits": {"seconds": maximum_seconds, "queries": maximum_queries,
                    "masterIterations": maximum_iterations, "observations": maximum_observations,
                    "branches": maximum_branches, "modelBytes": maximum_model_bytes,
                    "evidenceBytes": maximum_evidence_bytes, "conflicts": maximum_conflicts,
                    "depth": maximum_depth}, **({"partial": partial} if partial is not None else {})}
            return {**core, "sha256": canonical_sha256(core)}

        def remaining():
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled", stage="native_bid_information_set_search")
            if time.monotonic() >= deadline or charged >= maximum_queries:
                raise InformationSetCompilationStopped("maximum_seconds" if time.monotonic() >= deadline
                    else "maximum_queries", stage="native_bid_information_set_search")
            if len(canonical_json(receipt("incomplete")).encode("utf-8")) > maximum_evidence_bytes:
                raise InformationSetCompilationStopped("maximum_evidence_bytes", stage="native_bid_information_set_search")
            return deadline - time.monotonic()

        try:
            generation = self._solve_native_bid_policy_bounds(exact, prefix, Seat.parse(exact.next_actor).team,
                visibility="actor_information_set", selected_root_action=selected_action,
                required_threshold=upper_bound + 1, maximum_iterations=maximum_iterations,
                maximum_observations=maximum_observations, maximum_queries=maximum_queries,
                maximum_seconds=remaining(), progress_callback=progress_callback, **common)
            charged = generation["queriesUsed"]
            if type(charged) is not int or not 0 < charged <= maximum_queries:
                raise ValueError("information-set refutation query accounting exceeds its allowance")
            if generation["status"] != "scoped_strategy_threshold_refuted":
                if generation["status"] != "scoped_strategy_threshold_complete":
                    raise InformationSetCompilationStopped("threshold_not_completed", stage="native_bid_information_set_search")
                remaining()
                return None, receipt("threshold_not_refuted")
            stage = "fresh_verification"
            measured = []
            verified = validate_native_observed_bid_refuted_upper_v1(
                generation["strategy"], generation["preparation"], rules=self.rules, exact=exact,
                source_prefix=prefix, selected_action=selected_action, upper_bound=upper_bound,
                maximum_observations=maximum_observations, maximum_seconds=remaining(),
                maximum_queries=maximum_queries - charged, _on_queries_used=measured.append, **common)
            if (verified != upper_bound or len(measured) != 1 or type(measured[0]) is not int
                    or not 0 < measured[0] <= maximum_queries - charged):
                raise ValueError("information-set refutation fresh verification disagrees")
            verification_queries = measured[0]
            charged += verification_queries
            core = {"schemaVersion": 1, "kind": "native_observed_bid_refuted_upper_v1",
                "strategy": generation["strategy"], "preparation": generation["preparation"],
                "upperBound": upper_bound}
            premise = {"action": selected_action, "exact": exact, "sourcePrefix": prefix,
                "receipt": {**core, "sha256": canonical_sha256(core)}}
            final = receipt("upper_bound_proven")
            if (time.monotonic() >= deadline or cancel_check is not None and cancel_check()
                    or len(canonical_json({"search": final, "premise": {**premise,
                        "exact": exact.to_dict(), "sourcePrefix": prefix.to_dict()}}).encode("utf-8"))
                        > maximum_evidence_bytes):
                raise InformationSetCompilationStopped("completion_budget", stage="native_bid_information_set_search")
            return premise, final
        except InformationSetCompilationStopped as exc:
            # An interrupted reader may not expose every internal solve. Charge
            # its whole disjoint allowance, never reset it or imply actual usage.
            charged = maximum_queries
            return None, receipt("incomplete", exc.partial_receipt)

    def _search_observed_bid_alternative_upper(self, exact, prefix, *, selected_action, upper_bound,
            maximum_worlds, maximum_seconds, maximum_queries, maximum_world_seconds, maximum_world_queries,
            maximum_iterations, maximum_observations, maximum_branches, maximum_model_bytes, maximum_evidence_bytes,
            maximum_conflicts, maximum_depth, opportunity_order=None, cancel_check=None, progress_callback=None,
            world_ranking_id=None, strategy_play_fallback_id=None, minimax_budget=None):
        """Search for one freshly verified upper premise, not overall optimality.

        Per-world allowances include selection, fresh normalization and readback.
        Unknown child query usage is conservatively charged its whole allowance.
        Search exclusions never enter the returned strategic proof.
        """
        from .result_authority import validate_native_observed_bid_opponent_upper_v1

        if (any(type(value) is not int or value < 1 for value in (maximum_worlds, maximum_queries,
                maximum_world_queries, maximum_iterations, maximum_observations, maximum_branches,
                maximum_model_bytes, maximum_evidence_bytes, maximum_conflicts, maximum_depth))
                or any(type(value) not in {int, float} or not math.isfinite(value) or value <= 0
                    for value in (maximum_seconds, maximum_world_seconds))
                or type(upper_bound) is not int or not -(2**62) < upper_bound < 2**62
                or not self.native_cp_sat or not isinstance(exact, OrToolsNormalizedExactStateV1)
                or not isinstance(prefix, ExactTracePrefixV1)):
            raise ValueError("alternative world search requires native context and finite budgets")
        exact.require_integrity()
        if strategy_play_fallback_id is not None:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            native_observer_safe_policy_identity_v1(strategy_play_fallback_id, Seat.parse(exact.next_actor).team.other)
        if minimax_budget is not None and (not isinstance(minimax_budget, ScenarioProofBudgetV1)
                or strategy_play_fallback_id is None):
            raise ValueError("alternative minimax requires a typed caller budget and guarded strategy")
        from .scenario_contracts import NATIVE_BID_WORLD_RANKING_IDS
        if world_ranking_id is not None and world_ranking_id not in NATIVE_BID_WORLD_RANKING_IDS:
            raise ValueError("native bid world ranking must name a registered ordering")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("native proof progress callback must be callable")
        if (exact.decision_kind != "bid" or exact.rules_hash != self.rules.sha256
                or exact.prefix_hash != _effective_prefix_hash(prefix)
                or type(selected_action) is not str
                or selected_action not in {row["code"] for row in exact.legal_actions}):
            raise ValueError("alternative world search root mismatch")
        observer = canonical_observer_information_hash_v1(
            complete_state=exact.state, acting_seat=exact.next_actor)
        deadline = time.monotonic() + maximum_seconds
        excluded, attempts, charged = [], [], 0
        common = dict(maximum_branches=maximum_branches, maximum_model_bytes=maximum_model_bytes,
            maximum_conflicts=maximum_conflicts, maximum_depth=maximum_depth,
            opportunity_order=opportunity_order, cancel_check=cancel_check)

        def receipt(status, reason=None):
            core = {"schemaVersion": 1, "kind": "native_observed_bid_world_search_v1",
                "status": status, "strategicCoverage": False, "impossibilityProven": False,
                "observerRootHash": observer, "rulesHash": self.rules.sha256,
                "sourcePrefixHash": prefix.sha256, "action": selected_action, "upperBound": upper_bound,
                **({"worldRankingId": world_ranking_id} if world_ranking_id is not None else {}),
                "queriesCharged": charged, "attempts": attempts,
                **({"exhaustionReason": reason} if reason is not None else {}),
                "limits": {"worlds": maximum_worlds, "seconds": maximum_seconds, "queries": maximum_queries,
                    "worldSeconds": maximum_world_seconds, "worldQueries": maximum_world_queries,
                    "iterationsPerWorld": maximum_iterations, "observationsPerWorld": maximum_observations,
                    "branches": maximum_branches, "modelBytes": maximum_model_bytes,
                    "evidenceBytes": maximum_evidence_bytes,
                    "conflicts": maximum_conflicts, "depth": maximum_depth}}
            return {**core, "sha256": canonical_sha256(core)}

        def evidence_size(status="incomplete"):
            return len(canonical_json(receipt(status)).encode("utf-8"))

        def check_evidence():
            size = evidence_size()
            if size > maximum_evidence_bytes:
                raise InformationSetCompilationStopped("maximum_evidence_bytes", stage="native_bid_world_search",
                    evidence={"retainedBytes": size, "maximumBytes": maximum_evidence_bytes})

        for _ in range(maximum_worlds):
            if cancel_check is not None and cancel_check():
                return None, receipt("cancelled")
            if time.monotonic() >= deadline or charged >= maximum_queries:
                return None, receipt("not_found_within_budget")
            allowance = min(maximum_world_queries, maximum_queries - charged)
            world_deadline = min(deadline, time.monotonic() + maximum_world_seconds)
            used, models = 0, ()
            attempt = {"index": len(attempts), "queryAllowance": allowance, "stage": "selection"}
            attempts.append(attempt)

            def remaining(*, require_query=True):
                if cancel_check is not None and cancel_check():
                    raise InformationSetCompilationStopped("cancelled", stage="native_bid_world_search")
                if time.monotonic() >= world_deadline or require_query and used >= allowance:
                    raise InformationSetCompilationStopped("maximum_seconds" if time.monotonic() >= world_deadline
                        else "maximum_queries", stage="native_bid_world_search")
                return world_deadline - time.monotonic()

            def charge(count):
                nonlocal used
                if type(count) is not int or not 0 <= count <= allowance - used:
                    raise ValueError("alternative world query accounting exceeds its allowance")
                used += count

            try:
                check_evidence()
                world_prefix, selection = self._find_native_bid_compatible_world(exact, prefix,
                    selected_action=selected_action, excluded_decks=excluded, maximum_worlds=maximum_worlds,
                    maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts,
                    maximum_model_bytes=maximum_model_bytes, cancel_check=cancel_check,
                    **({"world_ranking_id": world_ranking_id} if world_ranking_id is not None else {}))
                charge(selection["queriesUsed"])
                attempt["selection"] = selection
                check_evidence()
                if world_prefix is None:
                    attempt.update(status=selection["status"], queriesCharged=used)
                    charged += used
                    return None, receipt("not_found_within_budget")
                excluded.append(world_prefix.deck_order)
                attempt["worldPrefix"] = world_prefix.to_dict()
                attempt["stage"] = "normalization"
                # A new native bid normalizer executes witness + uniqueness solves.
                # Reserve both even if the first solve stops; never reuse a cached certificate here.
                seconds = remaining()
                if allowance - used < 2:
                    raise InformationSetCompilationStopped("maximum_queries", stage="native_bid_world_search")
                charge(2)
                world = OrToolsExactStateNormalizerV1(self.rules, native_cp_sat=True).normalize(
                    world_prefix, decision_point={"kind": "bid"}, maximum_seconds=seconds,
                    cancel_check=cancel_check)
                if (world.next_actor != exact.next_actor
                        or canonical_observer_information_hash_v1(
                            complete_state=world.state, acting_seat=world.next_actor) != observer
                        or sorted(row["code"] for row in world.legal_actions)
                            != sorted(row["code"] for row in exact.legal_actions)):
                    raise ValueError("selected world differs from the observer root")
                attempt.update(worldPrefix=world_prefix.to_dict(), worldExactStateId=world.exact_state_id,
                    normalizationReceipt=world.normalization_receipt.to_dict())
                check_evidence()
                attempt["stage"] = "preparation"
                models, preparation = self._prepare_native_bid_branches(world, world_prefix,
                    visibility="actor_information_set", fixed_observed_world=True,
                    compact_purchase_invariant_ownership=True,
                    selected_root_action=selected_action, maximum_seconds=remaining(),
                    maximum_queries=allowance - used, **common)
                charge(preparation["queriesUsed"])
                attempt["preparation"] = preparation
                check_evidence()
                attempt["stage"] = "opponent_strategy"
                strategy = self._solve_symbolic_round_policy_bounds(models, Seat.parse(world.next_actor).team.other,
                    required_threshold=-upper_bound, maximum_iterations=maximum_iterations,
                    maximum_observations=maximum_observations, maximum_branches=maximum_branches,
                    maximum_conflicts=maximum_conflicts, maximum_seconds=remaining(),
                    maximum_queries=allowance - used, cancel_check=cancel_check,
                    progress_callback=(lambda row: progress_callback({**row,
                        "queryCount": charged + used + row["queryCount"]})) if progress_callback is not None else None,
                    shorten_counterexamples=strategy_play_fallback_id is None or minimax_budget is not None,
                    readback_witness_hints=world_ranking_id is not None and strategy_play_fallback_id is None,
                    **({"_play_fallback_id": strategy_play_fallback_id} if strategy_play_fallback_id is not None else {}),
                    **({"minimax_budget": minimax_budget} if minimax_budget is not None else {}))
                charge(strategy["queriesUsed"])
                attempt["strategy"] = strategy
                models = ()  # Release generation models before the independent reconstruction.
                check_evidence()
                if strategy["status"] == "scoped_strategy_threshold_refuted":
                    if (type(strategy.get("requiredThreshold")) is not int
                            or strategy["requiredThreshold"] != -upper_bound
                            or type(strategy.get("upperBound")) is not int
                            or strategy["upperBound"] >= -upper_bound):
                        raise ValueError("opponent threshold refutation contradicts its requested cap")
                    remaining(require_query=False)
                    attempt.update(status="opponent_threshold_refuted", queriesCharged=used)
                    charged += used
                    continue
                if strategy["status"] != "scoped_strategy_threshold_complete":
                    raise InformationSetCompilationStopped("opponent_threshold_not_proved", stage="native_bid_world_search")
                attempt["stage"] = "fresh_verification"
                measured = []
                verified = validate_native_observed_bid_opponent_upper_v1(strategy, preparation,
                    rules=self.rules, observer_exact=exact, exact=world, source_prefix=world_prefix,
                    action=selected_action, upper_bound=upper_bound, maximum_observations=maximum_observations,
                    maximum_seconds=remaining(), maximum_queries=allowance - used,
                    **({"strategy_play_fallback_id": strategy_play_fallback_id} if strategy_play_fallback_id is not None else {}),
                    **({"minimax_budget": minimax_budget} if minimax_budget is not None else {}),
                    _on_queries_used=measured.append, **common)
                if len(measured) != 1 or verified != upper_bound:
                    raise ValueError("alternative world fresh verification disagrees")
                charge(measured[0])
                remaining(require_query=False)
                proof = {"schemaVersion": 1, "kind": "native_observed_bid_opponent_policy_upper_v1",
                    "strategy": strategy, "preparation": preparation, "upperBound": upper_bound}
                attempt.update(status="upper_bound_proven", queriesCharged=used,
                    verificationQueriesUsed=measured[0])
                charged += used
                premise = {"action": selected_action, "exact": world, "sourcePrefix": world_prefix,
                    "receipt": {**proof, "sha256": canonical_sha256(proof)}}
                search_receipt = receipt("upper_bound_proven")
                envelope = {"premise": {**premise, "exact": world.to_dict(),
                    "sourcePrefix": world_prefix.to_dict()}, "searchReceipt": search_receipt}
                if len(canonical_json(envelope).encode("utf-8")) > maximum_evidence_bytes:
                    return None, receipt("not_found_within_budget", "maximum_evidence_bytes")
                return premise, search_receipt
            except (InformationSetCompilationStopped, ExactNormalizationStopped) as exc:
                attempt.update(status="incomplete", partial=exc.partial_receipt,
                    completedStageQueries=used, queriesCharged=allowance)
                # A stopped reader may not report all internal queries. Charging
                # its complete allowance is conservative, not an actual-use claim.
                charged += allowance
                if (exc.partial_receipt.get("exhaustionReason") == "maximum_evidence_bytes"
                        or evidence_size("not_found_within_budget") > maximum_evidence_bytes):
                    # Admission is checked after each stage. Preserve the
                    # offending diagnostic, but launch no further world.
                    return None, receipt("not_found_within_budget", "maximum_evidence_bytes")
                if cancel_check is not None and cancel_check():
                    return None, receipt("cancelled")
                if attempt["stage"] == "selection":
                    # No deck was obtained, so retrying would repeat the same
                    # deterministic selector with unchanged constraints.
                    return None, receipt("not_found_within_budget")
            finally:
                models = ()
        return None, receipt("not_found_within_budget")

    def _native_bid_branch_specs(self, dealer):
        """Exhaustive finite partition; native legality rejects empty cells."""
        return native_bid_branch_specs_v1(self.rules, dealer)

    def _build_native_bid_branch(self, exact, prefix, *, visibility, branch, cancel_check=None,
            opportunity_order=None, compact_known_initial_ownership=False, fixed_observed_world=False,
            compact_purchase_invariant_ownership=False):
        """Compile one complete deal cell from the verified bid root."""
        if branch not in self._native_bid_branch_specs(Seat.parse(prefix.dealer)):
            raise ValueError("unknown native bid branch")
        if (type(compact_known_initial_ownership) is not bool
                or compact_known_initial_ownership and visibility != "complete_state"):
            raise ValueError("initial ownership compaction requires a complete-state world")
        if (type(compact_purchase_invariant_ownership) is not bool
                or compact_purchase_invariant_ownership and (
                    not (visibility == "complete_state"
                         or visibility == "actor_information_set" and fixed_observed_world)
                    or compact_known_initial_ownership)):
            raise ValueError("purchase-invariant compaction requires a separate fixed-world mode")
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="native_bid_branch")
        scenario = self._native_bid_root_model(exact, prefix, visibility=visibility,
            fixed_observed_world=fixed_observed_world)
        if branch is None:
            scenario.add_native_redeal_continuation(match_score=prefix.match_score)
        else:
            contract, trump, leader, locked = branch
            known_owners = None
            if compact_purchase_invariant_ownership:
                fixed = {}
                for position, card in enumerate(prefix.deck_order):
                    owners = {row[position] for row in scenario._owners_by_recipient.values()}
                    if len(owners) == 1:
                        fixed[card] = Seat(next(iter(owners)))
                known_owners = tuple(fixed.get(card) for card in range(32))
            elif compact_known_initial_ownership:
                layout = DealLayoutV1.build(self.rules.structural_domain, dealer=Seat.parse(prefix.dealer))
                fixed = {}
                for seat in Seat:
                    cards = {self._card(card).id for card in exact.state["hands"][seat.name.lower()]}
                    expected = {prefix.deck_order[position] for position in layout.positions_for(seat, final=False)}
                    if cards != expected or fixed.keys() & cards:
                        raise ValueError("initial ownership differs from the normalized world")
                    fixed.update((card, seat) for card in cards)
                known_owners = tuple(fixed.get(card) for card in range(32))
            elif visibility == "actor_information_set":
                actor = self._nonplay_actor(exact)
                known_cards = {self._card(card).id for card in exact.state["hands"][actor.name.lower()]}
                known_owners = tuple(actor if card_id in known_cards else None for card_id in range(32))
            scenario.add_native_round_continuation(contract=contract, trump=trump,
                first_leader=leader, locked=locked, match_score=prefix.match_score,
                cancel_check=cancel_check, opportunity_order=opportunity_order,
                **({"fixed_card_owners": known_owners} if known_owners is not None else {}))
        return scenario

    @staticmethod
    def _validate_native_bid_information_bindings(model, record, *, opportunity_order=None, proto_hash=None):
        """Read structural provenance against an owned model, never admit proof."""
        # Reuse a digest only inside preparation's same, unmodified model state.
        if proto_hash is None:
            proto_hash = hashlib.sha256(str(model.model.proto).encode("utf-8")).hexdigest()
        information = record.get("informationBindings")
        expected_keys = {"schemaVersion", "kind", "cpModelProtoHash", "purchaseRecipient",
                         "opportunityOrder", "escalationOpportunities", "strategicCoverage", "sha256"}
        if (not isinstance(information, Mapping) or set(information) != expected_keys
                or type(information["schemaVersion"]) is not int or information["schemaVersion"] != 1
                or information["kind"] != "native_bid_information_bindings_v1"
                or information["strategicCoverage"] is not False
                or information["sha256"] != canonical_sha256({k: v for k, v in information.items() if k != "sha256"})
                or information["cpModelProtoHash"] != record.get("cpModelProtoHash")
                or information["cpModelProtoHash"] != proto_hash):
            raise ValueError("native bid information binding identity mismatch")
        expected_order = None if opportunity_order is None else [Seat.parse(seat).name.lower() for seat in opportunity_order]
        fragment = model._native_round_fragment
        if (information["opportunityOrder"] != expected_order
                or fragment is not None and fragment.scope["escalationOpportunityOrder"] != (
                    "symbolic_per_window" if expected_order is None else expected_order)):
            raise ValueError("native bid information opportunity order mismatch")

        def require_binding(value, variable):
            expected = {"index": variable.index, "name": variable.name,
                        "domain": list(model.model.proto.variables[variable.index].domain)}
            if not isinstance(value, Mapping) or canonical_sha256(value) != canonical_sha256(expected):
                raise ValueError("native bid information variable binding mismatch")

        require_binding(information["purchaseRecipient"], model.auction_fragment.states[-1]["recipient"])
        slots = () if fragment is None else fragment.escalation_opportunities
        rows = information["escalationOpportunities"]
        if not isinstance(rows, list) or len(rows) != len(slots):
            raise ValueError("native bid information opportunity coverage mismatch")
        for row, slot in zip(rows, slots, strict=True):
            if (not isinstance(row, Mapping) or set(row) != {"window", "ordinal", "actor", "active", "announced", "declined"}
                    or type(row["window"]) is not int or type(row["ordinal"]) is not int
                    or (row["window"], row["ordinal"]) != (slot["window"], slot["ordinal"])):
                raise ValueError("native bid information opportunity coverage mismatch")
            for key in ("actor", "active", "announced", "declined"):
                require_binding(row[key], slot[key])

    def _prepare_native_bid_branches(self, exact, prefix, *, visibility, maximum_branches,
            maximum_model_bytes, maximum_conflicts, maximum_seconds, cancel_check=None,
            selected_root_action=None, policy_team=None, nonplay_policy=None, maximum_queries=None, maximum_depth=None,
            opportunity_order=None, alternative_upper_bound=None, optimize_alternative_upper=False,
            opponent_play_policy_id=None, fixed_observed_world=False, compact_known_initial_ownership=False,
            compact_purchase_invariant_ownership=False, progress_callback=None, target_native_auction_policy_id=None,
            _refutation_threshold=None):
        """Construct every legal-universe cell under one explicit resource budget."""
        if (any(type(value) is not int or value < 1 for value in (maximum_branches, maximum_model_bytes, maximum_conflicts))
                or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
                or maximum_seconds <= 0):
            raise ValueError("native bid preparation requires positive finite budgets")
        if maximum_queries is not None and (type(maximum_queries) is not int or maximum_queries < 1):
            raise ValueError("native bid preparation requires a positive query budget")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("native proof progress callback must be callable")
        if (type(compact_known_initial_ownership) is not bool
                or compact_known_initial_ownership and visibility != "complete_state"):
            raise ValueError("initial ownership compaction requires a complete-state world")
        if maximum_depth is not None and (type(maximum_depth) is not int or maximum_depth < 1):
            raise ValueError("native bid preparation requires a positive depth budget")
        if opportunity_order is not None:
            if (not isinstance(opportunity_order, Sequence) or isinstance(opportunity_order, (str, bytes))
                    or any(type(seat) not in {Seat, str, int} for seat in opportunity_order)):
                raise ValueError("opportunity order must contain every seat once")
            opportunity_order = tuple(Seat.parse(seat) for seat in opportunity_order)
            if len(opportunity_order) != len(Seat) or len(set(opportunity_order)) != len(Seat):
                raise ValueError("opportunity order must contain every seat once")
        if (not self.native_cp_sat or not isinstance(exact, OrToolsNormalizedExactStateV1)
                or not isinstance(prefix, ExactTracePrefixV1) or exact.decision_kind != "bid"
                or visibility not in {"complete_state", "actor_information_set"}):
            raise ValueError("native bid preparation requires an exact bid root and visibility")
        if (type(fixed_observed_world) is not bool
                or fixed_observed_world and visibility != "actor_information_set"):
            raise ValueError("fixed observed world requires actor-information visibility")
        exact.require_integrity()
        if exact.rules_hash != self.rules.sha256 or exact.prefix_hash != _effective_prefix_hash(prefix):
            raise ValueError("native bid preparation differs from frozen root")
        if selected_root_action is not None and (type(selected_root_action) is not str
                or selected_root_action not in {row["code"] for row in exact.legal_actions}):
            raise ValueError("native bid preparation requires a legal selected root action")
        if target_native_auction_policy_id is not None:
            from .ortools_scenario_model import validate_native_observed_auction_policy_id_v1
            validate_native_observed_auction_policy_id_v1(target_native_auction_policy_id)
            if selected_root_action is None or visibility != "actor_information_set" or nonplay_policy is not None:
                raise ValueError("native target auction policy requires an actor-visible selected root and unmixed policies")
        if nonplay_policy is not None:
            if (not isinstance(policy_team, Team) or not isinstance(nonplay_policy, Mapping)
                    or set(nonplay_policy) != {"auction_action", "escalation_action"}
                    or any(value is not None and type(value) is not str for value in nonplay_policy.values())):
                raise ValueError("native bid preparation requires a typed nonplay policy")
        elif policy_team is not None:
            raise ValueError("native bid preparation policy team requires a policy")
        if type(optimize_alternative_upper) is not bool or optimize_alternative_upper and alternative_upper_bound is not None:
            raise ValueError("choose one streamed alternative bound mode")
        if _refutation_threshold is not None and (
                not optimize_alternative_upper or type(_refutation_threshold) is not int):
            raise ValueError("a refutation threshold requires native upper-bound optimization")
        streamed = alternative_upper_bound is not None or optimize_alternative_upper
        if (type(compact_purchase_invariant_ownership) is not bool
                or compact_purchase_invariant_ownership and (
                    not (visibility == "complete_state"
                         or fixed_observed_world and visibility == "actor_information_set")
                    or compact_known_initial_ownership)):
            raise ValueError("purchase-invariant compaction requires a fixed native world")
        if streamed:
            if (visibility != "complete_state" or selected_root_action is None
                    or nonplay_policy is None or (not optimize_alternative_upper and type(alternative_upper_bound) is not int)
                    or maximum_queries is None):
                raise ValueError("streamed bid alternatives require a complete-state root, policy, integer bound and query cap")
        if optimize_alternative_upper and policy_team != self._nonplay_actor(exact).team.other:
            raise ValueError("world upper bound requires the opponent policy team")
        play_policy = None
        if opponent_play_policy_id is not None:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            if not streamed or policy_team != self._nonplay_actor(exact).team.other:
                raise ValueError("opponent play policy requires a streamed opponent bound")
            play_policy = native_observer_safe_policy_identity_v1(opponent_play_policy_id, policy_team)
        deadline = time.monotonic() + maximum_seconds
        specs = self._native_bid_branch_specs(Seat.parse(prefix.dealer))
        scope = {"rulesHash": self.rules.sha256, "visibility": visibility,
            **({"ownershipCompaction": "purchase_recipient_invariant_v1"}
               if compact_purchase_invariant_ownership else {}),
            **({"compactKnownInitialOwnership": True} if compact_known_initial_ownership else {}),
            "rootHash": exact.state_hash if visibility == "complete_state" else
                canonical_observer_information_hash_v1(complete_state=exact.state,
                    acting_seat=self._nonplay_actor(exact).name.lower()),
            "matchScore": list(prefix.match_score), "continuationBoundary": "current_deal_completion",
            "nextDealAnalyzed": False,
            **({"targetNativeAuctionPolicyId": target_native_auction_policy_id}
               if target_native_auction_policy_id is not None else {}),
            **({"fixedObservedWorld": {"coverage": "single_compatible_deal",
                "stateHash": exact.state_hash, "prefixHash": exact.prefix_hash,
                "exactStateId": exact.exact_state_id}}
               if fixed_observed_world else {}),
            **({"escalationOpportunityOrder": [seat.name.lower() for seat in opportunity_order]}
               if opportunity_order is not None else {}),
            **({"selectedRootAction": selected_root_action} if selected_root_action is not None else {}),
            **({"nonplayPolicy": {"team": policy_team.name.lower(), **nonplay_policy}}
               if nonplay_policy is not None else {}),
            **({"opponentPlayPolicy": play_policy} if play_policy is not None else {}),
            "partition": [None if row is None else {"contract": row[0].value,
                "trump": row[1].name.lower() if row[1] is not None else None,
                "firstLeader": row[2].name.lower(), "locked": row[3]} for row in specs]}
        models, records, model_bytes = [], [], 0
        alternative_values = []
        queries_used = 0
        proof_queries_used, retained_model_bytes = 0, 0
        # Only solver-completed cells owned by this compiler can be resumed.
        # Keep one preparation, not an unbounded cross-candidate result cache.
        partial_key = canonical_sha256({"scope": scope, "prefix": prefix.sha256, "depth": maximum_depth,
            "conflicts": maximum_conflicts}) if optimize_alternative_upper else None
        cached_cells = (self._native_bid_partial_upper[1]
            if partial_key is not None and self._native_bid_partial_upper is not None
            and self._native_bid_partial_upper[0] == partial_key else {})
        completed_cells = {}

        def retain_cell(index, before):
            if partial_key is None:
                return
            row = records[-1]
            pending, seconds = [row], 0.0
            while pending:
                item = pending.pop()
                if isinstance(item, Mapping):
                    for key, value in item.items():
                        if key in {"maximumSeconds", "seconds"} and type(value) in {int, float}:
                            seconds = max(seconds, value)
                        elif isinstance(value, (Mapping, list, tuple)):
                            pending.append(value)
                elif isinstance(item, (list, tuple)):
                    pending.extend(item)
            completed_cells[index] = {"row": row, "seconds": seconds,
                "queries": queries_used - before[0], "proofQueries": proof_queries_used - before[1],
                "modelBytes": model_bytes - before[2], "retainedModelBytes": retained_model_bytes}

        def report_progress():
            if progress_callback is not None:
                try:
                    progress_callback({"kind": "quality_solver_progress_v1", "phase": "preparation",
                        "masterCount": 0, "queryCount": queries_used, "bindingCount": 0})
                except Exception:
                    pass  # Telemetry cannot change solver truth or qualification.

        def evidence(complete):
            core = {"schemaVersion": 1, "kind": ("native_bid_streamed_world_upper_v1" if optimize_alternative_upper
                    else "native_bid_streamed_alternative_preparation_v1" if streamed else "native_bid_branch_preparation_v1"),
                "scope": scope, "scopeHash": canonical_sha256(scope), "constructionComplete": complete,
                "strategicCoverage": False, "branches": list(records), "modelBytes": model_bytes,
                "queriesUsed": queries_used,
                "limits": {"branches": maximum_branches, "modelBytes": maximum_model_bytes,
                           "conflictsPerSolve": maximum_conflicts, "seconds": maximum_seconds,
                           "queries": maximum_queries,
                           **({"maximumDepth": maximum_depth} if maximum_depth is not None else {})},
                **({**({"alternativeUpperBound": alternative_upper_bound} if not optimize_alternative_upper else {}),
                    "proofQueriesUsed": proof_queries_used,
                    "retainedModelBytes": retained_model_bytes,
                    "modelByteUnit": "utf8_cp_model_proto", "includesSolverWorkingMemory": False}
                   if streamed else {}),
                **({"worldStateHash": exact.state_hash, "worldPrefixHash": prefix.sha256,
                    "worldExactStateId": exact.exact_state_id, "worldEffectivePrefixHash": exact.prefix_hash,
                    "targetTeam": policy_team.other.name.lower(),
                    "observerRootHash": canonical_observer_information_hash_v1(complete_state=exact.state,
                        acting_seat=self._nonplay_actor(exact).name.lower()),
                    "targetDecisionScope": "complete_information_cooperative_relaxation",
                    "upperBound": max(alternative_values) if complete and alternative_values else None}
                   if optimize_alternative_upper else {})}
            return {**core, "sha256": canonical_sha256(core)}

        def stop(reason):
            if partial_key is not None:
                # Copy before exposing the partial receipt to any caller.
                self._native_bid_partial_upper = ((partial_key, copy.deepcopy(completed_cells))
                    if len(json.dumps(completed_cells).encode("utf-8")) <= maximum_model_bytes else None)
            report_progress()
            raise InformationSetCompilationStopped(reason, stage="native_bid_preparation",
                evidence=evidence(False))

        def cancelled():
            return bool(cancel_check is not None and cancel_check()) or time.monotonic() >= deadline

        def check():
            if cancel_check is not None and cancel_check():
                stop("cancelled")
            if time.monotonic() >= deadline:
                stop("maximum_seconds")

        def reserve_query():
            nonlocal queries_used
            check()
            if maximum_queries is not None and queries_used >= maximum_queries:
                stop("maximum_queries")
            queries_used += 1

        check()
        if len(specs) > maximum_branches:
            stop("maximum_branches")
        # Reuse only this owner's last complete-state construction. Solves and
        # receipts remain fresh; no policy binding or proof result is cached.
        cacheable = visibility == "complete_state" and not streamed and nonplay_policy is None and target_native_auction_policy_id is None
        cache_key = canonical_sha256({"state": exact.state_hash, "prefix": prefix.sha256,
            "action": selected_root_action, "order": scope.get("escalationOpportunityOrder"),
            "knownOwnership": compact_known_initial_ownership,
            "purchaseInvariantOwnership": compact_purchase_invariant_ownership}) if cacheable else None
        cached_models = (self._native_bid_models[1] if cacheable and self._native_bid_models is not None
            and self._native_bid_models[0] == cache_key else {})
        self._native_bid_models = None
        completed_models = {}
        for index, spec in enumerate(specs):
            report_progress()
            check()
            before_cell = queries_used, proof_queries_used, model_bytes
            cached = cached_cells.get(index)
            if (cached is not None and cached["seconds"] <= maximum_seconds
                    and cached["retainedModelBytes"] <= maximum_model_bytes
                    and queries_used + cached["queries"] <= maximum_queries):
                row = copy.deepcopy(cached["row"])
                records.append(row)
                queries_used += cached["queries"]
                proof_queries_used += cached["proofQueries"]
                model_bytes += cached["modelBytes"]
                retained_model_bytes = max(retained_model_bytes, cached["retainedModelBytes"])
                if row["retained"]:
                    alternative_values.append(row["alternativeUpperBound"])
                retain_cell(index, before_cell)
                if (_refutation_threshold is not None and row["retained"]
                        and row["alternativeUpperBound"] >= _refutation_threshold):
                    return (), evidence(False)
                continue
            reachability = None
            if spec is not None or selected_root_action is not None or nonplay_policy is not None:
                probe = self._native_bid_root_model(exact, prefix, visibility=visibility,
                    fixed_observed_world=fixed_observed_world)
                escalation, opportunities, policy_bindings = None, (), []
                if selected_root_action is not None:
                    probe.constrain_native_auction_root_choice(selected_root_action, actor=self._nonplay_actor(exact))
                if target_native_auction_policy_id is not None:
                    choices = probe._native_visible_auction_policy_choices(
                        self._nonplay_actor(exact).team, target_native_auction_policy_id)
                    reserve_query()
                    binding = probe._bind_native_policy_choice(
                        observation_hash=canonical_sha256({"scope": scope, "stage": "auction"}),
                        action=target_native_auction_policy_id, choices=choices, kind="visible_auction", bindings={},
                        deadline=deadline, maximum_conflicts=maximum_conflicts, cancel_check=cancelled)
                    policy_bindings.append(binding)
                    if binding["status"] != "bound":
                        records.append({"partitionIndex": index, "retained": False,
                            "policyBindings": policy_bindings, "reachability": None,
                            "modelBytes": 0, "variables": 0, "constraints": 0})
                        check()
                        stop("policy_legality_not_proven")
                if spec is None:
                    probe.model.add(probe.auction_fragment.states[-1]["allPassed"] == 1)
                else:
                    contract, trump, leader, locked = spec
                    escalation, _, _ = probe._add_native_round_escalation(contract=contract, trump=trump,
                        first_leader=leader, locked=locked, match_score=prefix.match_score)
                if nonplay_policy is not None:
                    if escalation is not None:
                        opportunities = escalation.attach_opportunity_path()
                    for kind in ("auction", "escalation"):
                        action = nonplay_policy[kind + "_action"]
                        if action is None or kind == "escalation" and escalation is None:
                            continue
                        catalog = (probe.auction_fragment.legal_next_choice_literals if kind == "auction"
                            else opportunities[0]["legal"])
                        if action not in catalog:
                            raise ValueError("unsupported native preparation policy action")
                        choices = probe._native_constant_nonplay_choices(policy_team, kind, action,
                            escalation=escalation, opportunities=opportunities)
                        reserve_query()
                        binding = probe._bind_native_policy_choice(
                            observation_hash=canonical_sha256({"scope": scope, "stage": kind}),
                            action=action, choices=choices, kind="constant_" + kind, bindings={},
                            deadline=deadline, maximum_conflicts=maximum_conflicts, cancel_check=cancelled,
                            **({"record_bound_model_hash": True} if optimize_alternative_upper else {}),
                            **({"record_model_bytes": True} if streamed else {}))
                        policy_bindings.append(binding)
                        if streamed:
                            retained_model_bytes = max(retained_model_bytes,
                                binding["modelBytes"]["owner"] + binding["modelBytes"]["legalityProbe"])
                            if retained_model_bytes > maximum_model_bytes:
                                stop("maximum_model_bytes")
                        if binding["status"] != "bound":
                            records.append({"partitionIndex": index, "retained": False,
                                "policyBindings": policy_bindings, "reachability": None,
                                "modelBytes": 0, "variables": 0, "constraints": 0})
                            check()
                            stop("policy_legality_not_proven")
                check()
                solver = cp_model.CpSolver()
                solver.parameters.num_search_workers = 1
                solver.parameters.random_seed = 0
                solver.parameters.max_number_of_conflicts = maximum_conflicts
                solver.parameters.max_time_in_seconds = max(deadline - time.monotonic(), 1e-9)
                reserve_query()
                status = _solve_cp_sat_with_cancel_v1(solver, probe.model, cancelled)
                probe_proto = str(probe.model.proto).encode("utf-8")
                reachability = {"cpModelProtoHash": hashlib.sha256(probe_proto).hexdigest(),
                    "solverStatus": solver.status_name(status).lower(),
                    "branches": int(solver.num_branches), "conflicts": int(solver.num_conflicts)}
                if nonplay_policy is not None or target_native_auction_policy_id is not None:
                    reachability["policyBindings"] = policy_bindings
                if optimize_alternative_upper:
                    reachability["scope"] = {"preparationScopeHash": canonical_sha256(scope),
                        "partitionIndex": index, "partition": scope["partition"][index]}
                    reachability["boundModelHash"] = probe._model_hash(search_branching="automatic",
                        proto_hash=reachability["cpModelProtoHash"])
                if streamed:
                    reachability["modelBytes"] = len(probe_proto)
                    retained_model_bytes = max(retained_model_bytes, reachability["modelBytes"])
                    if retained_model_bytes > maximum_model_bytes:
                        stop("maximum_model_bytes")
                if optimize_alternative_upper:
                    reachability["sha256"] = canonical_sha256(reachability)
                del probe_proto, probe
                check()
                if status == cp_model.INFEASIBLE:
                    records.append({"partitionIndex": index, "retained": False,
                        "reachability": reachability, "modelBytes": 0, "variables": 0, "constraints": 0})
                    retain_cell(index, before_cell)
                    continue
                if status == cp_model.MODEL_INVALID:
                    raise ValueError("native bid reachability model is invalid")
                if status not in {cp_model.FEASIBLE, cp_model.OPTIMAL}:
                    records.append({"partitionIndex": index, "retained": False,
                        "reachability": reachability, "modelBytes": 0, "variables": 0, "constraints": 0})
                    stop("reachability_incomplete")
            expected_hash = None
            try:
                if index in cached_models:
                    model, expected_hash = cached_models[index]
                else:
                    model = self._build_native_bid_branch(exact, prefix, visibility=visibility,
                        branch=spec, cancel_check=cancelled, fixed_observed_world=fixed_observed_world,
                        **({"compact_purchase_invariant_ownership": True} if compact_purchase_invariant_ownership
                           else {"compact_known_initial_ownership": True} if streamed or compact_known_initial_ownership else {}),
                        **({"opportunity_order": opportunity_order} if opportunity_order is not None else {}))
                if selected_root_action is not None:
                    model.constrain_native_auction_root_choice(selected_root_action, actor=self._nonplay_actor(exact))
            except (PlayTailSolveStopped, InformationSetCompilationStopped):
                check()
                raise
            check()
            play_binding = None
            if play_policy is not None and spec is not None:
                before = hashlib.sha256(str(model.model.proto).encode("utf-8")).hexdigest()
                model._native_round_fragment.play.constrain_native_observer_safe_policy(
                    team=policy_team, policy_id=opponent_play_policy_id)
                check()
            proto = str(model.model.proto).encode("utf-8")
            model_bytes += len(proto)
            retained_model_bytes = max(retained_model_bytes, len(proto))
            proto_hash = hashlib.sha256(proto).hexdigest()
            if play_policy is not None and spec is not None:
                binding_core = {"schemaVersion": 1, "kind": "native_visible_play_policy_binding_v1",
                    "policy": play_policy, "sourceCpModelProtoHash": before,
                    "boundCpModelProtoHash": proto_hash}
                play_binding = {**binding_core, "sha256": canonical_sha256(binding_core)}
            records.append({"partitionIndex": index, "retained": True, "reachability": reachability,
                "sourceModelHash": model._model_hash(search_branching="automatic", proto_hash=proto_hash),
                "cpModelProtoHash": proto_hash,
                "modelBytes": len(proto), "variables": len(model.model.proto.variables),
                "constraints": len(model.model.proto.constraints)})
            if expected_hash is not None and records[-1]["sourceModelHash"] != expected_hash:
                raise ValueError("owned native bid construction was mutated")
            if play_binding is not None:
                records[-1]["playPolicyBinding"] = play_binding
            if visibility == "actor_information_set":
                # Variable bindings are structural evidence, not a strategy or
                # a claim that every value in a variable domain is reachable.
                def binding(variable):
                    return {"index": variable.index, "name": variable.name,
                            "domain": list(model.model.proto.variables[variable.index].domain)}

                round_fragment = model._native_round_fragment
                information = {"schemaVersion": 1,
                    "kind": "native_bid_information_bindings_v1",
                    "cpModelProtoHash": records[-1]["cpModelProtoHash"],
                    "purchaseRecipient": binding(model.auction_fragment.states[-1]["recipient"]),
                    "opportunityOrder": [seat.name.lower() for seat in opportunity_order]
                        if opportunity_order is not None else None,
                    "escalationOpportunities": [] if round_fragment is None else [
                        {"window": slot["window"], "ordinal": slot["ordinal"],
                         **{key: binding(slot[key]) for key in ("actor", "active", "announced", "declined")}}
                        for slot in round_fragment.escalation_opportunities],
                    "strategicCoverage": False}
                records[-1]["informationBindings"] = {**information,
                    "sha256": canonical_sha256(information)}
                self._validate_native_bid_information_bindings(model, records[-1],
                    opportunity_order=opportunity_order, proto_hash=proto_hash)
            if (retained_model_bytes if streamed else model_bytes) > maximum_model_bytes:
                stop("maximum_model_bytes")
            if maximum_depth is not None:
                reserve_query()
                depth_query = model.find_native_continuation_depth_above(maximum_depth=maximum_depth,
                    maximum_seconds=deadline - time.monotonic(), maximum_conflicts=maximum_conflicts,
                    cancel_check=cancelled, _source_proto=proto,
                    **({"record_model_bytes": True} if streamed else {}))
                records[-1]["depthQuery"] = depth_query
                if streamed:
                    retained_model_bytes = max(retained_model_bytes,
                        depth_query["modelBytes"]["owner"] + depth_query["modelBytes"]["depthQuery"])
                    if retained_model_bytes > maximum_model_bytes:
                        stop("maximum_model_bytes")
                check()
                if depth_query["status"] != "depth_bound_proven":
                    stop("maximum_depth" if depth_query["status"] == "over_depth_witness" else "depth_bound_incomplete")
            del proto
            if not streamed:
                models.append(model)
                if cacheable:
                    completed_models[index] = (model, records[-1]["sourceModelHash"])
            else:
                needed = (1 + int(nonplay_policy["auction_action"] is not None)
                    + int(spec is not None and nonplay_policy["escalation_action"] is not None))
                check()
                if maximum_queries is not None and queries_used + needed > maximum_queries:
                    stop("maximum_queries")
                threshold = (max(max(map(int, point.proto.domain)) for point in model._native_round_query_context()[1]) + 1
                    if optimize_alternative_upper else -alternative_upper_bound)
                query = model.find_native_round_with_constant_policy_below(
                    target=policy_team, policy_team=policy_team, threshold=threshold,
                    optimize_margin=optimize_alternative_upper,
                    maximum_seconds=deadline - time.monotonic(), maximum_conflicts=maximum_conflicts,
                    **({"record_policy_model_hashes": True} if optimize_alternative_upper else {}),
                    cancel_check=cancelled, record_model_bytes=True, **nonplay_policy)
                records[-1]["alternativeQuery"] = query
                used = len(query["bindings"]) + int(query["query"] is not None)
                queries_used += used
                proof_queries_used += used
                if query.get("sourceModelHash") != records[-1]["sourceModelHash"]:
                    raise ValueError("streamed bid alternative model ownership mismatch")
                retained_model_bytes = max(retained_model_bytes, query["modelBytes"]["retainedUpperBound"])
                del model
                check()
                if retained_model_bytes > maximum_model_bytes:
                    stop("maximum_model_bytes")
                outcome = query.get("query") or {}
                if optimize_alternative_upper:
                    if (query["status"] != "queried" or outcome.get("status") != "witness_found"
                            or outcome.get("solverStatus") != "optimal"
                            or type((outcome.get("candidate") or {}).get("utility")) is not int):
                        stop("alternative_upper_bound_not_certified")
                    upper = -outcome["candidate"]["utility"]
                    records[-1]["alternativeUpperBound"] = upper
                    alternative_values.append(upper)
                    retain_cell(index, before_cell)
                    if _refutation_threshold is not None and upper >= _refutation_threshold:
                        # One cell defeats this sufficient refutation attempt.
                        # Remaining cells cannot restore it. This partial receipt
                        # is NOT a world upper bound or strategic certificate.
                        report_progress()
                        return (), evidence(False)
                elif (query["status"] != "queried" or not isinstance(query.get("query"), Mapping)
                        or outcome["status"] != "no_matching_outcome"
                        or outcome["solverStatus"] != "infeasible"):
                    if (query["status"] == "queried" and outcome.get("status") == "witness_found"
                            and outcome.get("solverStatus") in {"optimal", "feasible"}
                            and type((outcome.get("candidate") or {}).get("utility")) is int
                            and outcome["candidate"]["utility"] < -alternative_upper_bound):
                        # This refutes only the proposed cap, not selected-action optimality.
                        stop("alternative_upper_bound_refuted")
                    stop("alternative_upper_bound_not_certified")
        check()
        report_progress()
        if cacheable:
            self._native_bid_models = (cache_key, completed_models)
        if partial_key is not None:
            self._native_bid_partial_upper = None
        return tuple(models), evidence(True)

    def _enumerate_hidden_groups(
        self,
        pool: Sequence[Card],
        group_sizes: Sequence[int],
        *,
        deadline: float,
        maximum_worlds: int,
        cancel_check: Callable[[], bool] | None,
        bounded_world_scope: bool = False,
        exact: OrToolsNormalizedExactStateV1 | None = None,
        bid_prefix: ExactTracePrefixV1 | None = None,
        _batch_allocations: bool = False,
    ) -> tuple[tuple[tuple[Card, ...], ...], ...]:
        if type(_batch_allocations) is not bool or (_batch_allocations and not bounded_world_scope):
            raise ValueError("batched allocations require an explicitly bounded world scope")
        cards = tuple(sorted(pool, key=lambda card: card.id))
        sizes = tuple(group_sizes)
        if sum(sizes) != len(cards):
            raise ValueError("hidden allocation group sizes do not match the pool")
        if not cards:
            return (tuple(() for _ in sizes),)
        bid_model = None
        if bid_prefix is not None:
            if not self.native_cp_sat or exact is None or exact.decision_kind != "bid":
                raise ValueError("bid allocation requires native observed bid context")
            exact.require_integrity()
            if exact.rules_hash != self.rules.sha256 or exact.prefix_hash != _effective_prefix_hash(bid_prefix):
                raise ValueError("bid allocation context differs from frozen rules or prefix")
            actor = self._nonplay_actor(exact)
            layout = DealLayoutV1.build(self.rules.structural_domain, dealer=Seat.parse(bid_prefix.dealer))
            groups = self._bid_hidden_group_positions(layout, actor)
            if sizes != tuple(map(len, groups)):
                raise ValueError("bid allocation groups differ from observed context")
            bid_model = self._native_bid_root_model(exact, bid_prefix, visibility="actor_information_set")
            model = bid_model.model
        else:
            model = cp_model.CpModel()
        owns = {
            (group, card): model.new_bool_var(f"hidden_{group}_{card.id}")
            for group in range(len(sizes))
            for card in cards
        }
        for card in cards:
            model.add_exactly_one(owns[group, card] for group in range(len(sizes)))
        for group, size in enumerate(sizes):
            model.add(sum(owns[group, card] for card in cards) == size)
        if bid_model is not None:
            for group, positions in enumerate(groups):
                for card in cards:
                    model.add_allowed_assignments(
                        [bid_model.position_of_card[card.id]], [(position,) for position in positions],
                    ).only_enforce_if(owns[group, card])
        elif self.native_cp_sat and exact is not None:
            fixed = self._public_fixed_ownership(exact)
            owners = tuple(model.new_int_var(0, 3, f"project_owner_{card.id}") for card in FULL_DECK)
            for card in FULL_DECK:
                if card in fixed:
                    model.add(owners[card.id] == int(fixed[card]))
                else:
                    for seat in Seat:
                        model.add(owners[card.id] == int(seat)).only_enforce_if(owns[int(seat), card])
            history = self._history(exact.state)
            # Before public declarations there is no project evidence to constrain
            # hidden ownership. Full candidate normalization still binds projects.
            if (
                any(row.get("declarationOrder", 999) < 999 for row in exact.state.get("projects", ()))
                or (self.rules.project_domain.declaration_mode != "optional"
                    and any(event.get("type") == "project_passed" for event in exact.state.get("events", ())))
                or len(history) > 4
            ) and not self._observer_project_facts_fixed(exact, fixed, history):
                fragment = attach_project_lifecycle_fragment_v1(
                    model, owner_of_card=owners,
                    contract=Contract.parse(exact.state.get("contract")),
                    dealer=Seat.parse(exact.state.get("dealer")),
                    trump=self._trump(exact.state), domain=self.rules.project_domain,
                    second_trick_leader=(Seat.parse(exact.state["completedTricks"][0]["winner"])
                        if self.rules.project_domain.tie_break_order == "second_trick_reveal"
                        and exact.state.get("completedTricks") else None),
                    fixed_card_owners=tuple(fixed.get(card) for card in FULL_DECK),
                    declaration_only=(self.rules.project_domain.tie_break_order == "second_trick_reveal"
                                      and not exact.state.get("completedTricks")),
                )
                self._bind_public_project_choices(model, fragment, exact.state, observer=exact.next_actor)
            if history:
                try:
                    self.graph_compiler.play_stage.build_model(
                        cards=FULL_DECK, ownership=None,
                        contract=Contract.parse(exact.state.get("contract")),
                        trump=self._trump(exact.state),
                        first_leader=Seat.parse(exact.state.get("firstLeader")),
                        play_domain=self.rules.play_domain,
                        structural_domain=self.rules.structural_domain,
                        prefix=history, locked=exact.state.get("locked") is True,
                        decision_prefix_length=len(history),
                        play_step_count=((len(history) + len(Seat) - 1) // len(Seat)) * len(Seat),
                        shared_model=model, shared_owner_of_card=owners,
                        _export_predicate_contexts=False, cancel_check=cancel_check,
                    )
                except PlayTailSolveStopped as exc:
                    raise InformationSetCompilationStopped(
                        "cancelled" if exc.partial_receipt["evidence"].get("cancelled") is True
                        else exc.solver_status, stage="hidden_world_nonplay_allocation",
                        evidence={"causeReceipt": exc.partial_receipt},
                    ) from exc
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise InformationSetCompilationStopped(
                "maximum_seconds", stage="hidden_world_nonplay_enumeration"
            )
        results: list[tuple[tuple[Card, ...], ...]] = []
        if _batch_allocations:
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = 0
            solver.parameters.max_time_in_seconds = remaining
            solver.parameters.enumerate_all_solutions = True
            seen = set()

            class CollectAllocations(cp_model.CpSolverSolutionCallback):
                def on_solution_callback(inner):
                    allocation = tuple(tuple(card for card in cards
                        if inner.boolean_value(owns[group, card])) for group in range(len(sizes)))
                    if allocation not in seen:
                        seen.add(allocation)
                        results.append(allocation)
                    if len(results) >= maximum_worlds:
                        inner.stop_search()

            status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check, CollectAllocations())
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled", stage="hidden_world_nonplay_enumeration")
            if (status not in {cp_model.INFEASIBLE, cp_model.OPTIMAL, cp_model.FEASIBLE}
                    or status == cp_model.FEASIBLE and len(results) < maximum_worlds):
                raise InformationSetCompilationStopped(
                    "maximum_seconds" if time.monotonic() >= deadline else "hidden_world_solver_incomplete",
                    stage="hidden_world_nonplay_enumeration",
                    evidence={"solverStatus": solver.status_name(status).lower()},
                )
            return tuple(sorted(results, key=lambda allocation: tuple(
                card.id for hand in allocation for card in hand)))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise InformationSetCompilationStopped(
                    "maximum_seconds", stage="hidden_world_nonplay_enumeration"
                )
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = 0
            solver.parameters.max_time_in_seconds = max(0.001, remaining)
            status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
            if status == cp_model.INFEASIBLE:
                break
            if status not in {cp_model.FEASIBLE, cp_model.OPTIMAL}:
                reason = (
                    "cancelled"
                    if cancel_check is not None and cancel_check()
                    else "maximum_seconds"
                    if time.monotonic() >= deadline
                    else "hidden_world_solver_incomplete"
                )
                raise InformationSetCompilationStopped(
                    reason,
                    stage="hidden_world_nonplay_enumeration",
                    evidence={"solverStatus": solver.status_name(status).lower()},
                )
            results.append(tuple(
                tuple(card for card in cards if solver.value(owns[group, card]))
                for group in range(len(sizes))
            ))
            selected = [
                owns[group, card]
                for group in range(len(sizes))
                for card in cards
                if solver.value(owns[group, card])
            ]
            model.add(sum(selected) <= len(cards) - 1)
            if bounded_world_scope and len(results) == maximum_worlds:
                break
            if len(results) > maximum_worlds:
                raise InformationSetCompilationStopped(
                    "maximum_hidden_worlds",
                    stage="hidden_world_nonplay_enumeration",
                    evidence={
                        "acceptedWorlds": len(results),
                        "maximumHiddenWorlds": maximum_worlds,
                    },
                )
        results.sort(key=lambda allocation: tuple(
            card.id for hand in allocation for card in hand
        ))
        return tuple(results)

    def _prefix_with_deck(
        self,
        source: ExactTracePrefixV1,
        deck_order: Sequence[int],
        *,
        project_actions: Sequence[Mapping[str, Any]] | None = None,
    ) -> ExactTracePrefixV1:
        raw = source.to_dict()
        raw.pop("sha256", None)
        raw["deckOrder"] = list(deck_order)
        # Assertions validate the authored complete state; they are not public
        # observations and must not eliminate otherwise indistinguishable worlds.
        raw["stateAssertions"] = {}
        raw["projectAssertions"] = []
        if project_actions is not None:
            raw["projectActions"] = [dict(item) for item in project_actions]
        return ExactTracePrefixV1.build(raw)

    def _canonical_final_deck(
        self,
        *,
        prefix: ExactTracePrefixV1,
        state: Mapping[str, Any],
        hands: Mapping[Seat, Sequence[Card]],
    ) -> tuple[int, ...]:
        purchase = self._card(state.get("purchaseCard"))
        recipient = Seat.parse(state.get("purchaseRecipient"))
        layout = DealLayoutV1.build(
            self.rules.structural_domain,
            dealer=Seat.parse(prefix.dealer),
            buyer=recipient,
        )
        deck: list[int | None] = [None] * len(FULL_DECK)
        deck[layout.purchase_position] = purchase.id
        for seat in Seat:
            cards = sorted(hands[seat], key=lambda card: card.id)
            if len(cards) != 8 or len(set(cards)) != 8:
                raise ValueError("hidden world must contain four unique full hands")
            if seat == recipient:
                if purchase not in cards:
                    raise ValueError(
                        "purchase recipient does not own the purchase card"
                    )
                cards.remove(purchase)
            positions = [
                position
                for position in layout.positions_for(seat, final=True)
                if position != layout.purchase_position
            ]
            if len(positions) != len(cards):
                raise AssertionError("canonical hidden-world deal layout is malformed")
            for position, card in zip(positions, cards, strict=True):
                deck[position] = card.id
        if any(card_id is None for card_id in deck):
            raise AssertionError("canonical hidden-world deck is incomplete")
        normalized = tuple(int(card_id) for card_id in deck)
        if sorted(normalized) != list(range(len(FULL_DECK))):
            raise ValueError("canonical hidden-world deck is not unique")
        return normalized

    def _bind_public_project_choices(self, model, fragment, state, *, observer=None) -> None:
        """Constrain public declarations without exposing their hidden cards."""
        if observer is not None:
            state = {**state, "currentPlayer": observer}
        declarations = {
            Seat.parse(row["owner"]): self._public_project_key(row, state)
            for row in state.get("projects", ())
            if row.get("declarationOrder", 999) < 999
        }
        passed = {
            Seat.parse(event["owner"])
            for event in state.get("events", ())
            if event.get("type") == "project_passed"
        }
        for owner in declarations.keys() | passed:
            key = declarations.get(owner)
            for project in fragment.projects:
                if project.owner != owner:
                    continue
                if (key is None or key[:3] != (owner, project.kind, project.value)
                        or (key[3] is not None and key[3] != project.high_rank)):
                    model.add(project.declared == 0)
            model.add(sum(project.declared for project in fragment.projects
                          if project.owner == owner) == int(key is not None))
        revealed_actors = {action.actor for action in self._history(state)[4:8]}
        if revealed_actors and any(project.awarded is None for project in fragment.projects):
            raise ValueError("revealed projects require a completed comparison model")
        visible = {
            (*self._public_project_key(row, state), tuple(sorted(self._card(card).id for card in row["cards"])))
            for row in state.get("projects", ()) if row.get("revealed") is True
        }
        for project in fragment.projects:
            if project.owner in revealed_actors:
                key = (project.owner, project.kind, project.value, project.high_rank,
                       tuple(sorted(card.id for card in project.cards)))
                reveal_source = (project.comparison_winner
                                 if self.rules.project_domain.reveal_scope == "comparison_winner"
                                 else project.awarded)
                model.add(reveal_source == int(key in visible))

    def _rebind_project_actions(
        self,
        prefix: ExactTracePrefixV1,
        *,
        hands: Mapping[Seat, Sequence[Card]],
        state: Mapping[str, Any],
        deadline: float,
        cancel_check: Callable[[], bool] | None = None,
        observer: str | None = None,
    ) -> tuple[Mapping[str, Any], ...] | None:
        if not prefix.project_actions_present:
            return None
        if self.native_cp_sat and not prefix.project_actions:
            return ()
        domain = self.rules.project_domain
        contract = Contract.parse(state.get("contract"))
        trump_raw = state.get("trumpSuit")
        trump = Suit.parse(trump_raw) if trump_raw is not None else None
        if self.native_cp_sat:
            def remaining_seconds() -> float:
                if cancel_check is not None and cancel_check():
                    raise InformationSetCompilationStopped("cancelled", stage="hidden_world_project_rebinding")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise InformationSetCompilationStopped("maximum_seconds", stage="hidden_world_project_rebinding")
                return remaining

            fixed = {card.id: seat for seat, cards in hands.items() for card in cards}
            if len(fixed) != 32 or sum(map(len, hands.values())) != 32:
                raise ValueError("hidden project world requires 32 unique cards")
            seats = domain.declaration_seats(Seat.parse(prefix.dealer))
            if len(prefix.project_actions) > len(seats):
                raise ValueError("project declaration prefix is too long")
            actions = tuple(
                ProjectDeclarationActionV1.build(raw, actor=seats[index])
                for index, raw in enumerate(prefix.project_actions)
            )
            if (domain.declaration_mode == "optional"
                    and all(action.action == "pass" for action in actions)
                    and not any(row.get("declarationOrder", 999) < 999 or row.get("revealed") is True
                                for row in state.get("projects", ()))
                    and state.get("projectWinningTeam") is None):
                return tuple(action.to_dict() for action in actions)
            remaining_seconds()
            model = cp_model.CpModel()
            fragment = attach_project_lifecycle_fragment_v1(
                model,
                owner_of_card=tuple(model.new_int_var(0, 3, f"owner_{i}") for i in range(32)),
                fixed_card_owners=tuple(fixed[i] for i in range(32)),
                contract=contract, dealer=prefix.dealer, trump=trump, domain=domain,
                declaration_only=(domain.tie_break_order == "second_trick_reveal"
                                  and not state.get("completedTricks")),
                second_trick_leader=(Seat.parse(state["completedTricks"][0]["winner"])
                    if domain.tie_break_order == "second_trick_reveal"
                    and state.get("completedTricks") else None),
            )
            self._bind_public_project_choices(model, fragment, state, observer=observer)
            for action in actions:
                model.add(
                    sum(p.declared for p in fragment.projects if p.owner == action.actor)
                    == int(action.action == "declare")
                )
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = 0
            solver.parameters.max_time_in_seconds = remaining_seconds()
            status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
            remaining_seconds()
            if status == cp_model.INFEASIBLE:
                raise ValueError("hidden world cannot reproduce the public project action pattern")
            if status not in {cp_model.FEASIBLE, cp_model.OPTIMAL}:
                raise InformationSetCompilationStopped(
                    "hidden_world_solver_incomplete", stage="hidden_world_project_rebinding",
                    evidence={"solverStatus": solver.status_name(status).lower()},
                )
            declared = {
                p.owner: p.project_id for p in fragment.projects
                if solver.boolean_value(p.declared)
            }
            return tuple(
                ProjectDeclarationActionV1(
                    action.actor, action.action,
                    declared[action.actor] if action.action == "declare" else None,
                ).to_dict()
                for action in actions
            )
        projects = domain.detect(hands, contract, Seat.parse(prefix.dealer))
        declaration = domain.declaration_state(
            projects,
            dealer=Seat.parse(prefix.dealer),
        )
        rebound: list[Mapping[str, Any]] = []
        for recorded in prefix.project_actions:
            legal = domain.legal_declaration_actions(
                declaration,
                trump_suit=trump,
            )
            action_kind = str(recorded.get("action"))
            candidates = [item for item in legal if item.action == action_kind]
            if len(candidates) != 1:
                raise ValueError(
                    "hidden world cannot reproduce the public project action pattern"
                )
            chosen = candidates[0]
            declaration, _announced = domain.apply_declaration_action(
                declaration,
                chosen,
                trump_suit=trump,
            )
            rebound.append(chosen.to_dict())
        return tuple(rebound)

    def enumerate_graph_world_inputs(
        self,
        exact: OrToolsNormalizedExactStateV1,
        exact_prefix: ExactTracePrefixV1,
        decision_point: Mapping[str, Any],
        *,
        maximum_worlds: int,
        maximum_seconds: float,
        cancel_check: Callable[[], bool] | None = None,
        bounded_world_scope: bool = False,
        _batch_allocations: bool = False,
    ) -> tuple[tuple[OrToolsNormalizedExactStateV1, ExactTracePrefixV1], ...]:
        """Normalize exhaustive or explicitly limited non-play allocations."""

        if type(bounded_world_scope) is not bool:
            raise TypeError("bounded world scope must be a boolean")
        if type(_batch_allocations) is not bool or (_batch_allocations and not bounded_world_scope):
            raise ValueError("batched allocations require an explicitly bounded world scope")
        if bounded_world_scope and not self.native_cp_sat:
            raise ValueError("bounded non-play worlds require native CP-SAT")
        if type(maximum_worlds) is not int or maximum_worlds < 1:
            raise ValueError("non-play hidden-world limit must be positive")
        if (
            type(maximum_seconds) not in {int, float}
            or maximum_seconds <= 0
            or not math.isfinite(float(maximum_seconds))
        ):
            raise ValueError("non-play hidden-world time limit must be positive")
        kind = str(decision_point.get("kind") or "").strip().lower()
        measurement = self.measure_nonplay_hidden_worlds(
            exact,
            exact_prefix,
            kind,
        )
        upper_bound = int(measurement["hiddenWorldUpperBound"])
        if upper_bound > maximum_worlds and not bounded_world_scope:
            raise InformationSetCompilationStopped(
                "maximum_hidden_worlds",
                stage="hidden_world_nonplay_preflight",
                evidence={
                    **measurement,
                    "maximumHiddenWorlds": maximum_worlds,
                },
            )
        deadline = time.monotonic() + float(maximum_seconds)
        actor = self._nonplay_actor(exact)
        prefixes: list[ExactTracePrefixV1] = []
        if kind == "bid":
            layout = DealLayoutV1.build(
                self.rules.structural_domain,
                dealer=Seat.parse(exact_prefix.dealer),
            )
            hands = exact.state.get("hands")
            assert isinstance(hands, Mapping)
            visible = tuple(
                sorted(
                    (self._card(raw) for raw in hands[actor.name.lower()]),
                    key=lambda card: card.id,
                )
            )
            purchase = self._card(exact.state.get("purchaseCard"))
            known = set(visible) | {purchase}
            pool = tuple(card for card in FULL_DECK if card not in known)
            group_positions = self._bid_hidden_group_positions(layout, actor)
            allocations = self._enumerate_hidden_groups(
                pool,
                tuple(len(positions) for positions in group_positions),
                deadline=deadline,
                maximum_worlds=maximum_worlds,
                cancel_check=cancel_check,
                bounded_world_scope=bounded_world_scope,
                exact=exact if self.native_cp_sat and not bounded_world_scope else None,
                bid_prefix=exact_prefix if self.native_cp_sat and not bounded_world_scope else None,
                _batch_allocations=_batch_allocations,
            )
            for allocation in allocations:
                deck: list[int | None] = [None] * len(FULL_DECK)
                for position, card in zip(
                    layout.positions_for(actor, final=False),
                    visible,
                    strict=True,
                ):
                    deck[position] = card.id
                deck[layout.purchase_position] = purchase.id
                for positions, cards in zip(
                    group_positions,
                    allocation,
                    strict=True,
                ):
                    for position, card in zip(
                        positions,
                        sorted(cards, key=lambda item: item.id),
                        strict=True,
                    ):
                        deck[position] = card.id
                if any(card_id is None for card_id in deck):
                    raise AssertionError("bid hidden-world deck is incomplete")
                prefixes.append(
                    self._prefix_with_deck(
                        exact_prefix,
                        tuple(int(card_id) for card_id in deck),
                    )
                )
        else:
            fixed = self._public_fixed_ownership(exact)
            fixed_by_seat = {
                seat: {card for card, owner in fixed.items() if owner == seat}
                for seat in Seat
            }
            pool = tuple(card for card in FULL_DECK if card not in fixed)
            sizes = tuple(8 - len(fixed_by_seat[seat]) for seat in Seat)
            allocations = self._enumerate_hidden_groups(
                pool,
                sizes,
                deadline=deadline,
                maximum_worlds=maximum_worlds,
                cancel_check=cancel_check,
                bounded_world_scope=bounded_world_scope,
                exact=exact,
                _batch_allocations=_batch_allocations,
            )
            for allocation in allocations:
                world_hands = {
                    seat: tuple(
                        sorted(
                            (*fixed_by_seat[seat], *allocation[int(seat)]),
                            key=lambda card: card.id,
                        )
                    )
                    for seat in Seat
                }
                try:
                    project_actions = self._rebind_project_actions(
                        exact_prefix,
                        hands=world_hands,
                        state=exact.state,
                        deadline=deadline,
                        cancel_check=cancel_check,
                        observer=exact.next_actor,
                    )
                    deck = self._canonical_final_deck(
                        prefix=exact_prefix,
                        state=exact.state,
                        hands=world_hands,
                    )
                    prefixes.append(
                        self._prefix_with_deck(
                            exact_prefix,
                            deck,
                            project_actions=project_actions,
                        )
                    )
                except ValueError:
                    # The exhaustive ownership allocation exists, but cannot
                    # reproduce the public project action pattern.
                    continue

        prefixes = list({prefix.sha256: prefix for prefix in prefixes}.values())
        normalizer = OrToolsExactStateNormalizerV1(
            self.rules,
            native_cp_sat=True,
        )
        expected_observation = canonical_observer_information_hash_v1(
            complete_state=exact.state,
            acting_seat=actor.name.lower(),
        )
        results: list[tuple[OrToolsNormalizedExactStateV1, ExactTracePrefixV1]] = []
        rejections: dict[str, int] = {}
        for world_prefix in prefixes:
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped(
                    "cancelled", stage="hidden_world_nonplay_normalization"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise InformationSetCompilationStopped(
                    "maximum_seconds",
                    stage="hidden_world_nonplay_normalization",
                    evidence={"normalizedWorlds": len(results)},
                )
            try:
                normalized = normalizer.normalize(
                    world_prefix,
                    decision_point=decision_point,
                    maximum_seconds=remaining,
                    cancel_check=cancel_check,
                )
            except ValueError:
                rejections["normalization"] = rejections.get("normalization", 0) + 1
                continue
            except RuntimeError as exc:
                raise InformationSetCompilationStopped(
                    "hidden_world_normalization_incomplete",
                    stage="hidden_world_nonplay_normalization",
                    evidence={
                        "normalizedWorlds": len(results),
                        **({"causeReceipt": exc.partial_receipt}
                           if isinstance(exc, ExactNormalizationStopped) else {}),
                    },
                ) from exc
            observation = canonical_observer_information_hash_v1(
                complete_state=normalized.state,
                acting_seat=actor.name.lower(),
            )
            if observation == expected_observation:
                results.append((normalized, world_prefix))
            else:
                rejections["observer_projection"] = rejections.get("observer_projection", 0) + 1
        if not results:
            raise InformationSetCompilationStopped(
                "no_compatible_hidden_worlds",
                stage="hidden_world_nonplay_normalization",
                evidence={
                    "enumeratedAllocations": len(prefixes),
                    "hiddenWorldUpperBound": upper_bound,
                    "rejections": rejections,
                },
            )
        return tuple(sorted(results, key=lambda item: item[0].sha256))

    def compile_nonplay(
        self,
        exact: OrToolsNormalizedExactStateV1,
        exact_prefix: ExactTracePrefixV1,
        decision_point: Mapping[str, Any],
        request: ScenarioProofRequestV1,
        *,
        graph_compiler: ActorInformationGraphCompilerV1,
        maximum_nodes: int,
        maximum_depth: int,
        maximum_worlds: int,
        maximum_seconds: float,
        cancel_check: Callable[[], bool] | None = None,
        graph_world_multiplier: int = 1,
        _compact_forced: bool = False,
    ) -> ActorInformationSetCompilationV1:
        """Compile a finite non-play information set through one graph callback."""

        if not callable(graph_compiler):
            raise TypeError("actor-information non-play graph compiler is required")
        if type(_compact_forced) is not bool or (_compact_forced and not self.native_cp_sat):
            raise ValueError("compact non-play information compilation requires native CP-SAT")
        if type(graph_world_multiplier) is not int or graph_world_multiplier < 1:
            raise ValueError("graph_world_multiplier must be a positive integer")
        if type(maximum_worlds) is not int or maximum_worlds < 1:
            raise ValueError("maximum_worlds must be a positive integer")
        allocation_limit = maximum_worlds // graph_world_multiplier
        if allocation_limit == 0:
            raise InformationSetCompilationStopped("maximum_combined_worlds",
                stage="information_graph_preflight", evidence={
                    "graphWorldMultiplier": graph_world_multiplier, "maximumWorlds": maximum_worlds,
                })
        bounded = request.mode == "bounded"
        if request.visibility != "actor_information_set" or request.mode not in {"exact", "bounded"}:
            raise ValueError(
                "actor-information non-play compiler requires exact/bounded actor visibility"
            )
        if bounded and not self.native_cp_sat:
            raise ValueError("bounded non-play compiler requires native CP-SAT")
        if request.claim == "winning_continuation_line_count":
            raise ValueError(
                "actor-information-set continuation counts are unsupported"
            )
        kind = str(decision_point.get("kind") or "").strip().lower()
        if (
            kind == "project"
            and self.rules.project_domain.declaration_mode != "optional"
            and not self.native_cp_sat
        ):
            raise ValueError(
                "actor-information project proof requires optional declaration"
            )
        started = time.monotonic()
        inputs = self.enumerate_graph_world_inputs(
            exact,
            exact_prefix,
            decision_point,
            maximum_worlds=allocation_limit,
            maximum_seconds=maximum_seconds,
            cancel_check=cancel_check,
            **({"bounded_world_scope": True} if bounded else {}),
        )
        worlds: list[ActorInformationGraphWorldV1] = []
        graph_nodes = 0
        if len(inputs) * graph_world_multiplier > maximum_worlds:
            raise InformationSetCompilationStopped("maximum_combined_worlds",
                stage="information_graph_preflight", evidence={
                    "hiddenAllocations": len(inputs), "graphWorldMultiplier": graph_world_multiplier,
                    "maximumWorlds": maximum_worlds,
                })
        for normalized, world_prefix in inputs:
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped(
                    "cancelled", stage="information_graph_world_build"
                )
            remaining = maximum_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise InformationSetCompilationStopped(
                    "maximum_seconds", stage="information_graph_world_build"
                )
            produced = graph_compiler(normalized, world_prefix, request)
            variants = (produced,) if isinstance(produced, ActorInformationGraphWorldV1) else produced
            if (not isinstance(variants, Sequence) or isinstance(variants, (str, bytes))
                    or any(not isinstance(world, ActorInformationGraphWorldV1) for world in variants)):
                raise TypeError("non-play graph callback must return ActorInformationGraphWorldV1 or a sequence thereof")
            if len(variants) != graph_world_multiplier:
                raise ValueError("non-play graph callback disagrees with graph_world_multiplier")
            for world in variants:
                if world.decision_kind != kind:
                    raise ValueError("non-play graph callback changed the decision kind")
                graph_nodes += (len(ProofGraphSequencePackV1.retained_node_ids(world.graph.nodes, world.graph.root_node_id))
                                if _compact_forced else len(world.graph.nodes))
                if graph_nodes > maximum_nodes:
                    raise InformationSetCompilationStopped(
                        "maximum_nodes", stage="information_graph_world_build",
                        evidence={"compiledWorlds": len(worlds), "graphNodes": graph_nodes,
                                  "maximumNodes": maximum_nodes},
                    )
                worlds.append(world)
        remaining = maximum_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise InformationSetCompilationStopped(
                "maximum_seconds", stage="information_graph_compilation"
            )
        compilation = self.compile_graph_worlds(
            worlds,
            request,
            maximum_nodes=maximum_nodes,
            maximum_depth=maximum_depth,
            maximum_worlds=maximum_worlds,
            maximum_seconds=remaining,
            cancel_check=cancel_check,
            _compact_forced=_compact_forced,
        )
        if graph_world_multiplier != 1:
            expansion = {"hiddenAllocationCount": len(inputs),
                         "graphWorldMultiplier": graph_world_multiplier,
                         "combinedWorldCount": len(worlds), "maximumWorlds": maximum_worlds,
                         "inputHashes": [canonical_sha256({"exact": normalized.sha256,
                             "prefix": world_prefix.sha256}) for normalized, world_prefix in inputs]}
            compilation = replace(compilation, graph=replace(compilation.graph,
                model_document={**compilation.graph.model_document, "worldExpansion": expansion}))
        if kind == "play":
            compilation = self._bind_observer_scope(compilation, exact, request,
                tuple(normalized.state for normalized, _prefix in inputs))
        return compilation

    @staticmethod
    def _card(raw: Any) -> Card:
        return Card.parse(raw if not isinstance(raw, Mapping) else raw.get("code"))

    def _history(self, state: Mapping[str, Any]) -> tuple[TailPrefixActionV1, ...]:
        rows: list[TailPrefixActionV1] = []
        for trick in [*state.get("completedTricks", []), state.get("currentTrick")]:
            if not isinstance(trick, Mapping):
                continue
            for play in trick.get("plays", []):
                rows.append(
                    TailPrefixActionV1.build(
                        play.get("seat"),
                        self._card(play.get("card")),
                        ekka=play.get("ekka") is True,
                    )
                )
        return tuple(rows)

    def _initial_hands(
        self,
        state: Mapping[str, Any],
        remaining: Mapping[Seat, tuple[Card, ...]],
    ) -> dict[Seat, tuple[Card, ...]]:
        played = {seat: [] for seat in Seat}
        for action in self._history(state):
            played[action.actor].append(action.card)
        return {
            seat: tuple(
                sorted((*played[seat], *remaining[seat]), key=lambda card: card.id)
            )
            for seat in Seat
        }

    @staticmethod
    def _trump(state: Mapping[str, Any]) -> Suit | None:
        contract = Contract.parse(state.get("contract"))
        raw = state.get("trumpSuit")
        if contract.effective == Contract.HOKUM:
            if raw is None:
                raise ValueError("Hokum information-set state requires trumpSuit")
            return Suit.parse(raw)
        if raw is not None:
            raise ValueError("Sun information-set state cannot carry trumpSuit")
        return None

    def _projects_v2(
        self,
        state: Mapping[str, Any],
        initial: Mapping[Seat, tuple[Card, ...]],
        *,
        maximum_seconds: float = 15.0,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[list[dict[str, Any]], Team | None]:
        contract = Contract.parse(state.get("contract"))
        trump = self._trump(state)
        solved = OrToolsProjectLifecycleModelV1().solve(
            hands=initial,
            contract=contract,
            dealer=Seat.parse(state.get("dealer")),
            trump=trump,
            domain=self.rules.project_domain,
            timeout_seconds=max(0.001, maximum_seconds),
            second_trick_leader=(Seat.parse(state["completedTricks"][0]["winner"])
                if self.rules.project_domain.tie_break_order == "second_trick_reveal"
                and state.get("completedTricks") else None),
            cancel_check=cancel_check,
        )
        if solved.solver_status != "optimal":
            raise InformationSetCompilationStopped(
                "project_model_incomplete",
                stage="hidden_world_project_compilation",
                evidence={"solverStatus": solved.solver_status},
            )
        history = self._history(state)
        first_trick_actors = {action.actor for action in history[:4]}
        second_trick_actors = {action.actor for action in history[4:8]}
        compared = len(history) >= 4
        model = cp_model.CpModel()
        reached: list[tuple[Any, Any, Any, Any]] = []
        for index, project in enumerate(solved.projects):
            reveal_source = (project.comparison_winner
                             if self.rules.project_domain.reveal_scope == "comparison_winner"
                             else project.awarded)
            declared = model.new_bool_var(f"reached_declared_{index}")
            scoring = model.new_bool_var(f"reached_scoring_{index}")
            revealed = model.new_bool_var(f"reached_revealed_{index}")
            order = model.new_int_var(0, 999, f"reached_order_{index}")
            model.add_bool_and(
                [
                    model.new_constant(int(project.declared)),
                    model.new_constant(int(project.owner in first_trick_actors)),
                ]
            ).only_enforce_if(declared)
            model.add_bool_or(
                [
                    model.new_constant(int(not project.declared)),
                    model.new_constant(int(project.owner not in first_trick_actors)),
                ]
            ).only_enforce_if(declared.Not())
            model.add_bool_and(
                [
                    model.new_constant(int(project.awarded)),
                    model.new_constant(int(compared)),
                ]
            ).only_enforce_if(scoring)
            model.add_bool_or(
                [
                    model.new_constant(int(not project.awarded)),
                    model.new_constant(int(not compared)),
                ]
            ).only_enforce_if(scoring.Not())
            model.add_bool_and(
                [
                    model.new_constant(int(reveal_source)),
                    model.new_constant(int(project.owner in second_trick_actors)),
                ]
            ).only_enforce_if(revealed)
            model.add_bool_or(
                [
                    model.new_constant(int(not reveal_source)),
                    model.new_constant(int(project.owner not in second_trick_actors)),
                ]
            ).only_enforce_if(revealed.Not())
            model.add(order == int(project.declaration_order or 0)).only_enforce_if(
                declared
            )
            model.add(order == 999).only_enforce_if(declared.Not())
            reached.append((declared, scoring, revealed, order))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_time_in_seconds = max(0.001, maximum_seconds)
        status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
        if status != cp_model.OPTIMAL:
            raise InformationSetCompilationStopped(
                "project_lifecycle_projection_incomplete",
                stage="hidden_world_project_compilation",
                evidence={"solverStatus": solver.status_name(status).lower()},
            )
        rows = [
            {
                "owner": project.owner.name.lower(),
                "kind": project.kind,
                "value": project.value,
                "highRank": project.high_rank,
                "declarationOrder": solver.value(reached[index][3]),
                "team": project.team.name.lower(),
                "cards": (
                    [card.to_dict() for card in project.cards]
                    if solver.value(reached[index][2])
                    else None
                ),
                "scoring": bool(solver.value(reached[index][1])),
                "revealed": bool(solver.value(reached[index][2])),
            }
            for index, project in enumerate(solved.projects)
            if project.selected
        ]
        return rows, solved.winning_team if compared else None

    @staticmethod
    def _public_project_key(
        row: Mapping[str, Any],
        state: Mapping[str, Any] | None = None,
    ) -> tuple[Seat, str, int, int | None]:
        rank_visible = state is None or project_rank_is_public(
            row, state.get("projects", ()), str(state.get("currentPlayer") or ""),
            state.get("projectRankVisibility"),
            comparison_complete=bool(state.get("completedTricks")) or any(
                event.get("type") == "projects_compared" for event in state.get("events", ())
            ),
        )
        try:
            owner = Seat.parse(row.get("owner"))
            kind = str(row["kind"])
            value = int(row["value"])
            high_rank = int(row["highRank"]) if rank_visible else None
        except (KeyError, TypeError, ValueError) as exc:
            raise InformationSetCompilationStopped(
                "native_project_public_history_invalid",
                stage="hidden_world_project_compilation",
            ) from exc
        if (
            not kind
            or type(row.get("value")) is not int
            or (rank_visible and type(row.get("highRank")) is not int)
        ):
            raise InformationSetCompilationStopped(
                "native_project_public_history_invalid",
                stage="hidden_world_project_compilation",
            )
        return owner, kind, value, high_rank

    def _projects_v3(
        self,
        state: Mapping[str, Any],
        initial: Mapping[Seat, tuple[Card, ...]],
        *,
        maximum_seconds: float,
        cancel_check: Callable[[], bool] | None,
    ) -> tuple[list[dict[str, Any]], Team | None, dict[str, Any]]:
        """Project one hidden world from the shared native lifecycle variables."""

        contract = Contract.parse(state.get("contract"))
        trump = self._trump(state)
        dealer = Seat.parse(state.get("dealer"))
        ownership: dict[Card, Seat] = {}
        for seat in Seat:
            hand = tuple(initial.get(seat, ()))
            if len(hand) != 8:
                raise ValueError("native project binding requires eight cards per seat")
            for card in hand:
                if card in ownership:
                    raise ValueError("native project binding cannot duplicate cards")
                ownership[card] = seat
        if set(ownership) != set(FULL_DECK):
            raise ValueError("native project binding requires the complete deck")

        model = cp_model.CpModel()
        owner_of_card = tuple(
            model.new_int_var(0, 3, f"information_project_owner_c{card.id}")
            for card in FULL_DECK
        )
        for card in FULL_DECK:
            model.add(owner_of_card[card.id] == int(ownership[card]))
        pending_comparison = (self.rules.project_domain.tie_break_order == "second_trick_reveal"
                              and not state.get("completedTricks"))
        fragment = attach_project_lifecycle_fragment_v1(
            model,
            owner_of_card=owner_of_card,
            contract=contract,
            dealer=dealer,
            trump=trump,
            domain=self.rules.project_domain,
            name="information_project",
            second_trick_leader=(Seat.parse(state["completedTricks"][0]["winner"])
                if self.rules.project_domain.tie_break_order == "second_trick_reveal"
                and state.get("completedTricks") else None),
            fixed_card_owners=tuple(ownership[card] for card in FULL_DECK),
            compact_fixed_states=True,
            declaration_only=pending_comparison,
        )
        required_bindings = (
            "eligible",
            "selected",
            "declarable",
            "declared",
            "declaration_order",
            "comparison_winner",
            "awarded",
        )
        if any(
            any(not hasattr(project, field) for field in required_bindings)
            for project in fragment.projects
        ):
            raise InformationSetCompilationStopped(
                "native_project_lifecycle_bindings_unavailable",
                stage="hidden_world_project_compilation",
            )

        history = self._history(state)
        first_trick_actors = {action.actor for action in history[:4]}
        second_trick_actors = {action.actor for action in history[4:8]}
        compared = len(history) >= 4
        raw_projects = state.get("projects")
        # Explicit native declarations can precede the owner's first card.
        declaration_actors = first_trick_actors | {
            Seat.parse(event.get("owner"))
            for event in state.get("events", ())
            if event.get("type") in {"project_declared", "project_passed"}
        }
        if declaration_actors and not isinstance(raw_projects, list):
            raise InformationSetCompilationStopped(
                "native_project_public_history_missing",
                stage="hidden_world_project_compilation",
                evidence={
                    "reachedActors": [
                        seat.name.lower()
                        for seat in sorted(first_trick_actors, key=int)
                    ]
                },
            )
        public_declarations: dict[Seat, tuple[tuple[Seat, str, int, int | None], int]] = {}
        for raw in raw_projects or ():
            if not isinstance(raw, Mapping):
                raise InformationSetCompilationStopped(
                    "native_project_public_history_invalid",
                    stage="hidden_world_project_compilation",
                )
            order = raw.get("declarationOrder")
            if type(order) is not int:
                raise InformationSetCompilationStopped(
                    "native_project_public_history_invalid",
                    stage="hidden_world_project_compilation",
                )
            if order >= 999:
                continue
            key = self._public_project_key(raw, state)
            owner = key[0]
            if owner not in declaration_actors or owner in public_declarations:
                raise InformationSetCompilationStopped(
                    "native_project_public_history_invalid",
                    stage="hidden_world_project_compilation",
                    evidence={
                        "owner": owner.name.lower(),
                        "declarationOrder": order,
                        "reachedActors": [
                            seat.name.lower()
                            for seat in sorted(declaration_actors, key=int)
                        ],
                        "duplicateOwner": owner in public_declarations,
                    },
                )
            public_declarations[owner] = (key, order)

        by_owner = {
            seat: tuple(
                project for project in fragment.projects if project.owner == seat
            )
            for seat in Seat
        }
        for actor in declaration_actors:
            declaration = public_declarations.get(actor)
            if declaration is None:
                model.add(sum(project.declared for project in by_owner[actor]) == 0)
                continue
            public_key, public_order = declaration
            matching = tuple(
                project
                for project in by_owner[actor]
                if (
                    project.owner,
                    project.kind,
                    project.value,
                )
                == public_key[:3]
                and (public_key[3] is None or project.high_rank == public_key[3])
            )
            nonmatching = tuple(
                project for project in by_owner[actor] if project not in matching
            )
            model.add(sum(project.declared for project in matching) == 1)
            model.add(sum(project.declared for project in nonmatching) == 0)
            for project in matching:
                model.add(project.declaration_order == public_order).only_enforce_if(
                    project.declared
                )

        reached: dict[str, tuple[Any, Any, Any, Any]] = {}
        unreached_order = model.new_int_var(999, 999, "information_unreached_order")
        for index, project in enumerate(fragment.projects):
            if list(project.eligible.proto.domain) == [0, 0]:
                not_reached = model.new_constant(0) if pending_comparison else project.awarded
                reached[project.project_id] = (
                    project.declared, not_reached, not_reached, unreached_order,
                )
                continue
            declared = model.new_bool_var(f"information_reached_declared_{index}")
            scoring = model.new_bool_var(f"information_reached_scoring_{index}")
            revealed = model.new_bool_var(f"information_reached_revealed_{index}")
            order = model.new_int_var(0, 999, f"information_reached_order_{index}")
            if project.owner in declaration_actors:
                model.add(declared == project.declared)
            else:
                model.add(declared == 0)
            if compared:
                model.add(scoring == project.awarded)
            else:
                model.add(scoring == 0)
            if project.owner in second_trick_actors:
                model.add(revealed == (project.comparison_winner
                          if self.rules.project_domain.reveal_scope == "comparison_winner"
                          else project.awarded))
            else:
                model.add(revealed == 0)
            model.add(order == project.declaration_order).only_enforce_if(declared)
            model.add(order == 999).only_enforce_if(declared.Not())
            reached[project.project_id] = (declared, scoring, revealed, order)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_time_in_seconds = max(0.001, maximum_seconds)
        status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
        if status == cp_model.INFEASIBLE:
            raise _IncompatibleProjectWorld(
                "public project history is incompatible with hidden world"
            )
        if status != cp_model.OPTIMAL:
            raise InformationSetCompilationStopped(
                "native_project_lifecycle_incomplete",
                stage="hidden_world_project_compilation",
                evidence={"solverStatus": solver.status_name(status).lower()},
            )
        evidence = None if pending_comparison else fragment.extract(solver)
        selected_projects = (tuple(project for project in fragment.projects
                                   if solver.boolean_value(project.selected)) if pending_comparison
                             else tuple(project for project in evidence.projects if project.selected))
        rows = [
            {
                "owner": project.owner.name.lower(),
                "kind": project.kind,
                "value": project.value,
                "highRank": project.high_rank,
                "declarationOrder": solver.value(reached[project.project_id][3]),
                "team": project.team.name.lower(),
                "cards": (
                    [card.to_dict() for card in project.cards]
                    if solver.boolean_value(reached[project.project_id][2])
                    else None
                ),
                "scoring": solver.boolean_value(reached[project.project_id][1]),
                "revealed": solver.boolean_value(reached[project.project_id][2]),
            }
            for project in selected_projects
        ]
        history_projection = {
            "plays": [
                {
                    "actor": action.actor.name.lower(),
                    "card": action.card.id,
                    "ekka": action.ekka,
                }
                for action in history[:8]
            ],
            "firstTrickActors": [
                seat.name.lower() for seat in sorted(first_trick_actors, key=int)
            ],
            "secondTrickActors": [
                seat.name.lower() for seat in sorted(second_trick_actors, key=int)
            ],
            "comparisonReached": compared,
            "publicDeclarations": [
                {
                    "owner": owner.name.lower(),
                    "kind": key[1],
                    "value": key[2],
                    "highRank": key[3],
                    "declarationOrder": order,
                }
                for owner, (key, order) in sorted(
                    public_declarations.items(), key=lambda item: int(item[0])
                )
            ],
        }
        if declaration_actors != first_trick_actors:
            history_projection["declarationActors"] = [
                seat.name.lower() for seat in sorted(declaration_actors, key=int)
            ]
        history_projection_hash = canonical_sha256(history_projection)
        proto_hash = hashlib.sha256(str(model.proto).encode("utf-8")).hexdigest()
        binding_rows = [
            {
                "projectId": project.project_id,
                "declared": project.declared.Name(),
                "comparisonWinner": project.comparison_winner.Name() if project.comparison_winner is not None else None,
                "awarded": project.awarded.Name() if project.awarded is not None else None,
                "reachedDeclared": reached[project.project_id][0].Name(),
                "reachedScoring": reached[project.project_id][1].Name(),
                "reachedRevealed": reached[project.project_id][2].Name(),
                "reachedOrder": reached[project.project_id][3].Name(),
            }
            for project in fragment.projects
        ]
        model_hash = canonical_sha256(
            {
                "cpModelProtoHash": proto_hash,
                "fragmentHash": fragment.fragment_hash,
                "historyProjectionHash": history_projection_hash,
            }
        )
        receipt_core = {
            "schemaVersion": 1,
            "kind": "actor_information_project_native_binding_v1",
            "authority": "cp_sat",
            "fragmentHash": fragment.fragment_hash,
            "fragmentEvidenceHash": (evidence.evidence_hash if evidence is not None else
                                     canonical_sha256({"declarations": rows, "comparisonPending": True})),
            "historyProjectionHash": history_projection_hash,
            "cpModelProtoHash": proto_hash,
            "modelHash": model_hash,
            "variableBindings": binding_rows,
            "solverStatus": solver.status_name(status).lower(),
        }
        if (evidence is not None and (state.get("currentTrick") or {}).get("index") == 1
                and any(row.get("scoring") is True and row.get("revealed") is not True
                        for row in rows)):
            receipt_core["selectedProjects"] = [project.to_dict() for project in evidence.projects if project.selected]
        receipt = {**receipt_core, "sha256": canonical_sha256(receipt_core)}
        return rows, evidence.winning_team if compared and evidence is not None else None, receipt

    def _projects_with_receipt(
        self,
        state: Mapping[str, Any],
        initial: Mapping[Seat, tuple[Card, ...]],
        *,
        maximum_seconds: float = 15.0,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[list[dict[str, Any]], Team | None, dict[str, Any] | None]:
        if self.native_cp_sat:
            return self._projects_v3(
                state,
                initial,
                maximum_seconds=maximum_seconds,
                cancel_check=cancel_check,
            )
        rows, winner = self._projects_v2(
            state,
            initial,
            maximum_seconds=maximum_seconds,
            cancel_check=cancel_check,
        )
        return rows, winner, None

    def _projects(
        self,
        state: Mapping[str, Any],
        initial: Mapping[Seat, tuple[Card, ...]],
        *,
        maximum_seconds: float = 15.0,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[list[dict[str, Any]], Team | None]:
        rows, winner, _receipt = self._projects_with_receipt(
            state,
            initial,
            maximum_seconds=maximum_seconds,
            cancel_check=cancel_check,
        )
        return rows, winner

    def _recorded_nonplay_reachable(
        self,
        state: Mapping[str, Any],
        initial: Mapping[Seat, tuple[Card, ...]],
        *,
        maximum_seconds: float,
        cancel_check: Callable[[], bool] | None,
    ) -> bool:
        """Require every hidden allocation to reach the recorded auction/escalation."""

        deadline = time.monotonic() + max(0.001, maximum_seconds)
        try:
            buyer = Seat.parse(state.get("buyer"))
            recipient = Seat.parse(state.get("purchaseRecipient") or buyer)
            contract = Contract.parse(state.get("contract"))
            trump = self._trump(state)
            purchase = self._card(state.get("purchaseCard"))
            auction_rows = []
            for raw in state.get("biddingHistory") or ():
                if not isinstance(raw, Mapping):
                    return False
                row = {
                    "roundIndex": raw.get("round", raw.get("roundIndex")),
                    "actor": raw.get("seat", raw.get("actor")),
                    "action": raw.get("action"),
                }
                if raw.get("trumpSuit") is not None:
                    row["trumpSuit"] = raw["trumpSuit"]
                auction_rows.append(row)

            cache_key = (self.rules.sha256, canonical_sha256(state)) if self.native_cp_sat else None
            cached = self._native_nonplay_model
            scenario = cached[1] if cached is not None and cached[0] == cache_key else None
            if scenario is None:
                scenario = OrToolsScenarioModelV1(
                    rules=self.rules,
                    dealer=Seat.parse(state.get("dealer")),
                    required_buyer=buyer,
                )
                if not self.native_cp_sat:
                    for seat in Seat:
                        for card in initial[seat]:
                            scenario.model.add(scenario.owner_of_card[card.id] == int(seat))
                scenario.model.add(scenario.purchase_card == purchase.id)
                scenario.model.add(scenario.purchase_recipient == int(recipient))
                if self.native_cp_sat:
                    fragment = scenario.add_native_auction_fragment(
                        required_actions=auction_rows, required_actions_complete=True,
                        allowed_contracts=(contract,),
                    )
                    scenario.model.add(fragment.trump == (int(trump) if trump is not None else len(Suit)))
                    if state.get("declarer") is not None:
                        scenario.model.add(fragment.declarer == int(Seat.parse(state["declarer"])))
                    self._native_nonplay_model = (cache_key, scenario)
                else:
                    fragment = scenario.add_auction_fragment(
                        required_actions=auction_rows,
                        prefix_length=len(auction_rows),
                        allowed_contracts=(contract,),
                    )
                    matching = [
                        fragment.selected[index]
                        for index, plan in enumerate(fragment.plans)
                        if plan.buyer == buyer
                        and plan.purchase_recipient == recipient
                        and plan.contract == contract
                        and plan.trump == trump
                    ]
                    if not matching:
                        return False
                    scenario.model.add_exactly_one(matching)
            model = scenario.model
            if self.native_cp_sat:
                # Reuse constraints, never solved truth or another world's fixed owners.
                model = model.clone()
                for seat in Seat:
                    for card in initial[seat]:
                        owner = model.get_int_var_from_proto_index(scenario.owner_of_card[card.id].index)
                        model.add(owner == int(seat))
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = 0
            solver.parameters.max_time_in_seconds = max(
                0.001, deadline - time.monotonic()
            )
            status = _solve_cp_sat_with_cancel_v1(
                solver,
                model,
                cancel_check,
            )
            if status == cp_model.INFEASIBLE:
                return False
            if status not in {cp_model.FEASIBLE, cp_model.OPTIMAL}:
                raise InformationSetCompilationStopped(
                    "auction_reachability_incomplete",
                    stage="hidden_world_nonplay_compilation",
                    evidence={"solverStatus": solver.status_name(status).lower()},
                )

            escalation_rows = []
            for raw in state.get("escalationHistory") or ():
                if not isinstance(raw, Mapping):
                    return False
                action = str(raw.get("action") or "").strip().lower()
                escalation_rows.append(
                    {
                        "actor": raw.get("actor", raw.get("seat")),
                        "action": action,
                        "mode": (
                            raw.get("requestedMode", raw.get("mode"))
                            if action in {"double", "four"}
                            else None
                        ),
                    }
                )
            escalation_domain = EscalationDomainIRV1.from_rules(self.rules.data)
            escalation = OrToolsEscalationStageV1(
                escalation_domain,
                declarer=Seat.parse(state.get("declarer") or buyer),
                contract=contract,
                match_score=tuple(int(item) for item in state.get("matchScore", ())),
                native_cp_sat=self.native_cp_sat,
            ).solve(
                required_actions=escalation_rows,
                required_actions_complete=True,
                required_level=str(state.get("escalation") or "normal"),
                required_mode="locked" if state.get("locked") is True else "open",
                required_response=escalation_domain.closure_action,
                maximum_seconds=max(0.001, deadline - time.monotonic()),
                cancel_check=cancel_check,
            )
            if escalation.status == "infeasible":
                return False
            if (
                escalation.status != "satisfied"
                or escalation.solver_status != "optimal"
            ):
                raise InformationSetCompilationStopped(
                    "escalation_reachability_incomplete",
                    stage="hidden_world_nonplay_compilation",
                    evidence={"solverStatus": escalation.solver_status},
                )
            return True
        except InformationSetCompilationStopped:
            raise
        except InterruptedError as exc:
            raise InformationSetCompilationStopped(
                "cancelled",
                stage="hidden_world_nonplay_compilation",
            ) from exc
        except (KeyError, TypeError, ValueError):
            return False

    def _legal_history(
        self,
        state: Mapping[str, Any],
        initial: Mapping[Seat, tuple[Card, ...]],
        *,
        maximum_seconds: float,
        cancel_check: Callable[[], bool] | None,
    ) -> bool:
        history = self._history(state)
        if self.native_cp_sat and not history:
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled", stage="hidden_world_play_reachability")
            return True
        contract = Contract.parse(state.get("contract"))
        trump = self._trump(state)
        ownership = {card: seat for seat in Seat for card in initial[seat]}
        try:
            solved = self.graph_compiler.play_stage.solve(
                cards=tuple(sorted(ownership, key=lambda card: card.id)),
                ownership=ownership,
                contract=contract,
                trump=trump,
                first_leader=Seat.parse(state.get("firstLeader")),
                play_domain=self.rules.play_domain,
                structural_domain=self.rules.structural_domain,
                scoring_domain=ScoringDomainIRV1.from_rules(self.rules.data),
                prefix=history,
                locked=state.get("locked") is True,
                decision_prefix_length=len(history),
                # History reachability needs no choices after its last trick.
                # Keep the frozen legacy model unchanged.
                play_step_count=(
                    max(len(Seat), ((len(history) + len(Seat) - 1) // len(Seat)) * len(Seat))
                    if self.native_cp_sat else None
                ),
                maximum_time_seconds=max(0.001, maximum_seconds),
                cancel_check=cancel_check,
            )
        except PlayTailSolveStopped as exc:
            if self.native_cp_sat and exc.solver_status != "infeasible":
                raise InformationSetCompilationStopped(
                    exc.solver_status, stage="hidden_world_play_reachability",
                ) from exc
            return False
        except ValueError:
            return False
        expected = tuple(
            Seat.parse(item.get("winner"))
            for item in state.get("completedTricks", ())
            if isinstance(item, Mapping)
        )
        return (
            solved.plays[: len(history)] == history
            and solved.trick_winners[: len(expected)] == expected
        )

    def _world_state(
        self,
        state: Mapping[str, Any],
        remaining: Mapping[Seat, tuple[Card, ...]],
        *,
        maximum_seconds: float,
        cancel_check: Callable[[], bool] | None,
        play_history_certified: bool = False,
        project_receipts: list[dict[str, Any]] | None = None,
        rejections: dict[str, int] | None = None,
    ) -> dict[str, Any] | None:
        if type(play_history_certified) is not bool:
            raise TypeError("play_history_certified must be a boolean")
        deadline = time.monotonic() + maximum_seconds

        def reject(reason: str) -> None:
            if rejections is not None:
                rejections[reason] = rejections.get(reason, 0) + 1

        def remaining_seconds() -> float:
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled", stage="hidden_world_reachability")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise InformationSetCompilationStopped("maximum_seconds", stage="hidden_world_reachability")
            return remaining

        initial = self._initial_hands(state, remaining)
        if any(len(initial[seat]) != 8 for seat in Seat):
            return reject("hand_sizes")
        if not self._recorded_nonplay_reachable(
            state, initial, maximum_seconds=remaining_seconds(), cancel_check=cancel_check,
        ):
            return reject("auction_escalation")
        if not play_history_certified:
            if not self._legal_history(
                state, initial, maximum_seconds=remaining_seconds(), cancel_check=cancel_check,
            ):
                return reject("play_history")
        try:
            projects, winner, project_receipt = self._projects_with_receipt(
                state,
                initial,
                maximum_seconds=remaining_seconds(),
                cancel_check=cancel_check,
            )
        except _IncompatibleProjectWorld:
            return reject("projects")
        except InformationSetCompilationStopped:
            raise
        world = copy.deepcopy(dict(state))
        world["hands"] = {
            seat.name.lower(): [card.to_dict() for card in remaining[seat]]
            for seat in Seat
        }
        world["initialHands"] = {
            seat.name.lower(): [card.to_dict() for card in initial[seat]]
            for seat in Seat
        }
        world["projects"] = projects
        world["projectWinningTeam"] = (
            winner.name.lower() if winner is not None else None
        )
        actor = str(state.get("currentPlayer") or "")
        expected = canonical_observer_information_hash_v1(
            complete_state=state, acting_seat=actor
        )
        actual = canonical_observer_information_hash_v1(
            complete_state=world, acting_seat=actor
        )
        if actual != expected:
            if rejections is not None:
                expected_view = build_canonical_observer_information_v1(complete_state=state, acting_seat=actor)["publicState"]
                actual_view = build_canonical_observer_information_v1(complete_state=world, acting_seat=actor)["publicState"]
                for field in expected_view:
                    if expected_view[field] != actual_view[field]:
                        reject(f"observer_projection.{field}")
            return reject("observer_projection")
        if project_receipts is not None and project_receipt is not None:
            retained = {"worldStateHash": canonical_sha256(world), **project_receipt}
            if "selectedProjects" in retained:
                retained["sha256"] = canonical_sha256({key: value for key, value in retained.items() if key != "sha256"})
            project_receipts.append(retained)
        return world

    def _enumerate_worlds(
        self,
        exact: OrToolsNormalizedExactStateV1,
        *,
        maximum_worlds: int,
        maximum_attempts: int,
        deadline: float,
        cancel_check: Callable[[], bool] | None,
        bounded_world_scope: bool = False,
        _batch_allocations: bool = False,
    ) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
        if type(_batch_allocations) is not bool or (_batch_allocations and not bounded_world_scope):
            raise ValueError("allocation batching requires bounded world scope")
        if type(bounded_world_scope) is not bool:
            raise TypeError("bounded_world_scope must be a boolean")
        if bounded_world_scope and not self.native_cp_sat:
            raise ValueError("bounded world analysis requires native CP-SAT")
        if type(maximum_worlds) is not int or maximum_worlds < 1:
            raise ValueError("maximum_worlds must be a positive integer")
        state = exact.state
        worlds: list[dict[str, Any]] = []
        project_receipts: list[dict[str, Any]] = []
        rejections: dict[str, int] = {}
        attempts = 0
        statuses: list[str] = []

        def stopped(reason: str) -> InformationSetCompilationStopped:
            return InformationSetCompilationStopped(
                reason,
                stage="hidden_world_enumeration",
                evidence={
                    "attempts": attempts,
                    "acceptedWorlds": len(worlds),
                    "solverStatuses": list(statuses),
                    "maximumWorlds": maximum_worlds,
                    "maximumAttempts": maximum_attempts,
                    "rejections": dict(rejections),
                    "playCount": len(self._history(state)),
                },
            )

        def check_setup_budget() -> bool:
            if cancel_check is not None and cancel_check():
                raise stopped("cancelled")
            if time.monotonic() >= deadline:
                raise stopped("maximum_seconds")
            return False

        check_setup_budget()
        actor = Seat.parse(state.get("currentPlayer"))
        raw_hands = state.get("hands")
        if not isinstance(raw_hands, Mapping):
            raise InformationSetCompilationStopped("hidden_world_hands_missing")
        actual = {
            seat: tuple(self._card(card) for card in raw_hands[seat.name.lower()])
            for seat in Seat
        }
        hidden_seats = tuple(seat for seat in Seat if seat != actor)
        pool = tuple(
            sorted(
                (card for seat in hidden_seats for card in actual[seat]),
                key=lambda c: c.id,
            )
        )
        fixed: dict[Card, Seat] = {}
        purchase = state.get("purchaseCard")
        recipient = state.get("purchaseRecipient") or state.get("buyer")
        if purchase is not None and recipient is not None:
            card = self._card(purchase)
            if card in pool:
                fixed[card] = Seat.parse(recipient)
        for project in state.get("projects", []):
            if not isinstance(project, Mapping) or project.get("revealed") is not True:
                continue
            owner = Seat.parse(project.get("owner"))
            for raw in project.get("cards") or []:
                card = self._card(raw)
                if card in pool:
                    fixed[card] = owner
        check_setup_budget()
        model = cp_model.CpModel()
        owns = {
            (seat, card): model.new_bool_var(f"owns_{seat.name.lower()}_{card.id}")
            for seat in hidden_seats
            for card in pool
        }
        for card in pool:
            model.add_exactly_one(owns[seat, card] for seat in hidden_seats)
            if card in fixed:
                model.add(owns[fixed[card], card] == 1)
        for seat in hidden_seats:
            model.add(sum(owns[seat, card] for card in pool) == len(actual[seat]))
        history = self._history(state)
        # Undeclared opponents' projects must not change the sampled universe.
        public_project_actions = any(
            row.get("declarationOrder", 999) < 999 or row.get("revealed") is True
            for row in state.get("projects", ())
        ) or any(
            event.get("type") in {"project_declared", "project_passed"}
            for event in state.get("events", ())
        )
        project_claim = (state.get("projectWinningTeam") is not None
            or bool(state.get("balootTeams"))
            or any(row.get("declarationOrder", 999) < 999 or row.get("revealed") is True
                   for row in state.get("projects", ()))
            or any(event.get("type") in {"project_declared", "baloot_declared", "baloot_included_in_project"}
                   for event in state.get("events", ())))
        check_setup_budget()
        # Unconstrained pre-play projects are certified per world below.
        if self.native_cp_sat and (history or public_project_actions):
            known = {card: actor for card in actual[actor]}
            known.update({action.card: action.actor for action in history})
            owners = tuple(model.new_int_var(0, 3, f"project_owner_{card.id}") for card in FULL_DECK)
            for card in FULL_DECK:
                if card in known:
                    model.add(owners[card.id] == int(known[card]))
                else:
                    for seat in hidden_seats:
                        model.add(owners[card.id] == int(seat)).only_enforce_if(owns[seat, card])
            if self.rules.project_domain.declaration_mode != "optional" or project_claim:
                check_setup_budget()
                fragment = attach_project_lifecycle_fragment_v1(
                    model, owner_of_card=owners,
                    contract=Contract.parse(state.get("contract")),
                    dealer=Seat.parse(state.get("dealer")), trump=self._trump(state),
                    domain=self.rules.project_domain,
                    second_trick_leader=(Seat.parse(state["completedTricks"][0]["winner"])
                        if self.rules.project_domain.tie_break_order == "second_trick_reveal"
                        and state.get("completedTricks") else None),
                    fixed_card_owners=tuple({**known, **fixed}.get(card) for card in FULL_DECK),
                    declaration_only=(self.rules.project_domain.tie_break_order == "second_trick_reveal"
                                      and not state.get("completedTricks")),
                    cancel_check=check_setup_budget,
                )
                check_setup_budget()
                self._bind_public_project_choices(model, fragment, state)
            if history:
                check_setup_budget()
                try:
                    self.graph_compiler.play_stage.build_model(
                        cards=FULL_DECK, ownership=None,
                        contract=Contract.parse(state.get("contract")), trump=self._trump(state),
                        first_leader=Seat.parse(state.get("firstLeader")),
                        play_domain=self.rules.play_domain,
                        structural_domain=self.rules.structural_domain,
                        prefix=history, locked=state.get("locked") is True,
                        decision_prefix_length=len(history),
                        play_step_count=((len(history) + len(Seat) - 1) // len(Seat)) * len(Seat),
                        shared_model=model, shared_owner_of_card=owners,
                        _export_predicate_contexts=False, cancel_check=check_setup_budget,
                    )
                except PlayTailSolveStopped as exc:
                    raise InformationSetCompilationStopped(
                        "cancelled" if exc.partial_receipt["evidence"].get("cancelled") is True
                        else exc.solver_status, stage="hidden_world_play_allocation",
                        evidence={"causeReceipt": exc.partial_receipt},
                    ) from exc
        check_setup_budget()
        bounded_termination_reason: str | None = None
        pending_allocations = []

        while True:
            if cancel_check is not None and cancel_check():
                raise stopped("cancelled")
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise stopped("maximum_seconds")
            if not pending_allocations:
                solver = cp_model.CpSolver()
                solver.parameters.num_search_workers = 1
                solver.parameters.random_seed = 0
                solver.parameters.max_time_in_seconds = remaining_seconds
                callback = None
                if _batch_allocations:
                    solver.parameters.enumerate_all_solutions = True
                    batch_limit = min(maximum_worlds - len(worlds), maximum_attempts - attempts)
                    if batch_limit < 1:
                        raise stopped("maximum_nodes")
                    seen_allocations = set()

                    class CollectAllocations(cp_model.CpSolverSolutionCallback):
                        def on_solution_callback(inner):
                            selected_keys = tuple(key for key, variable in owns.items()
                                if inner.boolean_value(variable))
                            if selected_keys not in seen_allocations:
                                seen_allocations.add(selected_keys)
                                pending_allocations.append(selected_keys)
                            if len(pending_allocations) >= batch_limit:
                                inner.stop_search()

                    callback = CollectAllocations()
                status = (_solve_cp_sat_with_cancel_v1(solver, model, cancel_check, callback)
                    if callback is not None else _solve_cp_sat_with_cancel_v1(solver, model, cancel_check))
                statuses.append(solver.status_name(status).lower())
                if cancel_check is not None and cancel_check():
                    raise stopped("cancelled")
                if status == cp_model.INFEASIBLE:
                    break
                if status not in {cp_model.FEASIBLE, cp_model.OPTIMAL}:
                    if bounded_world_scope and worlds:
                        bounded_termination_reason = "hidden_world_solver_incomplete"
                        break
                    raise stopped("hidden_world_solver_incomplete")
                if not _batch_allocations:
                    pending_allocations.append(tuple(key for key, variable in owns.items()
                        if solver.value(variable)))
                if not pending_allocations:
                    raise stopped("hidden_world_solver_incomplete")
            selected_keys = pending_allocations.pop(0)
            attempts += 1
            if attempts > maximum_attempts:
                raise stopped("maximum_nodes")
            allocation = {
                seat: tuple(card for owner, card in selected_keys if owner == seat)
                for seat in hidden_seats
            }
            remaining = {actor: actual[actor], **allocation}
            try:
                world = self._world_state(
                    state,
                    remaining,
                    maximum_seconds=max(0.001, deadline - time.monotonic()),
                    cancel_check=cancel_check,
                    play_history_certified=self.native_cp_sat,
                    project_receipts=project_receipts,
                    rejections=rejections,
                )
            except InformationSetCompilationStopped as exc:
                raise InformationSetCompilationStopped(
                    exc.reason, stage="hidden_world_enumeration",
                    evidence={**stopped(exc.reason).partial_receipt["evidence"],
                              "causeReceipt": exc.partial_receipt,
                              "playCount": len(self._history(state))},
                ) from exc
            if world is not None:
                worlds.append(world)
                if len(worlds) > maximum_worlds:
                    raise stopped("maximum_hidden_worlds")
            selected = [owns[key] for key in selected_keys]
            model.add(sum(selected) <= len(pool) - 1)
            if bounded_world_scope and len(worlds) == maximum_worlds:
                break
        if not worlds:
            raise stopped("no_compatible_hidden_worlds")
        model_document = {
            "schemaVersion": 1,
            "kind": "actor_information_set_allocation_model_v1",
            "exactStateId": exact.exact_state_id,
            "observer": actor.name.lower(),
            "pool": [card.id for card in pool],
            "handSizes": {
                seat.name.lower(): len(actual[seat]) for seat in hidden_seats
            },
            "fixedOwnership": {
                str(card.id): seat.name.lower()
                for card, seat in sorted(fixed.items(), key=lambda item: item[0].id)
            },
            "attempts": attempts,
            "acceptedWorlds": len(worlds),
            "terminalSolverStatus": statuses[-1],
            "reachability": "deal_auction_escalation_project_play_cp_sat_v1",
            "auctionDomainIrHash": AuctionDomainIRV1.from_rules(self.rules.data).sha256,
            "escalationDomainIrHash": EscalationDomainIRV1.from_rules(
                self.rules.data
            ).sha256,
        }
        if bounded_termination_reason is not None:
            model_document["boundedTerminationReason"] = bounded_termination_reason
        if self.native_cp_sat:
            ordered_receipts = sorted(
                project_receipts, key=lambda receipt: receipt["worldStateHash"]
            )
            model_document.update(
                {
                    "projectLifecycleAuthority": "cp_sat",
                    "projectLifecycleBindings": ordered_receipts,
                    "projectLifecycleBindingsHash": canonical_sha256(ordered_receipts),
                }
            )
        if bounded_world_scope:
            model_document["worldScope"] = {
                "method": ("deterministic_cp_sat_batched_prefix_v1" if _batch_allocations
                           else "deterministic_cp_sat_prefix_v1"),
                "complete": statuses[-1] == "infeasible",
                "maximumWorlds": maximum_worlds,
                "stopReason": (
                    "exhausted" if statuses[-1] == "infeasible" else "world_scope_limit"
                ),
                "probabilityModel": None,
            }
        return tuple(sorted(worlds, key=canonical_sha256)), model_document

    def _dynamic_state(
        self,
        base_state: Mapping[str, Any],
        start: Any,
        result: Any,
        path: tuple[TailPrefixActionV1, ...],
    ) -> dict[str, Any]:
        # Internal read-only view: replace changing fields; the canonical
        # observer projection owns its output and does not mutate history.
        state = dict(base_state)
        disclosures = getattr(result, "project_disclosures", None)
        if disclosures is not None:
            state["projects"] = [project.to_dict() for project in disclosures]
        state["hands"] = {
            seat.name.lower(): [
                card.to_dict() for card in result.remaining_hands[int(seat)]
            ]
            for seat in Seat
        }
        state["currentPlayer"] = (
            result.next_actor.name.lower() if result.next_actor is not None else None
        )
        plays = [*start.base_prefix, *path]
        recorded_baloot = {Card.parse(item["card"]["code"]): item.get("baloot", False)
            for item in base_state.get("currentTrick", {}).get("plays", [])}
        base_index = len(state.get("completedTricks", []))
        public_baloot = getattr(result, "public_baloot_events", None)
        if public_baloot is not None:
            teams = set(base_state.get("balootTeams", ()))
            for step, actor, kind in public_baloot:
                teams.add(actor.team.name.lower())
                local_step = step - base_index * 4
                if kind == "declared" and 0 <= local_step < len(plays):
                    recorded_baloot[plays[local_step].card] = True
            state["balootTeams"] = sorted(teams)
        leader = start.first_leader
        current: list[TailPrefixActionV1] = []
        completed: list[dict[str, Any]] = []
        for action in plays:
            current.append(action)
            if (start.trump is not None and action.card.suit == start.trump
                    and current[0].card.suit != start.trump):
                state["trumpBroken"] = True
            if len(current) == 4:
                solved_index = len(completed)
                winner = result.trick_winners[solved_index]
                raw = result.trick_raw_points[solved_index]
                completed.append(
                    {
                        "index": base_index + len(completed),
                        "leader": leader.name.lower(),
                        "plays": [
                            {
                                "seat": item.actor.name.lower(),
                                "card": item.card.to_dict(),
                                "trickIndex": base_index + len(completed),
                                "position": index,
                                "legalCards": [],
                                "ekka": item.ekka,
                                "baloot": recorded_baloot.get(item.card, False),
                            }
                            for index, item in enumerate(current)
                        ],
                        "winner": winner.name.lower(),
                        "rawPoints": raw,
                    }
                )
                leader = winner
                current = []
        state["completedTricks"] = [*state.get("completedTricks", []), *completed]
        state["currentTrick"] = {
            "index": base_index + len(completed),
            "leader": leader.name.lower(),
            "plays": [
                {
                    "seat": item.actor.name.lower(),
                    "card": item.card.to_dict(),
                    "trickIndex": base_index + len(completed),
                    "position": index,
                    "legalCards": [],
                    "ekka": item.ekka,
                    "baloot": recorded_baloot.get(item.card, False),
                }
                for index, item in enumerate(current)
            ],
            "winner": None,
            "rawPoints": 0,
        }
        return state

    def _path_observation_id(self, world_state, start, result, path, recall):
        actor = result.next_actor
        if actor is None:
            raise ValueError("a terminal path has no acting observation")
        return bind_escalation_recall_v1(canonical_observer_information_hash_v1(
            complete_state=self._dynamic_state(world_state, start, result, path),
            acting_seat=actor.name.lower()), recall.get(actor.name.lower(), ()))

    def _project_policy_observer(self, world_state, oracle, project_context, *, cache=None):
        from .ortools_nonplay_observation import _build_native_project_checkpoint_observer
        normalized, prefix, coverage = project_context
        if (coverage.get("status") != "complete"
                or coverage.get("sha256") != canonical_sha256({key: value for key, value in coverage.items() if key != "sha256"})
                or coverage["scope"]["sourceStateHash"] != oracle.start.source_state_hash
                or coverage["scope"]["rulesHash"] != self.rules.sha256
                or coverage["scope"]["prefixHash"] != prefix.sha256
                or canonical_sha256(world_state) != normalized.state_hash
                or not coverage["queries"] or coverage["queries"][-1]["status"] != "infeasible"
                or oracle.project_continuation_actions is None):
            raise ValueError("project policy requires owned complete declaration coverage")
        if [action.to_dict() for action in oracle.project_continuation_actions] not in coverage["sequences"]:
            raise ValueError("project branch is absent from declaration coverage")
        key = (normalized.state_hash, prefix.sha256, coverage["sha256"], oracle.model_hash)
        if cache is not None and key in cache:
            return cache[key]
        observer = _build_native_project_checkpoint_observer(normalized, prefix, path_oracle=oracle)
        if cache is not None:
            cache[key] = observer
        return observer

    def _observed_policy_bindings(self, world_state, oracle, witness, target, policy, recall,
                                  *, known_domains, project_context=None, maximum_seconds=None, cancel_check=None,
                                  _prefix_observations=None, _known_prefix_observations=None,
                                  _project_observer=None):
        """Bind only witnessed prefixes; returned domains are not exhaustive coverage.

        Unknown observations remain unbound. A proof caller must discover them
        or certify absence of adverse paths before claiming policy coverage.
        """
        bindings = {}
        witnessed_actions = set()
        # Copy the campaign's observation catalog: a rejected witness must not
        # partially mutate domains used by another world or master iteration.
        domains = dict(known_domains)
        project_observer = None
        if oracle.project_continuation_actions is not None and project_context is None:
            raise ValueError("project policy requires declaration coverage context")
        deadline = None
        if (maximum_seconds is not None or project_context is not None or oracle.public_baloot_binding is not None
                or getattr(oracle, "project_disclosure_bindings", None)):
            if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds):
                raise ValueError("native policy projection requires a finite time budget")
            deadline = time.monotonic() + maximum_seconds
        if project_context is not None:
            normalized, prefix, coverage = project_context
            branch = [action.to_dict() for action in oracle.project_continuation_actions]
            project_observer = _project_observer or self._project_policy_observer(
                world_state, oracle, project_context)
        for length in range(oracle.base_length, len(witness.plays)):
            if cancel_check is not None and cancel_check():
                raise PlayTailSolveStopped("cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                raise PlayTailSolveStopped("maximum_seconds")
            if witness.plays[length].actor.team != target:
                continue
            path = tuple(witness.plays[oracle.base_length:length])
            cached = (_known_prefix_observations or {}).get(path)
            # The caller owns this cache by model/world/recall branch, including
            # the fixed declaration sequence for project observations.
            if cached is not None and cached[2] is not None:
                project_observation, choice, observation, actions = cached
                if project_observation is not None:
                    if project_context is None:
                        raise ValueError("cached project observation lacks declaration coverage")
                    ordinal = length - len(prefix.project_actions)
                    project_domain = tuple(sorted({ProjectDeclarationActionV1.build(sequence[ordinal]).code
                        for sequence in coverage["sequences"] if sequence[:ordinal] == branch[:ordinal]}))
                    if domains.setdefault(project_observation, project_domain) != project_domain:
                        raise ValueError("observation_action_domain_mismatch")
                    witnessed_actions.add((project_observation, choice))
                    if project_observation in policy:
                        if policy[project_observation] not in project_domain:
                            raise ValueError("observed policy selected an illegal project action")
                        if policy[project_observation] != choice:
                            bindings[path] = None
                            if _prefix_observations is not None:
                                _prefix_observations[path] = cached
                            continue
                if domains.get(observation) != tuple(sorted(actions)):
                    raise ValueError("observation_action_domain_mismatch")
                played = witness.plays[length]
                played_code = played.card.code + ("+ekka" if played.ekka else "")
                if played_code not in actions:
                    raise ValueError("native witness contains an illegal observed action")
                witnessed_actions.add((observation, played_code))
                if observation in policy:
                    if policy[observation] not in actions:
                        raise ValueError("observed policy selected an illegal action")
                    bindings[path] = actions[policy[observation]]
                if _prefix_observations is not None:
                    _prefix_observations[path] = cached
                continue
            projected = witness.at_prefix(length)
            native_projection = None
            if oracle.public_baloot_binding is not None or getattr(oracle, "project_disclosure_bindings", None):
                native_projection = oracle.solve(path, maximum_seconds=deadline - time.monotonic(),
                    cancelled=cancel_check)
                result, _ = native_projection
            else:
                result = oracle._project(projected, length)
            project_observation, choice = None, None
            if project_observer is not None and length > oracle.base_length:
                if cancel_check is not None and cancel_check():
                    raise PlayTailSolveStopped("cancelled")
                if time.monotonic() >= deadline:
                    raise PlayTailSolveStopped("maximum_seconds")
                checkpoint, native_prefix = native_projection or oracle.solve(path, maximum_seconds=deadline - time.monotonic(),
                    cancelled=cancel_check)
                project_hash = project_observer(CompleteStateDecisionObservationV1(checkpoint.next_actor,
                    path, native_prefix, checkpoint.remaining_hands, checkpoint.current_trick,
                    checkpoint.trick_winners, checkpoint.trick_raw_points, (0, 0), (0, 0),
                    checkpoint.project_checkpoint, checkpoint.native_play))
                if length >= 4:
                    observation = bind_escalation_recall_v1(project_hash, recall.get(projected.next_actor.name.lower(), ()))
                else:
                    ordinal = length - len(prefix.project_actions)
                    choice = oracle.project_continuation_actions[ordinal].code
                    project_domain = tuple(sorted({ProjectDeclarationActionV1.build(sequence[ordinal]).code
                        for sequence in coverage["sequences"] if sequence[:ordinal] == branch[:ordinal]}))
                    project_observation = bind_escalation_recall_v1(project_hash,
                        recall.get(projected.next_actor.name.lower(), ()))
                    if domains.setdefault(project_observation, project_domain) != project_domain:
                        raise ValueError("observation_action_domain_mismatch")
                    witnessed_actions.add((project_observation, choice))
                    if project_observation in policy:
                        if policy[project_observation] not in project_domain:
                            raise ValueError("observed policy selected an illegal project action")
                        if policy[project_observation] != choice:
                            bindings[path] = None
                            if _prefix_observations is not None:
                                _prefix_observations[path] = (project_observation, choice, None, {})
                            continue
                    observation = bind_escalation_recall_v1(canonical_sha256({
                        "kind": "native_project_followup_play_v1", "priorObservation": project_hash,
                        "priorAction": choice}), recall.get(projected.next_actor.name.lower(), ()))
            else:
                observation = self._path_observation_id(world_state, oracle.start, result, path, recall)
            actions = {action.card.code + ("+ekka" if action.ekka else ""): action
                       for action in result.legal_play_actions}
            domain = tuple(sorted(actions))
            played = witness.plays[length]
            played_code = played.card.code + ("+ekka" if played.ekka else "")
            if played_code not in actions:
                raise ValueError("native witness contains an illegal observed action")
            witnessed_actions.add((observation, played_code))
            if domains.setdefault(observation, domain) != domain:
                raise ValueError("observation_action_domain_mismatch")
            if _prefix_observations is not None:
                _prefix_observations[path] = (project_observation, choice, observation, actions)
            if observation in policy:
                if policy[observation] not in actions:
                    raise ValueError("observed policy selected an illegal action")
                bindings[path] = actions[policy[observation]]
        return bindings, domains, tuple(sorted(witnessed_actions))

    def _solve_native_bid_policy_bounds(self, exact, prefix, target, *, visibility,
            maximum_iterations, maximum_queries, maximum_observations, maximum_branches,
            maximum_model_bytes, maximum_conflicts, maximum_seconds, cancel_check=None,
            selected_root_action=None, enumerate_root_actions=False, maximum_depth=None, opportunity_order=None,
            opponent_ceiling_policy=None, incumbent_play_policies=None, incumbent_only=False, incumbent_threshold=None,
            required_threshold=None, threshold_refutation=None, incumbent_native_play_policy_id=None,
            incumbent_nonplay_policy=None,
            _owned_incumbent_models=None, progress_callback=None,
            compact_purchase_invariant_ownership=False, strategy_play_fallback_id=None, minimax_budget=None):
        """Own root construction and strategy refinement under one deadline.

        This internal result is not a production certificate. In particular it
        does not certify subsequent decisions or select a root action count.
        """
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("native proof progress callback must be callable")
        if (type(compact_purchase_invariant_ownership) is not bool
                or compact_purchase_invariant_ownership and visibility != "complete_state"):
            raise ValueError("purchase-invariant compaction requires a complete-state world")
        if (not isinstance(target, Team)
                or any(type(value) is not int or value < 1 for value in (maximum_iterations,
                    maximum_queries, maximum_observations, maximum_branches, maximum_model_bytes, maximum_conflicts))
                or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
                or maximum_seconds <= 0):
            raise ValueError("native bid policy requires a team and positive finite budgets")
        deadline = time.monotonic() + maximum_seconds
        preparation, strategy = None, None
        _validate_native_opponent_ceiling_policy(opponent_ceiling_policy, target)
        if (opponent_ceiling_policy is not None and "native_play_policy_id" in opponent_ceiling_policy
                and (incumbent_only or incumbent_threshold is not None or required_threshold is not None)):
            raise ValueError("native opponent play policy requires active ceiling refinement")
        if incumbent_native_play_policy_id is not None:
            if incumbent_play_policies is not None:
                raise ValueError("incumbent policies must have one authority")
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            native_observer_safe_policy_identity_v1(incumbent_native_play_policy_id, target)
        has_incumbent = incumbent_play_policies is not None or incumbent_native_play_policy_id is not None
        if incumbent_nonplay_policy is not None and (incumbent_native_play_policy_id is None
                or not incumbent_only or visibility != "complete_state"):
            raise ValueError("nonplay incumbent requires native complete-state incumbent-only coverage")
        if incumbent_nonplay_policy is not None:
            from .scenario_contracts import validate_native_bid_nonplay_policy_v1
            incumbent_nonplay_policy = validate_native_bid_nonplay_policy_v1(incumbent_nonplay_policy)
        if _owned_incumbent_models is not None and (
                type(_owned_incumbent_models) is not list or _owned_incumbent_models
                or incumbent_native_play_policy_id is None or not incumbent_only
                or visibility != "complete_state" or selected_root_action is None or enumerate_root_actions):
            raise ValueError("owned incumbent models require one native complete-state floor")
        if type(incumbent_only) is not bool or incumbent_only and (
                not has_incumbent or enumerate_root_actions or selected_root_action is None):
            raise ValueError("incumbent-only bid bounds require one action and an explicit policy")
        if incumbent_threshold is not None and (type(incumbent_threshold) is not int or not incumbent_only):
            raise ValueError("incumbent threshold requires an integer and incumbent-only coverage")
        if required_threshold is not None and (type(required_threshold) is not int or incumbent_threshold is not None
                or incumbent_only or has_incumbent or opponent_ceiling_policy is not None
                or enumerate_root_actions or selected_root_action is None):
            raise ValueError("strategy threshold requires a separate integer synthesis request")
        if strategy_play_fallback_id is not None:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            native_observer_safe_policy_identity_v1(strategy_play_fallback_id, target)
            if required_threshold is None or threshold_refutation is not None or not self.native_cp_sat:
                raise ValueError("strategy play fallback requires an unmixed native threshold synthesis")
        if minimax_budget is not None and (not isinstance(minimax_budget, ScenarioProofBudgetV1)
                or strategy_play_fallback_id is None or required_threshold is None or threshold_refutation is not None):
            raise ValueError("bid minimax shortening requires a typed budget and guarded threshold synthesis")
        if threshold_refutation is not None and (required_threshold is None or visibility != "actor_information_set"
                or not self.native_cp_sat or not isinstance(prefix, ExactTracePrefixV1)
                or maximum_depth is not None and (type(maximum_depth) is not int or maximum_depth < 1)
                or not isinstance(threshold_refutation, Mapping) or set(threshold_refutation) != {
                    "receipt", "exact", "prefix", "opponentPolicy", "opponentPlayPolicyId"}):
            raise ValueError("threshold refutation requires a closed owned actor-information-set bound")
        if type(enumerate_root_actions) is not bool or (enumerate_root_actions and selected_root_action is not None):
            raise ValueError("choose one fixed bid action or enumerate all actions")
        if selected_root_action is not None or enumerate_root_actions:
            if not isinstance(exact, OrToolsNormalizedExactStateV1):
                raise ValueError("fixed bid action requires a normalized root")
            exact.require_integrity()
            if (exact.decision_kind != "bid"
                    or self._nonplay_actor(exact).team != target
                    or (not enumerate_root_actions and (type(selected_root_action) is not str
                        or selected_root_action not in {row["code"] for row in exact.legal_actions}))):
                raise ValueError("fixed bid action must be legal and owned by the target team")
        action_values, active_action = [], None
        queries_used = 0

        def evidence(status, cause=None):
            core = {"schemaVersion": 1, "kind": "native_bid_policy_bounds_v1", "status": status,
                "targetTeam": target.name.lower(), "visibility": visibility, "strategicCoverage": False,
                "preparation": preparation, "strategy": strategy,
                "queriesUsed": queries_used,
                "maximumSeconds": maximum_seconds,
                **({"selectedRootAction": selected_root_action} if selected_root_action is not None else {}),
                **({"rootActionValues": list(action_values), "activeRootAction": active_action,
                    "requestedRootActions": sorted(row["code"] for row in exact.legal_actions)}
                   if enumerate_root_actions else {}),
                **({"cause": cause} if cause is not None else {})}
            return {**core, "sha256": canonical_sha256(core)}

        def remaining():
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled", stage="native_bid_policy_bounds")
            if time.monotonic() >= deadline:
                raise InformationSetCompilationStopped("maximum_seconds", stage="native_bid_policy_bounds")
            return deadline - time.monotonic()

        try:
            if threshold_refutation is not None:
                from .result_authority import validate_native_bid_threshold_refutation_v1
                from .ortools_information_set_proof import build_native_threshold_refutation_model_v1
                remaining()
                if exact.rules_hash != self.rules.sha256 or exact.prefix_hash != _effective_prefix_hash(prefix):
                    raise ValueError("threshold refutation source root ownership mismatch")
                bound = threshold_refutation
                upper = validate_native_bid_threshold_refutation_v1(bound["receipt"], observer_exact=exact,
                    required_threshold=required_threshold, rules=self.rules, exact=bound["exact"],
                    source_prefix=bound["prefix"], action=selected_root_action,
                    opponent_policy=bound["opponentPolicy"], opponent_play_policy_id=bound["opponentPlayPolicyId"],
                    maximum_model_bytes=maximum_model_bytes, opportunity_order=opportunity_order,
                    maximum_depth=maximum_depth)
                if len(bound["receipt"]["branches"]) > maximum_branches:
                    raise InformationSetCompilationStopped("maximum_branches", stage="native_bid_threshold_refutation")
                model, claim = build_native_threshold_refutation_model_v1(upper, required_threshold)
                proto = str(model.proto)
                if len(proto.encode("utf-8")) > maximum_model_bytes:
                    raise InformationSetCompilationStopped("maximum_model_bytes", stage="native_bid_threshold_refutation")
                core = {"schemaVersion": 1, "kind": "native_bid_threshold_refutation_v1",
                    "status": "incomplete", "strategicCoverage": False,
                    "sourceStateHash": exact.state_hash, "observerRootHash": bound["receipt"]["observerRootHash"],
                    "selectedRootAction": selected_root_action, "requiredThreshold": required_threshold,
                    "upperBound": upper, "lowerBound": None, "worldUpperProof": copy.deepcopy(bound["receipt"]),
                    "modelProto": proto, "claimLiteralIndex": int(claim.index), "claimValue": None,
                    "solverStatus": "unknown", "queriesUsed": 0,
                    "reusedProofQueries": bound["receipt"]["queriesUsed"]}
                strategy = {**core, "sha256": canonical_sha256(core)}
                solver = cp_model.CpSolver()
                solver.parameters.num_search_workers = 1
                solver.parameters.random_seed = 0
                solver.parameters.max_number_of_conflicts = maximum_conflicts
                solver.parameters.max_time_in_seconds = remaining()
                status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
                queries_used += 1
                core.update(solverStatus=solver.status_name(status).lower(), queriesUsed=1)
                strategy = {**core, "sha256": canonical_sha256(core)}
                remaining()
                if status != cp_model.OPTIMAL:
                    raise InformationSetCompilationStopped("threshold_refutation_not_certified", stage="native_bid_threshold_refutation")
                if not solver.boolean_value(claim):
                    raise ValueError("native threshold refutation contradicts its certified bounds")
                core.update(status="scoped_strategy_threshold_refuted", claimValue=True)
                strategy = {**core, "sha256": canonical_sha256(core)}
                return evidence(strategy["status"])
            models, preparation = self._prepare_native_bid_branches(exact, prefix, visibility=visibility,
                compact_known_initial_ownership=visibility == "complete_state" and not compact_purchase_invariant_ownership,
                **({"compact_purchase_invariant_ownership": True} if compact_purchase_invariant_ownership else {}),
                maximum_branches=maximum_branches, maximum_model_bytes=maximum_model_bytes,
                maximum_queries=maximum_queries, maximum_depth=maximum_depth,
                **({"opportunity_order": opportunity_order} if opportunity_order is not None else {}),
                maximum_conflicts=maximum_conflicts, maximum_seconds=remaining(), cancel_check=cancel_check,
                **({"progress_callback": progress_callback} if progress_callback is not None else {}),
                **({"selected_root_action": selected_root_action} if selected_root_action is not None else {}))
            queries_used = preparation.get("queriesUsed", 0)
            if _owned_incumbent_models is not None:
                _owned_incumbent_models.extend(models)
            if incumbent_threshold is not None:
                utility_maximum = max(max(map(int, point.proto.domain)) for model in models
                    for point in model._native_round_query_context()[1])
                if not -utility_maximum <= incumbent_threshold <= utility_maximum:
                    raise InformationSetCompilationStopped("selected_threshold_out_of_domain", stage="native_bid_policy_bounds")
            observation = None
            if selected_root_action is not None or enumerate_root_actions:
                actor = self._nonplay_actor(exact)
                observation = models[0]._auction_observation_hash(actor,
                    [self._card(card) for card in exact.state["hands"][actor.name.lower()]],
                    self._card(exact.state["purchaseCard"]),
                    [row["action"] + (":" + row["trumpSuit"] if row.get("trumpSuit") is not None else "")
                     for row in prefix.auction_actions])
            actions = tuple(sorted(row["code"] for row in exact.legal_actions))
            iterations_used = observations_used = 0
            for active_action in actions if enumerate_root_actions else (selected_root_action,):
                remaining()
                for used, limit, reason in ((queries_used, maximum_queries, "maximum_queries"),
                        (iterations_used, maximum_iterations, "maximum_iterations"),
                        (observations_used, maximum_observations, "maximum_observations")):
                    if used >= limit:
                        raise InformationSetCompilationStopped(reason, stage="native_bid_action_values")
                action_models = models
                if active_action is not None and selected_root_action is None:
                    action_models = []
                    for model in models:
                        remaining()
                        conditioned = copy.copy(model)
                        conditioned.model = model.model.clone()
                        conditioned.constrain_native_auction_root_choice(active_action, actor=actor)
                        action_models.append(conditioned)
                strategy = self._solve_symbolic_round_policy_bounds(action_models, target,
                    maximum_iterations=maximum_iterations - iterations_used, maximum_queries=maximum_queries - queries_used,
                    maximum_observations=maximum_observations - observations_used, maximum_branches=maximum_branches,
                    maximum_conflicts=maximum_conflicts, maximum_seconds=remaining(), cancel_check=cancel_check,
                    progress_callback=(lambda row: progress_callback({**row,
                        "queryCount": queries_used + row["queryCount"]})) if progress_callback is not None else None,
                    fixed_root=(observation, actions, active_action) if active_action is not None else None,
                    # The cheap single-cell ceiling aid must not consume the
                    # entire multi-contract strategy budget before refinement.
                    tighten_utility_ceiling=len(action_models) == 1
                        and incumbent_threshold is None and required_threshold is None,
                    shorten_counterexamples=(minimax_budget is not None or strategy_play_fallback_id is None and (
                        visibility == "actor_information_set" and required_threshold is not None
                        or visibility == "complete_state" and not has_incumbent
                        and (len(action_models) > 1 or required_threshold is not None))),
                    **({"_play_fallback_id": strategy_play_fallback_id}
                       if strategy_play_fallback_id is not None else {}),
                    **({"minimax_budget": minimax_budget} if minimax_budget is not None else {}),
                    **({"required_threshold": required_threshold} if required_threshold is not None else {}),
                    **({"incumbent_play_policies": incumbent_play_policies}
                       if incumbent_play_policies is not None else {}),
                    **({"incumbent_native_play_policy_id": incumbent_native_play_policy_id}
                       if incumbent_native_play_policy_id is not None else {}),
                    **({"incumbent_nonplay_policy": incumbent_nonplay_policy}
                       if incumbent_nonplay_policy is not None else {}),
                    **({"incumbent_only": True} if incumbent_only else {}),
                    **({"incumbent_threshold": incumbent_threshold} if incumbent_threshold is not None else {}),
                    **({("opponent_ceiling_policy" if len(action_models) == 1 else
                        "_unavailable_opponent_ceiling_policy"): opponent_ceiling_policy}
                       if opponent_ceiling_policy is not None else {}))
                queries_used += strategy.get("queriesUsed", len(strategy["queries"]))
                if enumerate_root_actions:
                    action_values.append({"action": active_action, "strategy": strategy})
                    iterations_used += len(strategy["masters"])
                    observations_used += len(strategy["domains"])
                remaining()
            active_action = None
        except InformationSetCompilationStopped as exc:
            queries_used += exc.partial_receipt.get("evidence", {}).get("queriesUsed", 0)
            raise InformationSetCompilationStopped(exc.reason, stage="native_bid_policy_bounds",
                evidence=evidence("incomplete", exc.partial_receipt)) from exc
        return evidence("scoped_action_values_complete" if enumerate_root_actions else strategy["status"])

    def _prove_native_bid_selected_action(self, exact, prefix, target, *, selected_root_action,
            opponent_strategy, maximum_iterations, maximum_queries, maximum_observations,
            maximum_branches, maximum_model_bytes, maximum_conflicts, maximum_seconds,
            cancel_check=None, maximum_depth=None, opportunity_order=None,
            incumbent_play_policies=None, incumbent_maximum_seconds=None, incumbent_maximum_queries=None,
            visibility="complete_state", alternative_first=False, opponent_play_policy_id=None,
            incumbent_native_play_policy_id=None, incumbent_nonplay_policy=None,
            _owned_incumbent_models=None, progress_callback=None, _alternative_basis=None,
            _prior_alternatives=()):
        """Certify scoped bid dominance without exhaustive tie counts.

        This composes native current-deal bounds, not whole-round quality admission.
        A cooperative witness against the supplied opponent strategy means only
        that this certificate is incomplete, never that the selected action loses.
        Explicit incumbent inputs reserve a sub-budget for a policy floor instead
        of exact value convergence. Successful interval evidence leaves scores
        unknown; only selected-action optimality and zero regret are certified.
        """
        if (not self.native_cp_sat or not isinstance(exact, OrToolsNormalizedExactStateV1)
                or not isinstance(prefix, ExactTracePrefixV1) or not isinstance(target, Team)):
            raise ValueError("bid dominance requires owned native input and a target team")
        exact.require_integrity()
        if visibility not in {"complete_state", "actor_information_set"}:
            raise ValueError("bid dominance requires explicit supported visibility")
        if (exact.decision_kind != "bid" or exact.rules_hash != self.rules.sha256
                or exact.prefix_hash != _effective_prefix_hash(prefix)
                or self._nonplay_actor(exact).team != target):
            raise ValueError("bid dominance source or actor ownership mismatch")
        legal_actions = sorted(row["code"] for row in exact.legal_actions)
        observed = visibility == "actor_information_set"
        root_hash = (canonical_observer_information_hash_v1(complete_state=exact.state,
            acting_seat=self._nonplay_actor(exact).name.lower()) if observed else exact.state_hash)
        if (not isinstance(opponent_strategy, Mapping)
                or set(opponent_strategy) - {"auction_action", "escalation_action", "project_actions"}
                or not {"auction_action", "escalation_action"} <= opponent_strategy.keys()):
            raise ValueError("bid dominance requires a closed constant opponent strategy")
        if type(selected_root_action) is not str or selected_root_action not in legal_actions:
            raise ValueError("bid dominance requires one selected action")
        if any(opponent_strategy[key] is not None and type(opponent_strategy[key]) is not str
               for key in ("auction_action", "escalation_action")):
            raise ValueError("bid dominance requires typed strategy actions")
        project_actions = opponent_strategy.get("project_actions")
        if project_actions is not None and (not isinstance(project_actions, Mapping)
                or any(type(seat) is not str or type(action) is not str for seat, action in project_actions.items())):
            raise ValueError("bid dominance requires typed project actions")
        if not any(opponent_strategy[key] is not None for key in ("auction_action", "escalation_action")) and not project_actions:
            raise ValueError("bid dominance requires a nonempty opponent strategy")
        limits = {"maximumIterations": maximum_iterations, "maximumQueries": maximum_queries,
            "maximumObservations": maximum_observations, "maximumBranchesPerPreparation": maximum_branches,
            "maximumModelBytesPerPreparation": maximum_model_bytes, "maximumConflictsPerSolve": maximum_conflicts}
        if maximum_depth is not None:
            limits["maximumDepth"] = maximum_depth
        if any(type(value) is not int or value < 1 for value in limits.values()):
            raise ValueError("bid dominance limits must be positive integers")
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("bid dominance requires a finite positive deadline")
        native_incumbent = incumbent_native_play_policy_id is not None
        if incumbent_nonplay_policy is not None:
            from .scenario_contracts import validate_native_bid_nonplay_policy_v1
            incumbent_nonplay_policy = validate_native_bid_nonplay_policy_v1(incumbent_nonplay_policy)
            if not native_incumbent or visibility != "complete_state":
                raise ValueError("nonplay incumbent requires native complete-state coverage")
        if native_incumbent:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            native_observer_safe_policy_identity_v1(incumbent_native_play_policy_id, target)
            if incumbent_play_policies is not None or visibility != "complete_state":
                raise ValueError("native incumbent requires one complete-state policy authority")
        if _owned_incumbent_models is not None and (
                not native_incumbent or type(_owned_incumbent_models) is not list or _owned_incumbent_models):
            raise ValueError("owned incumbent models require a fresh native owner collection")
        owned_models = []
        interval = incumbent_play_policies is not None or native_incumbent
        resume = _alternative_basis is not None
        if resume and (interval or alternative_first or visibility != "complete_state"):
            raise ValueError("bid threshold resume requires complete-state synthesis without an incumbent")
        if opponent_play_policy_id is not None:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            native_observer_safe_policy_identity_v1(opponent_play_policy_id, target.other)
            if interval and (visibility != "complete_state" or project_actions):
                raise ValueError("paired native bid bounds require complete-state constant nonplay alternatives")
        if type(alternative_first) is not bool or alternative_first and (not interval or "project_actions" in opponent_strategy):
            raise ValueError("alternative-first dominance requires an incumbent and constant nonplay opponent policy")
        if ((not interval and (incumbent_maximum_seconds is not None or incumbent_maximum_queries is not None)) or interval and (
                type(incumbent_maximum_seconds) not in {int, float}
                or not math.isfinite(incumbent_maximum_seconds)
                or incumbent_maximum_seconds <= 0
                or type(incumbent_maximum_queries) is not int
                or not 0 < incumbent_maximum_queries < maximum_queries)):
            raise ValueError("incumbent bid bounds require positive time and queries below the total allowance")
        if interval:
            limits["incumbentMaximumSeconds"] = incumbent_maximum_seconds
            limits["incumbentMaximumQueries"] = incumbent_maximum_queries
        if opportunity_order is not None:
            if (not isinstance(opportunity_order, Sequence) or isinstance(opportunity_order, (str, bytes))
                    or any(type(seat) not in {Seat, str, int} for seat in opportunity_order)):
                raise ValueError("opportunity order must contain every seat once")
            opportunity_order = tuple(Seat.parse(seat).name.lower() for seat in opportunity_order)
            if len(opportunity_order) != len(Seat) or len(set(opportunity_order)) != len(Seat):
                raise ValueError("opportunity order must contain every seat once")
        deadline = time.monotonic() + maximum_seconds
        selected, active, preparation, qualification, action_quality = None, None, None, None, None
        alternatives = []
        queries_used = 0
        if resume:
            from .result_authority import validate_native_bid_dominance_resume_basis_v1
            validate_native_bid_dominance_resume_basis_v1(_alternative_basis, rules=self.rules,
                source_prefix=prefix, normalized=exact, selected_action=selected_root_action,
                opponent_policy=dict(opponent_strategy), opponent_play_policy_id=opponent_play_policy_id,
                limits=limits)
            if (_alternative_basis.get("escalationOpportunityOrder") != (
                    list(opportunity_order) if opportunity_order is not None else None)
                    or maximum_seconds > _alternative_basis["limits"]["maximumSeconds"]):
                raise ValueError("bid threshold resume order or deadline mismatch")
            alternatives = copy.deepcopy(_alternative_basis["alternatives"])

        def evidence(status, cause=None):
            core = {"schemaVersion": 1, "kind": ("native_bid_selected_action_dominance_v4" if resume else "native_bid_selected_action_dominance_v3" if alternative_first
                    else "native_bid_selected_action_dominance_v2" if interval else "native_bid_selected_action_dominance_v1"),
                "status": status, "visibility": visibility, "strategicCoverage": False,
                **({"observerRootHash": root_hash} if observed else {}),
                "rulesHash": self.rules.sha256, "sourcePrefixHash": prefix.sha256,
                "exactStateId": exact.exact_state_id, "stateHash": exact.state_hash,
                "targetTeam": target.name.lower(), "legalActions": legal_actions,
                "utility": "awarded_point_margin", "completionScope": "selected_bid_current_deal",
                "opponentStrategy": dict(opponent_strategy),
                **({"incumbentNativePlayPolicyId": incumbent_native_play_policy_id} if native_incumbent else
                   {"incumbentPlayPolicies": copy.deepcopy(incumbent_play_policies)} if interval else {}),
                **({"incumbentNonplayPolicy": incumbent_nonplay_policy} if incumbent_nonplay_policy is not None else {}),
                **({"opponentPlayPolicyId": opponent_play_policy_id}
                   if (interval or resume) and opponent_play_policy_id is not None else {}),
                **({"incumbentAttempt": copy.deepcopy(_alternative_basis)} if resume else {}),
                **({"escalationOpportunityOrder": list(opportunity_order)} if opportunity_order is not None else {}),
                "selectedAction": selected_root_action, "selectedValueProof": selected,
                "alternatives": alternatives, "activeAction": active, "activePreparation": preparation,
                "qualificationModel": qualification,
                "actionQuality": action_quality,
                "optimalActionCount": None, "optimalActionCountLowerBound": 1 if status == "selected_action_optimal" else 0,
                "optimalActionCountUpperBound": len(exact.legal_actions),
                "actualActionOptimal": True if status == "selected_action_optimal" else None,
                "maximumSeconds": maximum_seconds, "queriesUsed": queries_used,
                "limits": {**limits, "maximumSeconds": maximum_seconds, "workersPerSolve": 1,
                    **({"sharedDecisionMaximumSeconds": _alternative_basis["limits"]["maximumSeconds"]} if resume else {})},
                **({"cause": cause} if cause is not None else {})}
            return {**core, "sha256": canonical_sha256(core)}

        def remaining():
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled", stage="native_bid_dominance")
            if time.monotonic() >= deadline:
                raise InformationSetCompilationStopped("maximum_seconds", stage="native_bid_dominance")
            return deadline - time.monotonic()

        def complete_partition(receipt, action, nonplay_policy=None):
            branches = receipt.get("branches")
            specs = self._native_bid_branch_specs(Seat.parse(prefix.dealer))
            partition = [None if row is None else {"contract": row[0].value,
                "trump": row[1].name.lower() if row[1] is not None else None,
                "firstLeader": row[2].name.lower(), "locked": row[3]} for row in specs]
            scope = receipt.get("scope", {})
            count = len(specs)
            return (receipt.get("constructionComplete") is True
                and scope.get("rulesHash") == self.rules.sha256
                and scope.get("rootHash") == root_hash
                and scope.get("visibility") == visibility
                and scope.get("selectedRootAction") == action
                and scope.get("matchScore") == list(prefix.match_score)
                and scope.get("continuationBoundary") == "current_deal_completion"
                and scope.get("nextDealAnalyzed") is False
                and scope.get("partition") == partition
                and scope.get("nonplayPolicy") == nonplay_policy
                and scope.get("escalationOpportunityOrder") == (None if opportunity_order is None else list(opportunity_order))
                and isinstance(branches, list) and len(branches) == count
                and all(isinstance(row, Mapping) and type(row.get("partitionIndex")) is int
                    for row in branches)
                and [row["partitionIndex"] for row in branches] == list(range(count))
                and (not observed or all(not row.get("retained") or (
                    isinstance(row.get("informationBindings"), Mapping)
                    and type(row.get("cpModelProtoHash")) is str
                    and row["informationBindings"].get("kind") == "native_bid_information_bindings_v1"
                    and row["informationBindings"].get("strategicCoverage") is False
                    and row["informationBindings"].get("cpModelProtoHash") == row.get("cpModelProtoHash")
                    and row["informationBindings"].get("sha256") == canonical_sha256({
                        key: item for key, item in row["informationBindings"].items() if key != "sha256"})
                ) for row in branches))
                and all(type(row.get("retained")) is bool and (
                    row["retained"] or isinstance(row.get("reachability"), Mapping)
                    and row["reachability"].get("solverStatus") == "infeasible"
                    and all(binding.get("status") == "bound" for binding in
                        row["reachability"].get("policyBindings", ()))) for row in branches))

        def threshold_refutation():
            """Refute this sufficient threshold, never the selected bid itself."""
            nonlocal queries_used
            from .result_authority import validate_native_bid_world_upper_v1
            required = max(row["upperBound"] for row in alternatives)
            for prior in _prior_alternatives:
                if prior.get("scope", {}).get("selectedRootAction") != selected_root_action:
                    continue
                declared = prior.get("limits", {})
                caps = {"branches": maximum_branches, "modelBytes": maximum_model_bytes,
                    "conflictsPerSolve": maximum_conflicts, "queries": maximum_queries}
                if maximum_depth is not None:
                    caps["maximumDepth"] = maximum_depth
                if (any(type(declared.get(key)) is not int or not 0 < declared[key] <= cap
                        for key, cap in caps.items())
                        or type(declared.get("seconds")) not in {int, float}
                        or not 0 < declared["seconds"] <= maximum_seconds
                        or type(prior.get("queriesUsed")) is not int
                        or not 0 <= prior["queriesUsed"] <= maximum_queries - queries_used):
                    continue
                upper = validate_native_bid_world_upper_v1(prior, rules=self.rules, exact=exact,
                    source_prefix=prefix, action=selected_root_action, opponent_policy=opponent_strategy,
                    opponent_play_policy_id=opponent_play_policy_id,
                    maximum_model_bytes=maximum_model_bytes, opportunity_order=opportunity_order,
                    maximum_depth=maximum_depth)
                remaining()
                if upper < required:
                    queries_used += prior["queriesUsed"]
                    return evidence("incomplete", {"reason": "selected_threshold_exceeds_certified_upper",
                        "requiredThreshold": required, "selectedUpperBound": upper,
                        "selectedUpperPreparation": copy.deepcopy(prior)})
            return None

        try:
            threshold = max(row["upperBound"] for row in alternatives) if resume else None
            if alternative_first:
                from .result_authority import validate_native_bid_world_upper_v1, validate_native_incumbent_threshold_floor_v1
                for active in sorted(code for code in legal_actions if code != selected_root_action):
                    available_queries = maximum_queries - queries_used - incumbent_maximum_queries - 1
                    available_seconds = remaining() - incumbent_maximum_seconds
                    if available_queries < 1 or available_seconds <= 0:
                        raise InformationSetCompilationStopped("alternative_budget_exhausted", stage="native_bid_dominance")
                    reused = None
                    for prior in _prior_alternatives:
                        if prior.get("scope", {}).get("selectedRootAction") != active:
                            continue
                        declared = prior.get("limits", {})
                        caps = {"branches": maximum_branches, "modelBytes": maximum_model_bytes,
                            "conflictsPerSolve": maximum_conflicts, "queries": available_queries}
                        if maximum_depth is not None:
                            caps["maximumDepth"] = maximum_depth
                        if (any(type(declared.get(key)) is not int or not 0 < declared[key] <= cap
                                for key, cap in caps.items())
                                or type(declared.get("seconds")) not in {int, float}
                                or not 0 < declared["seconds"] <= maximum_seconds):
                            continue
                        upper = validate_native_bid_world_upper_v1(prior, rules=self.rules, exact=exact,
                            source_prefix=prefix, action=active, opponent_policy=opponent_strategy,
                            opponent_play_policy_id=opponent_play_policy_id,
                            maximum_model_bytes=maximum_model_bytes, opportunity_order=opportunity_order,
                            maximum_depth=maximum_depth)
                        remaining()
                        # Retain the original premise and conservative logical charge.
                        # Final admission/readback still reconstructs its solver proof.
                        queries_used += prior["queriesUsed"]
                        retained = [row for row in prior["branches"] if row["retained"]]
                        reused = {"action": active, "preparation": copy.deepcopy(prior), "upperBound": upper,
                            "queries": [row["alternativeQuery"] for row in retained],
                            "sourceModelHashes": [row["sourceModelHash"] for row in retained]}
                        alternatives.append(reused)
                        break
                    if reused is not None:
                        refuted = threshold_refutation()
                        if refuted is not None:
                            return refuted
                        continue
                    models, preparation = self._prepare_native_bid_branches(exact, prefix,
                        visibility="complete_state", selected_root_action=active,
                        policy_team=target.other, nonplay_policy=dict(opponent_strategy),
                        maximum_queries=available_queries, maximum_depth=maximum_depth,
                        maximum_branches=maximum_branches, maximum_model_bytes=maximum_model_bytes,
                        maximum_conflicts=maximum_conflicts, maximum_seconds=available_seconds,
                        cancel_check=cancel_check, optimize_alternative_upper=True,
                        progress_callback=(lambda row: progress_callback({**row,
                            "queryCount": queries_used + row["queryCount"]})) if progress_callback is not None else None,
                        **({"opponent_play_policy_id": opponent_play_policy_id,
                            "compact_purchase_invariant_ownership": True}
                           if opponent_play_policy_id is not None else {}),
                        **({"opportunity_order": opportunity_order} if opportunity_order is not None else {}))
                    if models:
                        raise ValueError("alternative world bounds must release streamed models")
                    upper = validate_native_bid_world_upper_v1(preparation, rules=self.rules, exact=exact,
                        source_prefix=prefix, action=active, opponent_policy=opponent_strategy,
                        opponent_play_policy_id=opponent_play_policy_id,
                        maximum_model_bytes=maximum_model_bytes, opportunity_order=opportunity_order,
                        maximum_depth=maximum_depth)
                    queries_used += preparation["queriesUsed"]
                    retained = [row for row in preparation["branches"] if row["retained"]]
                    alternatives.append({"action": active, "preparation": preparation, "upperBound": upper,
                        "queries": [row["alternativeQuery"] for row in retained],
                        "sourceModelHashes": [row["sourceModelHash"] for row in retained]})
                    remaining()
                    refuted = threshold_refutation()
                    if refuted is not None:
                        return refuted
                if not alternatives:
                    return evidence("incomplete", "alternative_threshold_unavailable")
                threshold = max(row["upperBound"] for row in alternatives)
                active, preparation = selected_root_action, None
            selected = self._solve_native_bid_policy_bounds(exact, prefix, target,
                visibility=visibility, selected_root_action=selected_root_action,
                **({"compact_purchase_invariant_ownership": True} if visibility == "complete_state" else {}),
                maximum_iterations=maximum_iterations,
                maximum_queries=incumbent_maximum_queries if interval else maximum_queries,
                maximum_observations=maximum_observations, maximum_branches=maximum_branches,
                maximum_model_bytes=maximum_model_bytes, maximum_conflicts=maximum_conflicts,
                maximum_seconds=min(remaining(), incumbent_maximum_seconds) if interval else remaining(),
                cancel_check=cancel_check, maximum_depth=maximum_depth,
                progress_callback=(lambda row: progress_callback({**row,
                    "queryCount": queries_used + row["queryCount"]})) if progress_callback is not None else None,
                **({"incumbent_only": True, "incumbent_native_play_policy_id": incumbent_native_play_policy_id,
                    **({"incumbent_nonplay_policy": incumbent_nonplay_policy}
                       if incumbent_nonplay_policy is not None else {}),
                    "_owned_incumbent_models": owned_models} if native_incumbent else
                   {"incumbent_play_policies": incumbent_play_policies, "incumbent_only": True} if interval else
                   {"required_threshold": threshold} if resume else
                   {"opponent_ceiling_policy": {key: opponent_strategy[key]
                    for key in ("auction_action", "escalation_action")}
                    | ({"native_play_policy_id": opponent_play_policy_id}
                       if opponent_play_policy_id is not None else {})}),
                **({"incumbent_threshold": threshold} if alternative_first else {}),
                **({"opportunity_order": opportunity_order} if opportunity_order is not None else {}))
            selected_usage = selected.get("queriesUsed")
            preparation_usage = selected.get("preparation", {}).get("queriesUsed")
            strategy_usage = (selected.get("strategy") or {}).get("queriesUsed")
            if (any(type(count) is not int or count < 0 for count in
                    (selected_usage, preparation_usage, strategy_usage))
                    or selected_usage != preparation_usage + strategy_usage):
                return evidence("incomplete", "selected_resource_accounting_mismatch")
            queries_used += selected_usage
            if _owned_incumbent_models is not None:
                _owned_incumbent_models.extend(owned_models)
            if selected_usage > (incumbent_maximum_queries if interval else maximum_queries) or queries_used > maximum_queries:
                return evidence("incomplete", "selected_query_budget_exceeded")
            selected_status = ("scoped_strategy_threshold_complete" if resume else "scoped_incumbent_threshold_complete" if alternative_first else
                "scoped_incumbent_bound_complete" if interval else "scoped_bound_converged")
            if selected["status"] != selected_status:
                return evidence("incomplete", "selected_value_not_complete")
            if (not complete_partition(selected.get("preparation", {}), selected_root_action)
                    or selected["strategy"].get("status") != selected_status
                    or type(selected["strategy"].get("lowerBound")) is not int
                    or type(selected["strategy"].get("upperBound")) is not int
                    or (selected["strategy"]["lowerBound"] > selected["strategy"]["upperBound"] if (interval or resume) else
                        selected["strategy"]["lowerBound"] != selected["strategy"]["upperBound"])):
                return evidence("incomplete", "selected_bounds_not_certified")
            value = selected["strategy"]["lowerBound"]
            if observed:
                root_model = self._native_bid_root_model(exact, prefix, visibility=visibility)
                actor = self._nonplay_actor(exact)
                observation = root_model._auction_observation_hash(actor,
                    [self._card(card) for card in exact.state["hands"][actor.name.lower()]],
                    self._card(exact.state["purchaseCard"]),
                    [row["action"] + (":" + row["trumpSuit"] if row.get("trumpSuit") is not None else "")
                     for row in prefix.auction_actions])
                if selected["strategy"].get("scope", {}).get("fixedRoot") != {
                    "observationHash": observation, "legalActions": legal_actions,
                    "selectedAction": selected_root_action,
                }:
                    return evidence("incomplete", "selected_observation_binding_mismatch")
                remaining()
            selected_models = [row.get("sourceModelHash") for row in selected["preparation"]["branches"] if row["retained"]]
            strategy_models = [row.get("modelHash") for row in selected["strategy"].get("scope", {}).get("branches", [])]
            if (not selected_models or any(type(item) is not str for item in selected_models)
                    or len(set(selected_models)) != len(selected_models)
                    or any(type(item) is not str for item in strategy_models)
                    or sorted(selected_models) != sorted(strategy_models)):
                return evidence("incomplete", "selected_model_coverage_mismatch")
            if alternative_first:
                certified = validate_native_incumbent_threshold_floor_v1(selected["strategy"],
                    target_team=target.name.lower(), play_policies=incumbent_play_policies,
                    **({"native_play_policy_id": incumbent_native_play_policy_id,
                        **({"nonplay_policy": incumbent_nonplay_policy} if incumbent_nonplay_policy is not None else {}),
                        "source_models": tuple(owned_models)} if native_incumbent else {}),
                    rules_hash=self.rules.sha256, source_model_hashes=tuple(strategy_models))
                if certified != threshold:
                    return evidence("incomplete", "selected_threshold_mismatch")
            if resume:
                from .result_authority import validate_native_strategy_threshold_v1
                scope = selected["strategy"]["scope"]
                certified = validate_native_strategy_threshold_v1(selected["strategy"], expected_scope={
                    **scope, "rulesHash": self.rules.sha256, "targetTeam": target.name.lower(),
                    "requiredThreshold": threshold, "objective": "strategy_threshold_satisfaction"})
                if certified != threshold:
                    return evidence("incomplete", "selected_threshold_mismatch")
            for active in (() if (alternative_first or resume) else sorted(row["code"] for row in exact.legal_actions if row["code"] != selected_root_action)):
                if queries_used >= maximum_queries:
                    raise InformationSetCompilationStopped("maximum_queries", stage="native_bid_dominance")
                # An exact selected value is also a certified alternative
                # threshold. Solve and release each branch instead of retaining
                # every round model before starting the upper-bound queries.
                streamed = not project_actions and not observed
                models, preparation = self._prepare_native_bid_branches(exact, prefix,
                    visibility=visibility, selected_root_action=active,
                    maximum_queries=maximum_queries - queries_used,
                    maximum_depth=maximum_depth,
                    **({"opportunity_order": opportunity_order} if opportunity_order is not None else {}),
                    policy_team=target.other, nonplay_policy={key: opponent_strategy[key]
                        for key in ("auction_action", "escalation_action")},
                    maximum_branches=maximum_branches, maximum_model_bytes=maximum_model_bytes,
                    maximum_conflicts=maximum_conflicts, maximum_seconds=remaining(), cancel_check=cancel_check,
                    progress_callback=(lambda row: progress_callback({**row,
                        "queryCount": queries_used + row["queryCount"]})) if progress_callback is not None else None,
                    **({"opponent_play_policy_id": opponent_play_policy_id,
                        "compact_purchase_invariant_ownership": True}
                       if streamed and interval and opponent_play_policy_id is not None else {}),
                    **({"alternative_upper_bound": value} if streamed else {}))
                queries_used += preparation.get("queriesUsed", 0)
                row = {"action": active, "preparation": preparation, "queries": [], "upperBound": None,
                    "sourceModelHashes": [model._model_hash(search_branching="automatic") for model in models]}
                alternatives.append(row)
                if not complete_partition(preparation, active, {"team": target.other.name.lower(),
                        **{key: opponent_strategy[key] for key in ("auction_action", "escalation_action")}}):
                    return evidence("incomplete", "alternative_partition_not_complete")
                if streamed:
                    if models or preparation.get("kind") != "native_bid_streamed_alternative_preparation_v1":
                        return evidence("incomplete", "alternative_stream_ownership_mismatch")
                    retained = [branch for branch in preparation["branches"] if branch["retained"]]
                    row["queries"] = [branch.get("alternativeQuery") for branch in retained]
                    row["sourceModelHashes"] = [branch.get("sourceModelHash") for branch in retained]
                    if not retained or any(not isinstance(query, Mapping) for query in row["queries"]):
                        return evidence("incomplete", "alternative_model_coverage_mismatch")
                    if any(query.get("sourceModelHash") != owner
                            for query, owner in zip(row["queries"], row["sourceModelHashes"])):
                        raise ValueError("bid dominance alternative model ownership mismatch")
                    if any(query.get("status") != "queried"
                            or query.get("query", {}).get("status") != "no_matching_outcome"
                            or query.get("query", {}).get("solverStatus") != "infeasible"
                            for query in row["queries"]):
                        return evidence("incomplete", "alternative_upper_bound_not_certified")
                    row["upperBound"] = value
                    continue
                if not models:
                    return evidence("incomplete", "alternative_reachability_disagreement")
                if sum(branch["retained"] for branch in preparation["branches"]) != len(models):
                    return evidence("incomplete", "alternative_model_coverage_mismatch")
                if row["sourceModelHashes"] != [branch.get("sourceModelHash")
                        for branch in preparation["branches"] if branch["retained"]]:
                    return evidence("incomplete", "alternative_preparation_model_mismatch")
                for model_index, model in enumerate(models):
                    remaining()
                    # Bindings themselves issue at most one legality query per stage/seat.
                    needed = 1 + sum(opponent_strategy[key] is not None for key in ("auction_action", "escalation_action"))
                    needed += len(opponent_strategy.get("project_actions") or {})
                    if queries_used + needed > maximum_queries:
                        raise InformationSetCompilationStopped("maximum_queries", stage="native_bid_dominance")
                    query = model.find_native_round_with_constant_policy_below(
                        target=target.other, policy_team=target.other, threshold=-value,
                        maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts,
                        cancel_check=cancel_check, **opponent_strategy)
                    if query.get("sourceModelHash") != row["sourceModelHashes"][model_index]:
                        raise ValueError("bid dominance alternative model ownership mismatch")
                    row["queries"].append(query)
                    queries_used += len(query["bindings"]) + int(query["query"] is not None)
                    remaining()
                    if (query["status"] != "queried" or query["query"]["status"] != "no_matching_outcome"
                            or query["query"]["solverStatus"] != "infeasible"):
                        return evidence("incomplete", "alternative_upper_bound_not_certified")
                row["upperBound"] = value
                del model, models
            active, preparation = None, None
            if queries_used >= maximum_queries:
                raise InformationSetCompilationStopped("maximum_queries", stage="native_bid_dominance")
            model, selected_value, claim, best_value, regret, bindings = build_native_bid_dominance_model_v1(value, alternatives)
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = 0
            solver.parameters.max_number_of_conflicts = maximum_conflicts
            solver.parameters.max_time_in_seconds = remaining()
            status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
            queries_used += 1
            qualification_core = {"schemaVersion": 1, "kind": "native_bid_dominance_qualification_v1",
                "selectedValueVariable": selected_value.name, "selectedProofHash": canonical_sha256(selected),
                "alternatives": bindings, "claimVariable": claim.name, "claimLiteralIndex": int(claim.index),
                "bestValueVariable": best_value.name, "regretVariable": regret.name,
                "modelProto": str(model.proto), "solverStatus": solver.status_name(status).lower()}
            qualification = {**qualification_core, "sha256": canonical_sha256(qualification_core)}
            remaining()
            if status == cp_model.MODEL_INVALID:
                raise ValueError("native bid dominance qualification model is invalid")
            if status != cp_model.OPTIMAL or not solver.boolean_value(claim):
                return evidence("incomplete", "dominance_qualification_not_proven")
            quality_core = {"schemaVersion": 1, "authority": "cp_sat",
                "scope": "compiled_actor_information_set" if observed else "compiled_complete_state",
                "rootActorRelation": "target", "actualAction": selected_root_action,
                "optimalActions": [selected_root_action], "optimalActionsComplete": False,
                "optimalActionCount": None, "optimalActionCountLowerBound": 1,
                "optimalActionCountUpperBound": len(legal_actions),
                "bestValue": None if (interval or resume) else int(solver.value(best_value)),
                "actualValue": None if (interval or resume) else int(solver.value(selected_value)),
                **({"selectedValueBounds": {"lower": value, "upper": selected["strategy"]["upperBound"]}} if (interval or resume) else {}),
                "regret": int(solver.value(regret)), "actualActionOptimal": bool(solver.boolean_value(claim)),
                "solverStatus": "optimal", "variableCount": len(model.proto.variables),
                "constraintCount": len(model.proto.constraints),
                "analysisScope": {"mode": "exact", "coverage": "selected_bid_current_deal",
                    "utility": "awarded_point_margin", "exactStateId": exact.exact_state_id},
                "qualificationModelHash": qualification["sha256"]}
            action_quality = {**quality_core, "sha256": canonical_sha256({**quality_core, "modelProto": str(model.proto)})}
            return evidence("selected_action_optimal")
        except InformationSetCompilationStopped as exc:
            partial = exc.partial_receipt.get("evidence", {})
            queries_used += partial.get("queriesUsed", 0)
            if exc.partial_receipt.get("stage") == "native_bid_preparation":
                preparation = partial
            return evidence("incomplete", exc.partial_receipt)

    def _prove_observed_bid_visible_policy_floor(self, exact, prefix, *, selected_action,
            native_play_policy_id, nonplay_policy, required_threshold, maximum_queries,
            maximum_branches, maximum_model_bytes, maximum_conflicts, maximum_seconds,
            maximum_depth, opportunity_order=None, cancel_check=None, observed_project_policy_id=None,
            observed_auction_policy_id=None):
        """Produce a selected-policy floor, not best-bid or whole-round admission.

        The caller must own the target policy and proposed threshold. Every
        actor-visible branch is checked; the existing fresh reader owns admission.
        """
        from .ortools_play_tail import native_observer_safe_policy_identity_v1

        if (not self.native_cp_sat or not isinstance(exact, OrToolsNormalizedExactStateV1)
                or not isinstance(prefix, ExactTracePrefixV1)
                or type(required_threshold) is not int or not -(2**62) <= required_threshold < 2**62
                or any(type(value) is not int or value < 1 for value in (maximum_queries,
                    maximum_branches, maximum_model_bytes, maximum_conflicts, maximum_depth))
                or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
                or maximum_seconds <= 0 or not isinstance(nonplay_policy, Mapping)
                or set(nonplay_policy) != {"auction_action", "escalation_action"}
                or any(value is not None and type(value) is not str for value in nonplay_policy.values())):
            raise ValueError("observed bid floor requires an owned policy, integer threshold, and finite limits")
        exact.require_integrity()
        if (exact.decision_kind != "bid" or exact.rules_hash != self.rules.sha256
                or exact.prefix_hash != _effective_prefix_hash(prefix)
                or type(selected_action) is not str
                or selected_action not in {row["code"] for row in exact.legal_actions}):
            raise ValueError("observed bid floor root ownership mismatch")
        target = self._nonplay_actor(exact).team
        native_observer_safe_policy_identity_v1(native_play_policy_id, target)
        project_metadata = {}
        if observed_auction_policy_id is not None:
            from .ortools_scenario_model import validate_native_observed_auction_policy_id_v1
            project_metadata["observedAuctionPolicyId"] = validate_native_observed_auction_policy_id_v1(
                observed_auction_policy_id)
            if nonplay_policy["auction_action"] is not None:
                raise ValueError("native observed auction strategy cannot mix auction authorities")
        if observed_project_policy_id is not None:
            from .ortools_project_fragment import validate_native_observed_project_policy_id_v1
            project_metadata["observedProjectPolicyId"] = validate_native_observed_project_policy_id_v1(
                observed_project_policy_id)
        deadline = time.monotonic() + maximum_seconds
        preparation, rows, used = None, [], 0

        def remaining():
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled", stage="native_observed_bid_floor")
            if time.monotonic() >= deadline:
                raise InformationSetCompilationStopped("maximum_seconds", stage="native_observed_bid_floor")
            return deadline - time.monotonic()

        try:
            models, preparation = self._prepare_native_bid_branches(exact, prefix,
                visibility="actor_information_set", selected_root_action=selected_action,
                # Historical preparation stays broad. An optional target auction
                # strategy can exclude its unreachable cells before full play build;
                # the owned floor query binds and verifies that same strategy below.
                maximum_queries=maximum_queries, maximum_branches=maximum_branches,
                maximum_model_bytes=maximum_model_bytes, maximum_conflicts=maximum_conflicts,
                maximum_depth=maximum_depth, maximum_seconds=remaining(),
                opportunity_order=opportunity_order, cancel_check=cancel_check,
                **({"target_native_auction_policy_id": observed_auction_policy_id}
                   if observed_auction_policy_id is not None else {}))
            used = preparation["queriesUsed"]
            if not models:
                raise ValueError("observed bid floor requires reachable branch coverage")
            for model in models:
                remaining()
                project_args = {}
                if observed_auction_policy_id is not None:
                    project_args["native_auction_policy_id"] = observed_auction_policy_id
                if observed_project_policy_id is not None:
                    from .ortools_project_fragment import NATIVE_FULL_HAND_PROJECT_POLICY_ID
                    actor = self._nonplay_actor(exact)
                    fragment = model._native_round_fragment
                    if observed_project_policy_id == NATIVE_FULL_HAND_PROJECT_POLICY_ID:
                        project_args["native_project_policy_id"] = observed_project_policy_id
                    else:
                        project_args["project_actions"] = (fragment.projects.known_project_policy_actions(
                            observed_project_policy_id, actor,
                            tuple(self._card(card) for card in exact.state["hands"][actor.name.lower()]))
                            if fragment is not None else {})
                cost = 1 + int(nonplay_policy["auction_action"] is not None) + int(
                    nonplay_policy["escalation_action"] is not None and model._native_round_fragment is not None)
                cost += len(project_args.get("project_actions", {}))
                cost += int(observed_auction_policy_id is not None)
                if used + cost > maximum_queries:
                    raise InformationSetCompilationStopped("maximum_queries", stage="native_observed_bid_floor")
                receipt = model.find_native_round_with_constant_policy_below(target=target, policy_team=target,
                    native_play_policy_id=native_play_policy_id, **nonplay_policy,
                    threshold=required_threshold, optimize_margin=False,
                    maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
                    record_model_bytes=True, **project_args)
                actual_cost = len(receipt["bindings"]) + int(receipt.get("query") is not None)
                if actual_cost > cost:
                    raise ValueError("observed bid floor query exceeded its reserved allowance")
                used += actual_cost
                rows.append({"sourceModelHash": model._model_hash(search_branching="automatic"), "receipt": receipt,
                    **({"projectActions": project_args["project_actions"]} if "project_actions" in project_args else {})})
                remaining()
                sizes = receipt["modelBytes"]
                if preparation["modelBytes"] - sizes["source"] + sizes["retainedUpperBound"] > maximum_model_bytes:
                    raise InformationSetCompilationStopped("maximum_model_bytes", stage="native_observed_bid_floor")
                outcome = receipt.get("query") or {}
                if (receipt.get("status") == "queried" and outcome.get("status") == "witness_found"
                        and outcome.get("solverStatus") in {"feasible", "optimal"}
                        and type((outcome.get("candidate") or {}).get("utility")) is int
                        and outcome["candidate"]["utility"] < required_threshold):
                    # Refutes this proposed floor, not the bid's optimality.
                    core = {"schemaVersion": 1, "kind": "native_observed_bid_policy_floor_refutation_v1",
                        "status": "refuted", "preparationHash": canonical_sha256(preparation),
                        "nativePlayPolicyId": native_play_policy_id, "nonplayPolicy": dict(nonplay_policy),
                        "requiredThreshold": required_threshold, "branches": rows, "queriesUsed": used,
                        **project_metadata}
                    return {**core, "sha256": canonical_sha256(core)}, preparation
                if (receipt.get("status") != "queried" or outcome.get("status") != "no_matching_outcome"
                        or outcome.get("solverStatus") != "infeasible" or outcome.get("candidate") is not None):
                    raise InformationSetCompilationStopped("selected_policy_floor_not_certified",
                        stage="native_observed_bid_floor")
        except InformationSetCompilationStopped as exc:
            if exc.partial_receipt.get("stage") == "native_bid_preparation":
                preparation = exc.partial_receipt.get("evidence")
                used = preparation.get("queriesUsed", 0) if isinstance(preparation, Mapping) else used
            raise InformationSetCompilationStopped(exc.reason, stage="native_observed_bid_floor",
                evidence={"preparation": preparation, "branches": rows, "queriesUsed": used,
                    "requiredThreshold": required_threshold, "nativePlayPolicyId": native_play_policy_id,
                    "cause": exc.partial_receipt, **project_metadata}) from exc
        core = {"schemaVersion": 1, "kind": "native_observed_bid_visible_policy_floor_v1",
            "preparationHash": canonical_sha256(preparation), "nativePlayPolicyId": native_play_policy_id,
            "nonplayPolicy": dict(nonplay_policy), "requiredThreshold": required_threshold, "branches": rows,
            **project_metadata}
        return {**core, "sha256": canonical_sha256(core)}, preparation

    def _probe_observed_bid_threshold_refutation(self, exact, prefix, *, selected_action, threshold,
            configuration, opponent_play_policy_id, opponent_escalation_action, maximum_evidence_bytes, **limits):
        """Try one legal counterworld; never refute optimality from a failed sufficient floor."""
        from .result_authority import reverify_native_bid_world_upper_v1

        deadline = time.monotonic() + configuration["maximumSeconds"]
        allowance = configuration["maximumQueries"]
        used, evidence, stage = 0, {}, "selection"

        def remaining():
            if limits.get("cancel_check") is not None and limits["cancel_check"]():
                raise InformationSetCompilationStopped("cancelled", stage="native_bid_threshold_probe")
            if time.monotonic() >= deadline or used >= allowance:
                raise InformationSetCompilationStopped("maximum_seconds" if time.monotonic() >= deadline
                    else "maximum_queries", stage="native_bid_threshold_probe")
            return deadline - time.monotonic()

        def charge(count):
            nonlocal used
            if type(count) is not int or not 0 <= count <= allowance - used:
                raise ValueError("native bid threshold probe exceeds its query allowance")
            used += count

        def receipt(status, cause=None):
            core = {"schemaVersion": 1, "kind": "native_bid_threshold_probe_v1", "status": status,
                "stage": stage, "configuration": dict(configuration), "selectedAction": selected_action,
                "requiredThreshold": threshold, "rulesHash": self.rules.sha256,
                "sourcePrefixHash": prefix.sha256, "strategicCoverage": False,
                "queriesCharged": used, "queryCountComplete": status != "incomplete", **evidence,
                **({"cause": cause} if cause is not None else {})}
            if len(canonical_json(core).encode("utf-8")) > maximum_evidence_bytes:
                raise InformationSetCompilationStopped("maximum_evidence_bytes", stage="native_bid_threshold_probe",
                    evidence={"queriesCharged": used})
            return {**core, "sha256": canonical_sha256(core)}

        common = {key: limits[key] for key in ("maximum_branches", "maximum_model_bytes", "maximum_conflicts",
            "maximum_depth", "opportunity_order", "cancel_check")}
        try:
            world_prefix, selection = self._find_native_bid_compatible_world(exact, prefix,
                selected_action=selected_action, world_ranking_id=configuration["worldRankingId"],
                maximum_worlds=1, maximum_seconds=remaining(), maximum_conflicts=limits["maximum_conflicts"],
                maximum_model_bytes=limits["maximum_model_bytes"], cancel_check=limits["cancel_check"])
            charge(selection["queriesUsed"])
            evidence["selection"] = selection
            receipt("incomplete")  # Fail before further work if retained evidence cannot fit.
            if world_prefix is None:
                return None, receipt("not_found_within_budget")
            stage = "normalization"
            seconds = remaining()
            charge(2)  # Canonical bid witness and uniqueness, including a stopped solve.
            world = OrToolsExactStateNormalizerV1(self.rules, native_cp_sat=True).normalize(
                world_prefix, decision_point={"kind": "bid"}, maximum_seconds=seconds,
                cancel_check=limits["cancel_check"])
            if canonical_observer_information_hash_v1(complete_state=world.state, acting_seat=world.next_actor) != (
                    canonical_observer_information_hash_v1(complete_state=exact.state, acting_seat=exact.next_actor)):
                raise ValueError("native bid threshold probe changed the observer root")
            evidence["worldPrefixHash"] = world_prefix.sha256
            evidence["worldPrefix"] = world_prefix.to_dict()
            policy = {"auction_action": configuration["opponentAuctionAction"],
                "escalation_action": opponent_escalation_action}
            stage = "world_upper"
            _, upper = self._prepare_native_bid_branches(world, world_prefix, visibility="complete_state",
                selected_root_action=selected_action, policy_team=Seat.parse(world.next_actor).team.other,
                nonplay_policy=policy, opponent_play_policy_id=opponent_play_policy_id,
                compact_purchase_invariant_ownership=True, optimize_alternative_upper=True,
                _refutation_threshold=threshold,
                maximum_seconds=remaining(), maximum_queries=allowance - used, **common)
            charge(upper["queriesUsed"])
            evidence["upperProof"] = upper
            receipt("incomplete")
            if any(row.get("alternativeUpperBound", threshold - 1) >= threshold
                   for row in upper["branches"] if row["retained"]):
                return None, receipt("threshold_not_refuted")
            stage = "fresh_verification"
            verified = reverify_native_bid_world_upper_v1(upper, rules=self.rules, exact=world, source_prefix=world_prefix,
                action=selected_action, opponent_policy=policy, opponent_play_policy_id=opponent_play_policy_id,
                maximum_seconds=remaining(), maximum_queries=allowance - used, _on_queries_used=charge, **common)
            if verified != upper["upperBound"]:
                raise ValueError("native bid threshold probe fresh cap disagrees")
            stage = "threshold_refutation"
            result = self._solve_native_bid_policy_bounds(exact, prefix, Seat.parse(exact.next_actor).team,
                visibility="actor_information_set", selected_root_action=selected_action, required_threshold=threshold,
                threshold_refutation=dict(receipt=upper, exact=world, prefix=world_prefix,
                    opponentPolicy=policy, opponentPlayPolicyId=opponent_play_policy_id),
                maximum_iterations=limits["maximum_iterations"], maximum_observations=limits["maximum_observations"],
                maximum_seconds=remaining(), maximum_queries=allowance - used, **common)
            charge(result["queriesUsed"])
            return result, receipt("threshold_refuted")
        except (InformationSetCompilationStopped, ExactNormalizationStopped) as exc:
            used = allowance  # A stopped nested reader may omit its final solve count.
            if exc.partial_receipt.get("exhaustionReason") == "cancelled":
                raise InformationSetCompilationStopped("cancelled", stage="native_bid_threshold_probe",
                    evidence={"queriesCharged": used}) from exc
            return None, receipt("incomplete", exc.partial_receipt)

    def _prove_observed_bid_dominance(self, exact, prefix, *, selected_action, opponent_policy,
            maximum_iterations, maximum_queries, maximum_observations, maximum_branches,
            maximum_model_bytes, maximum_conflicts, maximum_seconds, maximum_depth,
            opportunity_order=None, cancel_check=None, reserve_artifact_verification=False,
            opponent_play_policy_id=None, incumbent_proof=None, progress_callback=None,
            alternative_worlds=None, alternative_proof=None, maximum_evidence_bytes=None, threshold_refutation_probe=None,
            strategy_play_fallback_id=None, _prior_alternatives=None, _defer_bid_fallback=False,
            minimax_budget=None):
        """Produce and freshly reverify sufficient observed bid bounds, not full-round admission."""
        from .result_authority import (
            compose_native_observed_bid_dominance_v1, validate_native_bid_world_upper_v1,
        )
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("native proof progress callback must be callable")
        if (type(reserve_artifact_verification) is not bool or type(_defer_bid_fallback) is not bool
                or not self.native_cp_sat or not isinstance(exact, OrToolsNormalizedExactStateV1)
                or not isinstance(prefix, ExactTracePrefixV1)
                or any(type(value) is not int or value < 1 for value in (maximum_iterations,
                    maximum_queries, maximum_observations, maximum_branches, maximum_model_bytes,
                    maximum_conflicts, maximum_depth))
                or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
                or maximum_seconds <= 0):
            raise ValueError("observed bid dominance requires native context and finite budgets")
        exact.require_integrity()
        actions = sorted(row["code"] for row in exact.legal_actions)
        if (exact.decision_kind != "bid" or exact.rules_hash != self.rules.sha256
                or exact.prefix_hash != _effective_prefix_hash(prefix) or selected_action not in actions
                or not isinstance(opponent_policy, Mapping)
                or set(opponent_policy) != {"auction_action", "escalation_action"}
                or any(action is not None and type(action) is not str for action in opponent_policy.values())):
            raise ValueError("observed bid dominance root or opponent policy mismatch")
        target = self._nonplay_actor(exact).team
        if strategy_play_fallback_id is not None:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            native_observer_safe_policy_identity_v1(strategy_play_fallback_id, target)
        if minimax_budget is not None and (not isinstance(minimax_budget, ScenarioProofBudgetV1)
                or strategy_play_fallback_id is None):
            raise ValueError("observed bid minimax shortening requires a typed budget and governed fallback")
        if alternative_worlds is None:
            alternative_worlds = {}
        if alternative_proof is not None:
            from .scenario_contracts import build_native_bid_alternative_search_v1
            # Validate frozen ceilings independently of the smaller runtime
            # remainder. Actual stage calls below use min(remaining(), cap).
            alternative_proof = build_native_bid_alternative_search_v1(
                alternative_proof, maximum_seconds=math.inf)
            if (alternative_worlds or opponent_play_policy_id is not None
                    or type(maximum_evidence_bytes) is not int or maximum_evidence_bytes < 1
                    or alternative_proof["maximumQueries"] >= maximum_queries):
                raise ValueError("native alternative search requires unmixed policies and bounded parent resources")
        if (not isinstance(alternative_worlds, Mapping)
                or any(action not in actions or action == selected_action for action in alternative_worlds)
                or any(not isinstance(value, (tuple, list)) or len(value) != 2 for value in alternative_worlds.values())):
            raise ValueError("observed alternative worlds require legal alternative actions")
        if threshold_refutation_probe is not None:
            from .scenario_contracts import build_native_bid_refutation_probe_v1
            threshold_refutation_probe = build_native_bid_refutation_probe_v1(
                threshold_refutation_probe, maximum_seconds=math.inf)
            if (type(maximum_evidence_bytes) is not int
                    or maximum_evidence_bytes < 1 or threshold_refutation_probe["maximumQueries"] >= maximum_queries):
                raise ValueError("native bid threshold probe requires bounded parent resources")
        observer = canonical_observer_information_hash_v1(
            complete_state=exact.state, acting_seat=exact.next_actor)
        for world, world_prefix in alternative_worlds.values():
            if (not isinstance(world, OrToolsNormalizedExactStateV1)
                    or not isinstance(world_prefix, ExactTracePrefixV1)):
                raise ValueError("observed alternative requires a normalized legal prefix")
            world.require_integrity()
            if (world.decision_kind != "bid" or world.rules_hash != self.rules.sha256
                    or world.prefix_hash != _effective_prefix_hash(world_prefix)
                    or world.next_actor != exact.next_actor
                    or sorted(row["code"] for row in world.legal_actions) != actions
                    or canonical_observer_information_hash_v1(
                        complete_state=world.state, acting_seat=world.next_actor) != observer
                    or world_prefix != prefix and world_prefix != self._prefix_with_deck(prefix, world_prefix.deck_order)):
                raise ValueError("observed alternative world changes the public root")
        normalization_queries = 2 * len({world_prefix.sha256 for _, world_prefix in alternative_worlds.values()
            if world_prefix != prefix}) if reserve_artifact_verification else 0
        if incumbent_proof is not None:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            fields = {"schemaVersion", "nativePlayPolicyId", "maximumSeconds", "maximumQueries"}
            if (not isinstance(incumbent_proof, Mapping) or not fields <= set(incumbent_proof) <= fields | {"observedProjectPolicyId", "observedAuctionPolicyId"}
                    or type(incumbent_proof["schemaVersion"]) is not int or incumbent_proof["schemaVersion"] != 2
                    or type(incumbent_proof["maximumSeconds"]) not in {int, float}
                    or not math.isfinite(incumbent_proof["maximumSeconds"])
                    or not 0 < incumbent_proof["maximumSeconds"]
                    or type(incumbent_proof["maximumQueries"]) is not int
                    or not 0 < incumbent_proof["maximumQueries"] < maximum_queries):
                raise ValueError("observed incumbent requires a closed native policy and bounded sub-budget")
            native_observer_safe_policy_identity_v1(incumbent_proof["nativePlayPolicyId"], target)
            if "observedProjectPolicyId" in incumbent_proof:
                from .ortools_project_fragment import validate_native_observed_project_policy_id_v1
                validate_native_observed_project_policy_id_v1(incumbent_proof["observedProjectPolicyId"])
            if "observedAuctionPolicyId" in incumbent_proof:
                from .ortools_scenario_model import validate_native_observed_auction_policy_id_v1
                validate_native_observed_auction_policy_id_v1(incumbent_proof["observedAuctionPolicyId"])
        play_policy = None
        if opponent_play_policy_id is not None:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            play_policy = native_observer_safe_policy_identity_v1(opponent_play_policy_id, target.other)
        policy_args = {"opponent_play_policy_id": opponent_play_policy_id} if play_policy is not None else {}
        deadline = time.monotonic() + maximum_seconds
        used, alternatives, uppers, searches = 0, [], [], []
        search_deadline = None
        selected, certificate, active_preparation = None, None, None
        incumbent_attempt = None
        refutation_attempt = None
        stage = "alternative_bounds"

        def remaining():
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled", stage="observed_bid_dominance")
            seconds = deadline - time.monotonic()
            if seconds <= 0 or used >= maximum_queries:
                raise InformationSetCompilationStopped("maximum_seconds" if seconds <= 0 else "maximum_queries",
                    stage="observed_bid_dominance")
            return seconds

        common = dict(maximum_branches=maximum_branches, maximum_model_bytes=maximum_model_bytes,
            maximum_conflicts=maximum_conflicts, maximum_depth=maximum_depth,
            opportunity_order=opportunity_order, cancel_check=cancel_check)

        def charge(count):
            nonlocal used
            if type(count) is not int or not 0 <= count <= maximum_queries - used:
                raise ValueError("observed bid stage query accounting exceeds its allowance")
            used += count

        def evidence(status, cause=None):
            core = {"schemaVersion": 1, "kind": "native_observed_bid_dominance_attempt_v1",
                "status": status, "stage": stage, "rulesHash": self.rules.sha256,
                "sourcePrefixHash": prefix.sha256, "selectedAction": selected_action,
                "queriesUsed": used, "queryCountComplete": status != "incomplete",
                "maximumQueries": maximum_queries, "maximumSeconds": maximum_seconds,
                **({"opponentPlayPolicy": play_policy} if play_policy is not None else {}),
                **({"strategyPlayFallbackId": strategy_play_fallback_id}
                   if strategy_play_fallback_id is not None else {}),
                **({"incumbentProof": dict(incumbent_proof)} if incumbent_proof is not None else {}),
                **({"incumbentAttempt": incumbent_attempt} if incumbent_attempt is not None else {}),
                **({"thresholdRefutationProbe": refutation_attempt} if refutation_attempt is not None else {}),
                **({"artifactVerificationReserved": True} if reserve_artifact_verification else {}),
                **({"alternativeProof": dict(alternative_proof), "alternativeSearches": searches}
                   if alternative_proof is not None else {}),
                "limits": {"maximumIterations": maximum_iterations,
                    "maximumObservations": maximum_observations, "maximumBranches": maximum_branches,
                    "maximumModelBytes": maximum_model_bytes, "maximumConflicts": maximum_conflicts,
                    "maximumDepth": maximum_depth, "workersPerSolve": 1},
                "selectedProof": selected, "certificate": certificate,
                "alternatives": [{"action": row["action"], "receipt": row["receipt"]} for row in alternatives],
                "activePreparation": active_preparation,
                **({"cause": cause} if cause is not None else {})}
            return {**core, "sha256": canonical_sha256(core)}

        def prove_selected(threshold):
            nonlocal selected, active_preparation, incumbent_attempt, refutation_attempt
            if threshold_refutation_probe is not None:
                probe = {**threshold_refutation_probe,
                    "maximumSeconds": min(remaining(), threshold_refutation_probe["maximumSeconds"]),
                    "maximumQueries": min(maximum_queries - used, threshold_refutation_probe["maximumQueries"])}
                selected, refutation_attempt = self._probe_observed_bid_threshold_refutation(exact, prefix,
                    selected_action=selected_action, threshold=threshold, configuration=probe,
                    opponent_play_policy_id=opponent_play_policy_id, maximum_evidence_bytes=maximum_evidence_bytes,
                    opponent_escalation_action=opponent_policy["escalation_action"],
                    maximum_iterations=maximum_iterations, maximum_observations=maximum_observations, **common)
                charge(refutation_attempt["queriesCharged"])
                if selected is not None:
                    return False
            if incumbent_proof is not None:
                allowance = min(remaining(), incumbent_proof["maximumSeconds"])
                floor, interrupted = None, None
                try:
                    floor, active_preparation = self._prove_observed_bid_visible_policy_floor(exact, prefix,
                        selected_action=selected_action, native_play_policy_id=incumbent_proof["nativePlayPolicyId"],
                        nonplay_policy={"auction_action": None, "escalation_action": None},
                        required_threshold=threshold,
                        maximum_queries=min(maximum_queries - used, incumbent_proof["maximumQueries"]),
                        maximum_seconds=allowance, **common,
                        **({"observed_project_policy_id": incumbent_proof["observedProjectPolicyId"]}
                           if "observedProjectPolicyId" in incumbent_proof else {}),
                        **({"observed_auction_policy_id": incumbent_proof["observedAuctionPolicyId"]}
                           if "observedAuctionPolicyId" in incumbent_proof else {}))
                except InformationSetCompilationStopped as exc:
                    if (exc.partial_receipt.get("stage") != "native_observed_bid_floor"
                            or exc.reason not in {"maximum_seconds", "maximum_queries", "selected_policy_floor_not_certified"}):
                        raise
                    interrupted = exc.partial_receipt
                    active_preparation = interrupted["evidence"].get("preparation")
                count = (interrupted["evidence"]["queriesUsed"] if interrupted is not None else
                    floor["queriesUsed"] if floor.get("status") == "refuted" else
                    active_preparation["queriesUsed"] + sum(len(row["receipt"]["bindings"]) + 1
                                                            for row in floor["branches"]))
                if type(count) is not int or not 0 <= count <= incumbent_proof["maximumQueries"]:
                    raise ValueError("observed incumbent query accounting exceeds its allowance")
                core = {"schemaVersion": 1, "kind": "native_bid_policy_bounds_v1",
                    "status": "incomplete" if interrupted is not None or floor.get("status") == "refuted" else "scoped_strategy_threshold_complete",
                    "visibility": "actor_information_set", "targetTeam": target.name.lower(),
                    "strategicCoverage": False, "selectedRootAction": selected_action,
                    "strategy": floor, "preparation": active_preparation,
                    "queriesUsed": count, "maximumSeconds": allowance,
                    **({"partialReceipt": interrupted} if interrupted is not None else {})}
                selected = {**core, "sha256": canonical_sha256(core)}
                active_preparation = None
                charge(selected["queriesUsed"])
                if selected["status"] != "scoped_strategy_threshold_complete":
                    if _defer_bid_fallback:
                        incumbent_attempt = selected
                        return False
                    incumbent_attempt, selected = selected, None
            if selected is None:
                selected = self._solve_native_bid_policy_bounds(exact, prefix, target,
                    visibility="actor_information_set", selected_root_action=selected_action,
                    required_threshold=threshold, maximum_iterations=maximum_iterations,
                    maximum_observations=maximum_observations, maximum_queries=maximum_queries - used,
                    maximum_seconds=remaining(), progress_callback=(lambda row: progress_callback({**row,
                        "queryCount": used + row["queryCount"]})) if progress_callback is not None else None,
                    **({"strategy_play_fallback_id": strategy_play_fallback_id}
                       if strategy_play_fallback_id is not None else {}),
                    **({"minimax_budget": minimax_budget} if minimax_budget is not None else {}),
                    **common)
                charge(selected["queriesUsed"])
            return selected["status"] == "scoped_strategy_threshold_complete"

        try:
            if alternative_proof is not None:
                # Every searched upper premise uses this same cap. Prove the
                # selected floor first, rather than spending search budget on
                # premises that an unfinished selected proof cannot use.
                stage = "visible_strategy"
                if not prove_selected(alternative_proof["upperBound"]):
                    return evidence("incomplete", "sufficient_threshold_not_proved")
                search_deadline = min(deadline, time.monotonic() + alternative_proof["maximumSeconds"])
                stage = "alternative_bounds"
            for action in actions:
                if action == selected_action:
                    continue
                if alternative_proof is not None:
                    seconds = min(remaining(), search_deadline - time.monotonic())
                    allowance = alternative_proof["maximumQueries"] - sum(
                        row["queriesCharged"] for row in searches)
                    information_refutation = alternative_proof["schemaVersion"] == 2
                    worlds = (1 if information_refutation else alternative_proof["maximumWorlds"]
                        - sum(len(row["attempts"]) for row in searches))
                    if seconds <= 0 or allowance <= 0 or worlds <= 0:
                        return evidence("incomplete", "alternative_search_budget_exhausted")
                    if information_refutation:
                        premise, search = self._search_observed_bid_information_set_upper(exact, prefix,
                            selected_action=action, upper_bound=alternative_proof["upperBound"],
                            maximum_seconds=seconds, maximum_queries=min(allowance, maximum_queries - used),
                            maximum_iterations=maximum_iterations, maximum_observations=maximum_observations,
                            maximum_evidence_bytes=maximum_evidence_bytes,
                            progress_callback=(lambda row: progress_callback({**row,
                                "queryCount": used + row["queryCount"]})) if progress_callback is not None else None,
                            **common)
                    else:
                        premise, search = self._search_observed_bid_alternative_upper(exact, prefix,
                            selected_action=action, upper_bound=alternative_proof["upperBound"],
                            maximum_worlds=worlds, maximum_seconds=seconds,
                            maximum_queries=min(allowance, maximum_queries - used),
                            maximum_world_seconds=alternative_proof["maximumWorldSeconds"],
                            maximum_world_queries=alternative_proof["maximumWorldQueries"],
                            maximum_iterations=maximum_iterations, maximum_observations=maximum_observations,
                            maximum_evidence_bytes=maximum_evidence_bytes,
                            **({"world_ranking_id": alternative_proof["worldRankingId"]}
                               if "worldRankingId" in alternative_proof else {}),
                            **({"strategy_play_fallback_id": strategy_play_fallback_id} if strategy_play_fallback_id is not None else {}),
                            **({"minimax_budget": minimax_budget} if minimax_budget is not None else {}),
                            progress_callback=(lambda row: progress_callback({**row,
                                "queryCount": used + row["queryCount"]})) if progress_callback is not None else None, **common)
                    searches.append(search)
                    charge(search["queriesCharged"])
                    if premise is None:
                        return evidence("incomplete", "alternative_cap_not_proved")
                    alternatives.append(premise)
                    uppers.append(premise["receipt"]["upperBound"])
                    continue
                # One compatible world bounds an alternative above; only the selected
                # strategy must survive the entire observation-compatible universe.
                world, world_prefix = alternative_worlds.get(action, (exact, prefix))
                reused = None
                for prior_action, prior_receipt, _ in (_prior_alternatives or {}).values():
                    if prior_action != action or world_prefix != prefix:
                        continue
                    declared = prior_receipt.get("limits", {})
                    ceilings = {"branches": maximum_branches, "modelBytes": maximum_model_bytes,
                        "conflictsPerSolve": maximum_conflicts, "maximumDepth": maximum_depth,
                        "queries": maximum_queries - used}
                    if any(type(declared.get(key)) is not int or not 0 < declared[key] <= cap
                            for key, cap in ceilings.items()):
                        continue
                    upper = validate_native_bid_world_upper_v1(prior_receipt, rules=self.rules,
                        exact=world, source_prefix=world_prefix, action=action, opponent_policy=opponent_policy,
                        maximum_model_bytes=maximum_model_bytes, maximum_depth=maximum_depth,
                        opportunity_order=opportunity_order, **policy_args)
                    remaining()
                    # Logical charges remain conservative; final composition
                    # still reconstructs every premise, including this one.
                    charge(prior_receipt["queriesUsed"])
                    reused = {"action": action, "exact": world, "sourcePrefix": world_prefix,
                        "receipt": prior_receipt}
                    alternatives.append(reused)
                    uppers.append(upper)
                    break
                if reused is not None:
                    continue
                models, active_preparation = self._prepare_native_bid_branches(world, world_prefix,
                    visibility="complete_state",
                    selected_root_action=action, policy_team=target.other, nonplay_policy=opponent_policy,
                    compact_purchase_invariant_ownership=True,
                    optimize_alternative_upper=True, maximum_queries=maximum_queries - used,
                    maximum_seconds=remaining(),
                    **({"progress_callback": lambda row: progress_callback({**row,
                        "queryCount": used + row["queryCount"]})} if progress_callback is not None else {}),
                    **policy_args, **common)
                charge(active_preparation["queriesUsed"])
                if models:
                    raise ValueError("observed alternative upper must release streamed models")
                upper = validate_native_bid_world_upper_v1(active_preparation, rules=self.rules,
                    exact=world, source_prefix=world_prefix, action=action, opponent_policy=opponent_policy,
                    maximum_model_bytes=maximum_model_bytes, maximum_depth=maximum_depth,
                    opportunity_order=opportunity_order, **policy_args)
                alternatives.append({"action": action, "exact": world, "sourcePrefix": world_prefix,
                    "receipt": active_preparation})
                uppers.append(upper)
                active_preparation = None
            if not uppers:
                return evidence("incomplete", "no_alternative_bound")
            stage = "visible_strategy"
            threshold = max(uppers)
            if selected is None:
                prove_selected(threshold)
            if selected["status"] != "scoped_strategy_threshold_complete":
                return evidence("incomplete", "sufficient_threshold_refuted" if selected["status"] == (
                    "scoped_strategy_threshold_refuted") else "sufficient_threshold_not_proved")
            stage = "fresh_verification"
            if reserve_artifact_verification and alternative_proof is not None:
                normalization_queries = 2 * len({row["sourcePrefix"].sha256 for row in alternatives
                    if row["sourcePrefix"] != prefix})
            seconds = remaining()
            verification_queries = (maximum_queries - used - normalization_queries) // (2 if reserve_artifact_verification else 1)
            if verification_queries < len(actions) + 1:
                raise InformationSetCompilationStopped("maximum_queries", stage="observed_bid_dominance")
            certificate = compose_native_observed_bid_dominance_v1(rules=self.rules, exact=exact,
                source_prefix=prefix, selected_action=selected_action, strategy=selected["strategy"],
                preparation=selected["preparation"], required_threshold=threshold, alternatives=alternatives,
                opponent_policy=opponent_policy, maximum_observations=maximum_observations,
                **({"strategy_play_fallback_id": strategy_play_fallback_id}
                   if strategy_play_fallback_id is not None else {}),
                **({"minimax_budget": minimax_budget} if minimax_budget is not None else {}),
                maximum_queries=verification_queries, maximum_seconds=seconds, **policy_args, **common)
            charge(certificate["executionReceipt"]["queriesUsed"])
            if used > maximum_queries or time.monotonic() >= deadline or cancel_check is not None and cancel_check():
                return evidence("incomplete", "verification_budget_exhausted")
            return evidence(certificate["status"])
        except InformationSetCompilationStopped as exc:
            # Preserve the active stage's partial receipt; never reset its budget or retry silently.
            if exc.partial_receipt.get("stage") == "native_bid_threshold_probe":
                charge(exc.partial_receipt["evidence"]["queriesCharged"])
            if exc.partial_receipt.get("stage") in {"native_observed_bid_floor", "native_bid_preparation", "native_bid_policy_bounds"}:
                charge(exc.partial_receipt["evidence"]["queriesUsed"])
            if exc.partial_receipt.get("stage") == "native_bid_preparation" and stage == "alternative_bounds":
                active_preparation = exc.partial_receipt["evidence"]
            return evidence("incomplete", exc.partial_receipt)

    def _solve_symbolic_round_policy_bounds(self, scenarios, target, *, maximum_iterations,
            maximum_queries, maximum_observations, maximum_branches, maximum_conflicts,
            maximum_seconds, cancel_check=None, fixed_root=None, tighten_utility_ceiling=False,
            shorten_counterexamples=False, opponent_ceiling_policy=None, incumbent_play_policies=None,
            incumbent_only=False, incumbent_threshold=None, required_threshold=None,
            project_root_policy_probe=False, observed_play_path_bounds=False, retain_path_floor=False,
            incumbent_native_play_policy_id=None, incumbent_nonplay_policy=None,
            progress_callback=None, readback_witness_hints=False,
            _unavailable_opponent_ceiling_policy=None, _play_fallback_id=None, minimax_budget=None,
            _strategy_hint=None):
        """Refine shared strategy bounds over supplied native deal branches.

        This internal coordinator does not certify that supplied branches are
        the complete legal universe. Construction filters can exclude legal
        responses; a converged bound here must never admit an exact result.
        """
        if (not self.native_cp_sat or not isinstance(target, Team)
                or not isinstance(scenarios, (list, tuple)) or not scenarios):
            raise ValueError("symbolic policy bounds require native round branches and a team")
        if type(tighten_utility_ceiling) is not bool:
            raise ValueError("native utility ceiling selection must be boolean")
        if type(project_root_policy_probe) is not bool:
            raise ValueError("native project root policy probe selection must be boolean")
        if type(observed_play_path_bounds) is not bool:
            raise ValueError("native observed play path bounds selection must be boolean")
        if type(retain_path_floor) is not bool:
            raise ValueError("native path floor retention must be boolean")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("native proof progress callback must be callable")
        native_incumbent = None
        if incumbent_native_play_policy_id is not None:
            if incumbent_play_policies is not None:
                raise ValueError("incumbent policies must have one authority")
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            native_incumbent = native_observer_safe_policy_identity_v1(incumbent_native_play_policy_id, target)
        has_incumbent = incumbent_play_policies is not None or native_incumbent is not None
        if incumbent_nonplay_policy is not None:
            from .scenario_contracts import validate_native_bid_nonplay_policy_v1
            incumbent_nonplay_policy = validate_native_bid_nonplay_policy_v1(incumbent_nonplay_policy)
            if native_incumbent is None or not incumbent_only:
                raise ValueError("nonplay incumbent requires one native incumbent-only strategy")
        if required_threshold is not None and (type(required_threshold) is not int or incumbent_threshold is not None
                or incumbent_only or has_incumbent or opponent_ceiling_policy is not None
                or tighten_utility_ceiling):
            raise ValueError("strategy threshold requires a separate integer synthesis request")
        if type(shorten_counterexamples) is not bool:
            raise ValueError("native counterexample shortening selection must be boolean")
        if minimax_budget is not None and (not isinstance(minimax_budget, ScenarioProofBudgetV1)
                or not shorten_counterexamples or required_threshold is None):
            raise ValueError("minimax counterexample cuts require a typed budget and explicit threshold shortening")
        if type(readback_witness_hints) is not bool or readback_witness_hints and required_threshold is None:
            raise ValueError("readback witness hints require a native threshold strategy")
        fallback = None
        if _play_fallback_id is not None:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            fallback = native_observer_safe_policy_identity_v1(_play_fallback_id, target)
            if required_threshold is None or readback_witness_hints or (shorten_counterexamples and minimax_budget is None):
                raise ValueError("guarded fallback requires an unshortened threshold strategy without witness hints")
            if fixed_root is not None and any(model._native_auction_policy_root is None
                    or model._native_auction_policy_root[1:] != (fixed_root[2], target) for model in scenarios):
                raise ValueError("guarded fixed root must be an already constrained native auction action")
        if type(incumbent_only) is not bool or incumbent_only and not has_incumbent:
            raise ValueError("incumbent-only bounds require an explicit policy")
        if incumbent_threshold is not None and (type(incumbent_threshold) is not int or not incumbent_only):
            raise ValueError("incumbent threshold requires an integer and incumbent-only coverage")
        _validate_native_opponent_ceiling_policy(opponent_ceiling_policy, target)
        if _unavailable_opponent_ceiling_policy is not None:
            _validate_native_opponent_ceiling_policy(_unavailable_opponent_ceiling_policy, target)
            if len(scenarios) <= 1 or opponent_ceiling_policy is not None or tighten_utility_ceiling:
                raise ValueError("unavailable opponent ceiling requires direct multi-branch refinement")
        if (opponent_ceiling_policy is not None and "native_play_policy_id" in opponent_ceiling_policy
                and (not tighten_utility_ceiling or incumbent_only)):
            raise ValueError("native opponent play policy requires active ceiling refinement")
        if (any(type(value) is not int or value < 1 for value in (maximum_iterations,
                maximum_queries, maximum_observations, maximum_branches, maximum_conflicts))
                or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
                or maximum_seconds <= 0 or len(scenarios) > maximum_branches):
            raise ValueError("symbolic policy bounds require positive finite resource limits")
        deadline = time.monotonic() + maximum_seconds
        if fixed_root is not None and (not isinstance(fixed_root, tuple) or len(fixed_root) != 3
                or type(fixed_root[0]) is not str or not fixed_root[0]
                or not isinstance(fixed_root[1], tuple) or not fixed_root[1]
                or any(type(action) is not str for action in fixed_root[1])
                or len(set(fixed_root[1])) != len(fixed_root[1])
                or type(fixed_root[2]) is not str or fixed_root[2] not in fixed_root[1]):
            raise ValueError("fixed root requires an observation, legal domain, and selected action")
        ordered = []
        for scenario in scenarios:
            if (not isinstance(scenario, OrToolsScenarioModelV1) or scenario.rules.sha256 != self.rules.sha256
                    or any((scenario._observed_auction_bindings, scenario._observed_escalation_bindings,
                            scenario._observed_project_bindings, scenario._observed_play_bindings))):
                raise ValueError("symbolic policy bounds require unbound native round models")
            branch_scope, _ = scenario._native_round_query_context()
            if branch_scope.get("sha256") != canonical_sha256({key: value for key, value in branch_scope.items() if key != "sha256"}):
                raise ValueError("symbolic policy branch integrity mismatch")
            ordered.append((scenario._model_hash(search_branching="automatic"), scenario))
        ordered.sort(key=lambda item: item[0])
        if len({key for key, _ in ordered}) != len(ordered):
            raise ValueError("symbolic policy branches must be distinct")
        maximum = max(max(map(int, variable.proto.domain)) for _, scenario in ordered
                      for variable in scenario._native_round_query_context()[1])
        if incumbent_threshold is not None and not -maximum <= incumbent_threshold <= maximum:
            raise ValueError("incumbent threshold exceeds native utility bounds")
        # A guarantee may be below every outcome; keep arithmetic inside CP-SAT's safe range.
        if required_threshold is not None and not -(2**62) <= required_threshold < 2**62:
            raise InformationSetCompilationStopped("strategy_threshold_out_of_domain",
                stage="native_symbolic_round_policy_bounds",
                evidence={"requiredThreshold": required_threshold, "utilityBounds": [-maximum, maximum]})
        scope = {"kind": "native_symbolic_round_policy_scope_v1", "rulesHash": self.rules.sha256,
            "targetTeam": target.name.lower(), "coverage": "supplied_deal_branches_only",
            "utility": "awarded_point_margin", "utilityBounds": [-maximum, maximum],
            **({"requiredThreshold": required_threshold, "objective": "strategy_threshold_satisfaction"}
               if required_threshold is not None else {}),
            **({"cooperativeCeiling": True} if tighten_utility_ceiling else {}),
            **({"opponentCeilingUnavailable": {"reason": "multiple_contract_branches",
                "policy": dict(_unavailable_opponent_ceiling_policy)}}
               if _unavailable_opponent_ceiling_policy is not None else {}),
            **({"counterexampleShortening": True} if shorten_counterexamples else {}),
            **({"minimaxProofBudgetHash": minimax_budget.sha256} if minimax_budget is not None else {}),
            **({"readbackWitnessHints": "guarded_query_chain_v1"} if readback_witness_hints else {}),
            **({"playFallback": fallback, "policyRepresentation": "native_fallback_plus_deviations_v1"}
               if fallback is not None else {}),
            "branches": [{"modelHash": key, "scope": dict(scenario._native_round_query_context()[0])}
                         for key, scenario in ordered],
            **({"fixedRoot": {"observationHash": fixed_root[0], "legalActions": list(fixed_root[1]),
                              "selectedAction": fixed_root[2]}} if fixed_root is not None else {})}
        scope_hash = canonical_sha256(scope)
        domains, cuts, cut_hashes, masters, queries = {}, [], set(), [], []
        binding_queries = 0
        cut_bound_queries = {}
        incumbent_cut_queries = {}
        unconditional_bounds, reused_bounds = {}, []
        observation_hints = {}
        search_hints = {}  # One prior witness per bounded branch; hints never restrict its universe.
        minimax_preferences = {}
        outcome_cache = {}
        legality_cache = {}
        retained_refinements, final_branch_proofs, infeasible_branch_proofs = {}, [], {}
        fixed_actions = {}
        play_contexts = {}
        witnessed_actions = {}
        fallback_actions = {}
        if fixed_root is not None:
            domains[fixed_root[0]] = fixed_root[1]
            fixed_actions[fixed_root[0]] = fixed_root[2]
        policy, upper = dict(fixed_actions), maximum
        if _strategy_hint is not None:
            # Reuse proposed actions only. Bounds, cuts and query receipts must
            # be freshly proved; a hint cannot restrict the legal universe.
            fields = {"scopeHash", "domains", "policy", "playContexts", "sha256"}
            if (fallback is None or required_threshold is None
                    or not isinstance(_strategy_hint, Mapping) or set(_strategy_hint) != fields
                    or _strategy_hint["scopeHash"] != scope_hash
                    or any(not isinstance(_strategy_hint[key], Mapping)
                           or len(_strategy_hint[key]) > maximum_observations
                           for key in ("domains", "policy", "playContexts"))):
                raise ValueError("strategy hint requires owned guarded scope and bounded typed fields")
            hinted_domains, hinted_policy, hinted_contexts = (
                _strategy_hint[key] for key in ("domains", "policy", "playContexts"))
            if (any(type(key) is not str or len(key) != 64
                    or any(char not in "0123456789abcdef" for char in key)
                    or not isinstance(actions, (list, tuple)) or not actions
                    or any(type(action) is not str or not action for action in actions)
                    or len(set(actions)) != len(actions) for key, actions in hinted_domains.items())
                    or any(key not in hinted_domains or type(action) is not str
                           or action not in hinted_domains[key] for key, action in hinted_policy.items())
                    or any(key not in hinted_domains or not isinstance(info, Mapping)
                           for key, info in hinted_contexts.items())
                    or any(key in hinted_domains and set(hinted_domains[key]) != set(domains[key])
                           or key in hinted_policy and hinted_policy[key] != action
                           for key, action in fixed_actions.items())
                    or _strategy_hint["sha256"] != canonical_sha256({
                        key: value for key, value in _strategy_hint.items() if key != "sha256"})):
                raise ValueError("strategy hint action domains or fixed root mismatch")
            # Seed only proposed choices, not prior bounds/cuts or unused domains.
            # These are search variables, never legality or outcome evidence:
            # every effective choice still needs a fresh native branch proof.
            domains.update({key: tuple(hinted_domains[key]) for key in hinted_policy})
            play_contexts.update(copy.deepcopy({key: info for key, info in hinted_contexts.items()
                if key in hinted_policy}))
            witnessed_actions.update(hinted_policy)
        incumbent_lower = None
        path_floor = None
        fixed_world_ceiling = None
        # The native utility domains already cap every legal continuation.
        # A request above that cap is refuted, not a new enlarged game universe.
        ceiling = maximum if required_threshold is None else min(maximum, required_threshold)
        service = OrToolsActorInformationSetProofV1()

        def report_progress(phase, local_queries=0, local_bindings=0, solver_status=None):
            if progress_callback is None:
                return
            row = {"kind": "quality_solver_progress_v1", "phase": phase,
                "masterCount": len(masters), "queryCount": len(queries) + local_queries,
                "bindingCount": binding_queries + local_bindings,
                **({"solverStatus": solver_status} if solver_status is not None else {})}
            try:
                progress_callback(row)
            except Exception:
                pass  # Telemetry cannot change solver truth or qualification.

        def evidence(status, cause=None):
            floors = [value for value in (incumbent_lower, path_floor["value"] if path_floor else None)
                      if value is not None]
            core = {"schemaVersion": 1, "kind": ("native_guarded_round_strategy_threshold_v2" if fallback is not None
                    else "native_symbolic_round_strategy_threshold_v1"
                    if required_threshold is not None else "native_incumbent_threshold_floor_v1"
                    if incumbent_threshold is not None else "native_symbolic_round_policy_bounds_v2"),
                "status": status, "scope": scope, "scopeHash": scope_hash,
                "strategicCoverage": False, "upperBound": (maximum if required_threshold is not None
                    and status != "scoped_strategy_threshold_refuted" else upper),
                "lowerBound": required_threshold if status == "scoped_strategy_threshold_complete"
                    else upper if status == "scoped_bound_converged" else max(floors) if floors else None,
                **({"pathFloor": path_floor} if path_floor is not None else {}),
                **({"requiredThreshold": required_threshold,
                    "masterSearchValue": masters[-1].get("searchValue") if masters else None}
                   if required_threshold is not None else {}),
                "policy": dict(policy),
                **({"playContexts": dict(play_contexts)} if fallback is not None else {}),
                "domains": dict(domains), "masters": list(masters), "queries": list(queries),
                "queriesUsed": len(queries) + len(masters) + binding_queries,
                "bindingQueriesUsed": binding_queries,
                "refinements": dict(retained_refinements),
                "finalBranchProofs": list(final_branch_proofs),
                **({"reusedBranchProofs": list(reused_bounds)} if reused_bounds else {}),
                "cuts": [{"utility": cut.utility, "targetActions": [list(row) for row in cut.target_actions],
                          "sha256": cut.sha256,
                          **({"incumbentQuery": incumbent_cut_queries[cut.sha256]}
                             if cut.sha256 in incumbent_cut_queries else {}),
                          **({"prefixBoundQueryHash": cut_bound_queries[cut.sha256]}
                             if cut.sha256 in cut_bound_queries else {})} for cut in cuts],
                "limits": {"masterIterations": maximum_iterations, "queries": maximum_queries,
                    "observations": maximum_observations, "branches": maximum_branches,
                    "conflictsPerSolve": maximum_conflicts, "seconds": maximum_seconds},
                **({"cause": cause} if cause is not None else {})}
            return {**core, "sha256": canonical_sha256(core)}

        def stop(reason, cause=None):
            raise InformationSetCompilationStopped(reason, stage="native_symbolic_round_policy_bounds",
                evidence=evidence("incomplete", cause))

        def remaining():
            if cancel_check is not None and cancel_check():
                stop("cancelled")
            if time.monotonic() >= deadline:
                stop("maximum_seconds")
            return deadline - time.monotonic()

        if has_incumbent:
            if native_incumbent is None:
                scope["incumbentPlayPoliciesHash"] = canonical_sha256(incumbent_play_policies)
            else:
                scope["incumbentNativePlayPolicy"] = native_incumbent
            if incumbent_nonplay_policy is not None:
                scope["incumbentNonplayPolicy"] = incumbent_nonplay_policy
            if incumbent_threshold is not None:
                scope["incumbentThreshold"] = incumbent_threshold
            scope_hash = canonical_sha256(scope)
            floors = []
            incumbent_branches = (
                sorted(ordered, key=lambda row: (
                    len(row[1].model.proto.constraints), len(row[1].model.proto.variables), row[0]))
                if incumbent_threshold is not None else ordered
            )
            for branch_hash, scenario in incumbent_branches:
                remaining()
                reserved = 3 if incumbent_nonplay_policy is not None else 1
                if len(queries) + binding_queries + reserved > maximum_queries:
                    stop("maximum_queries")
                query = scenario.find_native_round_with_constant_policy_below(
                    target=target, policy_team=target,
                    auction_action=incumbent_nonplay_policy["auctionAction"] if incumbent_nonplay_policy else None,
                    escalation_action=incumbent_nonplay_policy["escalationAction"] if incumbent_nonplay_policy else None,
                    **({"native_project_policy_id": incumbent_nonplay_policy["projectPolicyId"]}
                       if incumbent_nonplay_policy else {}),
                    **({"record_policy_model_hashes": True} if incumbent_nonplay_policy else {}),
                    **({"play_policies": incumbent_play_policies} if native_incumbent is None else
                       {"native_play_policy_id": incumbent_native_play_policy_id}),
                    threshold=maximum + 1 if incumbent_threshold is None else incumbent_threshold,
                    maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts,
                    cancel_check=cancel_check,
                    **({"optimize_margin": False} if incumbent_threshold is not None else {}))
                queries.append({"branchModelHash": branch_hash, "purpose": (
                    "incumbent_threshold_floor" if incumbent_threshold is not None else "incumbent_lower_bound"),
                                "receipt": query})
                if incumbent_nonplay_policy is not None:
                    binding_queries += len(query.get("bindings", ()))
                outcome = query.get("query") or {}
                report_progress("preparation", solver_status=outcome.get("solverStatus"))
                if incumbent_threshold is not None:
                    remaining()
                    if (query.get("status") == "queried" and outcome.get("status") == "witness_found"
                            and outcome.get("solverStatus") in {"feasible", "optimal"}
                            and outcome.get("queryMode") == "feasibility_witness"
                            and outcome.get("threshold") == incumbent_threshold
                            and type((outcome.get("candidate") or {}).get("utility")) is int
                            and outcome["candidate"]["utility"] < incumbent_threshold):
                        # A failed sufficient policy is not a failed optimal-action claim.
                        stop("incumbent_threshold_refuted")
                    if (query.get("status") != "queried" or outcome.get("status") != "no_matching_outcome"
                            or outcome.get("solverStatus") != "infeasible"
                            or outcome.get("queryMode") != "feasibility_witness"
                            or outcome.get("threshold") != incumbent_threshold):
                        stop(query.get("exhaustionReason") or "incumbent_threshold_not_certified")
                    floors.append(incumbent_threshold)
                    continue
                if (incumbent_nonplay_policy is not None and query.get("status") == "queried"
                        and outcome.get("status") == "no_matching_outcome"
                        and outcome.get("solverStatus") == "infeasible"):
                    remaining()
                    continue  # This legal policy cannot reach the branch, not scenario impossibility.
                bound = outcome.get("objectiveLowerBound")
                if bound is None and outcome.get("solverStatus") == "optimal":
                    bound = outcome["candidate"]["utility"]
                if bound is not None:
                    if type(bound) is not int or not -maximum <= bound <= maximum:
                        raise ValueError("native incumbent objective bound is invalid")
                    floors.append(bound)
                    if len(floors) == len(ordered):
                        incumbent_lower = min(floors)
                remaining()
                if (query.get("status") != "queried" or bound is None
                        or outcome.get("solverStatus") not in {"optimal", "feasible"}):
                    stop(query.get("exhaustionReason") or "incumbent_bound_not_optimal")
            # Readers bind completed floor evidence in canonical scope order,
            # independently of the cheaper execution order.
            queries.sort(key=lambda row: row["branchModelHash"])
            if not floors:
                stop("incumbent_has_no_covered_continuation")
            incumbent_lower = min(floors)

        if incumbent_only:
            return evidence("scoped_incumbent_threshold_complete" if incumbent_threshold is not None
                            else "scoped_incumbent_bound_complete")

        if (tighten_utility_ceiling and len(ordered) == 1
                and (ordered[0][1]._native_project_root is not None
                     or ordered[0][1]._native_escalation_root is not None
                     or observed_play_path_bounds and ordered[0][1]._native_play_root is not None)
                and required_threshold is None and incumbent_threshold is None):
            # Every legal path bounds every observer-safe strategy below;
            # even cooperative play in one compatible world bounds it above.
            if len(queries) + len(masters) + binding_queries + 2 > maximum_queries:
                stop("maximum_queries")
            branch_hash, scenario = ordered[0]
            floor_scenario, root_policy = scenario, None
            source = getattr(scenario, "_native_play_source_query", None)
            if (source is not None and project_root_policy_probe
                    and (scenario._native_project_root is not None or scenario._native_escalation_root is not None)):
                escalation_root = scenario._native_escalation_root is not None
                kind = "escalation" if escalation_root else "project"
                owner = scenario._native_escalation_root if escalation_root else scenario._native_project_root
                if escalation_root:
                    root = next(row for row in source["candidate"]["escalationObservations"]
                        if scenario._escalation_observation_hash({key: value for key, value in row["information"].items()
                            if not key.startswith("complete")}) == owner["observationHash"])
                else:
                    root = source["candidate"]["projectObservations"][scenario._recorded_project_count]
                if len(root["legalActions"]) > 1:
                    if len(queries) + len(masters) + binding_queries + 3 > maximum_queries:
                        stop("maximum_queries")
                    info = {key: value for key, value in root["information"].items()
                            if not key.startswith("complete")}
                    key = (scenario._escalation_observation_hash(info) if escalation_root
                           else scenario._project_observation_hash(info))
                    if key != owner["observationHash"] or fixed_root is not None and key != fixed_root[0]:
                        raise ValueError("non-play policy probe differs from the observed root")
                    action = fixed_root[2] if fixed_root is not None else min(root["legalActions"])
                    floor_scenario = copy.copy(scenario)
                    floor_scenario.model = scenario.model.clone()
                    if escalation_root:
                        floor_scenario._observed_escalation_bindings = dict(scenario._observed_escalation_bindings)
                    else:
                        floor_scenario._observed_project_bindings = dict(scenario._observed_project_bindings)
                    bind = (floor_scenario.bind_native_escalation_observed_choice if escalation_root
                            else floor_scenario.bind_native_project_observed_choice)
                    binding = bind(information=info, action=action,
                        maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
                    queries.append({"branchModelHash": branch_hash, "purpose": f"{kind}_root_policy_binding",
                                    "receipt": binding})
                    report_progress("preparation", solver_status=binding.get("solverStatus"))
                    remaining()
                    if binding["status"] != "bound":
                        stop(f"{kind}_root_policy_binding_incomplete")
                    root_policy = (key, tuple(sorted(root["legalActions"])), action)
            floor_query = floor_scenario.find_native_round_margin_below(target=target,
                threshold=maximum + 1, maximum_seconds=remaining(),
                maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
            queries.append({"branchModelHash": branch_hash,
                            "purpose": "root_policy_path_floor" if root_policy else "universal_path_floor",
                            "receipt": floor_query})
            report_progress("preparation", solver_status=floor_query.get("solverStatus"))
            remaining()
            if floor_query["status"] == "witness_found" and floor_query["solverStatus"] == "optimal":
                floor = floor_query["candidate"]["utility"]
                if retain_path_floor:
                    path_floor = {"value": floor, "queryHash": floor_query["sha256"],
                        "branchModelHash": branch_hash, "policy": dict(fixed_actions)}
                    if root_policy is not None:
                        key, actions, action = root_policy
                        path_floor["policy"][key] = action
                        domains.setdefault(key, actions)
                bound_scenario = copy.copy(scenario)
                bound_scenario.model = scenario.model.clone()
                for variable, card in zip(bound_scenario.card_at_position,
                        floor_query["candidate"]["deckOrder"], strict=True):
                    bound_scenario.model.add(variable == card)
                native_world_policy = (opponent_ceiling_policy is not None
                    and "native_play_policy_id" in opponent_ceiling_policy and incumbent_lower is None)
                needed = 1 + (sum(opponent_ceiling_policy[key] is not None
                    for key in ("auction_action", "escalation_action")) if native_world_policy else 0)
                if len(queries) + len(masters) + binding_queries + needed > maximum_queries:
                    stop("maximum_queries")
                world_receipt = (bound_scenario.find_native_round_with_constant_policy_below(
                    target=target.other, policy_team=target.other, threshold=maximum + 1,
                    maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts,
                    cancel_check=cancel_check, **opponent_ceiling_policy) if native_world_policy else
                    bound_scenario.find_native_round_margin_below(target=target.other,
                        threshold=maximum + 1, maximum_seconds=remaining(),
                        maximum_conflicts=maximum_conflicts, cancel_check=cancel_check))
                world_query = (world_receipt.get("query") or {}) if native_world_policy else world_receipt
                if native_world_policy:
                    binding_queries += len(world_receipt.get("bindings", ()))
                queries.append({"branchModelHash": branch_hash, "purpose": (
                    "fixed_world_opponent_upper_bound" if native_world_policy else "adversarial_world_upper_bound"),
                    "sourceWorldQueryHash": floor_query["sha256"], "receipt": world_receipt})
                report_progress("preparation", solver_status=world_query.get("solverStatus"))
                remaining()
                if native_world_policy and world_receipt.get("status") != "queried":
                    stop(world_receipt.get("exhaustionReason") or "opponent_ceiling_not_certified")
                if world_query["status"] == "witness_found" and world_query["solverStatus"] == "optimal":
                    world_upper = -world_query["candidate"]["utility"]
                    world_source_hash = floor_query["sha256"]
                    if (floor < world_upper and scenario._complete_information_deck is None
                            and isinstance(source, Mapping) and incumbent_lower is None
                            and len(queries) + len(masters) + binding_queries < maximum_queries):
                        # Construct a separate adversarial witness, never alter
                        # the target's universe or its recorded deal/play line.
                        constructions = ("single_suit_hands_v1", "partner_without_projects_v1")
                        if scenario._native_play_root is not None:
                            constructions = tuple(reversed(constructions))
                        for construction in constructions:
                            if world_upper <= floor or len(queries) + len(masters) + binding_queries >= maximum_queries:
                                break
                            sampler = copy.copy(scenario)
                            sampler.model = scenario.model.clone()
                            if construction == "partner_without_projects_v1":
                                if scenario._native_escalation_root is not None:
                                    actor = next(row["actor"] for row in source["candidate"]["escalationObservations"]
                                        if scenario._escalation_observation_hash({key: value for key, value in row["information"].items()
                                            if not key.startswith("complete")}) == scenario._native_escalation_root["observationHash"])
                                else:
                                    observations, index = (source["candidate"]["playObservations"], scenario._recorded_play_count)
                                    if scenario._native_project_root is not None:
                                        observations, index = (source["candidate"]["projectObservations"], scenario._recorded_project_count)
                                    actor = observations[index]["actor"]
                                partner = Seat.parse(actor).partner
                                for project in sampler._native_round_fragment.projects.projects:
                                    if project.owner == partner:
                                        sampler.model.add(project.eligible == 0)
                            else:
                                suits = [sampler.model.new_int_var(0, len(Suit) - 1, f"bound_world_suit_{seat}")
                                    for seat in Seat]
                                sampler.model.add_all_different(suits)
                                for card, owner in zip(FULL_DECK, sampler.owner_of_card, strict=True):
                                    sampler.model.add_element(owner, suits, int(card.suit))
                            witness = sampler.find_native_round_margin_below(target=target, threshold=maximum + 1,
                                maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
                            queries.append({"branchModelHash": branch_hash, "purpose": "adversarial_world_witness",
                                "construction": construction, "receipt": witness})
                            report_progress("preparation", solver_status=witness.get("solverStatus"))
                            remaining()
                            if witness.get("status") != "witness_found":
                                continue
                            projects = {row["actor"]: row["selectedAction"]
                                for row in witness["candidate"]["projectObservations"]
                                if not row.get("authoredPrefix", False) and Seat.parse(row["actor"]).team == target.other}
                            bound_scenario = copy.copy(scenario)
                            bound_scenario.model = scenario.model.clone()
                            for variable, card in zip(bound_scenario.card_at_position,
                                    witness["candidate"]["deckOrder"], strict=True):
                                bound_scenario.model.add(variable == card)
                            policy_ids = ([opponent_ceiling_policy["native_play_policy_id"]]
                                if native_world_policy else sorted(
                                    NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS - {
                                        NATIVE_COMPACT_PARTNER_FEED_POLICY_ID, NATIVE_SECURE_CONTROL_POLICY_ID,
                                        NATIVE_GUARD_PROTECTION_POLICY_ID, NATIVE_CLOSED_TRICK_FEED_POLICY_ID}))
                            for policy_id in policy_ids:
                                if len(queries) + len(masters) + binding_queries + len(projects) + 1 > maximum_queries:
                                    break
                                template = {"auction_action": None, "escalation_action": None,
                                    "native_play_policy_id": policy_id}
                                bounded = bound_scenario.find_native_round_with_constant_policy_below(
                                    target=target.other, policy_team=target.other, threshold=maximum + 1,
                                    maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts,
                                    cancel_check=cancel_check, project_actions=projects, **template)
                                queries.append({"branchModelHash": branch_hash,
                                    "purpose": "fixed_world_opponent_upper_bound",
                                    "sourceWorldQueryHash": witness["sha256"], "receipt": bounded})
                                binding_queries += len(bounded.get("bindings", ()))
                                outcome = bounded.get("query") or {}
                                report_progress("preparation", solver_status=outcome.get("solverStatus"))
                                remaining()
                                if (bounded.get("status") == "queried" and outcome.get("status") == "witness_found"
                                        and outcome.get("solverStatus") == "optimal"
                                        and -outcome["candidate"]["utility"] < world_upper):
                                    world_receipt, world_query = bounded, outcome
                                    world_upper, world_source_hash = -outcome["candidate"]["utility"], witness["sha256"]
                                    opponent_ceiling_policy, native_world_policy = template, True
                                    scope["opponentCeilingProjectActions"] = projects
                                if world_upper <= floor:
                                    break
                    if native_world_policy:
                        scope["opponentCeilingPolicy"] = dict(opponent_ceiling_policy)
                        scope["opponentCeilingWorldQueryHash"] = world_source_hash
                        scope["utilityBounds"] = [-maximum, world_upper]
                        scope_hash = canonical_sha256(scope)
                    if floor > world_upper:
                        raise ValueError("native path floor exceeds compatible-world upper bound")
                    if floor == world_upper:
                        if root_policy is not None:
                            key, actions, action = root_policy
                            domains[key], policy[key] = actions, action
                        upper = floor
                        return evidence("scoped_bound_converged")
                    if (opponent_ceiling_policy is None or native_world_policy) and incumbent_lower is None:
                        # This compatible world (and policy, if supplied) gives
                        # the adversary an upper bound on every visible policy.
                        if not -maximum <= world_upper <= maximum:
                            raise ValueError("fixed-world cooperative ceiling exceeds utility bounds")
                        ceiling = upper = world_upper
                        fixed_world_ceiling = {"value": world_upper,
                            "queryHash": world_receipt["sha256"],
                            "sourceWorldQueryHash": world_source_hash}

        if tighten_utility_ceiling and fixed_world_ceiling is None:
            # A single observed branch gets a tighter native world/policy bound
            # below; optimizing its unrestricted ceiling first is redundant.
            observed_single = (len(ordered) == 1
                and ordered[0][1]._complete_information_deck is None and fixed_root is not None)
            use_incumbent_world = (observed_single and incumbent_lower is not None
                                   and opponent_ceiling_policy is None)
            world_ceiling_only = (observed_single and (opponent_ceiling_policy is not None or use_incumbent_world)
                or len(ordered) == 1 and opponent_ceiling_policy is not None
                and "native_play_policy_id" in opponent_ceiling_policy)
            feasible_branches = list(ordered) if world_ceiling_only else []
            branch_ceilings = [maximum] if world_ceiling_only else []
            for branch_hash, scenario in (() if world_ceiling_only else ordered):
                remaining()
                if len(queries) + len(masters) + binding_queries >= maximum_queries:
                    stop("maximum_queries")
                # Opposite-team minimization is a cooperative upper bound,
                # not minimax. The threshold includes the entire score domain.
                query = scenario.find_native_round_margin_below(target=target.other,
                    threshold=maximum + 1, maximum_seconds=remaining(),
                    maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
                queries.append({"branchModelHash": branch_hash, "purpose": "cooperative_upper_bound",
                                "receipt": query})
                report_progress("preparation", solver_status=query.get("solverStatus"))
                remaining()
                if query["status"] == "no_matching_outcome":
                    if query.get("solverStatus") != "infeasible":
                        raise ValueError("native branch exclusion lacks infeasibility proof")
                    infeasible_branch_proofs[branch_hash] = {"branchModelHash": branch_hash,
                        "infeasible": True, "queryHash": canonical_sha256(query)}
                    continue
                if query["status"] != "witness_found" or query["solverStatus"] != "optimal":
                    stop(query["exhaustionReason"] or "cooperative_bound_not_optimal")
                branch_ceilings.append(-query["candidate"]["utility"])
                feasible_branches.append((branch_hash, scenario))
            if not feasible_branches:
                raise ValueError("native utility ceiling found no feasible supplied branch")
            ordered = feasible_branches
            ceiling = upper = max(branch_ceilings)
            # Single-branch bound: the opponent policy is used only for this upper
            # bound. The adversary below still explores every legal response.
            if (opponent_ceiling_policy is not None or use_incumbent_world) and len(ordered) == 1:
                branch_hash, scenario = ordered[0]
                observed_world = scenario._complete_information_deck is None and fixed_root is not None
                world_query = queries[0]["receipt"]["query"] if use_incumbent_world else None
                ceiling_policy = opponent_ceiling_policy
                if use_incumbent_world:
                    observations = [{"information": row["information"], "action": row["selectedAction"]}
                        for row in world_query["candidate"]["escalationObservations"]
                        if Seat.parse(row["actor"]).team == target.other and not row.get("authoredPrefix", False)]
                    ceiling_policy = ({"auction_action": None, "escalation_action": None,
                        "escalation_observations": observations} if observations else None)
                needed = 1 + int(observed_world and not use_incumbent_world) + (
                    len(observations) if use_incumbent_world else
                    sum(opponent_ceiling_policy[key] is not None for key in ("auction_action", "escalation_action")))
                if len(queries) + len(masters) + binding_queries + needed > maximum_queries:
                    stop("maximum_queries")
                bound_scenario = scenario
                if observed_world:
                    if world_query is None:
                        world_query = scenario.find_native_round_margin_below(target=target,
                            threshold=maximum + 1, maximum_seconds=remaining(),
                            maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
                            optimize_margin=False)
                        queries.append({"branchModelHash": branch_hash, "purpose": "upper_bound_world_witness",
                            "receipt": world_query})
                    remaining()
                    if world_query["status"] != "witness_found":
                        stop(world_query["exhaustionReason"] or "world_witness_incomplete")
                    if not any(row["observationHash"] == fixed_root[0]
                            and row["selectedAction"] == fixed_root[2]
                            for row in world_query["candidate"]["auctionObservations"]):
                        stop("world_witness_root_mismatch")
                    bound_scenario = copy.copy(scenario)
                    bound_scenario.model = scenario.model.clone()
                    for variable, card in zip(bound_scenario.card_at_position,
                            world_query["candidate"]["deckOrder"], strict=True):
                        bound_scenario.model.add(variable == card)
                if ceiling_policy is not None:
                    query = bound_scenario.find_native_round_with_constant_policy_below(
                        target=target.other, policy_team=target.other, threshold=maximum + 1,
                        maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts,
                        cancel_check=cancel_check, **ceiling_policy)
                    outcome = query.get("query") or {}
                else:
                    query = bound_scenario.find_native_round_margin_below(target=target.other,
                        threshold=maximum + 1, maximum_seconds=remaining(),
                        maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
                    outcome = query
                queries.append({"branchModelHash": branch_hash, "purpose": (
                    "incumbent_world_opponent_upper_bound" if use_incumbent_world else
                    "fixed_world_opponent_upper_bound" if observed_world else "opponent_policy_upper_bound"),
                    "receipt": query})
                binding_queries += len(query.get("bindings", []))
                remaining()
                if ceiling_policy is not None and "native_play_policy_id" in ceiling_policy:
                    if outcome.get("solverStatus") == "infeasible":
                        raise ValueError("native opponent policy feasibility disagreement")
                    if (query.get("status") != "queried" or outcome.get("status") != "witness_found"
                            or outcome.get("solverStatus") != "optimal"):
                        stop(query.get("exhaustionReason") or "opponent_ceiling_not_certified")
                if query.get("status") in {"queried", "witness_found"} and outcome.get("solverStatus") == "optimal":
                    ceiling = upper = min(ceiling, -outcome["candidate"]["utility"])
                    scope["opponentCeilingPolicy"] = dict(ceiling_policy or {})
                    if world_query is not None:
                        scope["opponentCeilingWorldQueryHash"] = world_query["sha256"]
            scope["utilityBounds"] = [-maximum, ceiling]
            scope_hash = canonical_sha256(scope)
        elif fixed_world_ceiling is not None:
            scope["utilityBounds"] = [-maximum, ceiling]
            if opponent_ceiling_policy is not None:
                scope["opponentCeilingPolicy"] = dict(opponent_ceiling_policy)
                scope["opponentCeilingWorldQueryHash"] = fixed_world_ceiling["sourceWorldQueryHash"]
            else:
                scope["fixedWorldCooperativeCeiling"] = fixed_world_ceiling
            scope_hash = canonical_sha256(scope)

        # Keep scope identities in hash order, but seek counterexamples in the
        # smallest native models first. A hash is not a forecast of solve cost.
        refinement_branches = sorted(ordered, key=lambda row: (
            len(row[1].model.proto.constraints), len(row[1].model.proto.variables), row[0]))
        for _ in range(maximum_iterations):
            remaining()
            if len(queries) + len(masters) + binding_queries >= maximum_queries:
                stop("maximum_queries")
            if not masters and incumbent_lower is not None:
                # These witnesses were solved above in this execution. A policy
                # restriction preserves witness legality, not absence of outcomes.
                for entry in queries:
                    if entry.get("purpose") != "incumbent_lower_bound":
                        continue
                    owned_query = entry["receipt"]
                    query = owned_query["query"]
                    observation_hints[entry["branchModelHash"]] = owned_query
                    candidate = query["candidate"]
                    if not -maximum <= candidate["utility"] < upper:
                        continue
                    rows = [row for kind in ("auction", "escalation", "project", "play")
                        for row in candidate[kind + "Observations"]
                        if not row.get("authoredPrefix", False) and Seat.parse(row["actor"]).team == target]
                    if any(row["observationHash"] in fixed_actions
                            and row["selectedAction"] != fixed_actions[row["observationHash"]] for row in rows):
                        raise ValueError("native incumbent witness conflicts with fixed policy")
                    for row in rows:
                        key, actions = row["observationHash"], tuple(row["legalActions"])
                        if row["selectedAction"] not in actions or sorted(domains.setdefault(key, actions)) != sorted(actions):
                            raise ValueError("native incumbent witness has conflicting legal domains")
                    if len(domains) > maximum_observations:
                        stop("maximum_observations")
                    cut = _CounterexampleCut(candidate["utility"], tuple(sorted({
                        (row["observationHash"], row["selectedAction"]) for row in rows})))
                    if cut.sha256 not in cut_hashes:
                        cuts.append(cut)
                        cut_hashes.add(cut.sha256)
                        incumbent_cut_queries[cut.sha256] = {"branchModelHash": entry["branchModelHash"],
                            "ownedQueryHash": owned_query["sha256"], "queryHash": query["sha256"]}
            master = service._solve_master(scope_hash, "policy_bounds",
                _Catalog({("target", key): actions for key, actions in domains.items()},
                         len(domains), (min(-maximum, ceiling), ceiling)), cuts, deadline, cancel_check,
                canonicalize_policy=False,
                maximum_conflicts=maximum_conflicts, fixed_actions=fixed_actions,
                **({"preferred_actions": {key: action for key, action in
                    {**witnessed_actions, **policy, **minimax_preferences}.items() if key in domains}}
                   if fallback is not None else {}))
            masters.append({"modelHash": master.model_hash, "status": master.status})
            report_progress("master", solver_status=master.status)
            if required_threshold is not None:
                masters[-1]["searchValue"] = master.value if master.status == "optimal" else None
            if tighten_utility_ceiling:
                masters[-1]["upperBound"] = master.value if master.status == "optimal" else None
            remaining()
            if master.status != "optimal":
                stop("master_not_optimal")
            if path_floor is not None:
                if path_floor["value"] > master.value:
                    raise ValueError("native path floor exceeds the strategy upper bound")
                if path_floor["value"] == master.value:
                    policy, upper = dict(path_floor["policy"]), master.value
                    return evidence("scoped_bound_converged")
            policy, upper = dict(master.policy), master.value
            if fallback is not None:
                # Witnesses are compression hints, not action certificates. The
                # authoritative policy is the fallback plus these deviations;
                # every branch must freshly prove that exact effective strategy.
                policy = {key: action for key, action in policy.items()
                          if fallback_actions.get(key) != action}
            if required_threshold is not None and upper != required_threshold:
                if upper > required_threshold:
                    raise ValueError("native threshold master exceeded its search ceiling")
                # Native counterexample cuts cover every policy in this scope;
                # an optimal master below the request is refutation, not exhaustion.
                # Global admission still requires owned-source reconstruction.
                final_branch_proofs.clear()
                return evidence("scoped_strategy_threshold_refuted")
            if incumbent_lower is not None and incumbent_lower > upper:
                raise ValueError("native incumbent lower bound exceeds the strategy upper bound")
            refined = False
            final_branch_proofs.clear()
            prior_cut_hashes = set(cut_hashes)
            for branch_hash, scenario in refinement_branches:
                remaining()
                cached = unconditional_bounds.get(branch_hash)
                if cached is not None and upper <= cached["provedThreshold"]:
                    reused_bounds.append({"branchModelHash": branch_hash, "requestedThreshold": upper, **cached})
                    final_branch_proofs.append({"branchModelHash": branch_hash,
                        "refinementHash": cached["refinementHash"], "policyIndependent": True})
                    continue
                if len(queries) + len(masters) + binding_queries >= maximum_queries:
                    stop("maximum_queries")
                try:
                    receipt = self._refine_symbolic_round_policy(scenario, target, policy, threshold=upper,
                        maximum_queries=maximum_queries - len(queries) - len(masters) - binding_queries,
                        maximum_observations=maximum_observations,
                        maximum_conflicts=maximum_conflicts, maximum_seconds=remaining(), cancel_check=cancel_check,
                        shorten_counterexamples=shorten_counterexamples,
                        minimax_budget=minimax_budget,
                        optimize_prefix_bounds=shorten_counterexamples and required_threshold is None,
                        observation_hint=None if fallback is not None else observation_hints.get(branch_hash),
                        _outcome_cache=outcome_cache,
                        _legality_cache=legality_cache,
                        **({"_play_fallback_id": _play_fallback_id, "_play_contexts": play_contexts}
                           if fallback is not None else {}),
                        **({"_search_hint": search_hints.get(branch_hash)} if fallback is not None else {}),
                        # Bound witnesses also describe usable information states;
                        # only their outcomes remain conditional on the old policy.
                        _auction_contexts={row["observationHash"]: row
                            for entry in queries
                            for hint in (entry["receipt"].get("query") or entry["receipt"],)
                            if hint.get("status") == "witness_found"
                            for row in (hint.get("candidate") or {}).get("auctionObservations", ())
                            if not row.get("authoredPrefix", False)
                            and Seat.parse(row["actor"]).team == target
                            and row["observationHash"] in policy
                            and row["selectedAction"] != policy[row["observationHash"]]},
                        progress_callback=lambda row: report_progress(row["phase"],
                            row["queryCount"], row["bindingCount"], row.get("solverStatus")))
                except InformationSetCompilationStopped as exc:
                    unfinished = exc.partial_receipt.get("evidence", {})
                    queries.extend({"branchModelHash": branch_hash, "receipt": query}
                        for query in unfinished.get("queries", ()))
                    binding_queries += len(unfinished.get("bindings", ()))
                    stop(exc.reason, exc.partial_receipt)
                queries.extend({"branchModelHash": branch_hash, "receipt": query} for query in receipt["queries"])
                binding_queries += len(receipt["bindings"])
                remaining()
                for query in receipt["queries"]:
                    if (fallback is not None and scenario._native_round_fragment is not None
                            and query.get("status") == "witness_found"):
                        search_hints[branch_hash] = query
                    if (query.get("status") == "witness_found"
                            and query.get("baseModelHash") == branch_hash):
                        observation_hints[branch_hash] = query
                        break
                for key, actions in receipt["domains"].items():
                    actions = tuple(actions)
                    if sorted(domains.setdefault(key, actions)) != sorted(actions):
                        raise ValueError("symbolic policy observation has conflicting legal domains")
                if len(domains) > maximum_observations:
                    stop("maximum_observations", receipt)
                if fallback is not None:
                    for query in receipt["queries"]:
                        if (query.get("kind") != "native_counterexample_prefix_minimax_v1"
                                or query.get("status") != "upper_bound_complete" or query["upperBound"] < upper):
                            continue
                        row = receipt["candidate"]["playObservations"][query["prefixPlayCount"]]
                        action = query["proof"]["selectedAction"]
                        if Seat.parse(row["actor"]).team == target and action in domains.get(row["observationHash"], ()):
                            # One-world best moves guide search, never certify a
                            # visible policy. The master preserves its robust
                            # objective; every resulting policy needs fresh proof.
                            minimax_preferences[row["observationHash"]] = action
                    for query in receipt["queries"]:
                        candidate = query.get("candidate") or {}
                        for row in (row for kind in ("auction", "escalation", "project", "play")
                                    for row in candidate.get(kind + "Observations", ())):
                            if row.get("authoredPrefix", False) or Seat.parse(row["actor"]).team != target:
                                continue
                            witnessed_actions.setdefault(row["observationHash"], row["selectedAction"])
                        for row in candidate.get("playObservations", ()):
                            if row.get("authoredPrefix", False) or Seat.parse(row["actor"]).team != target:
                                continue
                            key, info = row["observationHash"], row["information"]
                            if play_contexts.setdefault(key, info) != info:
                                raise ValueError("guarded strategy play context changed")
                            if not any(binding["observationHash"] == key
                                       for binding in query.get("playPolicyBindings", ())):
                                fallback_actions.setdefault(key, row["selectedAction"])
                    if len(play_contexts) > maximum_observations:
                        stop("maximum_observations", receipt)
                if receipt["status"] == "no_adverse_round_in_branch":
                    terminal = receipt["queries"][-1] if receipt["queries"] else {}
                    inventories = [terminal.get(kind + "PolicyBindings")
                                   for kind in ("auction", "escalation", "project", "play")]
                    if (receipt.get("sha256") != canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})
                            or receipt.get("rulesHash") != self.rules.sha256
                            or receipt.get("scope") != scenario._native_round_query_context()[0]
                            or receipt.get("targetTeam") != target.name.lower()
                            or receipt.get("sourceModelHash") != branch_hash
                            or receipt.get("policyHash") != canonical_sha256(
                                {"fallback": fallback, "policy": policy} if fallback is not None else policy)
                            or type(receipt.get("threshold")) is not int or receipt["threshold"] != upper
                            or terminal.get("sha256") != canonical_sha256({k: v for k, v in terminal.items() if k != "sha256"})
                            or terminal.get("status") != "no_matching_outcome"
                            or terminal.get("solverStatus") != "infeasible"
                            or terminal.get("scope") != receipt["scope"]
                            or terminal.get("targetTeam") != target.name.lower()
                            or type(terminal.get("threshold")) is not int or terminal["threshold"] != upper
                            or terminal.get("baseModelHash") != receipt.get("boundModelHash")
                            or not isinstance(receipt.get("boundModelHash"), str)
                            or not receipt["boundModelHash"]
                            or any(not isinstance(rows, list) for rows in inventories)
                            or sorted(canonical_sha256(row) for rows in inventories for row in rows)
                                != sorted(canonical_sha256(row) for row in receipt["bindings"])
                            or not receipt["bindings"] and fallback is None and receipt["boundModelHash"] != branch_hash
                            or any(row.get("status") != "bound" for row in receipt["bindings"])):
                        raise ValueError("symbolic branch absence evidence mismatch")
                    # Retain the full receipt through references to queries
                    # already owned by this result, without duplicating payloads.
                    retained_refinements[receipt["sha256"]] = {
                        "core": {k: v for k, v in receipt.items() if k not in {"queries", "sha256"}},
                        "queryHashes": [canonical_sha256(row) for row in receipt["queries"]]}
                    final_branch_proofs.append({"branchModelHash": branch_hash,
                        "refinementHash": receipt["sha256"], "policyIndependent": not receipt["bindings"] and fallback is None})
                    if not receipt["bindings"] and fallback is None:
                        unconditional_bounds[branch_hash] = {"provedThreshold": upper,
                            "refinementHash": receipt["sha256"],
                            "queryHash": canonical_sha256(receipt["queries"][-1])}
                    continue
                if receipt["status"] not in {"policy_extension_required", "policy_counterexample_in_branch"}:
                    raise ValueError("symbolic policy refinement returned an unsupported status")
                candidate = receipt["candidate"]
                prefix_bound = None
                if receipt.get("prefixBoundQueryHash") is not None:
                    prefix_bound = next((query for query in receipt["queries"]
                        if query.get("sha256") == receipt["prefixBoundQueryHash"]), None)
                    if prefix_bound is None:
                        raise ValueError("counterexample prefix bound receipt is missing")
                actions = tuple(sorted({(row["observationHash"], row["selectedAction"])
                    for kind in ("auction", "escalation", "project", "play")
                    for row in candidate[kind + "Observations"]
                    if not row.get("authoredPrefix", False) and Seat.parse(row["actor"]).team == target}))
                cut_utility = candidate["utility"]
                if prefix_bound is not None:
                    if prefix_bound["status"] != "upper_bound_complete":
                        raise ValueError("counterexample prefix bound is incomplete")
                    actions = tuple(tuple(row) for row in prefix_bound["targetActions"])
                    cut_utility = prefix_bound["upperBound"]
                if not -maximum <= cut_utility < upper:
                    raise ValueError("symbolic counterexample utility is outside its native bounds")
                cut = _CounterexampleCut(cut_utility, actions)
                if cut.sha256 in prior_cut_hashes:
                    raise ValueError("symbolic policy repeated a counterexample cut")
                if cut.sha256 in cut_hashes:
                    continue
                cut_hashes.add(cut.sha256)
                cuts.append(cut)
                if prefix_bound is not None:
                    cut_bound_queries[cut.sha256] = prefix_bound["sha256"]
                refined = True
                # Refine before querying more branches with a defeated policy.
                break
            if not refined:
                if [row["branchModelHash"] for row in final_branch_proofs] != [key for key, _ in refinement_branches]:
                    raise ValueError("symbolic final policy lacks exact branch coverage")
                final_branch_proofs.extend(infeasible_branch_proofs.values())
                final_branch_proofs.sort(key=lambda row: row["branchModelHash"])
                if [row["branchModelHash"] for row in final_branch_proofs] != [row["modelHash"] for row in scope["branches"]]:
                    raise ValueError("symbolic final policy omits a prepared branch")
                return evidence("scoped_strategy_threshold_complete" if required_threshold is not None
                                else "scoped_bound_converged")
        stop("maximum_iterations")

    def _bound_native_counterexample_prefix(self, scenario, query, target, *, prefix_play_count,
            maximum_seconds, maximum_conflicts, cancel_check=None, required_threshold=None,
            minimax_budget=None, _normalizer=None):
        """Bound a witnessed prefix with an owned native suffix certificate.

        A fresh normalized model deliberately drops construction suffix filters.
        Matching target prefix choices lets an adversary follow the witnessed
        prefix; this bound then covers even cooperative target/opponent suffixes.
        No action is selected using the auxiliary complete-information world.
        Optional minimax covers one freshly normalized witness suffix only,
        even when its owning query ranges over hidden worlds. It is an upper
        bound for cutting that witnessed prefix, never an information-set
        strategy certificate, and is never enabled implicitly.
        """
        if (not self.native_cp_sat or not isinstance(scenario, OrToolsScenarioModelV1)
                or scenario.rules.sha256 != self.rules.sha256
                or not isinstance(target, Team)):
            raise ValueError("prefix bound requires native ownership and a target team")
        if _normalizer is not None and (not isinstance(_normalizer, OrToolsExactStateNormalizerV1)
                or not _normalizer.native_cp_sat or _normalizer.rules.sha256 != self.rules.sha256):
            raise ValueError("prefix bound requires a matching native normalizer")
        if (type(prefix_play_count) is not int or not len(Seat) <= prefix_play_count < len(FULL_DECK)
                or type(maximum_conflicts) is not int or maximum_conflicts < 1
                or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
                or maximum_seconds <= 0):
            raise ValueError("prefix bound requires a post-project prefix and positive finite budgets")
        deadline = time.monotonic() + maximum_seconds
        cpu_started = time.process_time()
        if required_threshold is not None and type(required_threshold) is not int:
            raise ValueError("prefix threshold must be an integer")
        if minimax_budget is not None:
            if not isinstance(minimax_budget, ScenarioProofBudgetV1) or required_threshold is not None:
                raise ValueError("minimax prefix bound requires a typed proof budget without a threshold query")
            deadline = min(deadline, time.monotonic() + min(
                minimax_budget.maximum_wall_seconds, minimax_budget.maximum_cpu_seconds))
        scope, _ = scenario._native_round_query_context()
        if (not isinstance(query, Mapping) or query.get("status") != "witness_found"
                or query.get("sha256") != canonical_sha256({key: value for key, value in query.items() if key != "sha256"})
                or query.get("scope") != scope or query.get("targetTeam") != target.name.lower()
                or query.get("baseModelHash") != scenario._model_hash(search_branching="automatic")):
            raise ValueError("prefix bound witness ownership or checksum mismatch")
        candidate = query["candidate"]
        if (len(candidate["play"]["plays"]) != len(FULL_DECK)
                or (scenario._complete_information_deck is not None
                    and tuple(candidate["deckOrder"]) != scenario._complete_information_deck)):
            raise ValueError("prefix bound witness must cover its complete owned deal")
        core = {"schemaVersion": 1, "kind": "native_counterexample_prefix_bound_v1",
            "scope": "fixed_world_cooperative_upper_bound", "strategicCoverage": False,
            "rulesHash": self.rules.sha256, "sourceQueryHash": query["sha256"],
            "targetTeam": target.name.lower(), "utility": "awarded_point_margin", "prefixPlayCount": prefix_play_count,
            "limits": {"seconds": maximum_seconds, "conflicts": maximum_conflicts}}
        if required_threshold is not None:
            core.update(kind="native_counterexample_prefix_threshold_v1",
                requiredThreshold=required_threshold, complementTargetTeam=target.other.name.lower(),
                complementThreshold=1 - required_threshold, queryMode="feasibility_v1")

        def resource_cancelled():
            return (cancel_check is not None and cancel_check()) or (
                minimax_budget is not None
                and time.process_time() - cpu_started >= minimax_budget.maximum_cpu_seconds)

        active_cancel = resource_cancelled if minimax_budget is not None else cancel_check

        def remaining():
            if cancel_check is not None and cancel_check():
                raise PlayTailSolveStopped("cancelled")
            if minimax_budget is not None and time.process_time() - cpu_started >= minimax_budget.maximum_cpu_seconds:
                raise PlayTailSolveStopped("maximum_cpu_seconds")
            if time.monotonic() >= deadline:
                raise PlayTailSolveStopped("maximum_seconds")
            return deadline - time.monotonic()

        try:
            remaining()
            prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": candidate["deckOrder"],
                "dealer": scenario.dealer.name.lower(), "matchScore": scope["matchScore"],
                "auctionActions": [{key: row[key] for key in ("actor", "action", "trumpSuit") if row.get(key) is not None}
                    for row in candidate["auctionActions"]],
                "escalationActions": [{key: row[key] for key in ("actor", "action", "requestedMode") if row.get(key) is not None}
                    for row in candidate["escalation"]["observation"]["actions"]],
                "projectActions": [ProjectDeclarationActionV1.build(row["selectedAction"],
                    actor=Seat.parse(row["actor"])).to_dict() for row in candidate["projectDecisions"]],
                "projectDeclarations": [], "playActions": candidate["play"]["plays"][:prefix_play_count]})
            core["prefixHash"] = prefix.sha256
            normalizer = _normalizer if _normalizer is not None else OrToolsExactStateNormalizerV1(self.rules, native_cp_sat=True)
            exact = normalizer.normalize(prefix,
                decision_point={"kind": "play"}, maximum_seconds=remaining(), cancel_check=active_cancel)
            core["exactState"] = exact.to_dict()
            if minimax_budget is not None:
                from .ortools_proof import OrToolsScenarioGraphProofV1

                core.update(kind="native_counterexample_prefix_minimax_v1",
                    scope="complete_state_minimax_upper_bound", proofBudgetHash=minimax_budget.sha256)
                request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "exact",
                    "visibility": "complete_state", "claim": "outcome_reachable",
                    "targetTeam": target.name.lower(), "utility": "awarded_point_margin",
                    "successThreshold": 0, "budgetProfile": minimax_budget.profile_id})
                compilation = self.graph_compiler.compile(exact, request,
                    expected_rules_hash=self.rules.sha256,
                    maximum_nodes=min(minimax_budget.maximum_nodes, minimax_budget.maximum_state_space),
                    maximum_depth=minimax_budget.maximum_depth, maximum_seconds=remaining(),
                    maximum_conflicts=maximum_conflicts,
                    cancelled=active_cancel)
                core["graphEvidence"] = {key: value for key, value in compilation.receipt.to_dict().items()
                    if key not in {"elapsedSeconds", "sha256"}}
                if not compilation.complete:
                    core.update(status="incomplete")
                    return {**core, "sha256": canonical_sha256(core)}
                graph = compilation.graph
                if len(canonical_json(graph.to_dict()).encode("utf-8")) > minimax_budget.maximum_cache_bytes:
                    core.update(status="incomplete", exhaustionReason="maximum_cache_bytes")
                    return {**core, "sha256": canonical_sha256(core)}
                proof = OrToolsScenarioGraphProofV1().prove(graph, request,
                    maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts,
                    cancel_check=active_cancel, native_action_quality=True)
                core.update(graphHash=graph.sha256, proof=proof.to_dict(), proofRequest=request.to_dict())
                # A disproved win claim still has exact values on this complete
                # graph. Keep claim truth separate from value-proof completion.
                if proof.completion_status not in {"exact_complete", "proof_failed"} or proof.solver_status != "optimal":
                    core.update(status="incomplete")
                    return {**core, "sha256": canonical_sha256(core)}
                # The native action-quality model selects the optimizing action
                # for the actual root team; extraction does not recompute minimax.
                value = next(row.value for row in proof.root_actions if row.action_code == proof.selected_action)
                core.update(status="upper_bound_complete", upperBound=value, modelHash=proof.model_hash,
                    targetActions=[list(row) for row in sorted({(row["observationHash"], row["selectedAction"])
                        for kind in ("auction", "escalation", "project", "play")
                        for row in (candidate[kind + "Observations"][:prefix_play_count]
                                    if kind == "play" else candidate[kind + "Observations"])
                        if not row.get("authoredPrefix", False) and Seat.parse(row["actor"]).team == target})])
                remaining()
                return {**core, "sha256": canonical_sha256(core)}
            start = self.graph_compiler._parse_start(exact, expected_rules_hash=self.rules.sha256)
            oracle = self.graph_compiler._path_oracle(start,
                target=target if required_threshold is None else target.other,
                utility="awarded_point_margin", cancel_check=cancel_check)
            tail, value, model_hash = oracle.find_path_below(
                oracle.query_utility_bounds()[1] + 1 if required_threshold is None else 1 - required_threshold,
                conditional_actions={}, maximum_seconds=remaining(), cancelled=cancel_check,
                maximum_conflicts=maximum_conflicts,
                **({"maximize_utility": True} if required_threshold is None else {"feasibility_only": True}))
            if tail is None and required_threshold is None:
                raise ValueError("normalized counterexample prefix has no legal terminal continuation")
            if required_threshold is not None:
                core.update(modelHash=model_hash, baseModelHash=oracle.model_hash,
                    solverStatus="infeasible" if tail is None else tail.solver_status,
                    status="upper_bound_complete" if tail is None else "threshold_reachable",
                    nativePlay=None if tail is None else tail.to_dict())
                if tail is not None:
                    remaining()
                    return {**core, "sha256": canonical_sha256(core)}
                value = required_threshold - 1
            core.update(status="upper_bound_complete", upperBound=value, modelHash=model_hash,
                baseModelHash=oracle.model_hash, nativePlay=None if tail is None else tail.to_dict(),
                targetActions=[list(row) for row in sorted({(row["observationHash"], row["selectedAction"])
                    for kind in ("auction", "escalation", "project", "play")
                    for row in (candidate[kind + "Observations"][:prefix_play_count]
                                if kind == "play" else candidate[kind + "Observations"])
                    if not row.get("authoredPrefix", False) and Seat.parse(row["actor"]).team == target})])
            remaining()
        except InterruptedError:
            core.pop("upperBound", None)
            core.pop("targetActions", None)
            core.update(status="incomplete", exhaustionReason=("maximum_cpu_seconds"
                if minimax_budget is not None and time.process_time() - cpu_started >= minimax_budget.maximum_cpu_seconds
                else "cancelled"))
        except (ExactNormalizationStopped, PlayTailSolveStopped) as exc:
            core.pop("upperBound", None)
            core.pop("targetActions", None)
            core.update(status="incomplete", cause=exc.partial_receipt)
        return {**core, "sha256": canonical_sha256(core)}

    def _refine_symbolic_round_policy(self, scenario, target, policy, *, threshold,
            maximum_queries, maximum_observations, maximum_conflicts, maximum_seconds,
            cancel_check=None, shorten_counterexamples=False, observation_hint=None,
            _outcome_cache=None, _legality_cache=None, progress_callback=None, observation_hints=None,
            _auction_contexts=None, optimize_prefix_bounds=False,
            _play_fallback_id=None, _play_contexts=None, minimax_budget=None, _search_hint=None):
        """Refine all observed choices in a deal query, not a proof.

        A compliant adverse line is a policy counterexample in this branch.
        A branch with no adverse line is not exhaustive auction or
        hidden-world certification. Never convert either into exact admission.
        """
        if (not self.native_cp_sat or not isinstance(scenario, OrToolsScenarioModelV1)
                or scenario.rules.sha256 != self.rules.sha256):
            raise ValueError("symbolic round refinement requires an owned native round model")
        if type(shorten_counterexamples) is not bool or type(optimize_prefix_bounds) is not bool:
            raise ValueError("counterexample shortening selection must be boolean")
        if minimax_budget is not None and (not isinstance(minimax_budget, ScenarioProofBudgetV1)
                or not shorten_counterexamples):
            raise ValueError("minimax shortening requires a typed budget and explicit shortening")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("native refinement progress callback must be callable")
        branch_scope, _ = scenario._native_round_query_context()
        if not isinstance(target, Team) or type(threshold) is not int:
            raise ValueError("symbolic round refinement requires a team and integer threshold")
        if (any(type(value) is not int or value < 1 for value in
                (maximum_queries, maximum_observations, maximum_conflicts))
                or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
                or maximum_seconds <= 0):
            raise ValueError("symbolic round refinement requires positive finite budgets")
        if (not isinstance(policy, Mapping) or any(type(key) is not str or len(key) != 64
                or any(char not in "0123456789abcdef" for char in key) or type(action) is not str
                for key, action in policy.items())):
            raise ValueError("symbolic round policy requires observation hashes and action codes")
        if len(policy) > maximum_observations:
            raise ValueError("symbolic round policy exceeds its observation budget")
        if _auction_contexts is not None and (not isinstance(_auction_contexts, Mapping)
                or len(_auction_contexts) > maximum_observations):
            raise ValueError("symbolic auction contexts exceed their observation budget")
        fallback, overrides = None, {}
        if _play_fallback_id is not None:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            fallback = native_observer_safe_policy_identity_v1(_play_fallback_id, target)
            if (not isinstance(_play_contexts, Mapping) or len(_play_contexts) > maximum_observations
                    or observation_hint is not None or observation_hints is not None):
                raise ValueError("guarded refinement requires bounded contexts and fresh queries")
            if scenario._native_round_fragment is not None:
                for key, info in sorted(_play_contexts.items()):
                    if key not in policy or scenario._round_observation_hash("play", info) != key:
                        continue  # A different contract/world is a different information set.
                    normalized, _ = scenario._normalize_play_information(info)
                    if (normalized != info or Seat.parse(info["actor"]).team != target
                            or len(info["publicPlays"]) < scenario._recorded_play_count):
                        raise ValueError("guarded refinement play context ownership mismatch")
                    overrides[key] = {"information": info, "action": policy[key]}
        elif _play_contexts is not None:
            raise ValueError("play contexts require a native fallback")
        if _search_hint is not None and (fallback is None or not isinstance(_search_hint, Mapping)
                or _search_hint.get("status") != "witness_found"
                or _search_hint.get("scope") != branch_scope
                or _search_hint.get("targetTeam") != target.name.lower()
                or _search_hint.get("sha256") != canonical_sha256(
                    {key: value for key, value in _search_hint.items() if key != "sha256"})
                or not isinstance(_search_hint.get("candidate"), Mapping)):
            raise ValueError("guarded search hint ownership or checksum mismatch")
        hint_deck, hint_plays = None, None
        if _search_hint is not None:
            try:
                hint_deck = _search_hint["candidate"]["deckOrder"]
                hint_plays = [Card.parse(row["card"]).id for row in _search_hint["candidate"]["play"]["plays"]]
                cards = set(self.rules.structural_domain.card_ids)
                if (not isinstance(hint_deck, (tuple, list)) or len(hint_deck) != len(cards)
                        or any(type(card) is not int for card in hint_deck)
                        or set(hint_deck) != cards or len(hint_plays) != len(cards) or set(hint_plays) != cards):
                    raise ValueError("invalid card sequence")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("guarded search hint requires complete legal card sequences") from exc
        deadline = time.monotonic() + maximum_seconds
        hint_hash = observation_hint.get("sha256") if isinstance(observation_hint, Mapping) else None
        # The source is immutable throughout this call; only its owned clone is bound.
        source_model_hash = scenario._model_hash(search_branching="automatic")
        if isinstance(observation_hint, Mapping) and observation_hint.get("kind") == "native_constant_policy_round_query_v1":
            parent = observation_hint
            observation_hint = parent.get("query")
            if (parent.get("status") != "queried"
                    or parent.get("sourceModelHash") != source_model_hash
                    or parent.get("targetTeam") != target.name.lower() or parent.get("scope") != branch_scope
                    or parent.get("sha256") != canonical_sha256({key: value for key, value in parent.items() if key != "sha256"})
                    or not isinstance(observation_hint, Mapping)
                    or not isinstance(parent.get("policy"), Mapping)
                    or parent.get("policy", {}).get("playPolicyModelHash") != observation_hint.get("baseModelHash")):
                raise ValueError("symbolic observation hint ownership or checksum mismatch")
            hint_model_hash = observation_hint["baseModelHash"]
        else:
            hint_model_hash = source_model_hash
        if observation_hint is not None:
            if (not isinstance(observation_hint, Mapping)
                    or observation_hint.get("status") != "witness_found"
                    or observation_hint.get("targetTeam") != target.name.lower()
                    or observation_hint.get("scope") != branch_scope
                    or observation_hint.get("baseModelHash") != hint_model_hash
                    or observation_hint.get("sha256") != canonical_sha256(
                        {key: value for key, value in observation_hint.items() if key != "sha256"})):
                raise ValueError("symbolic observation hint ownership or checksum mismatch")
        if observation_hints is not None and (observation_hint is not None
                or not isinstance(observation_hints, (tuple, list)) or len(observation_hints) > maximum_queries
                or any(not isinstance(hint, Mapping) or hint.get("status") != "witness_found"
                    or hint.get("targetTeam") != target.name.lower() or hint.get("scope") != branch_scope
                    or type(hint.get("baseModelHash")) is not str or not hint["baseModelHash"]
                    or hint.get("sha256") != canonical_sha256({k: v for k, v in hint.items() if k != "sha256"})
                    for hint in observation_hints)):
            raise ValueError("symbolic observation hint chain ownership or checksum mismatch")
        pending_hints, used_hint_hashes = list(observation_hints or ()), []
        # One fixed-policy query owns its mutable constraints; later master
        # iterations must start from the unchanged construction model.
        owned = copy.copy(scenario)
        owned.model = scenario.model.clone()
        owned._native_policy_legality_cache = _legality_cache
        owned._observed_auction_bindings = dict(scenario._observed_auction_bindings)
        owned._observed_escalation_bindings = dict(scenario._observed_escalation_bindings)
        owned._observed_project_bindings = dict(scenario._observed_project_bindings)
        owned._observed_play_bindings = dict(scenario._observed_play_bindings)
        domains, queries, bindings = {}, [], []
        bound_model_hash = source_model_hash
        prefix_bound_hash = None
        deck_hint, play_card_hint = hint_deck, hint_plays
        fallback_binding = None

        def report_progress(phase, solver_status=None):
            if progress_callback is None:
                return
            try:
                progress_callback({"phase": phase, "queryCount": len(queries),
                    "bindingCount": len(bindings),
                    **({"solverStatus": solver_status} if solver_status is not None else {})})
            except Exception:
                pass  # Operational progress never controls a proof.

        def evidence(status, candidate=None):
            core = {"schemaVersion": 1, "kind": ("native_guarded_round_refinement_v1" if fallback is not None
                else "native_symbolic_round_refinement_v1"),
                "status": status, "rulesHash": self.rules.sha256, "scope": dict(branch_scope),
                "sourceModelHash": source_model_hash,
                "boundModelHash": bound_model_hash,
                "targetTeam": target.name.lower(), "threshold": threshold,
                "policyHash": canonical_sha256({"fallback": fallback, "policy": dict(policy)}
                    if fallback is not None else dict(policy)), "strategicCoverage": False,
                **({"playFallback": fallback, "playOverrides": overrides,
                    "fallbackBinding": fallback_binding} if fallback is not None else {}),
                "domains": dict(domains), "queries": list(queries), "bindings": list(bindings),
                "queriesUsed": len(queries) + len(bindings),
                "candidate": candidate,
                **({"minimaxProofBudgetHash": minimax_budget.sha256} if minimax_budget is not None else {}),
                **({"observationHintHash": hint_hash} if hint_hash is not None else {}),
                **({"observationHintHashes": list(used_hint_hashes)} if observation_hints is not None else {}),
                **({"prefixBoundQueryHash": prefix_bound_hash} if prefix_bound_hash is not None else {})}
            return {**core, "sha256": canonical_sha256(core)}

        def remaining():
            reason = "cancelled" if cancel_check is not None and cancel_check() else (
                "maximum_seconds" if time.monotonic() >= deadline else None)
            if reason is not None:
                raise InformationSetCompilationStopped(reason, stage="native_symbolic_round_refinement",
                    evidence=evidence("incomplete"))
            return deadline - time.monotonic()

        if fallback is not None and owned._native_round_fragment is not None:
            before = hashlib.sha256(str(owned.model.proto).encode("utf-8")).hexdigest()
            batch, matches = owned._bind_native_play_overrides(
                tuple(overrides.values())[:maximum_queries], maximum_seconds=remaining(),
                maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
            bindings.extend(batch)
            for binding in batch:
                remaining()
                if binding["status"] != "bound":
                    raise InformationSetCompilationStopped("play_policy_binding_" + binding["status"],
                        stage="native_symbolic_round_refinement", evidence=evidence("incomplete"))
            if len(bindings) < len(overrides):
                raise InformationSetCompilationStopped("maximum_queries",
                    stage="native_symbolic_round_refinement", evidence=evidence("incomplete"))
            play = owned._native_round_fragment.play
            if owned._recorded_play_count < len(play._extract.actors):
                replace(play, model=owned.model).constrain_native_observer_safe_policy(
                    team=target, policy_id=_play_fallback_id, start_step=owned._recorded_play_count,
                    **({"_override_matches": matches} if matches else {}))
            after = hashlib.sha256(str(owned.model.proto).encode("utf-8")).hexdigest()
            core = {"schemaVersion": 1, "kind": "native_guarded_play_policy_binding_v1",
                "policy": fallback, "overridesHash": canonical_sha256(overrides),
                "sourceCpModelProtoHash": before, "boundCpModelProtoHash": after}
            fallback_binding = {**core, "sha256": canonical_sha256(core)}
            bound_model_hash = owned._model_hash(search_branching="automatic", proto_hash=after)
            remaining()

        # A foreign branch can describe an information state, never certify
        # this branch's reachability or outcome. Re-prove legality and install
        # only a conditional choice before searching for a complete round.
        contexts = _auction_contexts or {}
        for key, observation in contexts.items():
            if (not isinstance(observation, Mapping) or key not in policy
                    or not {"actor", "visibleCards", "purchaseCard", "publicChoices", "selectedAction"}.issubset(observation)
                    or Seat.parse(observation.get("actor")).team != target
                    or observation.get("authoredPrefix", False)
                    or owned._auction_observation_hash(observation["actor"], observation["visibleCards"],
                        observation["purchaseCard"], observation["publicChoices"]) != key):
                raise ValueError("symbolic auction context ownership mismatch")
        for key, observation in sorted(contexts.items(), key=lambda item: (len(item[1]["publicChoices"]), item[0])):
            if observation["selectedAction"] == policy[key]:
                continue
            if len(bindings) >= maximum_queries:
                raise InformationSetCompilationStopped("maximum_queries",
                    stage="native_symbolic_round_refinement", evidence=evidence("incomplete"))
            owned._last_policy_binding_proto_hash = None
            binding = owned.bind_native_auction_observed_choice(actor=Seat.parse(observation["actor"]),
                visible_cards=observation["visibleCards"], purchase_card=observation["purchaseCard"],
                public_choices=observation["publicChoices"], action=policy[key],
                maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
                _prefix_legality=True)
            bindings.append(binding)
            if (owned._last_policy_binding_proto_hash is not None
                    and owned._model_hash(search_branching="fixed",
                        proto_hash=owned._last_policy_binding_proto_hash) != binding.get("modelHash")):
                raise ValueError("policy binding digest ownership mismatch")
            bound_model_hash = owned._model_hash(search_branching="automatic",
                proto_hash=owned._last_policy_binding_proto_hash)
            report_progress("binding", binding.get("solverStatus"))
            remaining()
            if binding["status"] != "bound":
                raise InformationSetCompilationStopped("auction_policy_binding_" + binding["status"],
                    stage="native_symbolic_round_refinement", evidence=evidence("incomplete"))
            break  # Check reachability before binding a later, possibly excluded observation.

        if bindings and observation_hint is not None and observation_hint["baseModelHash"] != bound_model_hash:
            observation_hint = None  # The fresh guarded model supersedes its source witness.

        for _ in range(maximum_queries + len(pending_hints) + (observation_hint is not None)):
            if len(queries) + len(bindings) >= maximum_queries:
                break
            if observation_hint is None:
                for index, hint in enumerate(pending_hints):
                    if hint["baseModelHash"] == bound_model_hash:
                        observation_hint = pending_hints.pop(index)
                        used_hint_hashes.append(observation_hint["sha256"])
                        break
            from_hint = observation_hint is not None
            if from_hint:
                receipt, observation_hint = observation_hint, None
            else:
                remaining()
                # A legal below-threshold line is already a counterexample;
                # minimizing its score is not required to make the cut sound.
                optimize = False
                bound_hash = bound_model_hash if _outcome_cache is not None else None
                cache_key = (branch_scope["sha256"], bound_hash, target, threshold, optimize)
                receipt = None if _outcome_cache is None else _outcome_cache.get(cache_key)
                cached = receipt is not None
                if not cached:
                    source = getattr(scenario, "_native_play_source_query", None)
                    query_deck_hint = deck_hint if deck_hint is not None else (source["candidate"]["deckOrder"]
                        if not queries and not bindings and isinstance(source, Mapping)
                        and source.get("status") == "witness_found" else None)
                    receipt = owned.find_native_round_margin_below(target=target, threshold=threshold,
                        maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
                        optimize_margin=optimize, deck_hint=query_deck_hint,
                        **({"play_card_hint": play_card_hint} if play_card_hint is not None else {}))
                complete = (receipt.get("status"), receipt.get("solverStatus")) in {
                    ("witness_found", "optimal"), ("no_matching_outcome", "infeasible")}
                complete = complete or (not optimize and receipt.get("queryMode") == "feasibility_witness"
                    and (receipt.get("status"), receipt.get("solverStatus")) == ("witness_found", "feasible"))
                if cached or complete and _outcome_cache is not None:
                    if (not complete or receipt.get("baseModelHash") != bound_hash
                            or receipt.get("scope") != branch_scope
                            or receipt.get("targetTeam") != target.name.lower()
                            or receipt.get("queryMode") != (None if optimize else "feasibility_witness")
                            or type(receipt.get("threshold")) is not int or receipt["threshold"] != threshold
                            or receipt.get("sha256") != canonical_sha256(
                                {key: value for key, value in receipt.items() if key != "sha256"})):
                        raise ValueError("symbolic outcome cache identity mismatch")
                    # Local to one campaign and bounded by its query allowance.
                    # Reuse exact model results, never mutable policy bindings.
                    _outcome_cache[cache_key] = receipt
                queries.append(receipt)
                report_progress("counterexample", receipt.get("solverStatus"))
            remaining()
            if receipt["status"] == "incomplete":
                raise InformationSetCompilationStopped(receipt["exhaustionReason"] or "solver_unknown",
                    stage="native_symbolic_round_refinement", evidence=evidence("incomplete"))
            if receipt["status"] == "no_matching_outcome":
                return evidence("no_adverse_round_in_branch")
            candidate = receipt["candidate"]
            if "play" in candidate:
                # Soft hints accelerate the fresh solve; they neither bind a
                # continuation nor substitute for guarded policy evidence.
                play_card_hint = [Card.parse(row["card"]).id for row in candidate["play"]["plays"]]
                if fallback is not None:
                    deck_hint = candidate["deckOrder"]
            changed, unknown = False, False
            for kind, observation in ((kind, row) for kind in ("auction", "escalation", "project", "play")
                                      for row in candidate[kind + "Observations"]):
                if observation.get("authoredPrefix", False) or Seat.parse(observation["actor"]).team != target:
                    continue
                key, actions = observation["observationHash"], tuple(observation["legalActions"])
                if domains.setdefault(key, actions) != actions:
                    raise ValueError("symbolic round observation changed its legal action domain")
                if len(domains) > maximum_observations:
                    raise InformationSetCompilationStopped("maximum_observations",
                        stage="native_symbolic_round_refinement", evidence=evidence("incomplete", candidate))
                if key not in policy:
                    # A native singleton legal domain leaves no strategy choice
                    # to extend; the round constraints already force that action.
                    unknown = unknown or (len(actions) != 1 and (fallback is None or kind != "play"))
                    continue
                if fallback is not None and kind == "play" and key not in overrides:
                    raise ValueError("guarded strategy is missing a known play context")
                if policy[key] not in actions:
                    raise ValueError("symbolic round policy selected an illegal visible action")
                if observation["selectedAction"] == policy[key]:
                    continue
                if len(queries) + len(bindings) >= maximum_queries:
                    raise InformationSetCompilationStopped("maximum_queries",
                        stage="native_symbolic_round_refinement", evidence=evidence("incomplete", candidate))
                owned._last_policy_binding_proto_hash = None
                if kind == "auction":
                    binding = owned.bind_native_auction_observed_choice(actor=Seat.parse(observation["actor"]),
                        visible_cards=observation["visibleCards"], purchase_card=observation["purchaseCard"],
                        public_choices=observation["publicChoices"], action=policy[key],
                        maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
                        _prefix_legality=True)
                elif kind == "escalation":
                    binding = owned.bind_native_escalation_observed_choice(information=observation["information"],
                        action=policy[key], maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts,
                        cancel_check=cancel_check)
                elif kind == "project":
                    binding = owned.bind_native_project_observed_choice(information=observation["information"],
                        action=policy[key], maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts,
                        cancel_check=cancel_check)
                else:
                    binding = owned.bind_native_play_observed_choice(information=observation["information"],
                        action=policy[key], maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts,
                        cancel_check=cancel_check, _prefix_legality=True)
                bindings.append(binding)
                if (owned._last_policy_binding_proto_hash is not None
                        and owned._model_hash(search_branching="fixed",
                            proto_hash=owned._last_policy_binding_proto_hash) != binding.get("modelHash")):
                    raise ValueError("policy binding digest ownership mismatch")
                bound_model_hash = owned._model_hash(search_branching="automatic",
                    proto_hash=owned._last_policy_binding_proto_hash)
                report_progress("binding", binding.get("solverStatus"))
                remaining()
                if binding["status"] != "bound":
                    raise InformationSetCompilationStopped(kind + "_policy_binding_" + binding["status"],
                        stage="native_symbolic_round_refinement", evidence=evidence("incomplete", candidate))
                changed = True
                break
            if changed or from_hint:
                # Hints only install guarded choices; never certify or cut.
                continue
            if shorten_counterexamples and owned._native_round_fragment is not None:
                # All-pass ends at the redeal boundary; keep its auction-only
                # counterexample without inventing a play suffix.
                # Keep the witnessed choice in the prefix; before it, the
                # suffix can replace the mistake we need to exclude.
                # Moving later shrinks the legal suffix, so its maximum cannot
                # increase. Threshold feasibility can cheaply strengthen a cut;
                # score optimization retains its first proved bound.
                lengths = sorted({row["playIndex"] + 1 for row in candidate["playObservations"]
                    if len(Seat) <= row["playIndex"] + 1 < len(FULL_DECK) and len(row["legalActions"]) > 1
                    and not row.get("authoredPrefix", False)
                    and Seat.parse(row["actor"]).team == target})
                if minimax_budget is not None:
                    # Minimax values are not monotone along a witnessed line.
                    # Visit eligible boundaries in order; never binary-prune
                    # them using an opponent's potentially suboptimal choice.
                    lengths = sorted(set(lengths) | set(range(len(Seat), len(FULL_DECK), len(Seat))))
                    if len(Seat) <= owned._recorded_play_count < len(FULL_DECK):
                        lengths = sorted(set(lengths) | {owned._recorded_play_count})
                    lengths = [length for length in lengths
                        if length >= owned._recorded_play_count
                        and len(FULL_DECK) - length <= minimax_budget.maximum_depth]
                    # The native cooperative threshold query needs no minimax
                    # graph. Try the first fully declared prefix even when its
                    # suffix exceeds the optional graph's depth tier.
                    earliest = max(len(Seat), owned._recorded_play_count)
                    buyers = [row["buyer"] for row in candidate["auctionActions"] if row.get("buyer") is not None]
                    if (lengths and earliest < lengths[0]
                            and (not buyers or Seat.parse(buyers[-1]).team == target)):
                        lengths.insert(0, earliest)
                    if minimax_budget.native_observed is not None:
                        lengths = lengths[:minimax_budget.native_observed.maximum_bound_prefixes]
                lower, upper = 0, len(lengths) - 1
                # One witness owns these checks. Reuse the existing bounded,
                # hash-checked certificates, never a campaign-wide state cache.
                bound_normalizer = OrToolsExactStateNormalizerV1(self.rules, native_cp_sat=True) if lengths else None
                while lower <= upper:
                    if len(queries) + len(bindings) >= maximum_queries:
                        break
                    middle = lower if minimax_budget is not None else (lower + upper) // 2
                    length = lengths[middle]
                    if minimax_budget is not None:
                        # Share the existing allowance across the remaining
                        # optional query pairs and reserve one share for refinement.
                        # A large early graph must not starve every smaller suffix.
                        cheap = self._bound_native_counterexample_prefix(owned, receipt, target,
                            prefix_play_count=length, maximum_seconds=remaining() / (2 * (upper - lower + 1) + 1),
                            maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
                            required_threshold=threshold, _normalizer=bound_normalizer)
                        queries.append(cheap)
                        remaining()
                        if cheap["status"] == "upper_bound_complete":
                            prefix_bound_hash = cheap["sha256"]
                            break
                        if (cheap["status"] != "threshold_reachable"
                                or len(queries) + len(bindings) >= maximum_queries
                                or len(FULL_DECK) - length > minimax_budget.maximum_depth):
                            lower = middle + 1
                            continue
                    bound = self._bound_native_counterexample_prefix(owned, receipt, target,
                        # The cheap half of this pair is already charged. Share
                        # across remaining prefixes, reserving one for refinement.
                        prefix_play_count=length, maximum_seconds=(remaining() / (upper - lower + 2)
                            if minimax_budget is not None else remaining()),
                        maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
                        required_threshold=None if optimize_prefix_bounds or minimax_budget is not None else threshold,
                        _normalizer=bound_normalizer,
                        **({"minimax_budget": minimax_budget} if minimax_budget is not None else {}))
                    queries.append(bound)
                    remaining()
                    if bound["status"] == "threshold_reachable":
                        lower = middle + 1
                        continue
                    if bound["status"] != "upper_bound_complete":
                        if minimax_budget is not None and bound["status"] == "incomplete":
                            # Optional proof exhaustion does not exhaust the
                            # campaign. Keep its receipt, but never cut from it.
                            lower = middle + 1
                            continue
                        break
                    if bound["upperBound"] >= threshold:
                        lower = middle + 1
                    else:
                        prefix_bound_hash = bound["sha256"]
                        if optimize_prefix_bounds or minimax_budget is not None:
                            break
                        upper = middle - 1
                if prefix_bound_hash is not None:
                    bound = next(query for query in queries if query.get("sha256") == prefix_bound_hash)
                    retained = {key for key, _ in bound["targetActions"]}
                    # A proved prefix excludes every suffix; new suffix choices cannot escape this cut.
                    domains = {key: actions for key, actions in domains.items() if key in policy or key in retained}
            return evidence("policy_extension_required" if unknown else "policy_counterexample_in_branch", candidate)
        raise InformationSetCompilationStopped("maximum_queries", stage="native_symbolic_round_refinement",
            evidence=evidence("incomplete"))

    def _find_observed_policy_counterexample(self, world_state, oracle, target, policy, recall,
            *, threshold, known_domains, maximum_queries, maximum_prefixes,
            maximum_seconds, cancel_check=None, root_action=None, project_context=None,
            _prefix_observations=None, maximum_bound_prefixes=None, maximum_cached_prefixes=None,
            _project_observer_cache=None, maximum_evidence_bytes=None,
            counterexample_objective="feasibility"):
        """Refine one world's adverse-path query; never certify world coverage."""
        if counterexample_objective not in ("feasibility", "minimum_utility"):
            raise ValueError("unsupported native counterexample objective")
        if any(type(value) is not int or value < 1 for value in (maximum_queries, maximum_prefixes)):
            raise ValueError("counterexample query and prefix limits must be positive integers")
        if maximum_evidence_bytes is not None and (type(maximum_evidence_bytes) is not int or maximum_evidence_bytes < 1):
            raise ValueError("counterexample evidence limit must be a positive integer")
        maximum_bound_prefixes = maximum_prefixes if maximum_bound_prefixes is None else maximum_bound_prefixes
        maximum_cached_prefixes = maximum_prefixes if maximum_cached_prefixes is None else maximum_cached_prefixes
        if any(type(value) is not int or value < 1 for value in (maximum_bound_prefixes, maximum_cached_prefixes)):
            raise ValueError("native bound/cache prefix limits must be positive integers")
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds):
            raise ValueError("counterexample time limit must be finite")
        if oracle.project_continuation_actions is not None and project_context is None:
            raise ValueError("project policy requires declaration coverage context")
        deadline = time.monotonic() + maximum_seconds
        domains = dict(known_domains)
        if root_action is not None and root_action not in oracle.start.expected_legal_actions:
            raise ValueError("selected root action is not legal")
        bindings = {(): root_action} if root_action is not None else {}
        # This cache is local to one owned model/recall branch in the master.
        # Rebind observations to the new policy; never reuse old action choices.
        for prefix, (project_observation, choice, observation, actions) in (_prefix_observations or {}).items():
            if project_observation in policy and policy[project_observation] != choice:
                bindings[prefix] = None
            elif observation in policy:
                bindings[prefix] = actions[policy[observation]]
        query_hashes = []
        query_receipts = []
        recorded_bindings = set()
        query_evidence_bytes = 0
        phase = "project_observer" if project_context is not None else "path_query"

        def stop(reason, *, path_phase=None, cause=None):
            raise InformationSetCompilationStopped(reason, stage="native_observed_counterexample",
                evidence={"queryHashes": list(query_hashes), "boundPrefixes": len(bindings),
                          "observations": len(domains), "queries": tuple(query_receipts),
                          "queryPhase": phase,
                          **({"queryEvidenceBytes": query_evidence_bytes,
                              "maximumEvidenceBytes": maximum_evidence_bytes} if maximum_evidence_bytes is not None else {}),
                          **({"causeReceipt": cause} if cause is not None else {}),
                          **({"pathPhase": path_phase} if path_phase else {})})

        def retain_query(row):
            nonlocal query_evidence_bytes
            size = len(canonical_json(row).encode("utf-8"))
            if maximum_evidence_bytes is not None and query_evidence_bytes + size > maximum_evidence_bytes:
                stop("maximum_evidence_bytes", cause={"stage": "native_query_evidence_retention",
                    "retainedBytes": query_evidence_bytes, "candidateBytes": size,
                    "maximumBytes": maximum_evidence_bytes, "unretainedReceiptHash": row["sha256"]})
            query_receipts.append(row)
            query_evidence_bytes += size

        def check_stop():
            if cancel_check is not None and cancel_check():
                stop("cancelled")
            if time.monotonic() >= deadline:
                stop("maximum_seconds")

        def outcome(status, witness, utility, target_actions=()):
            return {"status": status, "witness": witness, "utility": utility,
                    "domains": domains, "queryHashes": tuple(query_hashes),
                    "boundPrefixes": len(bindings), "targetActions": target_actions,
                    "queries": tuple(query_receipts)}

        project_observer = None
        if project_context is not None:
            check_stop()
            project_observer = self._project_policy_observer(world_state, oracle, project_context,
                cache=_project_observer_cache)
            phase = "path_query"
        for _ in range(maximum_queries):
            check_stop()
            if len(bindings) > maximum_bound_prefixes:
                stop("maximum_bound_prefixes")
            try:
                # Any adverse witness is a valid master cut; only INFEASIBLE closes coverage.
                witness, utility, model_hash = oracle.find_path_below(threshold,
                    conditional_actions=bindings, maximum_seconds=deadline - time.monotonic(),
                    cancelled=cancel_check, feasibility_only=counterexample_objective == "feasibility")
            except PlayTailSolveStopped as exc:
                failed = exc.partial_receipt["evidence"]
                if failed.get("queryModelHash") is not None:
                    query_hashes.append(failed["queryModelHash"])
                    attempt = {"schemaVersion": 1, "status": "incomplete",
                        "solverStatus": exc.solver_status, "threshold": threshold, **failed}
                    retain_query({**attempt, "sha256": canonical_sha256(attempt)})
                stop(exc.solver_status, path_phase=failed.get("queryPhase"), cause=exc.partial_receipt)
            query_hashes.append(model_hash)
            additions = {prefix: action for prefix, action in bindings.items() if prefix not in recorded_bindings}
            query_document = {"schemaVersion": 2, "kind": "native_observed_path_query_v2",
                "baseModelHash": oracle.model_hash,
                "queryModelHash": model_hash, "threshold": threshold,
                **({"projectCoverageHash": project_context[2]["sha256"]} if project_context is not None else {}),
                "previousQueryReceiptHash": query_receipts[-1]["sha256"] if query_receipts else None,
                "conditionalActionCount": len(bindings),
                "conditionalActionsAdded": [{"prefix": [step.to_dict() for step in prefix],
                    "action": action.to_dict() if action is not None else None} for prefix, action in sorted(additions.items(),
                    key=lambda item: tuple((int(step.actor), step.card.id, step.ekka) for step in item[0]))],
                "status": "witness_found" if witness is not None else "infeasible",
                "utility": utility}
            if counterexample_objective != "feasibility":
                query_document.update(schemaVersion=3, kind="native_observed_path_query_v3",
                    counterexampleObjective=counterexample_objective)
            retain_query({**query_document, "sha256": canonical_sha256(query_document)})
            recorded_bindings.update(additions)
            check_stop()
            if witness is None:
                return outcome("no_counterexample", None, None)
            phase = "observation_binding"
            try:
                learned_prefixes = {}
                selected, domains, target_actions = self._observed_policy_bindings(world_state, oracle, witness,
                    target, policy, recall, known_domains=domains, project_context=project_context,
                    maximum_seconds=deadline - time.monotonic(), cancel_check=cancel_check,
                    _prefix_observations=learned_prefixes, _known_prefix_observations=_prefix_observations,
                    _project_observer=project_observer)
            except PlayTailSolveStopped as exc:
                stop(exc.solver_status)
            check_stop()
            if len(domains) > maximum_prefixes:
                stop("maximum_observations")
            if _prefix_observations is not None:
                for prefix, observation in learned_prefixes.items():
                    if prefix in _prefix_observations or len(_prefix_observations) < maximum_cached_prefixes:
                        _prefix_observations[prefix] = observation
            compliant = all(action is not None and witness.plays[oracle.base_length + len(prefix)] == action
                            for prefix, action in selected.items())
            if any(observation not in policy for observation in domains):
                # A fully recorded path already bounds policies choosing these
                # actions, even when the master has not seen its observations yet.
                return outcome("policy_extension_required", witness, utility,
                               target_actions if compliant else ())
            if compliant:
                return outcome("counterexample", witness, utility, target_actions)
            previous = len(bindings)
            for prefix, action in selected.items():
                if bindings.setdefault(prefix, action) != action:
                    raise ValueError("fixed observed policy changed its prefix action")
            if len(bindings) == previous:
                raise ValueError("native counterexample violated an already bound policy")
            if len(bindings) > maximum_bound_prefixes:
                stop("maximum_bound_prefixes")
            phase = "path_query"
        stop("maximum_queries")

    def _solve_observed_worlds(self, worlds, target, *, maximum_iterations=None, maximum_queries=None,
                              maximum_prefixes=None, maximum_seconds, cancel_check=None,
                              selected_root_action=None, project_contexts=None, native_limits=None,
                              _project_observer_cache=None, maximum_evidence_bytes=None,
                              admissible_root_actions=None,
                              _prefer_bounded_trick_enumeration=False):
        """Solve a supplied native world set, not its hidden-world exhaustiveness."""
        if not self.native_cp_sat:
            raise ValueError("observed world proof requires native CP-SAT")
        if maximum_evidence_bytes is not None and (type(maximum_evidence_bytes) is not int or maximum_evidence_bytes < 1):
            raise ValueError("native world evidence limit must be a positive integer")
        if native_limits is not None:
            if not isinstance(native_limits, NativeObservedProofLimitsV1):
                raise TypeError("native proof limits require the typed contract")
            if any(value is not None for value in (maximum_iterations, maximum_queries, maximum_prefixes)):
                raise ValueError("native proof limits cannot mix with legacy limit arguments")
            NativeObservedProofLimitsV1.build(native_limits.to_dict())
            maximum_iterations = native_limits.maximum_iterations
            maximum_queries = native_limits.maximum_queries
            maximum_prefixes = native_limits.maximum_observations
        bound_prefixes = native_limits.maximum_bound_prefixes if native_limits is not None else maximum_prefixes
        cached_prefixes = native_limits.maximum_cached_prefixes_per_branch if native_limits is not None else maximum_prefixes
        if any(type(value) is not int or value < 1 for value in
               (maximum_iterations, maximum_queries, maximum_prefixes)):
            raise ValueError("observed proof limits must be positive integers")
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds):
            raise ValueError("observed proof time limit must be finite")
        deadline = time.monotonic() + maximum_seconds
        if project_contexts is not None and not isinstance(project_contexts, Mapping):
            raise TypeError("project contexts must be keyed by native model hash")
        contexts = dict(project_contexts or {})
        branch_groups, world_ids = {}, {}
        ordered = []
        visible_root = None
        utility_scope = None
        for state, oracle, recall in worlds:
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled", stage="native_observed_world_preflight")
            if time.monotonic() >= deadline:
                raise InformationSetCompilationStopped("maximum_seconds", stage="native_observed_world_preflight")
            if oracle.target != target or oracle.start.rules_hash != self.rules.sha256:
                raise ValueError("native world target or rules disagree")
            if oracle.project_continuation_actions is not None:
                context = contexts.get(oracle.model_hash)
                if context is None:
                    raise ValueError("fixed project continuation lacks strategic policy coverage")
                self._project_policy_observer(state, oracle, context, cache=_project_observer_cache)
                coverage = context[2]
                key = (coverage["sha256"], canonical_sha256(recall))
                group = branch_groups.setdefault(key, {"coverage": coverage, "actual": set()})
                group["actual"].add(canonical_sha256([action.to_dict() for action in oracle.project_continuation_actions]))
            elif contexts:
                raise ValueError("project branch proof requires coverage for every supplied model")
            if oracle.start.source_state_hash != canonical_sha256(state):
                raise ValueError("native world state does not match its normalized model source")
            current_scope = {"utility": oracle.utility, "boundaryEvaluator": oracle.boundary_evaluator,
                "playStepCount": len(oracle.build.selected_play_indexes), "basePrefixLength": oracle.base_length,
                "cardCount": len(oracle.start.cards)}
            if utility_scope is not None and current_scope != utility_scope:
                raise ValueError("native worlds disagree on utility or horizon scope")
            utility_scope = current_scope
            actions = oracle.start.expected_legal_actions
            if not actions or any(action.actor != actions[0].actor for action in actions):
                raise ValueError("native world root requires one acting seat and legal actions")
            actor = actions[0].actor.name.lower()
            root = (actor, bind_escalation_recall_v1(canonical_observer_information_hash_v1(
                complete_state=state, acting_seat=actor), recall.get(actor, ())),
                tuple(sorted(action.card.code + ("+ekka" if action.ekka else "") for action in actions)))
            if visible_root is not None and root != visible_root:
                raise ValueError("native worlds disagree on visible root information or actions")
            visible_root = root
            identity = canonical_sha256({"state": state, "model": oracle.model_hash, "recall": recall})
            world_ids[identity] = canonical_sha256({"state": state, "recall": recall}) if contexts else identity
            ordered.append((identity, state, oracle, recall))
        ordered.sort(key=lambda item: item[0])
        if not ordered or len({item[0] for item in ordered}) != len(ordered):
            raise ValueError("observed proof requires distinct native worlds")
        if contexts and set(contexts) != {oracle.model_hash for _, _, oracle, _ in ordered}:
            raise ValueError("project context contains an unowned native model")
        for group in branch_groups.values():
            expected = {canonical_sha256(sequence) for sequence in group["coverage"]["sequences"]}
            if len(expected) != len(group["coverage"]["sequences"]) or group["actual"] != expected:
                raise ValueError("native project branch set does not cover every legal declaration sequence")
        if selected_root_action is not None and selected_root_action not in visible_root[2]:
            raise ValueError("selected root action is not legal")
        if admissible_root_actions is not None:
            if (selected_root_action is not None or type(admissible_root_actions) is not tuple
                    or not admissible_root_actions or Seat.parse(visible_root[0]).team != target
                    or any(type(action) is not str or action not in visible_root[2] for action in admissible_root_actions)
                    or len(set(admissible_root_actions)) != len(admissible_root_actions)):
                raise ValueError("admissible root domain requires distinct legal target actions")
            admissible_root_actions = tuple(sorted(admissible_root_actions))
        bounds = [oracle.query_utility_bounds() for _, _, oracle, _ in ordered]
        lower = min(bound[0] for bound in bounds)
        # Declaration branches are policy choices, not unavoidable hidden worlds.
        upper = (max if contexts else min)(bound[1] for bound in bounds)
        scope_document = {"kind": "native_observed_world_set_v1",
            "rulesHash": self.rules.sha256, "targetTeam": target.name.lower(),
            "rootActor": {"seat": visible_root[0], "observationHash": visible_root[1],
                          "relation": "target" if Seat.parse(visible_root[0]).team == target else "opponent"},
            "coverage": "supplied_worlds_only",
            "utilityScope": utility_scope,
            "worlds": sorted(set(world_ids.values())), "utilityBounds": [lower, upper],
            **({"declarationCoverage": {key[0]: group["coverage"] for key, group in sorted(branch_groups.items())},
                "branches": [{"branchId": identity, "worldId": world_ids[identity], "modelHash": oracle.model_hash,
                    "coverageHash": contexts[oracle.model_hash][2]["sha256"]} for identity, _, oracle, _ in ordered]}
               if contexts else {}),
            **({"selectedRootAction": selected_root_action} if selected_root_action is not None else {}),
            **({"admissibleRootActions": list(admissible_root_actions)} if admissible_root_actions is not None else {})}
        scope_hash = canonical_sha256(scope_document)
        domains, cuts, cut_hashes, queries, masters = {}, [], set(), [], []
        query_receipts = []
        cut_witnesses = {}
        prefix_observations = {identity: {} for identity, _, _, _ in ordered}
        fixed_actions = {}
        if (selected_root_action is not None or admissible_root_actions is not None) and Seat.parse(visible_root[0]).team == target:
            domains[visible_root[1]] = visible_root[2]
        if selected_root_action is not None and Seat.parse(visible_root[0]).team == target:
            fixed_actions[visible_root[1]] = selected_root_action
        service = OrToolsActorInformationSetProofV1()

        def cut_documents():
            return tuple({"utility": cut.utility,
                "targetActions": [list(item) for item in cut.target_actions]} for cut in cuts)

        def resources(cause=None):
            unfinished = cause.get("evidence", {}) if cause is not None else {}
            return {"schemaVersion": 1, "kind": "native_observed_proof_resources_v1",
                "limits": {"masterIterations": maximum_iterations, "counterexampleQueries": maximum_queries,
                    "observations": maximum_prefixes, "boundPrefixesPerQuery": bound_prefixes,
                    "cachedPrefixesPerBranch": cached_prefixes},
                "usage": {"masterIterations": len(masters),
                    "counterexampleQueries": len(queries) + len(unfinished.get("queryHashes", ())),
                    "observations": max(len(domains), unfinished.get("observations", 0)),
                    "cachedPrefixesByBranch": {identity: len(prefix_observations[identity])
                                               for identity in sorted(prefix_observations)},
                    "retainedCuts": len(cuts)}}

        def stop(reason, cause=None):
            raise InformationSetCompilationStopped(reason, stage="native_observed_worlds",
                evidence={"scopeHash": scope_hash, "scope": scope_document, "queryHashes": list(queries),
                          "masterHashes": list(masters), "cutHashes": sorted(cut_hashes),
                          "cuts": cut_documents(), "domains": domains, "cutWitnesses": cut_witnesses,
                          "queries": tuple(query_receipts),
                          "resources": resources(cause),
                          **({"causeReceipt": cause} if cause is not None else {})})

        def remaining_evidence_bytes():
            if maximum_evidence_bytes is None:
                return None
            size = len(canonical_json({"scope": scope_document, "queries": query_receipts,
                "domains": domains, "cuts": cut_documents(), "cutWitnesses": cut_witnesses}).encode("utf-8"))
            if size >= maximum_evidence_bytes:
                stop("maximum_evidence_bytes", {"retainedBytes": size, "maximumBytes": maximum_evidence_bytes})
            return maximum_evidence_bytes - size

        # Score-invariant continuations need no policy tree. Early unrestricted
        # single-world roots use the same extrema, without fixing an action.
        # Only singleton declaration coverage is forced. Optional declarations
        # remain strategic branches even when one branch has an invariant score.
        if (admissible_root_actions is None and native_limits is not None and (selected_root_action is not None
                or len(ordered) == 1 and utility_scope["cardCount"] == utility_scope["playStepCount"] == 32
                and utility_scope["basePrefixLength"] < 4)
                and (utility_scope["cardCount"] <= 12 or utility_scope["playStepCount"] == utility_scope["cardCount"])
                and all(len(group["coverage"]["sequences"]) == 1 for group in branch_groups.values())):
            uniform_worlds = []
            for world_identity, state, oracle, recall in ordered:
                fixed_path = () if selected_root_action is None else (next(
                    action for action in oracle.start.expected_legal_actions
                    if action.card.code + ("+ekka" if action.ekka else "") == selected_root_action),)
                extrema = []
                for maximize in (False, True):
                    if len(queries) >= maximum_queries:
                        stop("maximum_queries")
                    try:
                        witness, value, query_hash = oracle.find_path_below(
                            oracle.query_utility_bounds()[1] + 1, path=fixed_path, conditional_actions={},
                            maximum_seconds=deadline - time.monotonic(), cancelled=cancel_check,
                            maximize_utility=maximize)
                    except PlayTailSolveStopped as exc:
                        failed = exc.partial_receipt.get("evidence", {})
                        if failed.get("queryModelHash") is not None:
                            queries.append(failed["queryModelHash"])
                            attempt = {"schemaVersion": 1, "kind": "native_path_extremum_v1",
                                "worldId": world_ids[world_identity], "sourceStateHash": oracle.start.source_state_hash,
                                "selectedRootAction": selected_root_action, "solverStatus": exc.solver_status,
                                "status": "incomplete", **failed}
                            query_receipts.append({**attempt, "sha256": canonical_sha256(attempt)})
                        stop(exc.solver_status, exc.partial_receipt)
                    if witness is None:
                        raise ValueError("legal native root has no continuation")
                    query = {"schemaVersion": 1, "kind": "native_path_extremum_v1",
                        "worldId": world_ids[world_identity], "sourceStateHash": oracle.start.source_state_hash,
                        **({"branchId": world_identity} if contexts else {}),
                        "baseModelHash": oracle.model_hash, "queryModelHash": query_hash,
                        "selectedRootAction": selected_root_action,
                        "objective": "maximum" if maximize else "minimum", "value": value,
                        "solverStatus": "optimal", "nativePlay": witness.to_dict()}
                    queries.append(query_hash)
                    query_receipts.append({**query, "sha256": canonical_sha256(query)})
                    extrema.append((value, query_hash, witness))
                if extrema[0][0] != extrema[1][0]:
                    break
                uniform_worlds.append((world_identity, oracle, extrema))
            if len(uniform_worlds) == len(ordered):
                for identity, oracle, extrema in uniform_worlds:
                    cut = _CounterexampleCut(extrema[0][0], ())
                    if cut.sha256 in cut_hashes:
                        continue
                    cuts.append(cut)
                    cut_hashes.add(cut.sha256)
                    witness = {"schemaVersion": 1, "cutHash": cut.sha256,
                        "worldId": world_ids[identity], "sourceStateHash": oracle.start.source_state_hash,
                        **({"branchId": identity} if contexts else {}),
                        "baseModelHash": oracle.model_hash, "queryModelHash": extrema[0][1],
                        "uniformBoundQueryHashes": [row[1] for row in extrema],
                        "utility": cut.utility, "prefixLength": oracle.base_length,
                        "plays": [action.to_dict() for action in extrema[0][2].plays]}
                    cut_witnesses[cut.sha256] = {**witness, "sha256": canonical_sha256(witness)}
                master = service._solve_master(scope_hash, selected_root_action,
                    _Catalog({("target", key): actions for key, actions in domains.items()},
                        len(domains), (lower, upper)), cuts, deadline, cancel_check,
                    fixed_actions=fixed_actions, canonicalize_policy=False)
                masters.append(master.model_hash)
                if cancel_check is not None and cancel_check():
                    stop("cancelled")
                if time.monotonic() >= deadline:
                    stop("maximum_seconds")
                if master.status != "optimal":
                    stop("master_not_optimal")
                receipt = {"schemaVersion": 1, "status": "converged",
                    "proofMethod": "uniform_native_path_bounds_v1",
                    "strategyCoverage": "all_legal_continuations_value_invariant",
                    "scopeHash": scope_hash, "scope": scope_document,
                    "worldCount": len(ordered), **({"branchCount": len(ordered)} if contexts else {}),
                    "value": master.value, "policy": master.policy,
                    "domains": domains, "queryHashes": tuple(queries), "masterHashes": tuple(masters),
                    "queries": tuple(query_receipts), "resources": resources(),
                    "cutHashes": tuple(cut.sha256 for cut in cuts),
                    "cuts": cut_documents(), "cutWitnesses": cut_witnesses}
                return {**receipt, "sha256": canonical_sha256(receipt)}

        # The new native objective can use the existing exhaustive route when a
        # conservative one-trick path bound fits its frozen query allowance.
        # This is a cost forecast, not a legality or strategic-truth evaluator.
        bounded_trick_fits = False
        if (native_limits is not None and native_limits.counterexample_objective == "minimum_utility"
                and utility_scope["playStepCount"] <= len(Seat)):
            path_bound = 0
            for _, _, oracle, _ in ordered:
                actor = oracle.start.expected_legal_actions[0].actor
                bound = 1
                played = {action.card for action in oracle.start.base_prefix}
                for step in range(oracle.base_length, len(oracle.build.selected_play_indexes)):
                    choices = (1 if selected_root_action is not None else len(oracle.start.expected_legal_actions)) if step == oracle.base_length else sum(
                        1 for action_step, card_id, _ in oracle.action_literals
                        if action_step == step and Card(card_id) not in played
                        and oracle.start.ownership[Card(card_id)] == actor)
                    bound *= choices
                    actor = actor.next()
                path_bound += bound
                if path_bound > maximum_queries - len(queries):
                    break
            bounded_trick_fits = 0 < path_bound <= maximum_queries - len(queries)

        remaining_path_bound = 0
        if native_limits is not None:
            for _, _, oracle, _ in ordered:
                remaining_path_bound += _remaining_native_path_bound_v1(
                    oracle, selected_root_action, maximum_queries - len(queries))
                if remaining_path_bound > maximum_queries - len(queries):
                    break
        remaining_paths_fit = 0 < remaining_path_bound <= maximum_queries - len(queries)

        # Exhaustive path enumeration wins decisively for the measured late-game
        # workload; early-round models retain CEGIS instead of spending its budget
        # enumerating a much larger latent model.
        if admissible_root_actions is None and native_limits is not None and (
                bounded_trick_fits or remaining_paths_fit or utility_scope["cardCount"] <= 12
                or (_prefer_bounded_trick_enumeration
                    and utility_scope["boundaryEvaluator"] == "current_raw_point_margin_v1"
                    and utility_scope["playStepCount"] <= len(Seat))
                or (utility_scope["playStepCount"] <= 8
                    and utility_scope["basePrefixLength"] % 4 == 3)
                or (utility_scope["playStepCount"] <= 4
                    and utility_scope["basePrefixLength"] == 0 and len(ordered) <= 4)
                or (selected_root_action is not None
                    and utility_scope["playStepCount"] <= 4 and len(ordered) <= 4)):
            for _, _, oracle, _ in ordered:
                oracle.results = None
                oracle._enumerated_witnesses = None
            exhaustive_domains = dict(domains)
            exhaustive_cuts = {}
            exhaustive_witnesses = {}
            exhaustive_hashes = []
            enumerations = []
            remaining_paths = maximum_queries - len(queries)
            exhaustive = True
            exhaustive_prefix_observations = {identity: {} for identity, _, _, _ in ordered}
            for world_identity, state, oracle, recall in ordered:
                if remaining_paths < 1:
                    exhaustive = False
                    break
                reason = oracle.enumerate_paths(
                    maximum_nodes=maximum_queries,
                    maximum_solutions=remaining_paths,
                    maximum_seconds=deadline - time.monotonic(),
                    cancelled=cancel_check,
                    root_action=selected_root_action,
                )
                if reason in {"cancelled", "maximum_seconds"}:
                    stop(reason)
                if reason is not None:
                    if reason not in {"maximum_nodes", "maximum_solutions"}:
                        stop(reason)
                    exhaustive = False
                    break
                if oracle.results is None or oracle._enumerated_witnesses is None:
                    raise ValueError("complete native path enumeration lacks witnesses")
                branch_hashes = []
                project_observer = (
                    self._project_policy_observer(state, oracle, contexts[oracle.model_hash],
                        cache=_project_observer_cache)
                    if contexts else None
                )
                for path, witness in sorted(
                    oracle._enumerated_witnesses.items(),
                    key=lambda item: tuple(
                        (int(action.actor), action.card.id, action.ekka)
                        for action in item[0]
                    ),
                ):
                    path_document = {
                        "worldId": world_ids[world_identity],
                        **({"branchId": world_identity} if contexts else {}),
                        "baseModelHash": oracle.model_hash,
                        "plays": [action.to_dict() for action in witness.plays],
                    }
                    path_hash = canonical_sha256(path_document)
                    exhaustive_hashes.append(path_hash)
                    branch_hashes.append(path_hash)
                    if (
                        selected_root_action is not None
                        and Seat.parse(visible_root[0]).team != target
                        and (
                            not path
                            or path[0].card.code
                            + ("+ekka" if path[0].ekka else "")
                            != selected_root_action
                        )
                    ):
                        continue
                    try:
                        _, exhaustive_domains, target_actions = self._observed_policy_bindings(
                            state, oracle, witness, target, {}, recall,
                            known_domains=exhaustive_domains,
                            project_context=contexts.get(oracle.model_hash),
                            maximum_seconds=deadline - time.monotonic(),
                            cancel_check=cancel_check,
                            _prefix_observations=exhaustive_prefix_observations[world_identity],
                            _known_prefix_observations=exhaustive_prefix_observations[world_identity],
                            _project_observer=project_observer,
                        )
                    except PlayTailSolveStopped as exc:
                        # Retain completed policy work, but never promote a
                        # partial observation pass into an exhaustive proof.
                        domains.update(exhaustive_domains)
                        cuts.extend(exhaustive_cuts[key] for key in sorted(exhaustive_cuts))
                        cut_hashes.update(exhaustive_cuts)
                        cut_witnesses.update(exhaustive_witnesses)
                        queries.extend(exhaustive_hashes)
                        query_receipts.extend(enumerations)
                        prefix_observations.update(exhaustive_prefix_observations)
                        stop(exc.solver_status, {"stage": "native_exhaustive_policy_projection",
                            "worldId": world_ids[world_identity], "pathHash": path_hash,
                            "path": path_document})
                    cache = exhaustive_prefix_observations[world_identity]
                    if len(cache) > cached_prefixes:
                        retained = sorted(
                            cache.items(),
                            key=lambda item: tuple(
                                (int(action.actor), action.card.id, action.ekka)
                                for action in item[0]
                            ),
                        )[:cached_prefixes]
                        cache.clear()
                        cache.update(retained)
                    if len(exhaustive_domains) > maximum_prefixes:
                        stop("maximum_observations")
                    result = oracle.results[path]
                    utility = result.terminal_utility
                    if utility is None:
                        utility = result.boundary_utility
                    if utility is None:
                        raise ValueError("complete native path lacks a utility")
                    cut = _CounterexampleCut(utility, target_actions)
                    if cut.sha256 not in exhaustive_cuts:
                        witness_document = {
                            "schemaVersion": 1, "cutHash": cut.sha256,
                            "worldId": world_ids[world_identity],
                            **({"branchId": world_identity} if contexts else {}),
                            "sourceStateHash": oracle.start.source_state_hash,
                            "baseModelHash": oracle.model_hash,
                            "queryModelHash": path_hash,
                            "utility": utility, "prefixLength": oracle.base_length,
                            "plays": [action.to_dict() for action in witness.plays],
                        }
                        exhaustive_cuts[cut.sha256] = cut
                        exhaustive_witnesses[cut.sha256] = {
                            **witness_document,
                            "sha256": canonical_sha256(witness_document),
                        }
                remaining_paths -= len(branch_hashes)
                enumeration = {
                    "schemaVersion": 1,
                    "kind": "native_exhaustive_path_enumeration_v1",
                    "worldId": world_ids[world_identity],
                    **({"branchId": world_identity} if contexts else {}),
                    "baseModelHash": oracle.model_hash,
                    "solverStatus": "optimal",
                    "pathCount": len(branch_hashes),
                    "pathHashes": branch_hashes,
                }
                enumerations.append({**enumeration, "sha256": canonical_sha256(enumeration)})
            if exhaustive:
                ordered_cuts = tuple(exhaustive_cuts[key] for key in sorted(exhaustive_cuts))
                master = service._solve_master(
                    scope_hash,
                    selected_root_action or "policy_value",
                    _Catalog(
                        {("target", observation): actions
                         for observation, actions in exhaustive_domains.items()},
                        len(exhaustive_domains),
                        (lower, upper),
                    ),
                    ordered_cuts,
                    deadline,
                    cancel_check,
                    fixed_actions=fixed_actions,
                    canonicalize_policy=False,
                )
                if master.status != "optimal":
                    reason = (
                        "cancelled" if cancel_check is not None and cancel_check()
                        else "maximum_seconds" if time.monotonic() >= deadline
                        else "master_not_optimal"
                    )
                    stop(reason, {"stage": "native_exhaustive_policy_master",
                                  "modelHash": master.model_hash,
                                  "solverStatus": master.status})
                exhaustive_resources = {
                    "schemaVersion": 1,
                    "kind": "native_observed_proof_resources_v1",
                    "limits": {
                        "masterIterations": maximum_iterations,
                        "counterexampleQueries": maximum_queries,
                        "observations": maximum_prefixes,
                        "boundPrefixesPerQuery": bound_prefixes,
                        "cachedPrefixesPerBranch": cached_prefixes,
                    },
                    "usage": {
                        "masterIterations": 1,
                        "counterexampleQueries": len(queries) + len(exhaustive_hashes),
                        "observations": len(exhaustive_domains),
                        "cachedPrefixesByBranch": {
                            identity: len(exhaustive_prefix_observations[identity])
                            for identity in sorted(exhaustive_prefix_observations)
                        },
                        "retainedCuts": len(ordered_cuts),
                    },
                }
                receipt = {
                    "schemaVersion": 1, "status": "converged",
                    "proofMethod": "exhaustive_native_paths_v1",
                    "scopeHash": scope_hash, "scope": scope_document,
                    "worldCount": len(set(world_ids.values())),
                    **({"branchCount": len(ordered)} if contexts else {}),
                    "value": master.value, "policy": master.policy,
                    "domains": exhaustive_domains,
                    "queryHashes": (*queries, *exhaustive_hashes),
                    "masterHashes": (master.model_hash,),
                    "queries": (*query_receipts, *enumerations),
                    "resources": exhaustive_resources,
                    "cutHashes": tuple(cut.sha256 for cut in ordered_cuts),
                    "cuts": tuple({"utility": cut.utility,
                        "targetActions": [list(item) for item in cut.target_actions]}
                        for cut in ordered_cuts),
                    "cutWitnesses": exhaustive_witnesses,
                }
                return {**receipt, "sha256": canonical_sha256(receipt)}
            for _, _, oracle, _ in ordered:
                oracle.results = None
                oracle._enumerated_witnesses = None

        for _ in range(maximum_iterations):
            if cancel_check is not None and cancel_check():
                stop("cancelled")
            if time.monotonic() >= deadline:
                stop("maximum_seconds")
            if len(queries) >= maximum_queries:
                stop("maximum_queries")
            master = service._solve_master(scope_hash, selected_root_action or "policy_value",
                _Catalog({("target", observation): actions for observation, actions in domains.items()},
                         len(domains), (lower, upper)), cuts, deadline, cancel_check,
                fixed_actions=fixed_actions, canonicalize_policy=False,
                **({"allowed_actions": {visible_root[1]: admissible_root_actions}}
                   if admissible_root_actions is not None else {}))
            masters.append(master.model_hash)
            if master.status != "optimal":
                reason = ("cancelled" if cancel_check is not None and cancel_check() else
                          "maximum_seconds" if time.monotonic() >= deadline else "master_not_optimal")
                stop(reason, {"stage": "native_policy_master", "modelHash": master.model_hash,
                              "solverStatus": master.status})
            policy = dict(master.policy)
            refined = False
            prior_cut_hashes = set(cut_hashes)
            for world_identity, state, oracle, recall in ordered:
                root_action = next((action for action in oracle.start.expected_legal_actions
                    if action.card.code + ("+ekka" if action.ekka else "") == selected_root_action), None)
                remaining_queries = maximum_queries - len(queries)
                if remaining_queries <= 0:
                    stop("maximum_queries")
                try:
                    result = self._find_observed_policy_counterexample(state, oracle, target, policy, recall,
                        threshold=master.value, known_domains=domains, maximum_queries=remaining_queries,
                        maximum_prefixes=maximum_prefixes, maximum_seconds=deadline - time.monotonic(),
                        cancel_check=cancel_check, root_action=root_action,
                        project_context=contexts.get(oracle.model_hash),
                        _prefix_observations=prefix_observations[world_identity],
                        maximum_bound_prefixes=bound_prefixes, maximum_cached_prefixes=cached_prefixes,
                        **({"maximum_evidence_bytes": remaining_evidence_bytes()} if maximum_evidence_bytes is not None else {}),
                        _project_observer_cache=_project_observer_cache,
                        counterexample_objective=(native_limits.counterexample_objective
                            if native_limits is not None else "feasibility"))
                except InformationSetCompilationStopped as exc:
                    stop(exc.reason, exc.partial_receipt)
                queries.extend(result["queryHashes"])
                query_receipts.extend({"worldId": world_ids[world_identity],
                                      **({"branchId": world_identity} if contexts else {}), "receipt": receipt}
                                      for receipt in result["queries"])
                domains = result["domains"]
                remaining_evidence_bytes()
                if result["status"] == "policy_extension_required" and not result["targetActions"]:
                    refined = True
                    continue
                if result["status"] in {"counterexample", "policy_extension_required"}:
                    refined = True
                    cut = _CounterexampleCut(result["utility"], result["targetActions"])
                    if cut.sha256 in prior_cut_hashes:
                        raise ValueError("native observed proof repeated a counterexample cut")
                    if cut.sha256 in cut_hashes:
                        continue  # Another branch supplied the same cut in this iteration.
                    witness = result["witness"]
                    if witness.model_hash != result["queryHashes"][-1]:
                        raise ValueError("native cut witness belongs to another query model")
                    witness_document = {"schemaVersion": 1, "cutHash": cut.sha256,
                        "worldId": world_ids[world_identity],
                        **({"branchId": world_identity} if contexts else {}),
                        "sourceStateHash": oracle.start.source_state_hash,
                        "baseModelHash": oracle.model_hash, "queryModelHash": witness.model_hash,
                        "utility": result["utility"], "prefixLength": oracle.base_length,
                        "plays": [action.to_dict() for action in witness.plays]}
                    cut_witnesses[cut.sha256] = {**witness_document,
                        "sha256": canonical_sha256(witness_document)}
                    cuts.append(cut)
                    cut_hashes.add(cut.sha256)
            if not refined:
                receipt = {"schemaVersion": 1, "status": "converged", "scopeHash": scope_hash,
                    "scope": scope_document,
                    "worldCount": len(set(world_ids.values())),
                    **({"branchCount": len(ordered)} if contexts else {}),
                    "value": master.value, "policy": master.policy,
                    "domains": domains, "queryHashes": tuple(queries), "masterHashes": tuple(masters),
                    "queries": tuple(query_receipts),
                    "resources": resources(),
                    "cutHashes": tuple(cut.sha256 for cut in cuts),
                    "cuts": cut_documents(), "cutWitnesses": cut_witnesses}
                return {**receipt, "sha256": canonical_sha256(receipt)}
        stop("maximum_iterations")

    def _solve_observed_selected_action(self, worlds, target, *, native_limits, maximum_seconds,
                                       selected_action=None, cancel_check=None, project_contexts=None,
                                       admissible_root_actions=None,
                                       _exhaustive_bounded_trick_actions=None):
        """Own bounded best/selected proofs without claiming all-root coverage.

        Internal compilation evidence, not a qualification entry point. Consumers
        must reconstruct its worlds and bind its horizon before admitting it.
        """
        if not isinstance(native_limits, NativeObservedProofLimitsV1):
            raise TypeError("selected-action proof requires typed native limits")
        NativeObservedProofLimitsV1.build(native_limits.to_dict())
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("selected-action time limit must be finite and positive")
        worlds = tuple(worlds)
        if not worlds or any(action.actor.team != target
                for _, oracle, _ in worlds for action in oracle.start.expected_legal_actions):
            raise ValueError("selected-action proof requires the acting team's worlds")
        actions = tuple(sorted(action.card.code + ("+ekka" if action.ekka else "")
            for action in worlds[0][1].start.expected_legal_actions))
        if not actions or selected_action is not None and selected_action not in actions:
            raise ValueError("selected action is unavailable")
        if admissible_root_actions is not None:
            if (type(admissible_root_actions) is not tuple
                    or any(type(action) is not str or action not in actions for action in admissible_root_actions)
                    or len(set(admissible_root_actions)) != len(admissible_root_actions)
                    or selected_action is not None):
                raise ValueError("witness action hints require distinct legal generated actions")
            admissible_root_actions = tuple(sorted(admissible_root_actions))
        deadline = time.monotonic() + maximum_seconds
        proofs = {}
        queries = iterations = 0
        observer_cache = {}

        def remaining():
            reason = ("cancelled" if cancel_check is not None and cancel_check() else
                      "maximum_seconds" if time.monotonic() >= deadline else
                      "maximum_queries" if queries >= native_limits.maximum_queries else
                      "maximum_iterations" if iterations >= native_limits.maximum_iterations else None)
            if reason:
                raise InformationSetCompilationStopped(reason, stage="native_observed_selected_action",
                    evidence={"proofs": proofs, "limits": native_limits.to_dict()})
            return deadline - time.monotonic()

        def solve(action, allowed=None):
            nonlocal queries, iterations
            seconds = remaining()
            try:
                # Latent project branches can exhaust the deadline before any policy query.
                proof = self._solve_observed_worlds(worlds, target,
                    native_limits=replace(native_limits,
                        maximum_queries=native_limits.maximum_queries - queries,
                        maximum_iterations=native_limits.maximum_iterations - iterations),
                    maximum_seconds=seconds, cancel_check=cancel_check,
                    selected_root_action=action, project_contexts=project_contexts,
                    **({"admissible_root_actions": allowed} if allowed is not None else {}),
                    _project_observer_cache=observer_cache,
                    _prefer_bounded_trick_enumeration=(
                        native_limits.play_allocation_method == "deterministic_cp_sat_prefix_v1"
                        and not project_contexts
                        if _exhaustive_bounded_trick_actions is None
                        else action in _exhaustive_bounded_trick_actions))
            except InformationSetCompilationStopped as exc:
                raise InformationSetCompilationStopped(exc.reason, stage="native_observed_selected_action",
                    evidence={"proofs": proofs, "limits": native_limits.to_dict(),
                              "causeReceipt": exc.partial_receipt}) from exc
            proofs[proof["sha256"]] = proof
            queries += proof["resources"]["usage"]["counterexampleQueries"]
            iterations += proof["resources"]["usage"]["masterIterations"]
            return proof

        best = solve(None)
        if best["scope"]["utilityScope"].get("boundaryEvaluator") is None:
            raise ValueError("selected-action evidence requires a bounded evaluator")
        root = best["scope"]["rootActor"]["observationHash"]
        chosen = dict(best["policy"]).get(root)
        if chosen not in actions:
            raise InformationSetCompilationStopped("native_root_policy_missing",
                stage="native_observed_selected_action", evidence={"proofs": proofs})
        if admissible_root_actions is not None:
            # Witness constraints choose only among actions tied with the unrestricted best proof.
            selected = None
            if chosen in admissible_root_actions:
                selected_action, selected = chosen, best
            elif admissible_root_actions:
                candidate = solve(None, admissible_root_actions)
                action = dict(candidate["policy"]).get(root)
                if action not in admissible_root_actions:
                    raise ValueError("native admissible policy selected an action outside its root domain")
                if candidate["value"] > best["value"]:
                    raise ValueError("selected action exceeds the unrestricted best value")
                if candidate["value"] == best["value"]:
                    selected_action, selected = action, candidate
        elif selected_action is None:
            selected_action = chosen
            selected = best
        else:
            selected = best if selected_action == chosen else solve(selected_action)
        cancelled = cancel_check is not None and cancel_check()
        if cancelled or time.monotonic() >= deadline:
            raise InformationSetCompilationStopped(
                "cancelled" if cancelled else "maximum_seconds",
                stage="native_observed_selected_action", evidence={"proofs": proofs})
        if selected is None:
            raise InformationSetCompilationStopped("no_optimal_witness_action",
                stage="native_observed_selected_action", evidence={
                    "bestProofHash": best["sha256"], "proofs": proofs,
                    "admissibleRootActions": list(admissible_root_actions),
                    "limits": native_limits.to_dict(),
                    "usage": {"counterexampleQueries": queries, "masterIterations": iterations}})
        core = {"schemaVersion": 1, "kind": "native_observed_selected_action_v1",
            "coverage": "supplied_worlds_only", "status": "complete",
            "rulesHash": self.rules.sha256, "targetTeam": target.name.lower(),
            "legalActions": list(actions), "selectedAction": selected_action,
            "bestProofHash": best["sha256"], "selectedProofHash": selected["sha256"],
            "proofs": proofs, "limits": native_limits.to_dict(),
            "usage": {"counterexampleQueries": queries, "masterIterations": iterations},
            "optimalActionsComplete": False, "winningRootActionCount": None}
        if admissible_root_actions is not None:
            core["admissibleRootActions"] = list(admissible_root_actions)
            core["admissibleRootMethod"] = "joint_native_policy_domain_v1"
        return {**core, "sha256": canonical_sha256(core)}

    @staticmethod
    def _recorded_exhaustive_bounded_trick_actions(artifact):
        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("proofs"), Mapping):
            raise ValueError("recorded native path methods are malformed")
        actions = set()
        for proof in artifact["proofs"].values():
            if not isinstance(proof, Mapping) or not isinstance(proof.get("scope"), Mapping):
                raise ValueError("recorded native path methods are malformed")
            action = proof["scope"].get("selectedRootAction")
            if action is not None and not isinstance(action, str):
                raise ValueError("recorded native path methods are malformed")
            if proof.get("proofMethod") == "exhaustive_native_paths_v1":
                actions.add(action)
        return actions

    def _verify_observed_selected_action(self, artifact, worlds, target, *, native_limits,
                                        maximum_seconds, selected_action=None, cancel_check=None,
                                        project_contexts=None):
        """Reconstruct against caller-owned models, never artifact-supplied worlds."""
        if (not isinstance(artifact, Mapping)
                or artifact.get("kind") != "native_observed_selected_action_v1"
                or type(artifact.get("schemaVersion")) is not int or artifact["schemaVersion"] != 1
                or artifact.get("rulesHash") != self.rules.sha256
                or artifact.get("targetTeam") != target.name.lower()
                or artifact.get("limits") != native_limits.to_dict()
                or artifact.get("sha256") != canonical_sha256({
                    key: value for key, value in artifact.items() if key != "sha256"})):
            raise ValueError("selected-action artifact identity or budget mismatch")
        fresh = self._solve_observed_selected_action(worlds, target,
            native_limits=native_limits, maximum_seconds=maximum_seconds,
            selected_action=selected_action, cancel_check=cancel_check,
            project_contexts=project_contexts,
            **({"admissible_root_actions": tuple(artifact["admissibleRootActions"])}
               if isinstance(artifact.get("admissibleRootActions"), list) else {}),
            _exhaustive_bounded_trick_actions=self._recorded_exhaustive_bounded_trick_actions(artifact))
        if canonical_sha256(fresh) != canonical_sha256(artifact):
            raise ValueError("selected-action artifact differs from reconstructed proof")
        return fresh

    def _solve_observed_root_actions(self, worlds, target, *, native_limits, maximum_seconds,
                                     cancel_check=None, project_contexts=None):
        """Retain every legal play-root proof over the same supplied world set.

        This artifact certifies neither exhaustive hidden worlds nor a scenario
        claim. Queries and master iterations share one budget across all roots.
        """
        if not isinstance(native_limits, NativeObservedProofLimitsV1):
            raise TypeError("root proof requires typed native limits")
        NativeObservedProofLimitsV1.build(native_limits.to_dict())
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("root proof time limit must be finite and positive")
        worlds = tuple(worlds)
        if not worlds:
            raise ValueError("root proof requires native worlds")
        deadline = time.monotonic() + maximum_seconds
        actions = tuple(sorted(action.card.code + ("+ekka" if action.ekka else "")
                               for action in worlds[0][1].start.expected_legal_actions))
        if not actions or len(set(actions)) != len(actions):
            raise ValueError("root proof requires distinct legal actions")
        roots = {}
        observations = set()
        queries = iterations = 0
        observer_cache = {}
        preliminary_proof = None

        def document(status):
            return {"schemaVersion": 1, "kind": "native_observed_root_actions_v1", "status": status,
                "coverage": "supplied_worlds_only", "decisionKind": "play",
                "rulesHash": self.rules.sha256, "targetTeam": target.name.lower(),
                "legalActions": list(actions), "limits": native_limits.to_dict(),
                "usage": {"counterexampleQueries": queries, "masterIterations": iterations,
                          "completedRootObservations": len(observations)},
                "roots": dict(roots),
                **({"preliminaryProof": preliminary_proof} if preliminary_proof is not None else {})}

        def stop(reason, cause=None):
            raise InformationSetCompilationStopped(reason, stage="native_observed_root_actions",
                evidence={**document("incomplete"),
                          **({"causeReceipt": cause} if cause is not None else {})})

        # Small first-trick branch sets amortize enumeration across all roots.
        shared = None
        if (len(worlds) <= 4
                and worlds[0][1].start.expected_legal_actions[0].actor.team == target
                and all(oracle.base_length == 0 and len(oracle.build.selected_play_indexes) == 4
                        for _, oracle, _ in worlds)):
            try:
                shared = self._solve_observed_worlds(worlds, target, native_limits=native_limits,
                    maximum_seconds=deadline - time.monotonic(), cancel_check=cancel_check,
                    project_contexts=project_contexts, _project_observer_cache=observer_cache)
            except InformationSetCompilationStopped as exc:
                usage = exc.partial_receipt.get("evidence", {}).get("resources", {}).get("usage", {})
                queries += usage.get("counterexampleQueries", 0)
                iterations += usage.get("masterIterations", 0)
                stop(exc.reason, exc.partial_receipt)
            queries = shared["resources"]["usage"]["counterexampleQueries"]
            iterations = shared["resources"]["usage"]["masterIterations"]
            if shared.get("proofMethod") != "exhaustive_native_paths_v1":
                # A converged CEGIS proof does not cover every fixed root.
                # Retain its cost and use the ordinary root certification path.
                preliminary_proof = shared
                observations.update(shared["domains"])
                shared = None
        if shared is not None:
            root_observation = shared["scope"]["rootActor"]["observationHash"]
            domains = shared["domains"]
            if root_observation not in domains or set(domains[root_observation]) != set(actions):
                raise ValueError("shared exhaustive proof root domain disagrees with legal actions")
            cuts = tuple(_CounterexampleCut(row["utility"], tuple(map(tuple, row["targetActions"])))
                         for row in shared["cuts"])
            if [cut.sha256 for cut in cuts] != list(shared["cutHashes"]):
                raise ValueError("shared exhaustive proof cuts disagree with their identities")
            catalog = _Catalog({("target", key): tuple(value) for key, value in domains.items()},
                len(domains), tuple(shared["scope"]["utilityBounds"]))
            observations.update(domains)
            for action in actions:
                if cancel_check is not None and cancel_check():
                    stop("cancelled", shared)
                if time.monotonic() >= deadline:
                    stop("maximum_seconds", shared)
                if iterations >= native_limits.maximum_iterations:
                    stop("maximum_iterations", shared)
                scope = {**shared["scope"], "selectedRootAction": action}
                scope_hash = canonical_sha256(scope)
                master = OrToolsActorInformationSetProofV1()._solve_master(scope_hash, action,
                    catalog, cuts, deadline, cancel_check,
                    fixed_actions={root_observation: action}, canonicalize_policy=False)
                iterations += 1
                if master.status != "optimal":
                    stop("master_not_optimal", {"sharedEnumerationHash": shared["sha256"],
                        "action": action, "solverStatus": master.status})
                proof = {"schemaVersion": 1, "status": "converged",
                    "proofMethod": "shared_exhaustive_native_paths_v1",
                    "scopeHash": scope_hash, "scope": scope,
                    "worldCount": shared["worldCount"], "value": master.value,
                    "policy": master.policy, "domains": domains,
                    "sharedSourceHash": shared["sha256"],
                    "masterHashes": (master.model_hash,),
                    "resources": {"usage": {"counterexampleQueries": 0, "masterIterations": 1}}}
                roots[action] = {**proof, "sha256": canonical_sha256(proof)}
            if cancel_check is not None and cancel_check():
                stop("cancelled", shared)
            if time.monotonic() >= deadline:
                stop("maximum_seconds", shared)
            core = {**document("complete"), "sharedEnumeration": shared}
            return {**core, "sha256": canonical_sha256(core)}

        for action in actions:
            if cancel_check is not None and cancel_check():
                stop("cancelled")
            if time.monotonic() >= deadline:
                stop("maximum_seconds")
            if queries >= native_limits.maximum_queries:
                stop("maximum_queries")
            if iterations >= native_limits.maximum_iterations:
                stop("maximum_iterations")
            remaining = replace(native_limits,
                maximum_queries=native_limits.maximum_queries - queries,
                maximum_iterations=native_limits.maximum_iterations - iterations)
            try:
                proof = self._solve_observed_worlds(worlds, target, native_limits=remaining,
                    maximum_seconds=deadline - time.monotonic(), cancel_check=cancel_check,
                    selected_root_action=action, project_contexts=project_contexts,
                    _project_observer_cache=observer_cache)
            except InformationSetCompilationStopped as exc:
                usage = exc.partial_receipt.get("evidence", {}).get("resources", {}).get("usage", {})
                queries += usage.get("counterexampleQueries", 0)
                iterations += usage.get("masterIterations", 0)
                stop(exc.reason, exc.partial_receipt)
            roots[action] = proof
            observations.update(proof["domains"])
            queries += proof["resources"]["usage"]["counterexampleQueries"]
            iterations += proof["resources"]["usage"]["masterIterations"]
            if len(observations) > native_limits.maximum_observations:
                stop("maximum_observations")
        if cancel_check is not None and cancel_check():
            stop("cancelled")
        if time.monotonic() >= deadline:
            stop("maximum_seconds")
        core = document("complete")
        return {**core, "sha256": canonical_sha256(core)}

    def _certify_observed_root_actions(self, artifact, *, success_threshold, claim, selected_action=None,
                                       maximum_seconds, cancel_check=None, action_quality=False,
                                       quality_scope="compiled_actor_information_set"):
        """Certify a claim only within the artifact's declared supplied-world scope."""
        if type(success_threshold) is not int:
            raise ValueError("native root threshold must be an integer")
        if type(action_quality) is not bool:
            raise ValueError("native action-quality selection must be Boolean")
        if claim not in {"outcome_reachable", "outcome_forced", "selected_action_is_unique_winner",
                         "target_team_has_no_salvage", "winning_root_action_count"}:
            raise ValueError("native root claim is unsupported")
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("native root certification time must be finite and positive")
        deadline = time.monotonic() + maximum_seconds
        if (not isinstance(artifact, Mapping) or artifact.get("kind") != "native_observed_root_actions_v1"
                or type(artifact.get("schemaVersion")) is not int or artifact["schemaVersion"] != 1
                or artifact.get("decisionKind") != "play"
                or artifact.get("status") != "complete" or artifact.get("coverage") != "supplied_worlds_only"
                or artifact.get("rulesHash") != self.rules.sha256
                or artifact.get("sha256") != canonical_sha256({key: value for key, value in artifact.items() if key != "sha256"})):
            raise ValueError("native root certification requires an owned complete artifact")
        actions = artifact["legalActions"]
        if not actions or len(set(actions)) != len(actions) or set(actions) != set(artifact["roots"]):
            raise ValueError("native root artifact lacks complete legal-action coverage")
        shared = artifact.get("sharedEnumeration")
        if shared is not None and (shared.get("proofMethod") != "exhaustive_native_paths_v1"
                or shared.get("status") != "converged"
                or shared.get("sha256") != canonical_sha256({key: value for key, value in shared.items() if key != "sha256"})
                or artifact["usage"]["counterexampleQueries"] != shared["resources"]["usage"]["counterexampleQueries"]):
            raise ValueError("native shared enumeration identity disagrees")
        if selected_action is not None and selected_action not in actions:
            raise ValueError("selected native root action is unavailable")
        roots = []
        shared_scope = None
        for action in sorted(actions):
            proof = artifact["roots"][action]
            scope = proof["scope"]
            if (proof.get("status") != "converged" or type(proof.get("value")) is not int
                    or proof.get("sha256") != canonical_sha256({key: value for key, value in proof.items() if key != "sha256"})
                    or proof.get("scopeHash") != canonical_sha256(scope)
                    or (shared is not None and (proof.get("sharedSourceHash") != shared["sha256"]
                        or {key: value for key, value in scope.items() if key != "selectedRootAction"} != shared["scope"]))
                    or scope.get("selectedRootAction") != action
                    or scope.get("kind") != "native_observed_world_set_v1"
                    or scope.get("coverage") != "supplied_worlds_only"
                    or scope.get("rulesHash") != self.rules.sha256
                    or scope.get("targetTeam") != artifact["targetTeam"]):
                raise ValueError("native root proof identity or scope disagrees")
            current_scope = {key: value for key, value in scope.items() if key != "selectedRootAction"}
            if shared_scope is not None and shared_scope != current_scope:
                raise ValueError("native roots must share the same world and utility scope")
            shared_scope = current_scope
            roots.append(NativeRootValueCertificateV1(action, "exact", proof["value"], None,
                                                      proof["sha256"], "optimal"))
        preliminary = artifact.get("preliminaryProof")
        if preliminary is not None:
            if (shared is not None or preliminary.get("status") != "converged"
                    or preliminary.get("scope") != shared_scope
                    or preliminary.get("scopeHash") != canonical_sha256(shared_scope)
                    or preliminary.get("sha256") != canonical_sha256({
                        key: value for key, value in preliminary.items() if key != "sha256"})):
                raise ValueError("native preliminary proof identity or scope disagrees")
            for counter in ("counterexampleQueries", "masterIterations"):
                expected = preliminary["resources"]["usage"][counter] + sum(
                    proof["resources"]["usage"][counter] for proof in artifact["roots"].values())
                if artifact["usage"][counter] != expected:
                    raise ValueError("native preliminary proof resource accounting disagrees")
        qualified = OrToolsActorInformationSetProofV1()._certify_root_thresholds(roots,
            success_threshold=success_threshold, problem_hash=artifact["sha256"], claim=claim,
            selected_action=selected_action, deadline=deadline, cancel_check=cancel_check)
        complete = qualified.status == "optimal"
        quality = None
        if complete and action_quality:
            try:
                quality = _certify_action_quality_model_v1(
                    actions=[root.root_action for root in qualified.roots],
                    qualification_model_proto=qualified.model_proto,
                    qualification_variable_bindings=qualified.bindings,
                    qualification_model_hash=qualified.model_hash,
                    action_ids={action: LegalPlayActionV1.from_code(action).action_id for action in actions},
                    actor_relation=shared_scope["rootActor"]["relation"], selected_action=selected_action,
                    deadline=deadline, cancel_check=cancel_check, analysis_scope=shared_scope, scope=quality_scope)
            except InformationSetCompilationStopped as exc:
                raise InformationSetCompilationStopped(exc.reason, stage="native_root_action_quality",
                    evidence={"sourceArtifactHash": artifact["sha256"], "qualificationModelHash": qualified.model_hash,
                              "qualificationModelProto": qualified.model_proto, "bindings": qualified.bindings,
                              "causeReceipt": exc.partial_receipt}) from exc
        core = {"schemaVersion": 1, "kind": "native_observed_root_claim_v1",
            "status": ("satisfied" if qualified.claim_satisfied else "proof_failed") if complete else "incomplete",
            "coverage": artifact["coverage"], "sourceArtifactHash": artifact["sha256"],
            "scope": shared_scope, "claim": claim, "successThreshold": success_threshold,
            "selectedAction": selected_action, "claimSatisfied": qualified.claim_satisfied if complete else None,
            "winningRootActionCount": qualified.winning_count, "solverStatus": qualified.status,
            "modelHash": qualified.model_hash, "modelProto": qualified.model_proto,
            "bindings": [list(binding) for binding in qualified.bindings],
            "claimVariable": qualified.claim_variable, "claimLiteralIndex": qualified.claim_literal_index,
            "roots": [{"action": root.root_action, "value": root.robust_value, "wins": root.satisfies,
                       "certificateHash": root.model_hash} for root in qualified.roots],
            **({"actionQuality": quality} if action_quality else {})}
        if quality is not None and quality.get("modelStatus") == "quarantined":
            core["status"] = "model_error"
            core["claimSatisfied"] = None
        return {**core, "sha256": canonical_sha256(core)}

    @staticmethod
    def _require_observed_horizon(request, scope):
        base, cards, steps = scope["basePrefixLength"], scope["cardCount"], scope["playStepCount"]
        unit, value = request.horizon["unit"], request.horizon["value"]
        if unit in {"plies", "decisions"}:
            end = base + value
        elif unit == "tricks":
            end = (base // 4 + value) * 4
        elif unit == "round_end":
            end = cards
        else:
            raise ValueError("native observed horizon is unsupported")
        if (steps != min(cards, end) or scope["utility"] != request.utility
                or scope["boundaryEvaluator"] != request.boundary_evaluator):
            raise ValueError("native observed utility or achieved horizon disagrees with request")

    def _prepare_observed_play_worlds(self, exact, request, *, maximum_worlds, maximum_attempts,
                                      maximum_model_bytes, maximum_seconds, cancel_check=None, private_recall=None,
                                      project_prefix=None, project_contexts=None,
                                      maximum_horizon_actions=None, maximum_depth=None, _world_inputs=None,
                                      play_allocation_method="deterministic_cp_sat_prefix_v1"):
        """Prepare native bounded play worlds and retain their actual model inputs."""
        if play_allocation_method not in {"deterministic_cp_sat_prefix_v1", "deterministic_cp_sat_batched_prefix_v1"}:
            raise ValueError("unsupported play allocation method")
        if (not self.native_cp_sat or request.mode != "bounded" or request.visibility != "actor_information_set"
                or request.horizon is None or exact.decision_kind != "play"):
            raise ValueError("native play preparation requires bounded actor play")
        if any(type(value) is not int or value < 1 for value in (maximum_worlds, maximum_attempts, maximum_model_bytes)):
            raise ValueError("native play preparation limits must be positive integers")
        if any(value is not None and (type(value) is not int or value < 1)
               for value in (maximum_horizon_actions, maximum_depth)):
            raise ValueError("native play preparation depth limits must be positive integers")
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("native play preparation deadline must be finite and positive")
        recall = normalize_escalation_recall_v1(private_recall)
        deadline = time.monotonic() + maximum_seconds
        models, worlds = [], []
        model_bytes = 0

        def remaining():
            reason = ("cancelled" if cancel_check is not None and cancel_check() else
                      "maximum_seconds" if time.monotonic() >= deadline else None)
            if reason is not None:
                raise InformationSetCompilationStopped(reason, stage="native_play_preparation",
                    evidence={"sourceStateHash": exact.state_hash, "preparedModels": models,
                              "modelBytes": model_bytes})
            return deadline - time.monotonic()

        def construction_stopped():
            remaining()
            return False

        remaining()
        exact.require_integrity()
        if project_prefix is not None and exact.prefix_hash != _effective_prefix_hash(project_prefix):
            raise ValueError("native play preparation prefix ownership mismatch")
        pending_projects = (project_prefix is not None and project_prefix.project_actions_present
            and len(project_prefix.project_actions) < len(Seat))
        parse_request = request
        if pending_projects or (project_prefix is not None and len(project_prefix.play_actions) == 3
                and request.horizon["unit"] == "tricks" and request.horizon["value"] <= 8
                and request.boundary_evaluator == "current_raw_point_margin_v1"):
            # Only parse the current raw frontier here; explicit lifecycle branches
            # below supply the later declarations and the requested proof horizon.
            raw = request.to_dict()
            raw.pop("sha256")
            raw.update(horizon={"unit": "plies", "value": 1}, boundaryEvaluator="current_raw_point_margin_v1")
            parse_request = ScenarioProofRequestV1.build(raw)
        initial = self.graph_compiler._parse_start(exact, expected_rules_hash=self.rules.sha256, request=parse_request)
        steps = self.graph_compiler._bounded_play_step_count(request, initial)
        required_steps = steps - len(initial.base_prefix)
        if pending_projects:
            required_steps += max(0, min(steps, len(Seat)) - len(project_prefix.project_actions))
        for name, limit in (("maximum_horizon_actions", maximum_horizon_actions), ("maximum_depth", maximum_depth)):
            if limit is not None and required_steps > limit:
                raise InformationSetCompilationStopped(name, stage="native_play_preparation")
        self._require_observed_horizon(request, {"basePrefixLength": len(initial.base_prefix),
            "cardCount": len(initial.cards), "playStepCount": steps, "utility": request.utility,
            "boundaryEvaluator": request.boundary_evaluator})
        interleaved = (pending_projects and steps > len(initial.base_prefix) + 1) or (
            project_prefix is not None and project_prefix.project_actions_present
            and len(project_prefix.play_actions) == 3 and steps > 4)
        if interleaved:
            if project_contexts is None or project_contexts:
                raise ValueError("interleaved preparation requires an empty owned context map")
            if not 4 <= steps <= len(initial.cards) or (
                    steps > 4 and request.boundary_evaluator != "current_raw_point_margin_v1"):
                raise InformationSetCompilationStopped("native_project_horizon_unsupported", stage="native_play_preparation")
        if interleaved or _world_inputs is not None:
            inputs = _world_inputs if _world_inputs is not None else self.enumerate_graph_world_inputs(
                exact, project_prefix, {"kind": "play"}, maximum_worlds=maximum_worlds,
                maximum_seconds=remaining(), cancel_check=cancel_check, bounded_world_scope=True,
                _batch_allocations=play_allocation_method == "deterministic_cp_sat_batched_prefix_v1")
            if not inputs or len(inputs) > maximum_worlds:
                raise ValueError("native project worlds exceed declared coverage")
            visible = canonical_observer_information_hash_v1(complete_state=exact.state, acting_seat=exact.next_actor)
            identities = set()
            for normalized, world_prefix in inputs:
                normalized.require_integrity()
                if (normalized.rules_hash != self.rules.sha256 or normalized.decision_kind != "play"
                        or normalized.prefix_hash != _effective_prefix_hash(world_prefix)
                        or normalized.state_hash in identities
                        or canonical_observer_information_hash_v1(complete_state=normalized.state,
                            acting_seat=exact.next_actor) != visible):
                    raise ValueError("native project world ownership or observation mismatch")
                identities.add(normalized.state_hash)
            allocation = {"worldCount": len(inputs), "coverage": "supplied_worlds_only",
                "normalizations": [normalized.to_dict() for normalized, _ in inputs]}
            if _world_inputs is None and play_allocation_method == "deterministic_cp_sat_batched_prefix_v1":
                allocation["worldScope"] = {"method": play_allocation_method,
                    "complete": False, "maximumWorlds": maximum_worlds, "probabilityModel": None}
        else:
            states, allocation = self._enumerate_worlds(exact, maximum_worlds=maximum_worlds,
                maximum_attempts=maximum_attempts, deadline=deadline, cancel_check=cancel_check, bounded_world_scope=True,
                _batch_allocations=play_allocation_method == "deterministic_cp_sat_batched_prefix_v1")
            inputs = tuple((replace(exact, state=state, state_hash=canonical_sha256(state)), None) for state in states)
        target = Team[request.target_team.upper()]
        normalizer = OrToolsExactStateNormalizerV1(self.rules, native_cp_sat=True)
        project_receipts = {row["worldStateHash"]: row for row in allocation.get("projectLifecycleBindings", ())
                            if "selectedProjects" in row}
        for normalized, world_prefix in inputs:
            remaining()
            start = self.graph_compiler._parse_start(normalized, expected_rules_hash=self.rules.sha256,
                request=parse_request, project_receipt=project_receipts.get(normalized.state_hash))
            coverage = normalizer._enumerate_project_continuations(normalized, world_prefix,
                maximum_sequences=maximum_attempts, maximum_seconds=remaining(), cancel_check=cancel_check) if interleaved else None
            for sequence in coverage["sequences"] if coverage is not None else (None,):
                remaining()
                oracle = self.graph_compiler._path_oracle(start, target=target, utility=request.utility,
                    utility_encoding=request.utility_encoding,
                    play_step_count=steps, boundary_evaluator=request.boundary_evaluator,
                    cancel_check=construction_stopped,
                    **({"project_prefix": world_prefix, "project_continuation_actions": tuple(
                        ProjectDeclarationActionV1.build(row) for row in sequence)} if sequence is not None else {}))
                proto = str(oracle.build.model.proto)
                model_bytes += len(proto.encode("utf-8"))
                if model_bytes > maximum_model_bytes:
                    raise InformationSetCompilationStopped("maximum_model_bytes", stage="native_play_preparation",
                        evidence={"sourceStateHash": exact.state_hash, "preparedModels": models,
                                  "modelBytes": model_bytes, "maximumModelBytes": maximum_model_bytes})
                model_identity = {"model": proto}
                model_receipt = {"schemaVersion": 1, "kind": "native_play_model_text_v1",
                    "sourceModelHash": oracle.model_hash, "modelIdentity": model_identity,
                    "modelHash": canonical_sha256(model_identity)}
                models.append({"worldState": normalized.state, "worldStateHash": normalized.state_hash,
                    "modelHash": oracle.model_hash,
                    "modelReceipt": encode_native_model_receipt_v1({**model_receipt,
                        "sha256": canonical_sha256(model_receipt)}),
                    **({"projectCoverage": coverage, "projectPrefix": world_prefix.to_dict()} if interleaved else {})})
                worlds.append((normalized.state, oracle, recall))
                if interleaved:
                    project_contexts[oracle.model_hash] = (normalized, world_prefix, coverage)
        remaining()
        core = {"schemaVersion": 1, "kind": "native_observed_play_models_v1",
            "rulesHash": self.rules.sha256, "sourceStateHash": exact.state_hash,
            "exactStateId": exact.exact_state_id, "requestHash": request.sha256,
            "coverage": "supplied_worlds_only", "allocationModel": allocation,
            "privateHistoryCoverage": "declared_history_only", "privateRecall": recall,
            "privateRecallHash": canonical_sha256(recall),
            "models": models, "modelBytes": model_bytes, "maximumModelBytes": maximum_model_bytes}
        return tuple(worlds), {**core, "sha256": canonical_sha256(core)}

    def _prepare_observed_project_action(self, exact, prefix, request, action, *, maximum_worlds,
                                         maximum_attempts, maximum_model_bytes, maximum_seconds,
                                         cancel_check=None, private_recall=None, _parents=None):
        """Fix one legal declaration across the same observer-compatible parent worlds."""
        exact.require_integrity()
        if (not self.native_cp_sat or exact.decision_kind != "project"
                or exact.rules_hash != self.rules.sha256 or request.mode != "bounded"
                or request.visibility != "actor_information_set" or request.horizon is None
                or exact.prefix_hash != _effective_prefix_hash(prefix)
                or action not in {row["code"] for row in exact.legal_actions}):
            raise ValueError("native project action requires its owned legal root")
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("native project action deadline must be finite and positive")
        deadline = time.monotonic() + maximum_seconds
        children = []
        def remaining():
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled")
            if time.monotonic() >= deadline:
                raise InformationSetCompilationStopped("maximum_seconds")
            return deadline - time.monotonic()
        normalizer = OrToolsExactStateNormalizerV1(self.rules, native_cp_sat=True)
        parents = _parents if _parents is not None else self.enumerate_graph_world_inputs(exact, prefix,
            {"kind": "project", "selector": {"owner": exact.next_actor}}, maximum_worlds=maximum_worlds,
            maximum_seconds=remaining(), cancel_check=cancel_check, bounded_world_scope=True)
        if not parents or len(parents) > maximum_worlds:
            raise ValueError("native project parents exceed declared coverage")
        observation = canonical_observer_information_hash_v1(complete_state=exact.state, acting_seat=exact.next_actor)
        identities = set()
        for parent, parent_prefix in parents:
            remaining()
            parent.require_integrity()
            if (parent.rules_hash != self.rules.sha256 or parent.decision_kind != "project"
                    or parent.next_actor != exact.next_actor
                    or parent.prefix_hash != _effective_prefix_hash(parent_prefix)
                    or parent.state_hash in identities
                    or canonical_observer_information_hash_v1(complete_state=parent.state,
                        acting_seat=exact.next_actor) != observation):
                raise ValueError("native project parent ownership or observation mismatch")
            identities.add(parent.state_hash)
        for parent, parent_prefix in parents:
            remaining()
            child_prefix = normalizer.append_normalized_action(parent_prefix, parent, action)
            child = normalizer.normalize_play_or_project(child_prefix, maximum_seconds=remaining(), cancel_check=cancel_check)
            children.append((child, child_prefix))
        contexts = {}
        worlds, preparation = self._prepare_observed_play_worlds(children[0][0], request,
            maximum_worlds=maximum_worlds, maximum_attempts=maximum_attempts,
            maximum_model_bytes=maximum_model_bytes, maximum_seconds=remaining(), cancel_check=cancel_check,
            private_recall=private_recall, project_prefix=children[0][1], project_contexts=contexts,
            _world_inputs=tuple(children))
        core = {"kind": "native_observed_project_action_models_v1", "schemaVersion": 1,
            "sourceStateHash": exact.state_hash, "prefixHash": prefix.sha256, "action": action,
            "rulesHash": self.rules.sha256, "requestHash": request.sha256,
            "parents": [parent.to_dict() for parent, _ in parents], "preparation": preparation}
        return worlds, contexts, {**core, "sha256": canonical_sha256(core)}

    def _solve_observed_project_actions(self, exact, prefix, request, *, native_limits,
                                        maximum_worlds, maximum_attempts, maximum_model_bytes,
                                        maximum_seconds, cancel_check=None, private_recall=None):
        """Compare all declarations with shared native query/time ceilings."""
        if not isinstance(native_limits, NativeObservedProofLimitsV1):
            raise TypeError("project roots require typed native limits")
        NativeObservedProofLimitsV1.build(native_limits.to_dict())
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("project root deadline must be finite and positive")
        exact.require_integrity()
        if exact.decision_kind != "project" or exact.prefix_hash != _effective_prefix_hash(prefix):
            raise ValueError("project roots require their owned project state")
        deadline = time.monotonic() + maximum_seconds
        roots, preparations = {}, {}
        queries = iterations = 0
        observations = set()
        actions = sorted(row["code"] for row in exact.legal_actions)
        parent_ids = None
        def remaining():
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled")
            if time.monotonic() >= deadline:
                raise InformationSetCompilationStopped("maximum_seconds")
            return deadline - time.monotonic()
        try:
            parents = self.enumerate_graph_world_inputs(exact, prefix,
                {"kind": "project", "selector": {"owner": exact.next_actor}},
                maximum_worlds=maximum_worlds, maximum_seconds=remaining(),
                cancel_check=cancel_check, bounded_world_scope=True)
            for action in actions:
                remaining()
                if queries >= native_limits.maximum_queries or iterations >= native_limits.maximum_iterations:
                    raise InformationSetCompilationStopped("maximum_queries" if queries >= native_limits.maximum_queries else "maximum_iterations")
                worlds, contexts, preparation = self._prepare_observed_project_action(exact, prefix,
                    request, action, maximum_worlds=maximum_worlds, maximum_attempts=maximum_attempts,
                    maximum_model_bytes=maximum_model_bytes, maximum_seconds=remaining(),
                    cancel_check=cancel_check, private_recall=private_recall, _parents=parents)
                current_parents = sorted(parent["stateHash"] for parent in preparation["parents"])
                if parent_ids is not None and parent_ids != current_parents:
                    raise ValueError("project roots disagree on parent world coverage")
                parent_ids = current_parents
                preparations[action] = preparation
                proof = self._solve_observed_worlds(worlds, Team[request.target_team.upper()],
                    native_limits=replace(native_limits, maximum_queries=native_limits.maximum_queries - queries,
                        maximum_iterations=native_limits.maximum_iterations - iterations),
                    maximum_seconds=remaining(), cancel_check=cancel_check, project_contexts=contexts or None)
                roots[action] = proof
                if proof.get("status") != "converged" or type(proof.get("value")) is not int:
                    raise InformationSetCompilationStopped("project_root_incomplete")
                queries += proof["resources"]["usage"]["counterexampleQueries"]
                iterations += proof["resources"]["usage"]["masterIterations"]
                observations.update(proof["domains"])
                if len(observations) > native_limits.maximum_observations:
                    raise InformationSetCompilationStopped("maximum_observations")
            remaining()
        except InformationSetCompilationStopped as exc:
            unfinished = exc.partial_receipt.get("evidence", {})
            usage = unfinished.get("resources", {}).get("usage", {})
            queries += usage.get("counterexampleQueries", 0)
            iterations += usage.get("masterIterations", 0)
            observations.update(unfinished.get("domains", {}))
            raise InformationSetCompilationStopped(exc.reason, stage="native_observed_project_roots",
                evidence={"roots": roots, "preparations": preparations, "causeReceipt": exc.partial_receipt,
                    "sourceStateHash": exact.state_hash,
                    "usage": {"counterexampleQueries": queries, "masterIterations": iterations,
                        "observations": len(observations)}}) from exc
        core = {"kind": "native_observed_project_roots_v1", "schemaVersion": 1,
            "status": "complete", "coverage": "supplied_worlds_only", "rulesHash": self.rules.sha256,
            "sourceStateHash": exact.state_hash, "requestHash": request.sha256, "legalActions": actions,
            "parentStateHashes": parent_ids, "preparations": preparations, "roots": roots,
            "limits": native_limits.to_dict(), "usage": {"counterexampleQueries": queries,
                "masterIterations": iterations, "observations": len(observations)}}
        return {**core, "sha256": canonical_sha256(core)}

    def _certify_observed_project_actions(self, artifact, exact, request, *, maximum_seconds, cancel_check=None):
        """Bind project-root values to the existing CP-SAT claim certificate."""
        exact.require_integrity()
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("project certification deadline must be finite and positive")
        deadline = time.monotonic() + maximum_seconds
        if (artifact.get("kind") != "native_observed_project_roots_v1"
                or artifact.get("status") != "complete" or request.mode != "bounded"
                or artifact.get("coverage") != "supplied_worlds_only"
                or artifact.get("sourceStateHash") != exact.state_hash
                or artifact.get("rulesHash") != self.rules.sha256
                or artifact.get("requestHash") != request.sha256
                or artifact.get("sha256") != canonical_sha256({k: v for k, v in artifact.items() if k != "sha256"})):
            raise ValueError("project claim requires an owned complete root collection")
        actions = {row["code"] for row in exact.legal_actions}
        if (exact.decision_kind != "project" or set(artifact["legalActions"]) != actions
                or set(artifact["roots"]) != actions or set(artifact["preparations"]) != actions):
            raise ValueError("project claim legal root coverage mismatch")
        roots = []
        for action in sorted(actions):
            proof, preparation = artifact["roots"][action], artifact["preparations"][action]
            prepared = preparation["preparation"]
            for document in (proof, preparation, prepared):
                if document.get("sha256") != canonical_sha256({k: v for k, v in document.items() if k != "sha256"}):
                    raise ValueError("project claim nested evidence checksum mismatch")
            if (preparation["action"] != action or preparation["sourceStateHash"] != exact.state_hash
                    or preparation["requestHash"] != request.sha256
                    or sorted(parent["stateHash"] for parent in preparation["parents"]) != artifact["parentStateHashes"]
                    or proof["status"] != "converged" or type(proof["value"]) is not int
                    or proof["scopeHash"] != canonical_sha256(proof["scope"])
                    or proof["scope"]["targetTeam"] != request.target_team
                    or proof["scope"]["rulesHash"] != self.rules.sha256):
                raise ValueError("project claim action or proof scope mismatch")
            branches = proof["scope"].get("branches", [])
            models = prepared["models"]
            if branches:
                expected = {(row["modelHash"], row["projectCoverage"]["sha256"]) for row in models}
                matches = (len(expected) == len(models) == len(branches)
                    and {(row["modelHash"], row["coverageHash"]) for row in branches} == expected)
            else:
                expected = {canonical_sha256({"state": row["worldState"], "model": row["modelHash"],
                    "recall": prepared["privateRecall"]}) for row in models}
                matches = (len(expected) == len(models) and set(proof["scope"]["worlds"]) == expected
                    and all("projectCoverage" not in row for row in models))
            if (not expected or not matches
                    or proof["scope"].get("selectedRootAction") is not None
                    or proof["scope"]["rootActor"]["seat"] != exact.next_actor):
                raise ValueError("project claim declaration model binding mismatch")
            self._require_observed_horizon(request, proof["scope"]["utilityScope"])
            roots.append(NativeRootValueCertificateV1(action, "exact", proof["value"], None, proof["sha256"], "optimal"))
        qualified = OrToolsActorInformationSetProofV1()._certify_root_thresholds(roots,
            success_threshold=_integral_success_threshold(request.success_threshold),
            problem_hash=artifact["sha256"], claim=request.claim, selected_action=request.selected_action,
            deadline=deadline, cancel_check=cancel_check)
        quality = None
        if qualified.status == "optimal":
            quality = _certify_action_quality_model_v1(actions=sorted(actions),
                qualification_model_proto=qualified.model_proto,
                qualification_variable_bindings=qualified.bindings,
                qualification_model_hash=qualified.model_hash,
                action_ids={row["code"]: index for index, row in enumerate(exact.legal_actions)},
                actor_relation="target" if Seat.parse(exact.next_actor).team == Team[request.target_team.upper()] else "opponent",
                selected_action=request.selected_action, deadline=deadline, cancel_check=cancel_check,
                analysis_scope={"sourceArtifactHash": artifact["sha256"], "coverage": "supplied_worlds_only",
                    "requestHash": request.sha256, "decisionKind": "project"})
        core = {"kind": "native_observed_project_claim_v1", "schemaVersion": 1,
            "sourceArtifactHash": artifact["sha256"], "coverage": "supplied_worlds_only",
            "status": ("satisfied" if qualified.claim_satisfied else "proof_failed") if qualified.status == "optimal" else "incomplete",
            "claimSatisfied": qualified.claim_satisfied if qualified.status == "optimal" else None,
            "winningRootActionCount": qualified.winning_count, "modelHash": qualified.model_hash,
            "modelProto": qualified.model_proto, "bindings": [list(binding) for binding in qualified.bindings],
            "actionQuality": quality, "solverStatus": qualified.status,
            "claimVariable": qualified.claim_variable, "claimLiteralIndex": qualified.claim_literal_index,
            "scope": {"worlds": artifact["parentStateHashes"], "coverage": "supplied_worlds_only",
                "rootActor": {"seat": exact.next_actor, "observationHash": canonical_observer_information_hash_v1(
                    complete_state=exact.state, acting_seat=exact.next_actor)}},
            "roots": [{"action": root.root_action, "value": root.robust_value, "wins": root.satisfies}
                for root in qualified.roots]}
        if quality is not None and quality.get("modelStatus") == "quarantined":
            core.update(status="model_error", claimSatisfied=None)
        return {**core, "sha256": canonical_sha256(core)}

    def _prove_observed_project(self, exact, prefix, request, *, budget, maximum_seconds,
                                cancel_check=None, private_recall=None, maximum_worlds=None, maximum_depth=None):
        """Execute project roots and their certificate under one frozen deadline."""
        if budget.native_observed is None or request.budget_profile != budget.profile_id:
            raise ValueError("native project proof requires its selected native budget")
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or not 0 < maximum_seconds <= budget.maximum_wall_seconds:
            raise ValueError("native project runtime deadline exceeds its budget")
        for limit in (maximum_worlds, maximum_depth):
            if limit is not None and (type(limit) is not int or limit < 1):
                raise ValueError("native project execution ceilings must be positive integers")
        deadline = time.monotonic() + maximum_seconds
        cpu_deadline = time.process_time() + budget.maximum_cpu_seconds
        retained = {"sourceStateHash": exact.state_hash, "requestHash": request.sha256, "budgetHash": budget.sha256}
        def cancelled():
            return bool(cancel_check is not None and cancel_check()) or time.process_time() >= cpu_deadline
        def remaining():
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled")
            if time.process_time() >= cpu_deadline:
                raise InformationSetCompilationStopped("maximum_cpu_seconds")
            if time.monotonic() >= deadline:
                raise InformationSetCompilationStopped("maximum_seconds")
            return deadline - time.monotonic()
        try:
            remaining()
            exact.require_integrity()
            if request.horizon is None or request.horizon["unit"] != "tricks" or not 1 <= request.horizon["value"] <= 8:
                raise InformationSetCompilationStopped("native_project_horizon_unsupported")
            # First-trick declarations plus all requested play actions.
            required = 4 * request.horizon["value"] + 4 - 2 * len(prefix.play_actions)
            if required > budget.maximum_horizon_actions:
                raise InformationSetCompilationStopped("maximum_horizon_actions")
            if required > min(budget.maximum_depth, maximum_depth or budget.maximum_depth):
                raise InformationSetCompilationStopped("maximum_depth")
            artifact = self._solve_observed_project_actions(exact, prefix, request,
                native_limits=budget.native_observed,
                maximum_worlds=min(budget.maximum_hidden_worlds, maximum_worlds or budget.maximum_hidden_worlds),
                maximum_attempts=budget.maximum_state_space, maximum_model_bytes=budget.maximum_cache_bytes,
                maximum_seconds=remaining(), cancel_check=cancelled, private_recall=private_recall)
            retained["rootArtifact"] = artifact
            retained["preparation"] = artifact["preparations"]
            outcome = self._observed_project_outcome(artifact, exact, request,
                maximum_seconds=remaining(), cancel_check=cancelled)
            retained["outcome"] = outcome.to_dict()
            remaining()
        except InformationSetCompilationStopped as exc:
            reason = "maximum_cpu_seconds" if time.process_time() >= cpu_deadline and not (cancel_check is not None and cancel_check()) else exc.reason
            raise InformationSetCompilationStopped(reason, stage="native_observed_project",
                evidence={**retained, "causeReceipt": exc.partial_receipt}) from exc
        return outcome, {**retained, "sha256": canonical_sha256(retained)}

    def _prove_observed_play(self, exact, request, *, budget, maximum_seconds, cancel_check=None,
                             private_recall=None, maximum_worlds=None, maximum_depth=None, project_prefix=None):
        """Execute the native bounded play path under one frozen budget snapshot."""
        return self._execute_observed_play(exact, request, budget=budget,
            maximum_seconds=maximum_seconds, cancel_check=cancel_check, private_recall=private_recall,
            maximum_worlds=maximum_worlds, maximum_depth=maximum_depth, project_prefix=project_prefix,
            selected_action_only=False)

    def _compile_observed_play_selection(self, exact, request, *, admissible_root_actions=None, **context):
        """Bind best/selected evidence to owned preparation, not a scenario claim.

        This certifies only the selected-action relation, not root counts or a
        scenario claim. The result owner must reconstruct it before admission.
        """
        ScenarioProofRequestV1.build(request.to_dict())
        _, receipt = self._execute_observed_play(exact, request, selected_action_only=True,
            admissible_root_actions=admissible_root_actions, **context)
        return receipt

    def _verify_observed_play_selection(self, receipt, exact, request, **context):
        """Rebuild the preparation and certificate under the caller's authority."""
        ScenarioProofRequestV1.build(request.to_dict())
        if (not isinstance(receipt, Mapping)
                or receipt.get("kind") != "native_observed_play_selection_execution_v1"
                or receipt.get("sourceStateHash") != exact.state_hash
                or receipt.get("requestHash") != request.sha256
                or receipt.get("budgetHash") != context["budget"].sha256
                or receipt.get("sha256") != canonical_sha256({
                    key: value for key, value in receipt.items() if key != "sha256"})):
            raise ValueError("selected play execution identity or budget mismatch")
        # Replay the recorded method so a planner improvement cannot rewrite old receipts.
        fresh = self._compile_observed_play_selection(exact, request, **context,
            **({"admissible_root_actions": tuple(receipt["admissibleRootActions"])}
               if isinstance(receipt.get("admissibleRootActions"), list) else {}),
            _exhaustive_bounded_trick_actions=self._recorded_exhaustive_bounded_trick_actions(
                receipt.get("rootArtifact")))
        if canonical_sha256(fresh) != canonical_sha256(receipt):
            raise ValueError("selected play execution differs from owned reconstruction")
        return fresh

    def _execute_observed_play(self, exact, request, *, budget, maximum_seconds, selected_action_only,
                               cancel_check=None, private_recall=None, maximum_worlds=None,
                               maximum_depth=None, project_prefix=None,
                               admissible_root_actions=None,
                               _exhaustive_bounded_trick_actions=None):
        """Share source preparation and resource accounting between proof consumers."""
        if type(selected_action_only) is not bool:
            raise TypeError("native play evidence selection must be Boolean")
        if admissible_root_actions is not None:
            if (not selected_action_only or request.selected_action is not None
                    or type(admissible_root_actions) is not tuple
                    or any(type(action) is not str for action in admissible_root_actions)
                    or len(set(admissible_root_actions)) != len(admissible_root_actions)):
                raise ValueError("witness action hints require distinct generated play actions")
            admissible_root_actions = tuple(sorted(admissible_root_actions))
        if selected_action_only and (request.line_universe is not None
                or request.claim not in {"outcome_reachable", "winning_root_action_count"}
                or request.target_team != Seat.parse(exact.next_actor).team.name.lower()):
            raise ValueError("selected play evidence cannot certify root-count or scenario claims")
        if budget.native_observed is None or request.budget_profile != budget.profile_id:
            raise ValueError("native play proof requires its selected native budget")
        for limit in (maximum_worlds, maximum_depth):
            if limit is not None and (type(limit) is not int or limit < 1):
                raise ValueError("native execution ceilings must be positive integers")
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or not 0 < maximum_seconds <= budget.maximum_wall_seconds:
            raise ValueError("native play runtime deadline exceeds its budget")
        deadline = time.monotonic() + maximum_seconds
        cpu_deadline = time.process_time() + budget.maximum_cpu_seconds
        retained = {"sourceStateHash": exact.state_hash, "requestHash": request.sha256, "budgetHash": budget.sha256}
        if admissible_root_actions is not None:
            retained["admissibleRootActions"] = list(admissible_root_actions)

        def cancelled():
            return bool(cancel_check is not None and cancel_check()) or time.process_time() >= cpu_deadline

        def remaining():
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled")
            if time.process_time() >= cpu_deadline:
                raise InformationSetCompilationStopped("maximum_cpu_seconds")
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise InformationSetCompilationStopped("maximum_seconds")
            return seconds

        try:
            remaining()
            exact.require_integrity()
            if request.horizon is None:
                raise ValueError("native play proof requires a bounded horizon")
            contexts = {}
            worlds, preparation = self._prepare_observed_play_worlds(exact, request,
                play_allocation_method=budget.native_observed.play_allocation_method,
                maximum_worlds=min(budget.maximum_hidden_worlds, maximum_worlds or budget.maximum_hidden_worlds), maximum_attempts=budget.maximum_state_space,
                maximum_model_bytes=budget.maximum_cache_bytes, maximum_seconds=remaining(),
                cancel_check=cancelled, private_recall=private_recall,
                project_prefix=project_prefix, project_contexts=contexts,
                maximum_horizon_actions=budget.maximum_horizon_actions,
                maximum_depth=min(budget.maximum_depth, maximum_depth or budget.maximum_depth))
            retained["preparation"] = preparation
            solve = self._solve_observed_selected_action if selected_action_only else self._solve_observed_root_actions
            artifact = solve(worlds, Team[request.target_team.upper()],
                native_limits=budget.native_observed, maximum_seconds=remaining(), cancel_check=cancelled,
                project_contexts=contexts or None,
                **({"selected_action": request.selected_action,
                    "_exhaustive_bounded_trick_actions": _exhaustive_bounded_trick_actions}
                   if selected_action_only else {}),
                **({"admissible_root_actions": admissible_root_actions}
                   if admissible_root_actions is not None else {}))
            retained["rootArtifact"] = artifact
            outcome = None
            if selected_action_only:
                retained["kind"] = "native_observed_play_selection_execution_v1"
                best_proof = artifact["proofs"][artifact["bestProofHash"]]
                selected_proof = artifact["proofs"][artifact["selectedProofHash"]]
                alternatives = [{"action": action, "upperBound": best_proof["value"],
                    "sourceProofHash": best_proof["sha256"]}
                    for action in artifact["legalActions"] if action != artifact["selectedAction"]]
                model, selected, optimal, best, regret, bindings = build_native_bid_dominance_model_v1(
                    selected_proof["value"], alternatives)
                model.add(selected <= best_proof["value"])
                model.add(best == best_proof["value"])
                solver = cp_model.CpSolver()
                solver.parameters.num_search_workers = 1
                solver.parameters.random_seed = 0
                solver.parameters.max_time_in_seconds = remaining()
                status = _solve_cp_sat_with_cancel_v1(solver, model, cancelled)
                remaining()
                if status in {cp_model.MODEL_INVALID, cp_model.INFEASIBLE}:
                    raise ValueError("native selected play values disagree with their best proof")
                if status != cp_model.OPTIMAL:
                    raise InformationSetCompilationStopped("qualification_incomplete")
                quality = {"schemaVersion": 1, "authority": "cp_sat",
                    "scope": "compiled_actor_information_set", "rootActorRelation": "target",
                    "actualAction": artifact["selectedAction"],
                    "actualActionOptimal": bool(solver.boolean_value(optimal)),
                    "actualValue": int(solver.value(selected)), "bestValue": int(solver.value(best)),
                    "regret": int(solver.value(regret)),
                    "optimalActions": [artifact["selectedAction"]] if solver.boolean_value(optimal) else [],
                    "optimalActionsComplete": False, "optimalActionCount": None,
                    "optimalActionCountLowerBound": 1,
                    "optimalActionCountUpperBound": len(artifact["legalActions"]),
                    "qualificationModelHash": canonical_sha256(str(model.proto)),
                    "analysisScope": {"mode": "bounded", "coverage": "supplied_worlds_only",
                        "requestHash": request.sha256, "exactStateId": exact.exact_state_id,
                        "nativeScope": best_proof["scope"]}}
                retained["actionQuality"] = {**quality, "sha256": canonical_sha256(quality)}
                retained["qualityModelProto"] = str(model.proto)
                retained["qualityBindings"] = bindings
            else:
                outcome = self._observed_root_outcome(artifact, request,
                    maximum_seconds=remaining(), cancel_check=cancelled)
                retained["outcome"] = outcome.to_dict()
            remaining()
        except InformationSetCompilationStopped as exc:
            reason = ("maximum_cpu_seconds" if time.process_time() >= cpu_deadline
                      and not (cancel_check is not None and cancel_check()) else exc.reason)
            raise InformationSetCompilationStopped(reason, stage="native_observed_play",
                evidence={**retained, "causeReceipt": exc.partial_receipt}) from exc
        return outcome, {**retained, "sha256": canonical_sha256(retained)}

    def _prepare_complete_state_play(self, exact, prefix, request, *, budget,
                                   maximum_seconds, cancel_check=None, maximum_nodes=None,
                                   require_single_project_branch=False):
        """Own terminal models and exhaustive declaration branches for one fixed deal."""
        if (not self.native_cp_sat or not isinstance(exact, OrToolsNormalizedExactStateV1)
                or not isinstance(prefix, ExactTracePrefixV1)
                or not isinstance(request, ScenarioProofRequestV1)
                or not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None
                or request.budget_profile != budget.profile_id
                or request.mode != "exact" or request.visibility != "complete_state"
                or exact.decision_kind != "play" or request.horizon is not None
                or request.boundary_evaluator is not None
                or request.line_universe is not None or request.policies is not None
                or request.selected_action is not None and request.selected_action not in {
                    row["code"] for row in exact.legal_actions}
                or request.target_team != Seat.parse(exact.next_actor).team.name.lower()
                or not prefix.project_actions_present):
            raise ValueError("compact complete-state proof requires an owned exact play")
        exact.require_integrity()
        if exact.rules_hash != self.rules.sha256 or exact.prefix_hash != _effective_prefix_hash(prefix):
            raise ValueError("compact complete-state proof ownership mismatch")
        if (type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
                or maximum_seconds <= 0):
            raise ValueError("compact complete-state proof requires a finite positive deadline")
        if maximum_nodes is not None and (type(maximum_nodes) is not int or maximum_nodes < 1):
            raise ValueError("compact complete-state proof requires a positive node limit")
        node_limit = min(budget.maximum_nodes, budget.maximum_state_space,
            maximum_nodes if maximum_nodes is not None else budget.maximum_nodes)
        deadline = time.monotonic() + min(maximum_seconds, budget.maximum_wall_seconds, budget.maximum_cpu_seconds)
        models, contexts, worlds = [], {}, []

        def remaining():
            reason = ("cancelled" if cancel_check is not None and cancel_check() else
                      "maximum_seconds" if time.monotonic() >= deadline else None)
            if reason is not None:
                raise InformationSetCompilationStopped(reason, stage="native_complete_state_play",
                    evidence={"preparedModels": models})
            return deadline - time.monotonic()

        remaining()
        pending = len(prefix.project_actions) < len(Seat)
        interleaved = len(prefix.play_actions) < len(Seat)
        parse_request = request
        if interleaved:
            raw = request.to_dict()
            raw.pop("sha256")
            raw.update(mode="bounded", horizon={"unit": "plies", "value": 1},
                       boundaryEvaluator="current_raw_point_margin_v1")
            parse_request = ScenarioProofRequestV1.build(raw)
        start = self.graph_compiler._parse_start(exact, expected_rules_hash=self.rules.sha256, request=parse_request)
        required = len(start.cards) - len(start.base_prefix) + (len(Seat) - len(prefix.project_actions) if pending else 0)
        if required > min(budget.maximum_depth, budget.maximum_horizon_actions):
            raise InformationSetCompilationStopped("maximum_depth", stage="native_complete_state_play")
        coverage = (OrToolsExactStateNormalizerV1(self.rules, native_cp_sat=True)._enumerate_project_continuations(
            exact, prefix, maximum_sequences=min(budget.maximum_state_space, budget.maximum_cache_entries),
            maximum_seconds=remaining(), cancel_check=cancel_check) if interleaved else None)
        if require_single_project_branch and coverage is not None and len(coverage["sequences"]) != 1:
            raise InformationSetCompilationStopped("paired_project_branch_choices",
                stage="native_complete_state_policy_pair_preflight",
                evidence={"projectCoverage": coverage, "preparedModels": [],
                    "queriesUsed": 0, "iterationsUsed": 0})
        model_bytes = 0
        for sequence in coverage["sequences"] if coverage is not None else (None,):
            remaining()
            oracle = self.graph_compiler._path_oracle(start, target=Team[request.target_team.upper()],
                utility=request.utility, cancel_check=cancel_check,
                utility_encoding=request.utility_encoding,
                **({"play_step_count": len(start.cards), "project_prefix": prefix, "project_continuation_actions": tuple(
                    ProjectDeclarationActionV1.build(row) for row in sequence)} if sequence is not None else {}))
            model_bytes += len(str(oracle.build.model.proto).encode("utf-8"))
            if model_bytes > budget.maximum_cache_bytes:
                raise InformationSetCompilationStopped("maximum_model_bytes", stage="native_complete_state_play",
                    evidence={"preparedModels": models, "modelBytes": model_bytes})
            if len(oracle.build.selected_play_indexes) != len(start.cards) or oracle.boundary_evaluator is not None:
                raise ValueError("compact exact proof must reach terminal scoring")
            worlds.append((exact.state, oracle, {}))
            if coverage is not None:
                contexts[oracle.model_hash] = (exact, prefix, coverage)
            models.append({"modelHash": oracle.model_hash,
                "branchId": canonical_sha256({"state": exact.state, "model": oracle.model_hash, "recall": {}})})
        world = (canonical_sha256({"state": exact.state, "recall": {}}) if coverage is not None
                 else models[0]["branchId"])
        core = {"schemaVersion": 1, "kind": "native_complete_state_play_preparation_v1",
            "visibility": "complete_state", "coverage": "exact_fixed_deal_terminal_tail",
            "rulesHash": self.rules.sha256, "sourcePrefixHash": prefix.sha256,
            "exactStateId": exact.exact_state_id, "stateHash": exact.state_hash,
            "requestHash": request.sha256, "budgetHash": budget.sha256,
            "maximumNodeObservations": node_limit,
            "modelHash": canonical_sha256(models) if coverage is not None else models[0]["modelHash"],
            "modelBytes": model_bytes, "worldId": world,
            "remainingActions": required, "cardCount": len(start.cards), "basePrefixLength": len(start.base_prefix),
            **({"projectCoverage": coverage, "branches": sorted((
                {**model, "worldId": world, "coverageHash": coverage["sha256"]} for model in models),
                key=lambda row: row["branchId"])} if coverage is not None else {})}
        remaining()
        return tuple(worlds), contexts, {**core, "sha256": canonical_sha256(core)}

    def _synthesize_complete_state_policy_pair(self, exact, prefix, request, *, budget,
                                             maximum_seconds, cancel_check=None, maximum_nodes=None,
                                             native_play_policy_ids=None, play_policies=None, maximum_evidence_bytes=None):
        """Synthesize both unrestricted fixed-deal policies under one budget.

        The result still requires the owned pair verifier before use; it does
        not enumerate optimal roots or certify preceding round decisions.
        """
        if request.selected_action is not None or request.utility != "awarded_point_margin":
            raise ValueError("paired synthesis requires an unrestricted terminal play suffix")
        if maximum_evidence_bytes is not None and (type(maximum_evidence_bytes) is not int or maximum_evidence_bytes < 1):
            raise ValueError("paired synthesis evidence limit must be a positive integer")
        if native_play_policy_ids is not None and not isinstance(native_play_policy_ids, Mapping):
            raise ValueError("paired native policies require a team-to-policy mapping")
        native_play_policy_ids = dict(native_play_policy_ids or {})
        from .ortools_play_tail import native_observer_safe_policy_identity_v1
        if set(native_play_policy_ids) - {team.name.lower() for team in Team}:
            raise ValueError("paired native policies require canonical team identities")
        for team, policy_id in native_play_policy_ids.items():
            native_observer_safe_policy_identity_v1(policy_id, Team[team.upper()])
        ranked = {}
        if play_policies is not None:
            from .ortools_proof import DeclarativeActionPolicyV1
            if not isinstance(play_policies, Mapping) or not play_policies or set(play_policies) - {seat.name.lower() for seat in Seat}:
                raise ValueError("paired ranked policies require canonical seat identities")
            for team in Team:
                seats = {seat.name.lower() for seat in Seat if seat.team == team}
                selected = {seat: value for seat, value in sorted(play_policies.items()) if seat in seats}
                if not selected:
                    continue
                if set(selected) != seats or team.name.lower() in native_play_policy_ids:
                    raise ValueError("paired policies require one complete policy authority per team")
                for seat, value in selected.items():
                    parsed = DeclarativeActionPolicyV1.build(seat, value)
                    if parsed.choices or parsed.materialization is None:
                        raise ValueError("paired ranked policies require declarative templates")
                ranked[team.name.lower()] = copy.deepcopy(selected)
        started, cpu_started = time.monotonic(), time.process_time()
        proofs, preparation = {}, None
        used_queries = used_iterations = 0
        model_bytes = 0

        def remaining_evidence_bytes():
            if maximum_evidence_bytes is None:
                return None
            size = len(canonical_json({"preparation": preparation, "proofs": proofs}).encode("utf-8"))
            if size >= maximum_evidence_bytes:
                raise InformationSetCompilationStopped("maximum_evidence_bytes", stage="native_complete_state_policy_pair",
                    evidence={"queriesUsed": used_queries, "iterationsUsed": used_iterations,
                        "retainedBytes": size, "maximumBytes": maximum_evidence_bytes})
            return maximum_evidence_bytes - size

        def remaining():
            reason = ("cancelled" if cancel_check is not None and cancel_check() else
                "maximum_cpu_seconds" if time.process_time() - cpu_started >= budget.maximum_cpu_seconds else
                "maximum_seconds" if time.monotonic() - started >= min(maximum_seconds, budget.maximum_wall_seconds) else None)
            if reason is not None:
                raise InformationSetCompilationStopped(reason, stage="native_complete_state_policy_pair",
                    evidence={"preparation": preparation, "proofs": proofs,
                        "queriesUsed": used_queries, "iterationsUsed": used_iterations, "modelBytes": model_bytes})
            return min(maximum_seconds, budget.maximum_wall_seconds) - (time.monotonic() - started)

        worlds, contexts, preparation = self._prepare_complete_state_play(exact, prefix, request,
            budget=budget, maximum_seconds=remaining(), cancel_check=cancel_check, maximum_nodes=maximum_nodes,
            require_single_project_branch=True)
        if len(worlds) != 1:
            raise ValueError("paired synthesis requires one complete-state continuation model")
        start, target = worlds[0][1].start, Team[request.target_team.upper()]
        project_context = contexts.get(worlds[0][1].model_hash)
        oracle_args = ({"play_step_count": len(start.cards), "project_prefix": prefix,
            "project_continuation_actions": worlds[0][1].project_continuation_actions}
            if project_context is not None else {})
        model_bytes = preparation["modelBytes"]
        for team in (target, target.other):
            remaining()
            remaining_evidence_bytes()
            policy_id = native_play_policy_ids.get(team.name.lower())
            ranked_policy = ranked.get(team.name.lower())
            fixed_policy = policy_id is not None or ranked_policy is not None
            queries = budget.native_observed.maximum_queries - used_queries
            iterations = budget.native_observed.maximum_iterations - used_iterations
            if queries <= 0 or not fixed_policy and iterations <= 0:
                raise InformationSetCompilationStopped("native_pair_budget_exhausted",
                    stage="native_complete_state_policy_pair", evidence={"preparation": preparation,
                        "proofs": proofs, "queriesUsed": used_queries, "iterationsUsed": used_iterations,
                        "modelBytes": model_bytes})
            oracle = worlds[0][1] if team == target else self.graph_compiler._path_oracle(
                start, target=team, utility=request.utility, utility_encoding=request.utility_encoding,
                cancel_check=cancel_check, **oracle_args)
            if team != target:
                model_bytes += len(str(oracle.build.model.proto).encode("utf-8"))
            if model_bytes > budget.maximum_cache_bytes:
                raise InformationSetCompilationStopped("maximum_model_bytes",
                    stage="native_complete_state_policy_pair", evidence={"proofs": proofs, "modelBytes": model_bytes,
                        "queriesUsed": used_queries, "iterationsUsed": used_iterations})
            try:
                if not fixed_policy:
                    limits = replace(budget.native_observed, maximum_queries=queries,
                        maximum_iterations=iterations, maximum_observations=min(
                            budget.native_observed.maximum_observations, preparation["maximumNodeObservations"]))
                    proof = self._solve_observed_worlds(((exact.state, oracle, {}),), team,
                        native_limits=limits, maximum_seconds=remaining(),
                        **({"maximum_evidence_bytes": remaining_evidence_bytes()} if maximum_evidence_bytes is not None else {}),
                        project_contexts={oracle.model_hash: project_context} if project_context is not None else None,
                        cancel_check=lambda: bool(cancel_check and cancel_check())
                            or time.process_time() - cpu_started >= budget.maximum_cpu_seconds)
                else:
                    bound = oracle.with_native_play_policy(team=team, policy_id=policy_id, play_policies=ranked_policy)
                    model_bytes += len(str(bound.build.model.proto).encode("utf-8"))
                    if model_bytes > budget.maximum_cache_bytes:
                        raise InformationSetCompilationStopped("maximum_model_bytes", stage="native_complete_state_policy_pair")
                    used_queries += 1
                    witness, value, query_hash = bound.find_path_below(bound.query_utility_bounds()[1] + 1,
                        path=(), conditional_actions={}, maximum_seconds=remaining(),
                        cancelled=lambda: bool(cancel_check and cancel_check())
                            or time.process_time() - cpu_started >= budget.maximum_cpu_seconds)
                    if witness is None:
                        raise ValueError("native paired policy has no legal continuation")
                    scope = {"kind": "native_visible_path_policy_floor_scope_v1", "rulesHash": self.rules.sha256,
                        "targetTeam": team.name.lower(), "coverage": "supplied_worlds_only",
                        "worlds": [canonical_sha256({"state": exact.state, "recall": {}}) if project_context is not None
                            else canonical_sha256({"state": exact.state, "model": oracle.model_hash, "recall": {}})],
                        **({"declarationCoverage": {project_context[2]["sha256"]: project_context[2]},
                            "branches": [{"branchId": canonical_sha256({"state": exact.state, "model": oracle.model_hash, "recall": {}}),
                                "worldId": canonical_sha256({"state": exact.state, "recall": {}}),
                                "modelHash": oracle.model_hash, "coverageHash": project_context[2]["sha256"]}]}
                            if project_context is not None else {}),
                        "utilityScope": {"utility": request.utility, "boundaryEvaluator": None,
                            "playStepCount": len(start.cards), "basePrefixLength": oracle.base_length,
                            "cardCount": len(start.cards)}}
                    proof = {"schemaVersion": 1, "kind": "native_visible_path_policy_floor_v1",
                        "status": "converged", "value": value, "worldCount": 1,
                        "scope": scope, "scopeHash": canonical_sha256(scope), "policy": [], "domains": {},
                        **({"nativePlayPolicy": native_observer_safe_policy_identity_v1(policy_id, team)}
                            if ranked_policy is None else {"rankedPlayPolicies": ranked_policy}),
                        "nativePolicyModelHash": bound.model_hash, "queryHashes": [query_hash],
                        "resources": {"usage": {"counterexampleQueries": 1, "masterIterations": 0}}}
                    proof["sha256"] = canonical_sha256(proof)
            except PlayTailSolveStopped as exc:
                raise InformationSetCompilationStopped(exc.solver_status, stage="native_complete_state_policy_pair",
                    evidence={"preparation": preparation, "proofs": proofs,
                        "queriesUsed": used_queries, "iterationsUsed": used_iterations,
                        "modelBytes": model_bytes, "causeReceipt": exc.partial_receipt}) from exc
            except InformationSetCompilationStopped as exc:
                unfinished = exc.partial_receipt.get("evidence", {}).get("resources", {}).get("usage", {})
                raise InformationSetCompilationStopped(exc.reason, stage="native_complete_state_policy_pair",
                    evidence={"preparation": preparation, "proofs": proofs,
                        "queriesUsed": used_queries + unfinished.get("counterexampleQueries", 0),
                        "iterationsUsed": used_iterations + unfinished.get("masterIterations", 0),
                        "modelBytes": model_bytes,
                        "causeReceipt": exc.partial_receipt}) from exc
            proofs[team.name.lower()] = proof
            if not fixed_policy:
                used_queries += proof["resources"]["usage"]["counterexampleQueries"]
            used_iterations += proof["resources"]["usage"]["masterIterations"]
        remaining()
        remaining_evidence_bytes()
        if proofs[target.name.lower()]["value"] != -proofs[target.other.name.lower()]["value"]:
            raise InformationSetCompilationStopped("policy_bounds_not_closed", stage="native_complete_state_policy_pair",
                evidence={"preparation": preparation, "proofs": proofs, "queriesUsed": used_queries,
                    "iterationsUsed": used_iterations, "modelBytes": model_bytes})
        core = {"schemaVersion": 1, "kind": "native_complete_state_policy_pair_synthesis_v1",
            "status": "complete", "preparation": preparation, "proofs": proofs,
            "queriesUsed": used_queries, "iterationsUsed": used_iterations, "modelBytes": model_bytes}
        return {**core, "sha256": canonical_sha256(core)}

    def _prove_last_project_dominance(self, exact, prefix, request, *, maximum_seconds,
                                      cancel_check=None, budget=None, maximum_conflicts=None, maximum_depth=None):
        """Certify strict root dominance, not root values, on a shared Sun play tree.

        Only the last declaration is covered: no later declaration choice may
        change the action tree. Complete information permits coupling identical
        play histories despite the different public project announcements.
        """
        from .ortools_exact_state import _native_project_action_literals
        from .ortools_scoring_stage import attach_scoring_fragment_v1

        if (not self.native_cp_sat or request.mode != "exact"
                or request.visibility != "complete_state" or request.utility != "awarded_point_margin"
                or request.line_universe is not None or request.policies is not None
                or request.horizon is not None or request.boundary_evaluator is not None
                or exact.decision_kind != "project" or exact.state.get("contract") != "sun"
                or not prefix.project_actions_present or len(prefix.project_actions) != 3
                or len(prefix.play_actions) != 3 or exact.rules_hash != self.rules.sha256
                or exact.prefix_hash != _effective_prefix_hash(prefix)):
            raise ValueError("project dominance requires an owned exact final Sun declaration")
        exact.require_integrity()
        actor = Seat.parse(exact.next_actor)
        order = self.rules.project_domain.declaration_seats(Seat.parse(prefix.dealer))
        actions = tuple(sorted(row["code"] for row in exact.legal_actions))
        if (actor != order[-1] or request.target_team != actor.team.name.lower()
                or request.selected_action not in actions or len(actions) < 2):
            raise ValueError("project dominance requires a legal selected root and every alternative")
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("project dominance deadline must be finite and positive")
        limits = None
        cpu_deadline = math.inf
        if budget is not None:
            if (not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None
                    or budget.profile_id != request.budget_profile
                    or type(maximum_conflicts) is not int or maximum_conflicts < 1):
                raise ValueError("project dominance requires its frozen native proof budget")
            if 2 * (len(actions) - 1) > budget.native_observed.maximum_queries:
                raise InformationSetCompilationStopped("maximum_queries", stage="last_project_dominance")
            if maximum_depth is not None and (type(maximum_depth) is not int or maximum_depth < 1):
                raise ValueError("project dominance caller depth must be positive")
            effective_depth = min(budget.maximum_depth, budget.maximum_horizon_actions,
                                  maximum_depth if maximum_depth is not None else budget.maximum_depth)
            if 32 - len(prefix.play_actions) + 1 > effective_depth:
                raise InformationSetCompilationStopped("maximum_depth", stage="last_project_dominance")
            maximum_seconds = min(maximum_seconds, budget.maximum_wall_seconds)
            cpu_deadline = time.process_time() + budget.maximum_cpu_seconds
            limits = {"profileId": budget.profile_id, "budgetHash": budget.sha256,
                "maximumQueries": budget.native_observed.maximum_queries,
                "maximumModelBytes": budget.maximum_cache_bytes, "maximumConflicts": maximum_conflicts,
                "maximumDepth": effective_depth}
        elif maximum_conflicts is not None:
            raise ValueError("project conflict allowance requires a frozen budget")
        deadline = time.monotonic() + maximum_seconds
        queries = []

        def cancelled():
            return bool(cancel_check is not None and cancel_check()) or time.process_time() >= cpu_deadline

        def remaining():
            reason = ("cancelled" if cancel_check is not None and cancel_check() else
                      "maximum_cpu_seconds" if time.process_time() >= cpu_deadline else
                      "maximum_seconds" if time.monotonic() >= deadline else None)
            if reason is not None:
                raise InformationSetCompilationStopped(reason, stage="last_project_dominance",
                    evidence={"queries": queries})
            return min(deadline - time.monotonic(), cpu_deadline - time.process_time())

        normalizer = OrToolsExactStateNormalizerV1(self.rules, native_cp_sat=True)
        remaining()
        owned_root = normalizer.normalize(prefix, decision_point={"kind": "project",
            "selector": {"owner": actor.name.lower()}}, maximum_seconds=remaining(), cancel_check=cancelled)
        if (owned_root.state_hash != exact.state_hash or owned_root.exact_state_id != exact.exact_state_id
                or owned_root.legal_actions != exact.legal_actions or owned_root.next_actor != exact.next_actor):
            raise ValueError("project dominance root differs from native prefix normalization")
        selected_prefix = normalizer.append_normalized_action(prefix, exact, request.selected_action)
        child = normalizer.normalize_play_or_project(selected_prefix,
            maximum_seconds=remaining(), cancel_check=cancelled)
        raw = request.to_dict()
        raw.pop("sha256")
        raw.pop("selectedAction", None)
        raw.update(mode="bounded", claim="outcome_forced", horizon={"unit": "plies", "value": 1},
                   boundaryEvaluator="current_raw_point_margin_v1")
        start = self.graph_compiler._parse_start(child, expected_rules_hash=self.rules.sha256,
            request=ScenarioProofRequestV1.build(raw))
        for alternative in actions:
            if alternative == request.selected_action:
                continue
            remaining()
            oracle = self.graph_compiler._path_oracle(start, target=actor.team,
                utility=request.utility, play_step_count=len(start.cards), project_prefix=selected_prefix,
                utility_encoding=request.utility_encoding,
                project_continuation_actions=(), cancel_check=cancelled)
            model = oracle.build.model
            projects = attach_project_lifecycle_fragment_v1(model,
                owner_of_card=oracle.build.initial_owner_variables,
                fixed_card_owners=tuple(start.ownership[Card(i)] for i in range(32)),
                contract=start.contract, dealer=Seat.parse(prefix.dealer), trump=start.trump,
                domain=self.rules.project_domain, name="alternative_projects", compact_fixed_states=True,
                second_trick_leader=(oracle.build._extract.winner_seats[0]
                    if self.rules.project_domain.tie_break_order == "second_trick_reveal" else None))
            alternate_prefix = normalizer.append_normalized_action(prefix, exact, alternative)
            for action in alternate_prefix.project_actions:
                model.add_bool_and(_native_project_action_literals(projects, ProjectDeclarationActionV1.build(action)))
            scoring = attach_scoring_fragment_v1(model, oracle.scoring_domain,
                replace(oracle.scoring_fragment.inputs, project_points=projects.combined_awarded_value),
                source_hash=alternate_prefix.sha256, native_conversion=True)
            # Sun has no Baloot. All ownership, card/Ekka choices, raw points,
            # winners, escalation and match inputs are shared without relaxation.
            alternative_margin = scoring.awarded_points[int(actor.team)] - scoring.awarded_points[int(actor.team.other)]
            for phase in ("nonvacuity", "counterexample"):
                remaining()
                if phase == "counterexample":
                    model.add(oracle.terminal_utility_variable <= alternative_margin)
                model_bytes = str(model.proto).encode("utf-8")
                if budget is not None and len(model_bytes) > budget.maximum_cache_bytes:
                    raise InformationSetCompilationStopped("maximum_model_bytes", stage="last_project_dominance",
                        evidence={"queries": queries, "modelBytes": len(model_bytes)})
                model_hash = hashlib.sha256(model_bytes).hexdigest()
                solver = cp_model.CpSolver()
                solver.parameters.num_search_workers = 1
                solver.parameters.random_seed = 0
                solver.parameters.max_time_in_seconds = remaining()
                if maximum_conflicts is not None:
                    solver.parameters.max_number_of_conflicts = maximum_conflicts
                status = _solve_cp_sat_with_cancel_v1(solver, model, cancelled)
                remaining()
                row = {"alternative": alternative, "phase": phase, "modelHash": model_hash,
                       "solverStatus": solver.status_name(status).lower(),
                       "modelBytes": len(model_bytes),
                       "sharedPlayModelHash": oracle.model_hash,
                       "selectedProjectFragmentHash": oracle.project_lifecycle.fragment_hash,
                       "alternativeProjectFragmentHash": projects.fragment_hash}
                if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
                    row.update(selectedMargin=int(solver.value(oracle.terminal_utility_variable)),
                               alternativeMargin=int(solver.value(alternative_margin)))
                queries.append(row)
                if phase == "nonvacuity" and status == cp_model.INFEASIBLE:
                    raise ValueError("project dominance paired model is infeasible")
                if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.INFEASIBLE}:
                    raise InformationSetCompilationStopped(row["solverStatus"], stage="last_project_dominance",
                        evidence={"queries": queries})
                if phase == "counterexample" and status != cp_model.INFEASIBLE:
                    break
            if queries[-1]["phase"] == "counterexample" and queries[-1]["solverStatus"] != "infeasible":
                break
        dominant = (len(queries) == 2 * (len(actions) - 1)
            and all(row["solverStatus"] == "infeasible" for row in queries if row["phase"] == "counterexample"))
        core = {"schemaVersion": 1, "kind": "native_last_project_dominance_v1",
            "status": "strictly_dominant" if dominant else "counterexample_found",
            "rulesHash": self.rules.sha256, "stateHash": exact.state_hash, "prefixHash": prefix.sha256,
            "exactStateId": exact.exact_state_id,
            "requestHash": request.sha256, "visibility": "complete_state", "utility": request.utility,
            "selectedAction": request.selected_action, "legalActions": list(actions), "queries": queries,
            "limits": limits,
            "rootValues": None, "winningRootActionCount": None, "wholeRoundCertified": False}
        return {**core, "sha256": canonical_sha256(core)}

    def _select_last_project_dominance(self, exact, prefix, request, *, budget, maximum_seconds,
                                       maximum_conflicts, maximum_depth=None, cancel_check=None):
        """Find a strictly dominant generated choice with owned candidate requests."""
        if (request.selected_action is not None or not isinstance(budget, ScenarioProofBudgetV1)
                or budget.native_observed is None or request.budget_profile != budget.profile_id
                or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
                or maximum_seconds <= 0):
            raise ValueError("generated project selection requires an unselected request and frozen budget")
        exact.require_integrity()
        actions = tuple(sorted(row["code"] for row in exact.legal_actions))
        if len(actions) < 2 or len(set(actions)) != len(actions):
            raise ValueError("generated project selection requires distinct legal alternatives")
        if 2 * len(actions) * (len(actions) - 1) > budget.native_observed.maximum_queries:
            raise InformationSetCompilationStopped("maximum_queries", stage="last_project_selection")
        deadline = time.monotonic() + min(maximum_seconds, budget.maximum_wall_seconds)
        cpu_deadline = time.process_time() + budget.maximum_cpu_seconds
        attempts, query_count, peak_bytes = [], 0, 0

        def cancelled():
            return bool(cancel_check is not None and cancel_check()) or time.process_time() >= cpu_deadline

        def remaining():
            reason = ("cancelled" if cancel_check is not None and cancel_check() else
                      "maximum_cpu_seconds" if time.process_time() >= cpu_deadline else
                      "maximum_seconds" if time.monotonic() >= deadline else None)
            if reason is not None:
                raise InformationSetCompilationStopped(reason, stage="last_project_selection",
                    evidence={"attempts": attempts, "queryCount": query_count})
            return min(deadline - time.monotonic(), cpu_deadline - time.process_time())

        selected = None
        for action in actions:
            raw = request.to_dict()
            raw.pop("sha256")
            raw["selectedAction"] = action
            candidate = ScenarioProofRequestV1.build(raw)
            try:
                proof = self._prove_last_project_dominance(exact, prefix, candidate,
                    budget=budget, maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts,
                    maximum_depth=maximum_depth, cancel_check=cancelled)
            except (InformationSetCompilationStopped, ExactNormalizationStopped, PlayTailSolveStopped) as exc:
                partial = exc.partial_receipt.get("evidence", {})
                partial_queries = partial.get("queries", [])
                query_count += len(partial_queries)
                peak_bytes = max(peak_bytes, partial.get("modelBytes", 0),
                                 *(row.get("modelBytes", 0) for row in partial_queries))
                reason = ("cancelled" if cancel_check is not None and cancel_check() else
                          "maximum_cpu_seconds" if time.process_time() >= cpu_deadline else exc.partial_receipt["exhaustionReason"])
                raise InformationSetCompilationStopped(reason, stage="last_project_selection",
                    evidence={"attempts": attempts, "queryCount": query_count,
                        "peakModelBytes": peak_bytes,
                        "candidateRequest": candidate.to_dict(), "causeReceipt": exc.partial_receipt}) from exc
            attempts.append({"request": candidate.to_dict(), "proof": proof})
            query_count += len(proof["queries"])
            peak_bytes = max(peak_bytes, *(row["modelBytes"] for row in proof["queries"]))
            remaining()
            if query_count > budget.native_observed.maximum_queries:
                raise InformationSetCompilationStopped("maximum_queries", stage="last_project_selection",
                    evidence={"attempts": attempts, "queryCount": query_count})
            if proof["status"] == "strictly_dominant":
                selected = action
                break
            if proof["status"] != "counterexample_found":
                raise ValueError("generated project selection received an unsupported proof result")
        core = {"schemaVersion": 1, "kind": "native_last_project_selection_v1",
            "status": "strictly_dominant" if selected is not None else "no_strictly_dominant_action",
            "request": request.to_dict(), "rulesHash": self.rules.sha256,
            "stateHash": exact.state_hash, "prefixHash": prefix.sha256, "exactStateId": exact.exact_state_id,
            "legalActions": list(actions), "attempts": attempts, "selectedAction": selected,
            "selectedProofHash": attempts[-1]["proof"]["sha256"] if selected is not None else None,
            "queryCount": query_count, "peakModelBytes": peak_bytes,
            "limits": attempts[0]["proof"]["limits"], "wholeRoundCertified": False}
        return {**core, "sha256": canonical_sha256(core)}

    def _verify_last_project_dominance(self, artifact, exact, prefix, request, *, maximum_seconds,
                                       cancel_check=None, budget=None, maximum_conflicts=None, maximum_depth=None):
        """Rebuild owned native queries; a caller-resealed receipt is not evidence."""
        if (not isinstance(artifact, Mapping)
                or artifact.get("sha256") != canonical_sha256({k: v for k, v in artifact.items() if k != "sha256"})
                or artifact.get("rulesHash") != self.rules.sha256
                or artifact.get("stateHash") != exact.state_hash
                or artifact.get("prefixHash") != prefix.sha256
                or artifact.get("requestHash") != request.sha256):
            raise ValueError("project dominance artifact ownership mismatch")
        fresh = self._prove_last_project_dominance(exact, prefix, request,
            maximum_seconds=maximum_seconds, cancel_check=cancel_check,
            budget=budget, maximum_conflicts=maximum_conflicts, maximum_depth=maximum_depth)
        if fresh != artifact:
            raise ValueError("project dominance artifact differs from fresh native proof")
        return fresh

    def _prove_complete_state_play(self, exact, prefix, request, *, budget,
                                   maximum_seconds, cancel_check=None, maximum_nodes=None):
        """Prove all play roots without fixing future declaration choices."""
        started = time.monotonic()
        worlds, contexts, preparation = self._prepare_complete_state_play(exact, prefix, request,
            budget=budget, maximum_seconds=maximum_seconds, cancel_check=cancel_check, maximum_nodes=maximum_nodes)
        deadline = started + min(maximum_seconds, budget.maximum_wall_seconds, budget.maximum_cpu_seconds)
        native_limits = replace(budget.native_observed, maximum_observations=min(
            budget.native_observed.maximum_observations, preparation["maximumNodeObservations"]))
        roots = None

        def remaining():
            reason = ("cancelled" if cancel_check is not None and cancel_check() else
                      "maximum_seconds" if time.monotonic() >= deadline else None)
            if reason is not None:
                raise InformationSetCompilationStopped(reason, stage="native_complete_state_play",
                    evidence={"preparation": preparation, "roots": roots})
            return deadline - time.monotonic()

        try:
            roots = self._solve_observed_root_actions(worlds, Team[request.target_team.upper()],
                native_limits=native_limits, maximum_seconds=remaining(), cancel_check=cancel_check,
                project_contexts=contexts or None)
            for proof in roots["roots"].values():
                scope = proof["scope"]
                if (scope["worlds"] != [preparation["worldId"]]
                        or scope["utilityScope"] != {"utility": request.utility, "boundaryEvaluator": None,
                            "playStepCount": preparation["cardCount"], "basePrefixLength": preparation["basePrefixLength"],
                            "cardCount": preparation["cardCount"]}
                        or scope.get("branches") != preparation.get("branches")
                        or scope.get("declarationCoverage") != ({preparation["projectCoverage"]["sha256"]:
                            preparation["projectCoverage"]} if contexts else None)):
                    raise ValueError("compact complete-state root coverage mismatch")
            certificate = self._certify_observed_root_actions(roots,
                success_threshold=_integral_success_threshold(request.success_threshold), claim=request.claim,
                selected_action=request.selected_action, maximum_seconds=remaining(), cancel_check=cancel_check,
                action_quality=True, quality_scope="compiled_complete_state")
            if certificate["status"] == "incomplete":
                raise InformationSetCompilationStopped("qualification_incomplete", stage="native_complete_state_play",
                    evidence={"certificate": certificate})
            if certificate["status"] == "model_error":
                raise ValueError("compact complete-state qualification model disagreement")
        except InformationSetCompilationStopped as exc:
            raise InformationSetCompilationStopped(exc.reason, stage="native_complete_state_play",
                evidence={"preparation": preparation, "roots": roots, "causeReceipt": exc.partial_receipt}) from exc
        remaining()
        core = {"schemaVersion": 1, "kind": "native_complete_state_play_roots_v1",
            "status": certificate["status"], "strategicCoverage": False,
            "preparation": preparation, "roots": roots, "certificate": certificate}
        return {**core, "sha256": canonical_sha256(core)}

    def _observed_root_outcome(self, artifact, request, *, maximum_seconds, cancel_check=None):
        """Adapt scoped native roots to the existing outcome, not a fake finite graph."""
        if (request.mode != "bounded" or request.visibility != "actor_information_set"
                or request.target_team != artifact.get("targetTeam") or request.horizon is None):
            raise ValueError("native observed outcome requires matching bounded actor scope")
        if not artifact.get("roots"):
            raise ValueError("native observed outcome requires root evidence")
        scope = next(iter(artifact["roots"].values()))["scope"]["utilityScope"]
        self._require_observed_horizon(request, scope)
        certificate = self._certify_observed_root_actions(artifact,
            success_threshold=_integral_success_threshold(request.success_threshold),
            claim=request.claim, selected_action=request.selected_action,
            maximum_seconds=maximum_seconds, cancel_check=cancel_check, action_quality=True)
        return self._native_certificate_outcome(artifact, request, certificate,
            {action: LegalPlayActionV1.from_code(action).action_id for action in artifact["legalActions"]})

    def _observed_project_outcome(self, artifact, exact, request, *, maximum_seconds, cancel_check=None):
        certificate = self._certify_observed_project_actions(artifact, exact, request,
            maximum_seconds=maximum_seconds, cancel_check=cancel_check)
        return self._native_certificate_outcome(artifact, request, certificate,
            {row["code"]: index for index, row in enumerate(exact.legal_actions)})

    def _native_certificate_outcome(self, artifact, request, certificate, action_ids):
        """Serialize either owned native root certificate through one outcome contract."""
        complete_state = artifact.get("kind") == "native_complete_state_play_roots_v1"
        if complete_state:
            preparation = artifact["preparation"]
            utility_scope = certificate["scope"]["utilityScope"]
            if (request.mode != "exact" or request.visibility != "complete_state"
                    or preparation.get("coverage") != "exact_fixed_deal_terminal_tail"
                    or preparation.get("requestHash") != request.sha256
                    or preparation.get("rulesHash") != self.rules.sha256
                    or preparation.get("sha256") != canonical_sha256({
                        key: value for key, value in preparation.items() if key != "sha256"})
                    or certificate != artifact.get("certificate")
                    or certificate.get("sourceArtifactHash") != artifact["roots"].get("sha256")
                    or certificate["scope"]["worlds"] != [preparation["worldId"]]
                    or utility_scope["playStepCount"] != utility_scope["cardCount"]
                    or utility_scope["boundaryEvaluator"] is not None
                    or certificate["actionQuality"].get("scope") != "compiled_complete_state"):
                raise ValueError("complete-state root outcome ownership or terminal coverage mismatch")
        elif request.mode != "bounded" or request.visibility != "actor_information_set":
            raise ValueError("supplied-world roots cannot claim exact complete-state coverage")
        status = {"satisfied": "exact_complete" if complete_state else "bounded_complete", "proof_failed": "proof_failed",
                  "incomplete": "proof_incomplete", "model_error": "model_error"}[certificate["status"]]
        quality = certificate["actionQuality"]
        proto = quality.get("modelProto", certificate["modelProto"]) if quality is not None else certificate["modelProto"]
        model = cp_model.CpModel()
        if not model.proto.parse_text_format(proto):
            raise ValueError("native observed outcome lacks its qualification model")
        model_document = {"schemaVersion": 1, "kind": "native_complete_state_root_outcome_v1" if complete_state else "native_observed_root_outcome_v1",
            "rootActorSeat": certificate["scope"]["rootActor"]["seat"],
            "requestHash": request.sha256, "sourceArtifactHash": artifact["sha256"],
            "nativeRootArtifactHash": artifact["sha256"], "rootClaimCertificate": certificate,
            "completionStatus": status, "rootActionCountScope": "exact_fixed_deal_terminal_tail" if complete_state else "declared_worlds_only",
            "qualificationSource": "cp_sat_cegis_root_certificates_v2",
            "qualificationModelHash": certificate["modelHash"],
            "analysisScope": {"mode": request.mode, "requestHash": request.sha256,
                "horizon": dict(request.horizon) if request.horizon is not None else None,
                "boundaryEvaluator": request.boundary_evaluator,
                "coverage": "exact_fixed_deal_terminal_tail" if complete_state else "supplied_worlds_only",
                "nativeScope": certificate["scope"]},
            "rootActionVariables": [{"action": action, "actionId": action_ids[action],
                "valueVariable": value_name, "winningVariable": winning_name}
                for action, value_name, winning_name in certificate["bindings"]],
            "winningRootActionCountVariable": "winning_root_action_count",
            "claimVariable": certificate["claimVariable"],
            "claimEvidenceBinding": {"kind": request.claim, "variable": certificate["claimVariable"],
                "literalIndex": certificate["claimLiteralIndex"], "winningCountVariable": "winning_root_action_count"},
            "modelProto": proto,
            **({"actionQuality": {key: value for key, value in quality.items() if key != "modelProto"}}
               if quality is not None else {})}
        return OrToolsGraphProofOutcomeV1.build(completion_status=status,
            solver_status=quality.get("solverStatus", certificate["solverStatus"]) if quality is not None else certificate["solverStatus"],
            claim_satisfied=certificate["claimSatisfied"] is True,
            selected_action=quality.get("actualAction") if quality is not None else None,
            winning_root_action_count=certificate["winningRootActionCount"],
            root_actions=tuple(ProofRootActionV1(action_ids[row["action"]],
                row["action"], row["value"], row["wins"] is True) for row in certificate["roots"]),
            principal_variation=(), continuation_lines=None, policies=(),
            graph_hash=artifact["sha256"], model_hash=canonical_sha256(model_document), request_hash=request.sha256,
            variable_count=len(model.proto.variables), constraint_count=len(model.proto.constraints),
            model_document=model_document, hidden_worlds=None if complete_state else {"method": "supplied_worlds_only",
                "count": len(certificate["scope"]["worlds"]), "complete": False,
                "coverage": {"scopeHash": canonical_sha256(certificate["scope"]["worlds"]),
                    "observerInformationHash": certificate["scope"]["rootActor"]["observationHash"]}},
            proof_engine_id="ortools_native_complete_state_cegis_v1" if complete_state else "ortools_native_observed_cegis_v1",
            proof_engine_version="1.0.0")

    def _compile_play_worlds(
        self, exact, request, world_states, *, maximum_nodes, maximum_depth,
        deadline, cancel_check, requested_recall, histories,
        project_receipts=None,
    ):
        target = Team[str(request.target_team).upper()]
        root_actor = Seat.parse(exact.state.get('currentPlayer'))
        bounded = request.mode == 'bounded'
        recall = requested_recall
        node_count = 0
        boundary_count = 0
        transition_models: list[dict[str, Any]] = []
        worlds: list[FiniteInformationWorldV1] = []

        def stopped(reason: str) -> InformationSetCompilationStopped:
            return InformationSetCompilationStopped(
                reason,
                stage="information_graph_compilation",
                evidence={
                    "compiledWorlds": len(worlds),
                    "compatibleWorlds": len(world_states),
                    "nodeCount": node_count,
                    "maximumNodes": maximum_nodes,
                    "maximumDepth": maximum_depth,
                },
            )

        def check_build_budget():
            if cancel_check is not None and cancel_check():
                raise stopped("cancelled")
            if time.monotonic() >= deadline:
                raise stopped("maximum_seconds")
            return False

        for world_state in world_states:
            check_build_budget()
            world_exact = replace(
                exact, state=world_state, state_hash=canonical_sha256(world_state)
            )
            start = self.graph_compiler._parse_start(
                world_exact, expected_rules_hash=self.rules.sha256, request=request,
                project_receipt=(project_receipts or {}).get(world_exact.state_hash),
            )
            path_oracle = self.graph_compiler._path_oracle(
                start,
                target=target,
                utility=request.utility,
                utility_encoding=request.utility_encoding,
                play_step_count=self.graph_compiler._bounded_play_step_count(request, start),
                boundary_evaluator=request.boundary_evaluator if bounded else None,
                cancel_check=check_build_budget,
            )
            check_build_budget()
            enumeration_reason = path_oracle.enumerate_paths(
                maximum_nodes=max(1, maximum_nodes - node_count),
                maximum_seconds=max(0.001, deadline - time.monotonic()),
                cancelled=cancel_check,
            )
            if enumeration_reason is not None:
                raise stopped(enumeration_reason)
            if bounded:
                transition_models.append({
                    "worldStateHash": canonical_sha256(world_state),
                    "modelHash": path_oracle.model_hash,
                    "boundaryVariables": [variable.name for variable in path_oracle.boundary_utilities],
                    "boundaryEvaluator": request.boundary_evaluator,
                })

            def visit(
                path: tuple[TailPrefixActionV1, ...], depth: int
            ) -> FiniteInformationNodeV1:
                nonlocal node_count, boundary_count
                if cancel_check is not None and cancel_check():
                    raise stopped("cancelled")
                if time.monotonic() >= deadline:
                    raise stopped("maximum_seconds")
                node_count += 1
                if node_count > maximum_nodes:
                    raise stopped("maximum_nodes")
                try:
                    result, prefix = path_oracle.solve(
                        path,
                        maximum_seconds=max(0.001, deadline - time.monotonic()),
                        cancelled=cancel_check,
                    )
                except PlayTailSolveStopped as exc:
                    raise stopped(exc.solver_status) from exc
                if len(prefix) == len(start.cards):
                    utility = self.graph_compiler._terminal_utility(result)
                    return FiniteInformationNodeV1.terminal(utility)
                if self.graph_compiler._boundary_reached(
                    request, depth=depth, prefix_length=len(prefix),
                    base_prefix_length=len(start.base_prefix), terminal=False,
                ):
                    if result.boundary_utility is None:
                        raise stopped("native_boundary_variable_missing")
                    boundary_count += 1
                    return FiniteInformationNodeV1.terminal(result.boundary_utility)
                if depth >= maximum_depth:
                    raise stopped("maximum_depth")
                actor = result.next_actor
                if actor is None:
                    raise stopped("missing_actor")
                return FiniteInformationNodeV1.decision_observed(
                    "target" if actor.team == target else "opponent",
                    observation_id=self._path_observation_id(world_state, start, result, path, recall),
                    children={
                        (
                            f"{action.card.code}+ekka"
                            if action.ekka
                            else action.card.code
                        ): visit((*path, action), depth + 1)
                        for action in result.legal_play_actions
                    },
                )

            try:
                initial, _prefix = path_oracle.solve(
                    (),
                    maximum_seconds=max(0.001, deadline - time.monotonic()),
                    cancelled=cancel_check,
                )
            except PlayTailSolveStopped as exc:
                raise stopped(exc.solver_status) from exc
            if initial.next_actor != root_actor:
                raise stopped("root_actor_mismatch")
            for past in histories if histories is not None else (None,):
                recall = past["privateRecall"] if past is not None else requested_recall
                roots = {
                    (f"{action.card.code}+ekka" if action.ekka else action.card.code): visit((action,), 1)
                    for action in initial.legal_play_actions
                }
                identity = canonical_sha256(world_state)
                if past is not None:
                    identity = canonical_sha256({"state": identity, "priorHistory": past["worldId"]})
                worlds.append(FiniteInformationWorldV1.build(identity, roots))
        return worlds, node_count, boundary_count, transition_models, recall

    def _compile_play_worlds_parallel(
        self, exact, request, world_states, *, graph_workers, maximum_nodes, maximum_depth,
        deadline, cancel_check, requested_recall, histories,
        worker_cpu_callback=None,
        project_receipts=None,
    ):
        context = multiprocessing.get_context("spawn")
        cancelled = context.Event()
        worker_count = min(graph_workers, len(world_states))
        cpu_slots = context.Array("d", worker_count)
        next_slot = context.Value("i", 0)
        pool = ProcessPoolExecutor(
            max_workers=worker_count, mp_context=context,
            initializer=_initialize_world_compile_worker, initargs=(cancelled, cpu_slots, next_slot))
        worlds, transitions = [], []
        node_count = boundary_count = 0
        recall = requested_recall
        reported_cpu = 0.0

        def report_cpu():
            nonlocal reported_cpu
            with cpu_slots.get_lock():
                consumed = sum(cpu_slots[:])
            if worker_cpu_callback is not None and consumed > reported_cpu:
                worker_cpu_callback(consumed - reported_cpu)
            reported_cpu = max(reported_cpu, consumed)

        def stopped(reason):
            return InformationSetCompilationStopped(reason, stage="information_graph_compilation",
                evidence={"compiledWorlds": len(worlds), "compatibleWorlds": len(world_states),
                    "nodeCount": node_count, "maximumNodes": maximum_nodes, "maximumDepth": maximum_depth})

        limits = dict(maximum_nodes=maximum_nodes, maximum_depth=maximum_depth, deadline=deadline,
            requested_recall=requested_recall, histories=histories)
        try:
            futures = []
            for state in world_states:
                receipt = (project_receipts or {}).get(canonical_sha256(state)) if project_receipts else None
                worker_limits = (limits if receipt is None else
                    {**limits, "project_receipts": {receipt["worldStateHash"]: receipt}})
                futures.append(pool.submit(_compile_play_world_fragment,
                    (self.rules.data, self.native_cp_sat, exact, request, state, worker_limits)))
            # Merge in enumeration order, regardless of worker completion order.
            for future in futures:
                while True:
                    report_cpu()
                    if cancel_check is not None and cancel_check():
                        raise stopped("cancelled")
                    if time.monotonic() >= deadline:
                        raise stopped("maximum_seconds")
                    try:
                        fragment, nodes, boundaries, models, recall = future.result(timeout=0.05)
                        break
                    except FutureTimeout:
                        continue
                    except InformationSetCompilationStopped as error:
                        node_count += error.partial_receipt["evidence"].get("nodeCount", 0)
                        raise stopped(error.reason) from error
                node_count += nodes
                if node_count > maximum_nodes:
                    raise stopped("maximum_nodes")
                worlds.extend(fragment)
                boundary_count += boundaries
                transitions.extend(models)
            return worlds, node_count, boundary_count, transitions, recall
        finally:
            cancelled.set()
            pool.shutdown(wait=True, cancel_futures=True)
            report_cpu()

    def compile(
        self,
        exact: OrToolsNormalizedExactStateV1,
        request: ScenarioProofRequestV1,
        *,
        maximum_nodes: int,
        maximum_depth: int,
        maximum_worlds: int,
        maximum_seconds: float,
        cancel_check: Callable[[], bool] | None = None,
        private_recall: Mapping[str, Sequence[str]] | None = None,
        prior_history_worlds: Sequence[Mapping[str, Any]] | None = None,
        graph_workers: int = 1,
        worker_cpu_callback: Callable[[float], None] | None = None,
    ) -> ActorInformationSetCompilationV1:
        if type(graph_workers) is not int or not 1 <= graph_workers <= 4:
            raise ValueError("graph_workers must be an integer within the granted 1..4 worker ceiling")
        if worker_cpu_callback is not None and not callable(worker_cpu_callback):
            raise TypeError("worker CPU callback must be callable")
        recall = normalize_escalation_recall_v1(private_recall)
        if type(maximum_worlds) is not int or maximum_worlds < 1:
            raise ValueError("maximum_worlds must be a positive integer")
        requested_recall = recall
        histories = None
        if prior_history_worlds is not None:
            if not self.native_cp_sat or not isinstance(prior_history_worlds, (list, tuple)) or not prior_history_worlds:
                raise ValueError("history worlds require nonempty native-certified enumeration")
            if len(prior_history_worlds) > maximum_worlds:
                raise InformationSetCompilationStopped("maximum_combined_worlds", stage="information_graph_preflight")
            histories = []
            for row in prior_history_worlds:
                if not isinstance(row, Mapping) or set(row) != {"worldId", "privateRecall"}:
                    raise ValueError("history world requires an identity and recall")
                remembered = normalize_escalation_recall_v1(row["privateRecall"])
                if row["worldId"] != canonical_sha256(remembered):
                    raise ValueError("history world checksum mismatch")
                if remembered.get(exact.next_actor, ()) != recall.get(exact.next_actor, ()):
                    raise ValueError("history worlds disagree with observer recall")
                histories.append({"worldId": row["worldId"], "privateRecall": remembered})
            if len({row["worldId"] for row in histories}) != len(histories):
                raise ValueError("history world identities must be unique")
            histories.sort(key=lambda row: row["worldId"])
        if recall and not self.native_cp_sat:
            raise ValueError("private recall requires native CP-SAT compilation")
        private_history_incomplete = histories is None and (
            any(seat != exact.next_actor for seat in recall)
            or (self.native_cp_sat and exact.state.get("escalation") not in {None, "normal"}))
        if private_history_incomplete and request.mode == "exact":
            raise InformationSetCompilationStopped("private_history_coverage_unavailable", stage="observed_decision",
                evidence={"privateHistoryScopeHash": canonical_sha256(recall)})
        bounded = request.mode == "bounded"
        if request.visibility != "actor_information_set" or request.mode not in {
            "exact", "bounded",
        }:
            raise ValueError(
                "actor-information-set compiler supports exact visibility only"
            )
        if bounded and not self.native_cp_sat:
            raise ValueError(
                "bounded actor analysis requires native CP-SAT"
            )
        if request.claim == "winning_continuation_line_count":
            raise ValueError(
                "actor-information-set continuation counts are unsupported"
            )
        started = time.monotonic()
        deadline = started + maximum_seconds
        world_states, allocation_model = self._enumerate_worlds(
            exact,
            maximum_worlds=maximum_worlds // (len(histories) if histories is not None else 1),
            maximum_attempts=maximum_nodes,
            deadline=deadline,
            cancel_check=cancel_check,
            bounded_world_scope=bounded,
        )
        target = Team[str(request.target_team).upper()]
        root_actor = Seat.parse(exact.state.get("currentPlayer"))
        root_actor_relation = "target" if root_actor.team == target else "opponent"
        compile_worlds = self._compile_play_worlds_parallel if graph_workers > 1 and len(world_states) > 1 else self._compile_play_worlds
        worlds, node_count, boundary_count, transition_models, recall = compile_worlds(
            exact, request, world_states, maximum_nodes=maximum_nodes, maximum_depth=maximum_depth,
            deadline=deadline, cancel_check=cancel_check, requested_recall=requested_recall, histories=histories,
            project_receipts={row["worldStateHash"]: row
                              for row in allocation_model.get("projectLifecycleBindings", ())
                              if "selectedProjects" in row},
            **({"graph_workers": graph_workers, "worker_cpu_callback": worker_cpu_callback}
                if graph_workers > 1 and len(world_states) > 1 else {}))
        problem = FiniteInformationSetProblemV1.build(
            problem_id=f"actor-information-{canonical_sha256({'exactStateId': exact.exact_state_id, 'request': request.sha256, 'worlds': [world.world_id for world in worlds]})}",
            worlds=worlds,
            success_threshold=_integral_success_threshold(request.success_threshold),
        )
        coverage = {
            "count": len(worlds),
            "complete": True,
            "coverage": {
                "method": "exhaustive_cp_sat_hidden_allocations_v1",
                "observerInformationHash": canonical_observer_information_hash_v1(
                    complete_state=exact.state,
                    acting_seat=str(exact.state.get("currentPlayer")),
                ),
                "scopeHash": canonical_sha256([world.world_id for world in problem.worlds]),
            },
        }
        analysis_scope = None
        if bounded:
            world_scope = allocation_model["worldScope"]
            coverage["complete"] = world_scope["complete"]
            coverage["coverage"]["method"] = world_scope["method"]
            analysis_scope = {
                "mode": "bounded",
                "requestHash": request.sha256,
                "horizon": dict(request.horizon),
                "boundaryEvaluator": request.boundary_evaluator,
                "boundaryEvaluated": boundary_count > 0,
                "boundaryNodeCount": boundary_count,
                "continuationCoverage": "requested_horizon" if boundary_count else "round_end",
                "worldCoverageComplete": world_scope["complete"],
                "worldScopeHash": coverage["coverage"]["scopeHash"],
                "worldLimit": maximum_worlds,
                "probabilityModel": None,
            }
            allocation_model["transitionModels"] = transition_models
            analysis_scope["transitionModelsHash"] = canonical_sha256(transition_models)
        if histories is not None:
            allocation_model["priorHistoryWorldsHash"] = canonical_sha256(histories)
        elif requested_recall:
            allocation_model["privateRecallHash"] = canonical_sha256(requested_recall)
        if requested_recall:
            coverage["coverage"]["observerInformationHash"] = bind_escalation_recall_v1(
                coverage["coverage"]["observerInformationHash"], requested_recall.get(root_actor.name.lower(), ()))
        if private_history_incomplete:
            coverage["complete"] = False
            analysis_scope = {**(analysis_scope or {}), "worldCoverageComplete": False,
                "privateHistoryCoverage": "declared_history_only", "privateHistoryScopeHash": canonical_sha256(recall)}
        compilation = ActorInformationSetCompilationV1(
            ActorInformationSetGraphV1(
                problem,
                coverage,
                allocation_model,
                root_actor=root_actor.name.lower(),
                root_actor_team=root_actor.team.name.lower(),
                root_actor_relation=root_actor_relation,
                analysis_scope=analysis_scope,
            ),
            node_count,
        )
        return self._bind_observer_scope(compilation, exact, request, world_states)


def _certify_actor_action_quality_v1(
    receipt: ActorInformationSetProofReceiptV1,
    *,
    action_ids: Mapping[str, int],
    actor_relation: str,
    selected_action: str | None,
    deadline: float,
    cancel_check: Callable[[], bool] | None,
    analysis_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extend native root certificates with best choice and actual-action regret.

    Values come from the certified CP-SAT model, not a Python best-move
    evaluator. Regret is in the proof utility's units, not a win probability.
    """
    if receipt.completion_status != "exact_complete":
        raise ValueError("action quality requires complete root certificates")
    return _certify_action_quality_model_v1(actions=[row.root_action for row in receipt.roots],
        qualification_model_proto=receipt.qualification_model_proto,
        qualification_variable_bindings=receipt.qualification_variable_bindings,
        qualification_model_hash=receipt.qualification_model_hash, action_ids=action_ids,
        actor_relation=actor_relation, selected_action=selected_action, deadline=deadline,
        cancel_check=cancel_check, analysis_scope=analysis_scope)


def _certify_action_quality_model_v1(*, actions, qualification_model_proto, qualification_variable_bindings,
                                    qualification_model_hash, action_ids, actor_relation, selected_action,
                                    deadline, cancel_check, analysis_scope=None,
                                    scope="compiled_actor_information_set", maximum_conflicts=None):
    """Shared CP-SAT quality lowering for certified native and finite root models."""
    if maximum_conflicts is not None and (type(maximum_conflicts) is not int or maximum_conflicts < 1):
        raise ValueError("action quality conflict limit must be a positive integer")
    if actor_relation not in {"target", "opponent"}:
        raise ValueError("action quality requires an actor team relation")
    if scope not in {"compiled_actor_information_set", "compiled_complete_state"}:
        raise ValueError("action quality scope is unsupported")
    if set(action_ids) != set(actions) or len(set(action_ids.values())) != len(actions):
        raise ValueError("action quality IDs must cover every legal root exactly once")
    if selected_action is not None and selected_action not in action_ids:
        raise ValueError("action quality selected action is not legal")
    model = cp_model.CpModel()
    if (
        not qualification_model_proto
        or not model.proto.parse_text_format(qualification_model_proto)
    ):
        raise ValueError("action quality requires a native qualification model")
    by_name = {variable.name: index for index, variable in enumerate(model.proto.variables)}
    values = {
        action: model.get_int_var_from_proto_index(by_name[value_name])
        for action, value_name, _winning_name in qualification_variable_bindings
    }
    if set(values) != set(actions):
        raise ValueError("action quality requires every root value binding")
    # The native proto container does not implement Python negative indexing.
    domains = [tuple(model.proto.variables[value.index].domain) for value in values.values()]
    lower = min(domain[0] for domain in domains)
    upper = max(domain[-1] for domain in domains)
    best_value = model.new_int_var(lower, upper, "quality_best_value")
    if actor_relation == "target":
        model.add_max_equality(best_value, list(values.values()))
    else:
        model.add_min_equality(best_value, list(values.values()))
    best_flags = {}
    tie_candidates = []
    sentinel = max(action_ids.values()) + 1
    for ordinal, action in enumerate(actions):
        optimal = model.new_bool_var(f"quality_optimal_{ordinal}")
        model.add(values[action] == best_value).only_enforce_if(optimal)
        model.add(values[action] != best_value).only_enforce_if(optimal.Not())
        tie = model.new_int_var(0, sentinel, f"quality_tie_{ordinal}")
        model.add(tie == action_ids[action]).only_enforce_if(optimal)
        model.add(tie == sentinel).only_enforce_if(optimal.Not())
        tie_candidates.append(tie)
        best_flags[action] = optimal
    best_action = model.new_int_var(0, sentinel - 1, "quality_best_action_id")
    model.add_min_equality(best_action, tie_candidates)
    actual_value = model.new_int_var(lower, upper, "quality_actual_value")
    if selected_action is None:
        model.add(actual_value == best_value)
    else:
        model.add(actual_value == values[selected_action])
    regret = model.new_int_var(0, upper - lower, "quality_actual_regret")
    model.add(regret == (
        best_value - actual_value if actor_relation == "target"
        else actual_value - best_value
    ))
    optimal_actual = model.new_bool_var("quality_actual_optimal")
    model.add(regret == 0).only_enforce_if(optimal_actual)
    model.add(regret > 0).only_enforce_if(optimal_actual.Not())
    proto = str(model.proto)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise InformationSetCompilationStopped("wall_time", stage="action_quality")
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = remaining
    if maximum_conflicts is not None:
        solver.parameters.max_number_of_conflicts = maximum_conflicts
    status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
    if status in {cp_model.INFEASIBLE, cp_model.MODEL_INVALID}:
        return {
            "schemaVersion": 1,
            "kind": "actor_information_set_action_quality_disagreement_v1",
            "modelStatus": "quarantined",
            "quarantineReason": "action_quality_certificate_disagreement",
            "sourceQualificationModelHash": qualification_model_hash,
            "solverStatus": solver.status_name(status).lower(),
            "modelProto": proto,
        }
    if status != cp_model.OPTIMAL:
        raise InformationSetCompilationStopped(
            "action_quality_not_optimal", stage="action_quality",
            evidence={"solverStatus": solver.status_name(status).lower()},
        )
    selected_id = int(solver.value(best_action))
    core = {
        "schemaVersion": 1,
        "authority": "cp_sat",
        "scope": scope,
        "sourceQualificationModelHash": qualification_model_hash,
        "rootActorRelation": actor_relation,
        "actualAction": selected_action or next(
            action for action, identifier in action_ids.items() if identifier == selected_id
        ),
        "bestActionId": selected_id,
        "optimalActions": [action for action in actions if solver.value(best_flags[action])],
        "bestValue": int(solver.value(best_value)),
        "actualValue": int(solver.value(actual_value)),
        "regret": int(solver.value(regret)),
        "actualActionOptimal": bool(solver.value(optimal_actual)),
        "variableBindings": {
            "bestValue": best_value.name,
            "bestActionId": best_action.name,
            "actualValue": actual_value.name,
            "regret": regret.name,
            "actualActionOptimal": optimal_actual.name,
        },
        "modelProto": proto,
        "variableCount": len(model.proto.variables),
        "constraintCount": len(model.proto.constraints),
        "solverStatus": "optimal",
    }
    if analysis_scope is not None:
        core["analysisScope"] = copy.deepcopy(dict(analysis_scope))
    if selected_action is None and len(core["optimalActions"]) > 1:
        core["selectionTieBreak"] = "stable_action_id"
    return {**core, "sha256": canonical_sha256(core)}


def prove_actor_information_set_v1(
    compilation: ActorInformationSetCompilationV1,
    request: ScenarioProofRequestV1,
    *,
    maximum_nodes: int,
    maximum_worlds: int,
    maximum_seconds: float,
    cancel_check: Callable[[], bool] | None = None,
    native_action_quality: bool = False,
    declared_maximum_seconds: float | None = None,
    policy_search: str | None = None,
    native_opponent_stop: bool = False,
    _root_witnesses: list[dict[str, Any]] | None = None,
) -> OrToolsGraphProofOutcomeV1:
    if _root_witnesses is not None and (type(_root_witnesses) is not list or _root_witnesses):
        raise ValueError("root witness sink must be an empty list")
    if type(native_opponent_stop) is not bool:
        raise TypeError("native opponent stopping flag must be boolean")
    if native_opponent_stop and (not native_action_quality or request.mode != "exact"
            or request.utility != "kabout" or compilation.graph.decision_kind != "play"):
        raise ValueError("native opponent stopping requires exact actor Kabout play")
    scope = compilation.graph.analysis_scope
    coverage = compilation.graph.hidden_worlds
    if request.visibility != "actor_information_set":
        raise ValueError("actor-information proof requires actor_information_set visibility")
    if coverage.get("count") != len(compilation.graph.problem.worlds):
        raise ValueError("actor-information world coverage count disagrees with graph")
    if compilation.graph.problem.success_threshold != _integral_success_threshold(request.success_threshold):
        raise ValueError("actor-information success threshold disagrees with request")
    if request.mode == "exact":
        if scope is not None or coverage.get("complete") is not True:
            raise ValueError("exact actor-information proof requires exhaustive coverage")
    elif request.mode == "bounded":
        if (
            not native_action_quality
            or scope is None
            or scope.get("mode") != "bounded"
            or scope.get("requestHash") != request.sha256
            or scope.get("horizon") != request.horizon
            or scope.get("boundaryEvaluator") != request.boundary_evaluator
            or scope.get("continuationCoverage") not in {"round_end", "requested_horizon"}
            or type(scope.get("boundaryEvaluated")) is not bool
            or scope.get("boundaryEvaluated") != (scope.get("continuationCoverage") == "requested_horizon")
            or type(scope.get("boundaryNodeCount")) is not int
            or scope["boundaryNodeCount"] < 0
            or scope["boundaryEvaluated"] != (scope["boundaryNodeCount"] > 0)
            or type(scope.get("worldLimit")) is not int
            or scope["worldLimit"] < len(compilation.graph.problem.worlds)
            or scope.get("probabilityModel") is not None
            or type(scope.get("worldCoverageComplete")) is not bool
            or scope.get("worldCoverageComplete") != coverage.get("complete")
            or scope.get("worldScopeHash") != canonical_sha256(
                [world.world_id for world in compilation.graph.problem.worlds]
            )
        ):
            raise ValueError("bounded actor analysis requires matching native scope evidence")
        transition_models = compilation.graph.model_document.get("transitionModels")
        if transition_models is not None and scope.get("transitionModelsHash") != canonical_sha256(transition_models):
            raise ValueError("bounded actor analysis transition model identities disagree")
    else:
        raise ValueError("actor-information proof mode is unsupported")
    deadline = time.monotonic() + maximum_seconds
    budget = InformationSetProofBudgetV1(
        maximum_nodes=maximum_nodes,
        maximum_worlds=maximum_worlds,
        maximum_iterations=maximum_nodes,
        maximum_wall_seconds=maximum_seconds if declared_maximum_seconds is None else declared_maximum_seconds,
        policy_search=policy_search,
    )
    receipt: ActorInformationSetProofReceiptV1 = (
        OrToolsActorInformationSetProofV1().prove(
            compilation.graph.problem,
            budget=budget,
            claim=request.claim,
            selected_action=request.selected_action,
            cancel_check=cancel_check,
            **({"runtime_maximum_seconds": maximum_seconds} if declared_maximum_seconds is not None else {}),
        )
    )

    def action_id(action_code: str) -> int:
        if compilation.graph.root_action_ids is not None:
            try:
                return int(compilation.graph.root_action_ids[action_code])
            except KeyError as exc:
                raise ValueError(
                    "actor-information proof action lacks a certified root ID"
                ) from exc
        return LegalPlayActionV1.from_code(action_code).action_id

    roots = tuple(
        ProofRootActionV1(
            action_id(row.root_action),
            row.root_action,
            int(row.robust_value or 0),
            row.satisfies is True,
        )
        for row in receipt.roots
    )
    selected_code = request.selected_action
    action_quality = None
    if native_action_quality and receipt.completion_status == "exact_complete":
        action_quality = _certify_actor_action_quality_v1(
            receipt,
            action_ids={row.root_action: action_id(row.root_action) for row in receipt.roots},
            actor_relation=compilation.graph.root_actor_relation,
            selected_action=selected_code,
            deadline=deadline,
            cancel_check=cancel_check,
            analysis_scope=scope,
        )
        if action_quality.get("modelStatus") == "quarantined":
            return OrToolsGraphProofOutcomeV1.build(
                completion_status="model_error", solver_status="model_error",
                claim_satisfied=False, selected_action=None,
                winning_root_action_count=None, root_actions=(), principal_variation=(),
                continuation_lines=None, policies=(),
                graph_hash=compilation.graph.to_dict()["sha256"],
                model_hash=canonical_sha256(action_quality), request_hash=request.sha256,
                variable_count=receipt.compiled_nodes,
                constraint_count=receipt.qualification_constraint_count,
                model_document=action_quality, hidden_worlds=compilation.graph.hidden_worlds,
                proof_engine_id="ortools_actor_information_set_cegis_v1",
                proof_engine_version="1.3.0",
            )
        selected_code = action_quality["actualAction"]
    if (
        selected_code is None
        and receipt.completion_status == "exact_complete"
        and receipt.roots
    ):
        values = [
            int(row.robust_value)
            for row in receipt.roots
            if row.robust_value is not None
        ]
        if len(values) != len(receipt.roots):
            raise ValueError("exact actor-information proof lacks a root value")
        best_value = (
            max(values)
            if compilation.graph.root_actor_relation == "target"
            else min(values)
        )
        selected_code = min(
            (
                row.root_action
                for row in receipt.roots
                if row.robust_value == best_value
            ),
            key=action_id,
        )
    count = receipt.winning_root_action_count
    claim_satisfied = bool(receipt.qualification_claim_satisfied)
    completion_status = receipt.completion_status
    if completion_status == "exact_complete" and not claim_satisfied:
        completion_status = "proof_failed"
    elif completion_status == "exact_complete" and request.mode == "bounded":
        completion_status = "bounded_complete"
    selected_receipt = next(
        (row for row in receipt.roots if row.root_action == selected_code),
        None,
    )
    binding_model_proto = receipt.qualification_model_proto
    binding_variables = receipt.qualification_variable_count
    binding_constraints = receipt.qualification_constraint_count
    if action_quality is not None:
        binding_model_proto = action_quality["modelProto"]
        binding_variables = action_quality["variableCount"]
        binding_constraints = action_quality["constraintCount"]
    root_action_variables = [
        {
            "actionId": action_id(action),
            "action": action,
            "valueVariable": value_variable,
            "winningVariable": winning_variable,
        }
        for action, value_variable, winning_variable in receipt.qualification_variable_bindings
    ]
    if receipt.completion_status == "exact_complete":
        if (
            binding_model_proto is None
            or receipt.qualification_model_hash is None
            or len(root_action_variables) != len(receipt.roots)
        ):
            raise ValueError("exact actor-information proof lacks root certificates")
    model_document = {
        "schemaVersion": 1,
        "kind": "actor_information_set_cegis_model_v1",
        "problemHash": receipt.problem_hash,
        "budgetHash": receipt.budget_hash,
        "completionStatus": completion_status,
        "receiptCompletionStatus": receipt.completion_status,
        "modelStatus": receipt.model_status,
        "quarantineReason": receipt.quarantine_reason,
        "claimSatisfied": claim_satisfied,
        "rootActor": {
            "seat": compilation.graph.root_actor,
            "team": compilation.graph.root_actor_team,
            "relation": compilation.graph.root_actor_relation,
        },
        "rootActorSeat": compilation.graph.root_actor,
        "rootActorTeam": compilation.graph.root_actor_team,
        "rootActorRelation": compilation.graph.root_actor_relation,
        "targetTeam": request.target_team,
        "roots": [
            {
                "action": row.root_action,
                "completionStatus": row.completion_status,
                "solverStatus": row.solver_status,
                "robustValue": row.robust_value,
                "satisfies": row.satisfies,
                "exhaustionReason": row.exhaustion_reason,
                "targetPolicyCount": row.target_policy_count,
                "opponentPolicyCount": row.opponent_policy_count,
                "masterModelHash": row.master_model_hash,
                "adversaryModelHash": row.adversary_model_hash,
                "cutHashes": list(row.cut_hashes),
                "iterations": row.iterations,
            }
            for row in receipt.roots
        ],
    }
    if compilation.graph.decision_kind != "play":
        model_document["decisionKind"] = compilation.graph.decision_kind
    if scope is not None:
        model_document["analysisScope"] = copy.deepcopy(dict(scope))
        model_document["rootActionCountScope"] = "declared_worlds_only"
    if action_quality is not None:
        model_document["actionQuality"] = {
            key: value for key, value in action_quality.items() if key != "modelProto"
        }
    if binding_model_proto is not None:
        model_document["qualificationSource"] = (
            "cp_sat_cegis_root_certificates_v2"
            if receipt.completion_status == "exact_complete"
            else "cp_sat_cegis_root_binding_model_v2"
        )
        model_document["qualificationModelHash"] = receipt.qualification_model_hash
        model_document["rootActionVariables"] = root_action_variables
        model_document["winningRootActionCountVariable"] = "winning_root_action_count"
        if receipt.qualification_claim_binding is None:
            raise ValueError("actor-information proof lacks a claim binding")
        model_document["claimVariable"] = receipt.qualification_claim_binding[0]
        model_document["claimEvidenceBinding"] = {
            "kind": request.claim,
            "variable": receipt.qualification_claim_binding[0],
            "literalIndex": receipt.qualification_claim_binding[1],
            "winningCountVariable": "winning_root_action_count",
        }
        model_document["modelProto"] = binding_model_proto
    stop_nodes = stop_constraints = 0
    if native_opponent_stop:
        companion = {"scope": {
            "graphHash": compilation.graph.to_dict()["sha256"], "problemHash": receipt.problem_hash,
            "requestHash": request.sha256, "qualificationModelHash": receipt.qualification_model_hash,
            "worldIds": [world.world_id for world in compilation.graph.problem.worlds],
            "roots": [[row.action_code, row.action_id] for row in roots],
            "transform": "swap_policy_roles_negate_native_kabout_v1",
        }, "completionStatus": "proof_incomplete"}
        if receipt.completion_status == "exact_complete":
            source_model = cp_model.CpModel()
            if not source_model.proto.parse_text_format(binding_model_proto):
                raise ValueError("opponent stopping lacks the original native model")
            companion["sourcePrefix"] = {
                "variables": len(source_model.proto.variables), "constraints": len(source_model.proto.constraints),
                "sha256": canonical_sha256({"variables": list(map(str, source_model.proto.variables)),
                    "constraints": list(map(str, source_model.proto.constraints))}),
            }
            prover = OrToolsActorInformationSetProofV1()
            try:
                dual = prover._opponent_stop_problem(compilation.graph.problem,
                    maximum_nodes=maximum_nodes, deadline=deadline, cancel_check=cancel_check)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _FiniteProofBudgetExceeded("cancelled_or_wall_time")
                stop = prover.prove(dual, budget=budget, claim="winning_root_action_count",
                    cancel_check=cancel_check, runtime_maximum_seconds=remaining)
                companion.update(completionStatus=stop.completion_status, problemHash=stop.problem_hash,
                    budgetHash=stop.budget_hash, roots=[row.to_dict() for row in stop.roots])
                stop_nodes = stop.compiled_nodes
                if stop.completion_status == "exact_complete":
                    qualified = prover._certify_root_thresholds(stop.roots, success_threshold=1,
                        problem_hash=stop.problem_hash, claim="winning_root_action_count", selected_action=None,
                        deadline=deadline, cancel_check=cancel_check, model=source_model,
                        variable_prefix="opponent_stop_")
                    if qualified.status == "optimal":
                        companion.update(solverStatus="optimal", modelHash=qualified.model_hash,
                            modelCore=qualified.model_core,
                            rootActionVariables=[{"action": action, "actionId": action_id(action),
                                "valueVariable": value, "winningVariable": winning}
                                for action, value, winning in qualified.bindings])
                        model_document["modelProto"] = qualified.model_proto
                        binding_variables, binding_constraints = qualified.variable_count, qualified.constraint_count
                        stop_constraints = sum(len(row.cut_hashes) + 2 for row in stop.roots)
                    else:
                        companion["completionStatus"] = "proof_incomplete"
            except _FiniteProofBudgetExceeded as exc:
                companion["exhaustionReason"] = str(exc)
        companion["sha256"] = canonical_sha256(companion)
        model_document["opponentStopCertificate"] = companion
    model_hash = canonical_sha256(model_document)
    graph_hash = compilation.graph.to_dict()["sha256"]
    if _root_witnesses is not None and scope is None and receipt.completion_status == "exact_complete":
        # Keep already-solved adversarial lines outside historical proof/model identities.
        _root_witnesses.extend({
            "action": row.root_action,
            "worstCaseValue": row.robust_value,
            "worldId": row.worst_world_id,
            "targetPolicyHash": row.target_policy_hash,
            "opponentPolicyHash": row.worst_opponent_policy_hash,
            "principalVariation": list(row.principal_variation),
        } for row in receipt.roots)
    return OrToolsGraphProofOutcomeV1.build(
        completion_status=completion_status,
        solver_status=(
            "optimal"
            if completion_status in {"exact_complete", "bounded_complete", "proof_failed"}
            else "model_error"
            if completion_status == "model_error"
            else "unknown"
        ),
        claim_satisfied=claim_satisfied,
        selected_action=selected_code,
        winning_root_action_count=count,
        winning_root_action_lower_bound=receipt.winning_root_action_lower_bound,
        winning_root_action_upper_bound=receipt.winning_root_action_upper_bound,
        root_actions=roots,
        principal_variation=(
            selected_receipt.principal_variation if selected_receipt else ()
        ),
        continuation_lines=None,
        policies=(),
        graph_hash=graph_hash,
        model_hash=model_hash,
        request_hash=request.sha256,
        variable_count=receipt.compiled_nodes + binding_variables + stop_nodes,
        constraint_count=(
            sum(len(row.cut_hashes) + 2 for row in receipt.roots) + binding_constraints + stop_constraints
        ),
        model_document=model_document,
        hidden_worlds=compilation.graph.hidden_worlds,
        proof_engine_id="ortools_actor_information_set_cegis_v1",
        proof_engine_version=(
            "1.4.0" if scope is not None
            else "1.3.0" if native_action_quality else "1.2.0"
        ),
    )


__all__ = [
    "ActorInformationGraphCompilerV1",
    "ActorInformationGraphWorldV1",
    "ActorInformationSetCompilationV1",
    "ActorInformationSetGraphV1",
    "InformationSetCompilationStopped",
    "OrToolsActorInformationSetCompilerV1",
    "_integral_success_threshold",
    "hidden_allocation_upper_bound_v1",
    "prove_actor_information_set_v1",
]
