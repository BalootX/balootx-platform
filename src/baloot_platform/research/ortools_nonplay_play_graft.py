"""Graft exhaustive Domain-IR play graphs onto exact-input non-play roots."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.models import LegalPlayActionV1, Project
from baloot_platform.core.project_ir import ProjectDeclarationActionV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1

from .ortools_complete_state_graph import (
    CompleteStateDecisionObserverV1,
    CompleteStateGraphSeedV1,
    OrToolsCompleteStateGraphCompilerV1,
)
from .ortools_exact_state import (
    ExactNormalizationStopped,
    OrToolsExactStateNormalizerV1,
    OrToolsNormalizedExactStateV1,
    _cp_model_hash,
    _native_forced_pass_certificate,
    _certified_project_followup_rows,
    _effective_prefix_hash,
)
from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1
from .ortools_nonplay_graphs import EscalationGraphActionV1, bind_escalation_recall_v1
from .ortools_play_tail import TailPrefixActionV1
from .ortools_proof import ProofGraphEdgeV1, ProofGraphNodeV1, ProofGraphV1, ProofGraphSequencePackV1
from .ortools_project_fragment import (
    ProjectLifecycleFragmentEvidenceV1,
    attach_project_lifecycle_fragment_v1,
)
from .scenario_contracts import (
    ExactTracePrefixV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)


@dataclass(frozen=True, slots=True)
class NonPlayPlayStartV1:
    prefix: ExactTracePrefixV1
    projects: tuple[Project, ...]
    project_evidence: ProjectLifecycleFragmentEvidenceV1 | None = None
    normalization_parent: OrToolsNormalizedExactStateV1 | None = None


def build_nonplay_play_start_prefix_v1(
    rules: RuleBook,
    prefix: ExactTracePrefixV1,
    *,
    buyer: Seat,
    contract: Contract,
    trump: Suit | None,
    escalation_actions: Sequence[EscalationGraphActionV1],
    projects: Sequence[Project] | None = None,
    project_actions: Sequence[ProjectDeclarationActionV1] = (),
) -> NonPlayPlayStartV1:
    """Build only the public action syntax for the first play decision."""

    # Compatibility inputs are intentionally not semantic inputs. The native
    # lifecycle solve below owns holdings, declarations, comparison and awards.
    del rules, buyer, contract, trump, projects
    value = prefix.to_dict()
    value.pop("sha256", None)
    value.update(
        {
            "escalationActions": [
                {
                    "actor": item.actor.name.lower(),
                    "action": item.action,
                    **({"mode": item.mode} if item.mode is not None else {}),
                }
                for item in escalation_actions
            ],
            "projectDeclarations": [],
            "projectAssertions": [],
            "stateAssertions": {},
        }
    )
    if project_actions:
        value["projectActions"] = [
            action.to_dict() for action in project_actions
        ]
    return NonPlayPlayStartV1(
        ExactTracePrefixV1.build(value),
        (),
    )


def selected_projects_from_exact_receipt_v1(
    normalized: OrToolsNormalizedExactStateV1,
) -> tuple[Project, ...]:
    """Read CP-selected project structure from an exact-normalization receipt."""

    receipt = normalized.normalization_receipt.payload
    document = receipt.get("derivedProjects")
    if not isinstance(document, Mapping) or document.get("solverStatus") != "optimal":
        raise ValueError("non-play project graph requires optimal CP project evidence")
    rows = document.get("projects")
    if not isinstance(rows, list):
        raise ValueError("non-play project receipt lacks project rows")
    projects = []
    for row in rows:
        if not isinstance(row, Mapping) or type(row.get("selected")) is not bool:
            raise ValueError("non-play project receipt row is invalid")
        if not row["selected"]:
            continue
        cards = row.get("cards")
        if not isinstance(cards, list):
            raise ValueError("non-play project receipt row lacks cards")
        projects.append(
            Project(
                Seat.parse(row.get("owner")),
                str(row.get("kind")),
                tuple(Card.parse(card) for card in cards),
                int(row.get("value")),
                int(row.get("highRank")),
                int(row.get("categoryPriority")),
            )
        )
    dealer = Seat.parse(normalized.state.get("dealer"))
    order = (dealer.next(), dealer.next().next(), dealer.previous(), dealer)
    projects.sort(
        key=lambda project: (
            order.index(project.owner),
            -project.value,
            -project.category_priority,
            -project.high_rank,
        )
    )
    return tuple(projects)


class OrToolsNonPlayPlayGraphGraftV1:
    """Own one globally budgeted non-play-root plus exhaustive-play graph."""

    def __init__(
        self,
        rules: RuleBook,
        request: ScenarioProofRequestV1,
        *,
        maximum_nodes: int,
        maximum_depth: int,
        maximum_seconds: float,
        native_cp_sat: bool = False,
        root_decision_kind: str | None = None,
        project_continuation: Callable[..., tuple[ProofGraphV1, Mapping[str, str]]
            | tuple[ProofGraphV1, Mapping[str, str], ProofGraphV1]] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        _normalizer: OrToolsExactStateNormalizerV1 | None = None,
        _compact_forced: bool = False,
        project_counterfactual: bool = False,
    ) -> None:
        if type(project_counterfactual) is not bool:
            raise TypeError("project_counterfactual must be boolean")
        if project_counterfactual and (not native_cp_sat or request.mode != "exact" or request.utility != "awarded_point_margin"):
            raise ValueError("project counterfactual requires exact native margin proof")
        self.project_counterfactual = project_counterfactual
        self.project_counterfactual_graphs: dict[str, ProofGraphV1] = {}
        self.project_compilation_evidence: dict[str, dict[str, Any]] = {}
        if request.mode not in {"exact", "bounded"}:
            raise ValueError("non-play play graft supports exact or bounded proof")
        if request.visibility != "complete_state":
            raise ValueError("non-play play graft requires complete-state visibility")
        if type(native_cp_sat) is not bool:
            raise TypeError("native_cp_sat must be bool")
        if type(_compact_forced) is not bool or (_compact_forced and not native_cp_sat):
            raise ValueError("compact graft requires native CP-SAT")
        self._compact_forced = _compact_forced
        self.rules = rules
        self.request = request
        self.native_cp_sat = native_cp_sat
        self._normalizer = _normalizer or OrToolsExactStateNormalizerV1(rules, native_cp_sat=native_cp_sat)
        if self._normalizer.rules.sha256 != rules.sha256 or self._normalizer.native_cp_sat != native_cp_sat:
            raise ValueError("shared normalizer differs from graft authority")
        self.root_decision_kind = root_decision_kind
        self.project_continuation = project_continuation
        if cancel_check is not None and not callable(cancel_check):
            raise TypeError("cancel_check must be callable")
        self.cancel_check = cancel_check
        self.maximum_nodes = maximum_nodes
        self.maximum_depth = maximum_depth
        self.deadline = time.perf_counter() + maximum_seconds
        self._terminals: dict[int, tuple[int, Callable[[], NonPlayPlayStartV1], tuple[tuple[str, str], ...]]] = {}
        self._next_sentinel = -2_000_000_000

    def terminal_utility(
        self,
        *,
        nonplay_depth: int,
        prefix_factory: Callable[[], NonPlayPlayStartV1],
        private_history: tuple[tuple[str, str], ...] = (),
    ) -> int:
        sentinel = self._next_sentinel
        self._next_sentinel -= 1
        self._terminals[sentinel] = (nonplay_depth, prefix_factory, private_history)
        return sentinel

    def _check_budget(self, *, seconds, nodes, depth, required_depth, stage):
        reason = (
            "cancelled" if self.cancel_check is not None and self.cancel_check()
            else "maximum_seconds" if seconds <= 0
            else "maximum_nodes" if nodes < 1
            else "maximum_depth" if depth < required_depth else None
        )
        if reason is not None:
            raise ExactNormalizationStopped(reason, evidence={
                "sourceStage": stage, "requestHash": self.request.sha256,
                "rulesHash": self.rules.sha256,
                "remainingSeconds": seconds, "remainingNodes": nodes,
                "remainingDepth": depth, "requiredDepth": required_depth,
            })

    def graft(self, base: ProofGraphV1) -> ProofGraphV1:
        graph, _observations = self._graft(base)
        return graph

    def graft_observed(
        self,
        base: ProofGraphV1,
        *,
        base_observation_ids: Mapping[str, str],
        decision_observer_factory: Callable[
            [OrToolsNormalizedExactStateV1, NonPlayPlayStartV1],
            CompleteStateDecisionObserverV1,
        ],
        project_observer_factory: Callable | None = None,
    ) -> tuple[ProofGraphV1, Mapping[str, str]]:
        """Graft play while preserving every pre-transposition observation ID."""

        if not callable(decision_observer_factory):
            raise TypeError("decision_observer_factory must be callable")
        return self._graft(
            base,
            base_observation_ids=base_observation_ids,
            decision_observer_factory=decision_observer_factory,
            project_observer_factory=project_observer_factory,
        )

    def _graft(
        self,
        base: ProofGraphV1,
        *,
        base_observation_ids: Mapping[str, str] | None = None,
        decision_observer_factory: Callable[
            [OrToolsNormalizedExactStateV1, NonPlayPlayStartV1],
            CompleteStateDecisionObserverV1,
        ]
        | None = None,
        project_observer_factory: Callable | None = None,
    ) -> tuple[ProofGraphV1, Mapping[str, str]]:
        observed = decision_observer_factory is not None
        if observed and self.project_counterfactual:
            raise ValueError("project counterfactual currently requires complete-state grafting")
        if observed != (base_observation_ids is not None):
            raise ValueError(
                "observed graft requires both base observations and a projector"
            )
        if base_observation_ids is not None:
            base_decisions = {
                node.node_id for node in base.nodes if not node.is_leaf
            }
            if set(base_observation_ids) != base_decisions:
                raise ValueError(
                    "base observation IDs must cover every base decision exactly"
                )
        replacements: dict[str, ProofGraphV1] = {}
        replacement_observations: dict[str, Mapping[str, str]] = {}
        boundary_replacements: set[str] = set()
        normalized_cache: dict[str, OrToolsNormalizedExactStateV1] = {}
        graph_cache: dict[tuple[str, str, int], ProofGraphV1] = {}
        used_nodes = self._node_count(base)
        play_compiler = OrToolsCompleteStateGraphCompilerV1(
            play_domain=self.rules.play_domain,
            scoring_domain=ScoringDomainIRV1.from_rules(self.rules.data),
            structural_domain=self.rules.structural_domain,
            project_domain=self.rules.project_domain,
            native_conversion=self.native_cp_sat,
        )
        for node in base.nodes:
            if self.cancel_check is not None and self.cancel_check():
                raise ExactNormalizationStopped("cancelled")
            if not node.is_leaf or node.utility not in self._terminals:
                continue
            if replacements:
                used_nodes = self._replacement_node_count(base, replacements)
            depth, prefix_factory, private_history = self._terminals[int(node.utility)]
            remaining_depth = self.maximum_depth - depth
            adjusted = self._adjusted_request(depth)
            if adjusted is None:
                boundary_replacements.add(node.node_id)
                continue
            if remaining_depth < 1:
                self._check_budget(seconds=self.deadline - time.perf_counter(),
                                   nodes=self.maximum_nodes - used_nodes, depth=remaining_depth,
                                   required_depth=1, stage="combined_nonplay_frontier")
            if (
                self.root_decision_kind == "escalation"
                and self._raw_margin_is_invariant_before_first_trick(adjusted)
            ):
                boundary_replacements.add(node.node_id)
                continue
            remaining_seconds = self.deadline - time.perf_counter()
            # The continuation replaces this sentinel; it does not append to it.
            remaining_nodes = self.maximum_nodes - used_nodes + 1
            self._check_budget(seconds=remaining_seconds, nodes=remaining_nodes,
                               depth=remaining_depth, required_depth=1, stage="combined_nonplay_frontier")
            start = prefix_factory()

            declaration_count = len(
                self.rules.project_domain.declaration_seats(
                    Seat.parse(start.prefix.dealer)
                )
            )
            if (
                self.native_cp_sat
                and start.prefix.project_actions_present
                and len(start.prefix.play_actions) < declaration_count
                and len(start.prefix.project_actions) == len(start.prefix.play_actions)
            ):
                child, child_observations = self._continue_project(
                    start.prefix,
                    adjusted,
                    remaining_nodes,
                    remaining_depth,
                    remaining_seconds,
                    retained=replacements,
                )
                replacements[node.node_id] = self._namespace_paired_graph(
                    child, node.node_id
                )
                if observed:
                    replacement_observations[node.node_id] = (
                        self._namespace_observations(
                            child,
                            node.node_id,
                            child_observations,
                            private_history=private_history,
                        )
                    )
                used_nodes += self._node_count(child) - 1
                continue

            def retain_forced_graph(graph):
                self._check_budget(seconds=self.deadline - time.perf_counter(),
                    nodes=remaining_nodes - self._node_count(graph) + 1, depth=remaining_depth, required_depth=1,
                    stage="forced_first_trick_frontier")
                replacements[node.node_id] = self._namespace_graph(graph, node.node_id)
                if observed:
                    project_root = next(item for item in base.nodes if item.node_id == base.root_node_id)
                    prior_action = next(edge.action_code for edge in project_root.edges if edge.child_node_id == node.node_id)
                    replacement_observations[node.node_id] = self._namespace_observations(
                        graph, node.node_id, {graph.root_node_id: canonical_sha256({
                            "kind": "native_project_followup_play_v1",
                            "priorObservation": base_observation_ids[base.root_node_id],
                            "priorAction": prior_action,
                        })}, private_history=private_history)
                return self._node_count(graph) - 1

            if not self.project_counterfactual and start.normalization_parent is not None and (not observed or project_observer_factory is not None):
                forced_graph = self._final_project_play_graph(start.normalization_parent, start.prefix, adjusted,
                    compiler=play_compiler, maximum_nodes=remaining_nodes, maximum_depth=remaining_depth,
                    graph_cache=graph_cache)
                if forced_graph is not None:
                    used_nodes += retain_forced_graph(forced_graph)
                    continue
            parent_rows = (None if self.project_counterfactual else
                           self._third_project_parent_rows(start.normalization_parent, start.prefix, adjusted))
            normalized = (start.normalization_parent if parent_rows is not None
                          else normalized_cache.get(start.prefix.sha256))
            if normalized is None:
                if start.normalization_parent is not None:
                    start.normalization_parent.require_integrity()
                normalized = self._normalizer.normalize(
                    start.prefix,
                    decision_point={"kind": "play"},
                    maximum_seconds=remaining_seconds,
                    cancel_check=self.cancel_check,
                    _play_template=(start.normalization_parent._play_template
                                    if start.normalization_parent is not None else None),
                )
                normalized_cache[start.prefix.sha256] = normalized
            if not self.project_counterfactual and (not observed or project_observer_factory is not None):
                root_play_observation = None
                if observed:
                    project_root = next(item for item in base.nodes if item.node_id == base.root_node_id)
                    prior_action = next(edge.action_code for edge in project_root.edges if edge.child_node_id == node.node_id)
                    root_play_observation = canonical_sha256({"kind": "native_project_followup_play_v1",
                        "priorObservation": base_observation_ids[base.root_node_id], "priorAction": prior_action})
                batched = self._forced_project_sequence_graph(
                    normalized, start.prefix, adjusted, play_compiler,
                    remaining_nodes, remaining_depth,
                    project_observer_factory=project_observer_factory if observed else None,
                    root_play_observation=root_play_observation,
                    legal_action_rows=parent_rows,
                    shared_boundary_values=frozenset(
                        item.utility for graph in replacements.values()
                        if graph.scope == "bounded" and graph.boundary_evaluator == adjusted.boundary_evaluator
                        for item in graph.nodes if item.is_leaf and item.leaf_kind == "boundary"))
                if batched is not None:
                    graph, graph_observations = batched
                    replacements[node.node_id] = self._namespace_graph(graph, node.node_id)
                    if observed:
                        replacement_observations[node.node_id] = self._namespace_observations(
                            graph, node.node_id, graph_observations, private_history=private_history)
                    used_nodes += self._node_count(graph) - 1
                    continue
            if (
                self.native_cp_sat and self.root_decision_kind == "project"
                and start.prefix.project_actions_present
                and len(start.prefix.project_actions) < len(self.rules.project_domain.declaration_seats(Seat.parse(start.prefix.dealer)))
                and (adjusted.mode == "exact" or (
                    adjusted.mode == "bounded" and adjusted.horizon is not None
                    and adjusted.horizon["unit"] in {"plies", "decisions"}
                    and adjusted.boundary_evaluator == "current_raw_point_margin_v1"))
            ):
                # Only the declaring actor plays before this frontier. Preserve
                # its prior observation and chosen declaration across worlds.
                project_root = next(item for item in base.nodes if item.node_id == base.root_node_id)
                prior_action = next(edge.action_code for edge in project_root.edges if edge.child_node_id == node.node_id)

                def observe_followup(value):
                    if value.path or value.actor.name.lower() != project_root.actor_seat:
                        raise ValueError("single-play project frontier crossed a decision boundary")
                    assert base_observation_ids is not None
                    return canonical_sha256({
                        "kind": "native_project_followup_play_v1",
                        "priorObservation": base_observation_ids[base.root_node_id],
                        "priorAction": prior_action,
                    })

                continuation_request = self._adjusted_request(depth + 1)
                if continuation_request is not None:
                    # The native normalizer already proved the complete legal
                    # action set. Every successor is normalized below; compiling
                    # a disposable one-play graph would solve those edges twice.
                    normalized.require_integrity()
                    if normalized.next_actor != project_root.actor_seat:
                        raise ValueError("single-play project frontier crossed a decision boundary")
                    actor = Seat.parse(normalized.next_actor)
                    root_id = canonical_sha256({"kind": "native_project_followup_v1",
                        "normalization": normalized.normalization_receipt.sha256})
                    edges, leaves = [], []
                    for row in (normalized.legal_actions if parent_rows is None else parent_rows):
                        action = LegalPlayActionV1.from_code(str(row["code"]))
                        action_id = action.card.id + 32 * int(action.ekka)
                        child_id = canonical_sha256({"root": root_id, "action": action_id})
                        edges.append(ProofGraphEdgeV1.build(action_id, str(row["code"]), child_id,
                                                           play_trick_index=0))
                        leaves.append(ProofGraphNodeV1.leaf(child_id, 0))
                    self._check_budget(seconds=self.deadline - time.perf_counter(),
                        nodes=remaining_nodes - len(leaves), depth=remaining_depth,
                        required_depth=2, stage="project_continuation")
                    graph = ProofGraphV1.build(scope="exact" if adjusted.mode == "exact" else "bounded", root_node_id=root_id,
                        boundary_evaluator=adjusted.boundary_evaluator,
                        nodes=[*leaves, ProofGraphNodeV1.decision(root_id, actor.team.name.lower(),
                                                                edges, actor_seat=actor.name.lower())])
                    graph_observations = ({root_id: canonical_sha256({
                        "kind": "native_project_followup_play_v1",
                        "priorObservation": base_observation_ids[base.root_node_id],
                        "priorAction": prior_action,
                    })} if observed else {})
                else:
                    single_play = adjusted.to_dict()
                    single_play.pop("sha256", None)
                    single_play["horizon"] = {"unit": adjusted.horizon["unit"], "value": 1}
                    compilation = play_compiler.compile(
                        normalized, ScenarioProofRequestV1.build(single_play), expected_rules_hash=self.rules.sha256,
                        maximum_nodes=remaining_nodes, maximum_depth=remaining_depth,
                        _compact_forced=self._compact_forced,
                        maximum_seconds=max(0.001, self.deadline - time.perf_counter()),
                        decision_observer=observe_followup if observed else None,
                        cancelled=self.cancel_check,
                    )
                    if not compilation.complete or compilation.graph is None:
                        if self.cancel_check is not None and self.cancel_check():
                            raise ExactNormalizationStopped("cancelled")
                        raise ExactNormalizationStopped(
                            str(compilation.receipt.exhaustion_reason),
                            evidence={"sourceStage": "single_play_project_frontier",
                                      "causeReceipt": compilation.receipt.to_dict()},
                        )
                    graph = compilation.graph
                    graph_observations = dict(compilation.decision_observation_ids)
                if continuation_request is not None:
                    if self.project_continuation is None:
                        raise ValueError("project frontier requires a continuation compiler")
                    continuations = {}
                    continuation_nodes = self._node_count(graph)
                    play_root = next(item for item in graph.nodes if item.node_id == graph.root_node_id)
                    normalizer = self._normalizer
                    for edge in play_root.edges:
                        seconds = self.deadline - time.perf_counter()
                        continuation_nodes = self._replacement_node_count(graph, continuations)
                        nodes_left = remaining_nodes - continuation_nodes + 1
                        self._check_budget(seconds=seconds, nodes=nodes_left,
                                           depth=remaining_depth, required_depth=2, stage="project_continuation")
                        if parent_rows is None:
                            advanced = normalizer.append_normalized_action(start.prefix, normalized, edge.action_code)
                        else:
                            action = LegalPlayActionV1.from_code(edge.action_code)
                            raw = start.prefix.to_dict()
                            raw.pop("sha256")
                            raw["stateAssertions"] = {}
                            raw["playActions"].append({"actor": normalized.next_actor,
                                "card": action.card.id, "ekka": action.ekka})
                            advanced = ExactTracePrefixV1.build(raw)
                        child, child_observations = self._continue_project(
                            advanced, continuation_request, nodes_left, remaining_depth - 1, seconds,
                            retained=continuations,
                        )
                        continuations[edge.child_node_id] = self._namespace_paired_graph(child, edge.child_node_id)
                        if observed:
                            graph_observations.update(self._namespace_observations(
                                child, edge.child_node_id, child_observations,
                            ))
                        continuation_nodes += self._node_count(child) - 1
                    graph = self._replace_leaves(graph, continuations)
                replacements[node.node_id] = self._namespace_paired_graph(graph, node.node_id)
                if observed:
                    replacement_observations[node.node_id] = self._namespace_observations(
                        graph, node.node_id, graph_observations, private_history=private_history,
                    )
                used_nodes += self._node_count(graph) - 1
                continue
            forced_graph = (None if self.project_counterfactual else
                            self._final_project_play_graph(normalized, start.prefix, adjusted))
            if forced_graph is not None:
                used_nodes += retain_forced_graph(forced_graph)
                continue
            start, seed = self.native_project_play_seed(normalized, start.prefix)
            cache_key = (
                self._play_projection_hash(seed),
                adjusted.sha256,
                remaining_depth,
            )
            graph = (
                graph_cache.get(cache_key)
                if not observed
                and self._project_independent_boundary(adjusted, seed)
                else None
            )
            graph_observations: Mapping[str, str] = {}
            if graph is None:
                decision_observer = (
                    decision_observer_factory(normalized, start)
                    if decision_observer_factory is not None
                    else None
                )
                if (project_observer_factory is not None and self.root_decision_kind == "project"
                        and len(start.prefix.play_actions) == 3
                        and adjusted.mode == "bounded" and adjusted.horizon is not None
                        and adjusted.horizon["unit"] in {"plies", "decisions"}
                        and adjusted.horizon["value"] == 1
                        and adjusted.boundary_evaluator == "current_raw_point_margin_v1"):
                    def decision_observer(value, actor=normalized.next_actor, identity=root_play_observation):
                        if value.path or value.actor.name.lower() != actor:
                            raise ValueError("project follow-up observation crossed its play boundary")
                        return identity
                compilation = play_compiler.compile(
                    seed,
                    adjusted,
                    expected_rules_hash=self.rules.sha256,
                    maximum_nodes=remaining_nodes,
                    _compact_forced=self._compact_forced,
                    maximum_depth=remaining_depth,
                    maximum_seconds=max(0.001, self.deadline - time.perf_counter()),
                    decision_observer=decision_observer,
                    cancelled=self.cancel_check,
                    **({"project_counterfactual": True} if self.project_counterfactual else {}),
                )
                if not compilation.complete or compilation.graph is None:
                    if self.cancel_check is not None and self.cancel_check():
                        raise ExactNormalizationStopped("cancelled")
                    raise ExactNormalizationStopped(
                        str(compilation.receipt.exhaustion_reason),
                        evidence={"sourceStage": "combined_nonplay_graph",
                                  "causeReceipt": compilation.receipt.to_dict()},
                    )
                graph = compilation.graph
                if self.project_counterfactual:
                    if compilation.project_counterfactual_graph is None:
                        raise ValueError("native play compilation omitted project counterfactual")
                    document = compilation.model_document
                    if document is None or canonical_sha256(document) != compilation.receipt.model_hash:
                        raise ValueError("native play compilation omitted owned model evidence")
                    receipt = compilation.receipt.to_dict()
                    # Timings belong to execution receipts, not semantic identities.
                    receipt.pop("elapsedSeconds")
                    receipt.pop("sha256")
                    evidence = {"model": document, "receipt": receipt,
                                "seed": seed.to_dict(), "request": adjusted.to_dict(),
                                "exactTracePrefix": start.prefix.to_dict(),
                                "graph": graph.to_dict(),
                                "counterfactualGraph": compilation.project_counterfactual_graph.to_dict()}
                    self.project_compilation_evidence[canonical_sha256(evidence)] = evidence
                    self.project_counterfactual_graphs[graph.sha256] = compilation.project_counterfactual_graph
                graph_observations = compilation.decision_observation_ids
                if not observed and self._project_independent_boundary(
                    adjusted, seed
                ):
                    graph_cache[cache_key] = graph
            namespaced = self._namespace_paired_graph(graph, node.node_id)
            replacements[node.node_id] = namespaced
            if observed:
                replacement_observations[node.node_id] = (
                    self._namespace_observations(
                        graph,
                        node.node_id,
                        graph_observations,
                        private_history=private_history,
                    )
                )
            used_nodes += self._node_count(namespaced) - 1
        combined = self._replace_leaves(
            base,
            replacements,
            boundary_replacements=boundary_replacements,
        )
        if not observed:
            return combined, {}
        observations = dict(base_observation_ids or {})
        for values in replacement_observations.values():
            overlap = set(observations).intersection(values)
            if overlap:
                raise AssertionError("combined observation identity collision")
            observations.update(values)
        decision_nodes = {node.node_id for node in combined.nodes if not node.is_leaf}
        if set(observations) != decision_nodes:
            raise AssertionError(
                "combined observations must cover every decision node exactly"
            )
        return combined, dict(sorted(observations.items()))

    @staticmethod
    def _project_independent_boundary(
        request: ScenarioProofRequestV1,
        seed: CompleteStateGraphSeedV1,
    ) -> bool:
        if (
            request.mode != "bounded"
            or request.horizon is None
            or request.boundary_evaluator != "current_raw_point_margin_v1"
        ):
            return False
        unit, value = str(request.horizon["unit"]), int(request.horizon["value"])
        return unit in {"plies", "decisions"} and value < sum(
            len(hand) for hand in seed.remaining_hands
        )

    @staticmethod
    def _play_projection_hash(seed: CompleteStateGraphSeedV1) -> str:
        projection = seed.to_dict()
        for field in (
            "exactStateId",
            "awardedProjectPoints",
            "balootPoints",
            "proofStateProjectionHash",
        ):
            projection.pop(field)
        return canonical_sha256(projection)

    @staticmethod
    def _raw_margin_is_invariant_before_first_trick(
        request: ScenarioProofRequestV1,
    ) -> bool:
        """Skip play construction when the boundary cannot yet change.

        Non-play roots enter play before its first card.  Fewer than four play
        plies cannot complete a trick, so the governed current-raw-margin
        evaluator remains exactly zero for every continuation.
        """

        if (
            request.mode != "bounded"
            or request.horizon is None
            or request.boundary_evaluator != "current_raw_point_margin_v1"
        ):
            return False
        return (
            str(request.horizon["unit"]) in {"plies", "decisions"}
            and int(request.horizon["value"]) < 4
        )

    def graft_terminal_graphs(
        self,
        base: ProofGraphV1,
        terminal_graphs: Mapping[int, ProofGraphV1],
    ) -> ProofGraphV1:
        """Replace sentinel terminal leaves with their exhaustive next-stage graphs."""

        replacements = {
            node.node_id: self._namespace_graph(
                terminal_graphs[int(node.utility)], node.node_id
            )
            for node in base.nodes
            if node.is_leaf and node.utility in terminal_graphs
        }
        combined = self._replace_leaves(base, replacements)
        if self._node_count(combined) > self.maximum_nodes:
            raise RuntimeError(
                "combined non-play graph exhausted its global node budget"
            )
        return combined

    def graft_terminal_graph_pairs(
        self, base: ProofGraphV1,
        terminal_graphs: Mapping[int, tuple[ProofGraphV1, ProofGraphV1]],
    ) -> tuple[ProofGraphV1, ProofGraphV1]:
        """Carry native project interventions across the same non-play choices."""
        if not self.native_cp_sat or self.request.mode != "exact" or self.request.utility != "awarded_point_margin":
            raise ValueError("paired graft requires exact native margin proof")
        required = {node.utility for node in base.nodes if node.is_leaf and node.utility in self._terminals}
        if set(terminal_graphs) != required:
            raise ValueError("paired graft requires every registered continuation exactly once")
        if any(node.is_leaf and node.utility not in required and node.utility != 0 for node in base.nodes):
            raise ValueError("paired graft cannot retain an unpaired scoring terminal")
        for actual, counterfactual in terminal_graphs.values():
            actual.require_same_legal_graph(counterfactual)
        results = []
        for side in (0, 1):
            self._check_budget(seconds=self.deadline - time.perf_counter(), nodes=self.maximum_nodes,
                               depth=self.maximum_depth, required_depth=1, stage="paired_nonplay_graft")
            results.append(self.graft_terminal_graphs(base, {
                sentinel: pair[side] for sentinel, pair in terminal_graphs.items()
            }))
        self._check_budget(seconds=self.deadline - time.perf_counter(), nodes=self.maximum_nodes,
                           depth=self.maximum_depth, required_depth=1, stage="paired_nonplay_graft")
        results[0].require_same_legal_graph(results[1])
        return results[0], results[1]

    def graft_terminal_graphs_observed(
        self,
        base: ProofGraphV1,
        terminal_graphs: Mapping[
            int, tuple[ProofGraphV1, Mapping[str, str]]
        ],
        *,
        base_observation_ids: Mapping[str, str],
    ) -> tuple[ProofGraphV1, Mapping[str, str]]:
        """Replace stage leaves and rekey their certified observation bindings."""

        base_decisions = {node.node_id for node in base.nodes if not node.is_leaf}
        if set(base_observation_ids) != base_decisions:
            raise ValueError(
                "base observation IDs must cover every base decision exactly"
            )
        replacements: dict[str, ProofGraphV1] = {}
        observations = dict(base_observation_ids)
        for node in base.nodes:
            if not node.is_leaf or node.utility not in terminal_graphs:
                continue
            graph, child_observations = terminal_graphs[int(node.utility)]
            replacements[node.node_id] = self._namespace_graph(
                graph, node.node_id
            )
            namespaced = self._namespace_observations(
                graph,
                node.node_id,
                child_observations,
            )
            overlap = set(observations).intersection(namespaced)
            if overlap:
                raise AssertionError("combined observation identity collision")
            observations.update(namespaced)
        combined = self._replace_leaves(base, replacements)
        if self._node_count(combined) > self.maximum_nodes:
            raise RuntimeError(
                "combined non-play graph exhausted its global node budget"
            )
        decision_nodes = {node.node_id for node in combined.nodes if not node.is_leaf}
        if set(observations) != decision_nodes:
            raise AssertionError(
                "combined observations must cover every decision node exactly"
            )
        return combined, dict(sorted(observations.items()))

    def native_project_play_seed(self, normalized, prefix):
        """Derive the play seed through the same native declaration lifecycle."""
        if (self.native_cp_sat and self.rules.project_domain.tie_break_order == "second_trick_reveal"
                and not normalized.state.get("completedTricks")):
            projects, declarations, evidence_hash = self._native_project_lifecycle_evidence(
                normalized, prefix, declaration_only=True)
            start = NonPlayPlayStartV1(prefix, projects)
            return start, self._seed(normalized, projects,
                awarded_project_points=None, project_evidence_hash=evidence_hash,
                pending_projects={"dealer": Seat.parse(prefix.dealer).name.lower(),
                                  "declarations": declarations})
        evidence = self._native_project_lifecycle_evidence(normalized, prefix)
        start = NonPlayPlayStartV1(prefix, tuple(
            Project(item.owner, item.kind, item.cards, item.value, item.high_rank,
                    item.category_priority, item.declared,
                    item.declaration_order if item.declaration_order is not None else 999,
                    item.awarded)
            for item in evidence.projects if item.selected), evidence)
        return start, self._seed(normalized, start.projects,
            awarded_project_points=evidence.combined_awarded_value,
            project_evidence_hash=evidence.evidence_hash)

    def _seed(
        self,
        normalized: OrToolsNormalizedExactStateV1,
        projects: Sequence[Project],
        *,
        awarded_project_points: tuple[int, int] | None,
        project_evidence_hash: str,
        legal_action_rows=None,
        pending_projects=None,
    ) -> CompleteStateGraphSeedV1:
        state = normalized.state
        current = state.get("currentTrick")
        hands = state.get("hands")
        if not isinstance(current, Mapping) or not isinstance(hands, Mapping):
            raise ValueError("non-play continuation lacks an exact play state")
        next_actor = Seat.parse(normalized.next_actor)
        history = state.get("escalationHistory")
        if not isinstance(history, list):
            raise ValueError("non-play continuation lacks escalation history")
        last_team = None
        if history:
            final = history[-1]
            if not isinstance(final, Mapping):
                raise ValueError("non-play escalation history item must be an object")
            last_team = Team[str(final.get("team")).strip().upper()]
        core = {
            "normalizedExactStateId": normalized.exact_state_id,
            "projects": [project.to_dict() for project in projects],
            "projectEvidenceHash": project_evidence_hash,
        }
        return CompleteStateGraphSeedV1.build(
            exact_state_id="STATE-" + canonical_sha256(core)[:24].upper(),
            rules_hash=normalized.rules_hash,
            remaining_hands=tuple(
                tuple(Card.parse(card) for card in hands[seat.name.lower()])
                for seat in Seat
            ),
            current_trick=tuple(
                TailPrefixActionV1.build(
                    item.get("seat"),
                    item.get("card"),
                    ekka=item.get("ekka") is True,
                )
                for item in current.get("plays", ())
            ),
            first_leader=Seat.parse(current.get("leader")),
            next_actor=next_actor,
            legal_actions=tuple(
                TailPrefixActionV1.build(
                    next_actor,
                    item.get("card") or item.get("code"),
                    ekka=item.get("ekka") is True,
                )
                for item in (normalized.legal_actions if legal_action_rows is None else legal_action_rows)
            ),
            contract=Contract.parse(state.get("contract")),
            trump=(
                Suit.parse(state["trumpSuit"])
                if state.get("trumpSuit") is not None
                else None
            ),
            buyer=Seat.parse(state.get("buyer")),
            locked=bool(state.get("locked")),
            base_raw=tuple(
                sum(
                    int(trick["rawPoints"])
                    for trick in state.get("completedTricks", ())
                    if Seat.parse(trick["winner"]).team == team
                )
                for team in Team
            ),
            base_tricks=tuple(
                sum(
                    1
                    for trick in state.get("completedTricks", ())
                    if Seat.parse(trick["winner"]).team == team
                )
                for team in Team
            ),
            awarded_project_points=awarded_project_points,
            baloot_points=(0, 0),
            escalation=str(state.get("escalation")),
            last_escalating_team=last_team,
            starting_match_score=tuple(int(item) for item in state["matchScore"]),
            pending_projects=pending_projects,
        )

    def _forced_project_sequence_graph(self, normalized, prefix, request, compiler, nodes, depth,
                                       *, project_observer_factory=None, root_play_observation=None,
                                       shared_boundary_values=frozenset(), legal_action_rows=None):
        """Compile one raw-point trick, retaining its forced declaration nodes."""
        played = len(prefix.play_actions)
        if not (self.native_cp_sat and self.root_decision_kind == "project"
                and played < 3 and len(prefix.project_actions) == played + 1
                and request.mode == "bounded" and request.horizon is not None
                and request.horizon["unit"] in {"plies", "decisions"}
                and request.horizon["value"] == 2 * (4 - played) - 1
                and request.boundary_evaluator == "current_raw_point_margin_v1"):
            return None
        normalized.require_integrity()
        receipt = normalized.normalization_receipt.payload
        certificate = _native_forced_pass_certificate(normalized)
        if certificate is None:
            return None
        if certificate["modelHash"] != receipt["derivedPlay"]["modelHash"]:
            raise ValueError("forced project sequence differs from native play authority")
        order = self.rules.project_domain.declaration_seats(Seat.parse(prefix.dealer))
        stop = next((index for index in range(played + 1, 4)
                     if order[index].name.lower() not in certificate["seats"]), 4)
        if stop == played + 1:
            return None
        if stop < 4 and self.project_continuation is None:
            raise ValueError("forced segment requires its optional project continuation")
        self._check_budget(seconds=self.deadline - time.perf_counter(), nodes=nodes,
                           depth=depth, required_depth=2 * (4 - played) - 1,
                           stage="forced_project_sequence")
        seed = self._seed(normalized, selected_projects_from_exact_receipt_v1(normalized),
            awarded_project_points=tuple(receipt["derivedProjects"]["combinedAwardedValue"]),
            project_evidence_hash=normalized.normalization_receipt.sha256,
            legal_action_rows=legal_action_rows)
        raw = request.to_dict()
        raw.pop("sha256")
        raw["horizon"] = {"unit": "plies", "value": stop - played}
        project_observations = {}
        observer = None
        if project_observer_factory is not None:
            project_observer = project_observer_factory(normalized, prefix)

            def observer(value):
                if not value.path:
                    return root_play_observation
                project_hash = project_observer(value)
                play_hash = canonical_sha256({"kind": "native_project_followup_play_v1",
                    "priorObservation": project_hash, "priorAction": "pass"})
                project_observations[play_hash] = project_hash
                return play_hash

        compiled = compiler.compile(seed, ScenarioProofRequestV1.build(raw),
            expected_rules_hash=self.rules.sha256,
            maximum_nodes=nodes + (len(shared_boundary_values) if stop == 4 else 0),
            _merge_final_boundary=stop == 4,
            _compact_forced=self._compact_forced,
            maximum_depth=stop - played, maximum_seconds=self.deadline - time.perf_counter(),
            cancelled=self.cancel_check, decision_observer=observer,
            **({"_project_prefix": prefix, "_project_normalization": normalized} if observer is not None else {}))
        if not compiled.complete or compiled.graph is None:
            raise ExactNormalizationStopped(str(compiled.receipt.exhaustion_reason), evidence={
                "sourceStage": "forced_project_sequence", "causeReceipt": compiled.receipt.to_dict()})
        graph = compiled.graph
        pass_ids = {node.node_id: canonical_sha256({"forcedProjectPass": node.node_id})
                    for node in graph.nodes if not node.is_leaf and node.node_id != graph.root_node_id}
        expanded = [replace(node, edges=tuple(replace(edge,
            child_node_id=pass_ids.get(edge.child_node_id, edge.child_node_id)) for edge in node.edges))
            if not node.is_leaf else node for node in graph.nodes]
        for node in graph.nodes:
            if node.node_id in pass_ids:
                expanded.append(ProofGraphNodeV1.decision(pass_ids[node.node_id], node.actor_team,
                    [ProofGraphEdgeV1.build(0, "pass", node.node_id)], actor_seat=node.actor_seat))
        if stop == 4:
            # Only final utility leaves are interchangeable. Earlier frontiers
            # still need their distinct public histories for continuation.
            representatives = {}
            leaf_names = {}
            for node in sorted(expanded, key=lambda item: item.node_id):
                if node.is_leaf:
                    leaf_names[node.node_id] = representatives.setdefault(
                        (node.leaf_kind, node.utility), node.node_id)
            expanded = [replace(node, edges=tuple(replace(edge,
                child_node_id=leaf_names.get(edge.child_node_id, edge.child_node_id)) for edge in node.edges))
                if not node.is_leaf else node for node in expanded
                if not node.is_leaf or leaf_names[node.node_id] == node.node_id]
        self._check_budget(seconds=self.deadline - time.perf_counter(),
                           nodes=nodes - (len(ProofGraphSequencePackV1.retained_node_ids(expanded, graph.root_node_id))
                               if self._compact_forced else len(expanded)) + 1 + (sum(
                               node.is_leaf and node.leaf_kind == "boundary" and node.utility in shared_boundary_values
                               for node in expanded) if stop == 4 else 0), depth=depth,
                           required_depth=2 * (4 - played) - 1, stage="forced_project_sequence")
        observations = dict(compiled.decision_observation_ids)
        if observer is not None:
            observations.update({pass_id: project_observations[observations[node_id]]
                                 for node_id, pass_id in pass_ids.items()})
        combined = ProofGraphV1.build(scope=graph.scope, root_node_id=graph.root_node_id,
            nodes=expanded, boundary_evaluator=graph.boundary_evaluator, utility_scale=graph.utility_scale)
        if stop < 4:
            lookup = {node.node_id: node for node in graph.nodes}
            pending = [(graph.root_node_id, ())]
            leaf_paths = {}
            while pending:
                node_id, path = pending.pop()
                node = lookup[node_id]
                if node.is_leaf:
                    if node_id in leaf_paths and leaf_paths[node_id] != path:
                        raise ValueError("forced segment merged different public play histories")
                    leaf_paths[node_id] = path
                    continue
                for edge in node.edges:
                    action = LegalPlayActionV1.from_code(edge.action_code)
                    pending.append((edge.child_node_id, (*path, TailPrefixActionV1(
                        Seat.parse(node.actor_seat), action.card, action.ekka))))
            consumed = 2 * (stop - played) - 1
            child_raw = request.to_dict()
            child_raw.pop("sha256")
            child_raw["horizon"] = {"unit": request.horizon["unit"],
                                    "value": request.horizon["value"] - consumed}
            child_request = ScenarioProofRequestV1.build(child_raw)
            continuations = {}
            used = self._node_count(combined)
            for leaf, path in sorted(leaf_paths.items()):
                remaining_seconds = self.deadline - time.perf_counter()
                used = self._replacement_node_count(combined, continuations)
                self._check_budget(seconds=remaining_seconds, nodes=nodes - used + 1,
                    depth=depth - consumed, required_depth=1, stage="optional_project_after_forced_segment")
                advanced = prefix.to_dict()
                advanced.pop("sha256")
                advanced["stateAssertions"] = {}
                advanced["projectAssertions"] = []
                advanced["playActions"].extend({"actor": action.actor.name.lower(), "card": action.card.id,
                                                "ekka": action.ekka} for action in path)
                advanced["projectActions"].extend({"actor": action.actor.name.lower(), "action": "pass"}
                                                   for action in path[1:])
                child, child_observations = self._continue_project(ExactTracePrefixV1.build(advanced),
                    child_request, nodes - used + 1, depth - consumed, remaining_seconds,
                    retained=continuations)
                continuations[leaf] = self._namespace_graph(child, leaf)
                if observer is not None:
                    observations.update(self._namespace_observations(child, leaf, child_observations))
                used += self._node_count(child) - 1
            combined = self._replace_leaves(combined, continuations)
        return combined, observations

    def _third_project_parent_rows(self, normalized, prefix, request):
        if not (self.native_cp_sat and self.root_decision_kind == "project"
                and normalized is not None and normalized.decision_kind == "project"
                and len(prefix.play_actions) == 2 and len(prefix.project_actions) == 3
                and request.mode == "bounded" and request.horizon is not None
                and request.horizon["unit"] in {"plies", "decisions"}
                and request.horizon["value"] == 3
                and request.boundary_evaluator == "current_raw_point_margin_v1"):
            return None
        if normalized.rules_hash != self.rules.sha256:
            raise ValueError("project follow-up parent differs from rules")
        return _certified_project_followup_rows(normalized, prefix) or None

    def _final_project_play_graph(self, normalized, prefix, request, *, compiler=None,
                                maximum_nodes=None, maximum_depth=None, graph_cache=None):
        if not (self.native_cp_sat and self.root_decision_kind == "project"
                and len(prefix.play_actions) == 3 and len(prefix.project_actions) == 4
                and request.mode == "bounded"
                and request.horizon is not None and request.horizon["unit"] in {"plies", "decisions"}
                and request.horizon["value"] == 1
                and request.boundary_evaluator == "current_raw_point_margin_v1"):
            return None
        normalized.require_integrity()
        receipt = normalized.normalization_receipt.payload
        native = receipt.get("nativePlayNormalization") or {}
        if normalized.decision_kind == "project":
            legal = native.get("legalPlayActions", ())
            if not legal or (len(legal) != 1 and compiler is None):
                return None
            parent = prefix.to_dict()
            parent.pop("sha256")
            chosen = parent["projectActions"].pop()
            parent = ExactTracePrefixV1.build(parent)
            if normalized.rules_hash != self.rules.sha256 or normalized.prefix_hash != _effective_prefix_hash(parent):
                raise ValueError("forced play parent differs from declaration prefix")
            declaration = ProjectDeclarationActionV1.build(chosen, actor=normalized.next_actor)
            if not any(ProjectDeclarationActionV1.build(row["code"], actor=normalized.next_actor) == declaration
                       for row in normalized.legal_actions):
                raise ValueError("forced play follows an uncertified declaration")
            if any(Seat.parse(row["actor"]).name.lower() != normalized.next_actor for row in legal):
                raise ValueError("forced play belongs to another actor")
            if len(legal) > 1:
                # Projects cannot change this raw-only frontier. Compile every
                # native legal play, without normalizing the declaration again.
                cache_key = ("final_project:" + normalized.normalization_receipt.sha256,
                             request.sha256, maximum_depth)
                if graph_cache is not None and cache_key in graph_cache:
                    graph = graph_cache[cache_key]
                    self._check_budget(seconds=self.deadline - time.perf_counter(),
                        nodes=maximum_nodes - self._node_count(graph) + 1,
                        depth=maximum_depth, required_depth=1, stage="final_project_play")
                    return graph
                seed = self._seed(normalized, selected_projects_from_exact_receipt_v1(normalized),
                    awarded_project_points=tuple(receipt["derivedProjects"]["combinedAwardedValue"]),
                    project_evidence_hash=normalized.normalization_receipt.sha256,
                    legal_action_rows=legal)
                compilation = compiler.compile(seed, request, expected_rules_hash=self.rules.sha256,
                    maximum_nodes=maximum_nodes, maximum_depth=maximum_depth,
                    maximum_seconds=self.deadline - time.perf_counter(),
                    _compact_forced=self._compact_forced, cancelled=self.cancel_check)
                if not compilation.complete or compilation.graph is None:
                    raise ExactNormalizationStopped(str(compilation.receipt.exhaustion_reason), evidence={
                        "sourceStage": "final_project_play", "causeReceipt": compilation.receipt.to_dict()})
                if graph_cache is not None:
                    graph_cache[cache_key] = compilation.graph
                return compilation.graph
            action = LegalPlayActionV1(Card.parse(legal[0]["card"]), legal[0].get("ekka") is True)
        elif normalized.decision_kind == "play" and len(normalized.legal_actions) == 1:
            action = LegalPlayActionV1.from_code(str(normalized.legal_actions[0]["code"]))
        else:
            return None
        margins = native.get("witnessRawMargins")
        if margins is None:
            return None
        play = receipt["derivedPlay"]
        if margins["modelHash"] != play["modelHash"] or native["uniquenessSolverStatus"] != "infeasible":
            raise ValueError("forced play lacks owned unique native utility evidence")
        witness = play["plays"][3]
        if (Card.parse(witness["card"]) != action.card
                or Seat.parse(witness["actor"]).name.lower() != normalized.next_actor
                or bool(witness.get("ekka")) != action.ekka):
            raise ValueError("forced play witness differs from its sole legal action")
        actor = Seat.parse(normalized.next_actor)
        root_id = canonical_sha256({"kind": "native_forced_first_trick_frontier_v1",
                                   "normalization": receipt["sha256"], "request": request.sha256,
                                   "prefix": prefix.sha256})
        leaf_id = canonical_sha256({"root": root_id, "boundary": True})
        trick = play["tricks"][0]
        edge = ProofGraphEdgeV1.build(action.card.id + 32 * int(action.ekka),
            action.code, leaf_id, play_trick_index=0,
            completed_trick_index=0, completed_trick_winner_seat=trick["winner"],
            completed_trick_winning_card_id=Card.parse(trick["winningCard"]).id)
        return ProofGraphV1.build(scope="bounded", root_node_id=root_id,
            boundary_evaluator=request.boundary_evaluator, nodes=[
                ProofGraphNodeV1.decision(root_id, actor.team.name.lower(), [edge], actor_seat=actor.name.lower()),
                ProofGraphNodeV1.leaf(leaf_id, margins["values"][int(Team[request.target_team.upper()])], kind="boundary"),
            ])

    def _native_project_lifecycle_evidence(
        self,
        normalized: OrToolsNormalizedExactStateV1,
        prefix: ExactTracePrefixV1,
        *, declaration_only: bool = False,
    ) -> ProjectLifecycleFragmentEvidenceV1 | tuple[tuple[Project, ...], list[list[str | int]], str]:
        """Solve declarations, deferring awards until the reveal leader is known."""

        def remaining_seconds():
            seconds = self.deadline - time.perf_counter()
            reason = ("cancelled" if self.cancel_check is not None and self.cancel_check()
                      else "maximum_seconds" if seconds <= 0 else None)
            if reason is not None:
                raise ExactNormalizationStopped(reason, evidence={"sourceStage": "nonplay_project_lifecycle"})
            return seconds

        remaining_seconds()

        structural_projects = selected_projects_from_exact_receipt_v1(normalized)
        initial_hands = normalized.state.get("initialHands")
        if not isinstance(initial_hands, Mapping):
            raise ValueError("non-play continuation lacks exact initial hands")
        owners: list[Seat | None] = [None] * 32
        for seat in Seat:
            cards = initial_hands.get(seat.name.lower())
            if not isinstance(cards, list):
                raise ValueError("non-play continuation has invalid initial hands")
            for raw in cards:
                card = Card.parse(raw)
                if owners[card.id] is not None:
                    raise ValueError("non-play continuation duplicates an initial card")
                owners[card.id] = seat
        if any(owner is None for owner in owners):
            raise ValueError("non-play continuation lacks a complete initial deal")

        model = cp_model.CpModel()
        owner_variables = tuple(
            model.new_int_var(0, 3, f"nonplay_project_owner_{card_id}")
            for card_id in range(32)
        )
        contract = Contract.parse(normalized.state.get("contract"))
        trump_raw = normalized.state.get("trumpSuit")
        fragment = attach_project_lifecycle_fragment_v1(
            model,
            owner_of_card=owner_variables,
            fixed_card_owners=tuple(owner for owner in owners if owner is not None),
            contract=contract,
            dealer=prefix.dealer,
            trump=Suit.parse(trump_raw) if trump_raw is not None else None,
            domain=self.rules.project_domain,
            name="nonplay_projects",
            second_trick_leader=(Seat.parse(normalized.state["completedTricks"][0]["winner"])
                if self.rules.project_domain.tie_break_order == "second_trick_reveal"
                and normalized.state.get("completedTricks") else None),
            compact_fixed_states=self.native_cp_sat,
            declaration_only=declaration_only,
        )
        actions = tuple(
            ProjectDeclarationActionV1.build(raw) for raw in prefix.project_actions
        )
        declaration_seats = self.rules.project_domain.declaration_seats(
            Seat.parse(prefix.dealer)
        )
        if actions and tuple(action.actor for action in actions) != declaration_seats:
            raise ValueError("play-start project actions must complete declaration order")
        if self.rules.project_domain.declaration_mode == "optional" and not actions:
            raise ValueError("optional projects require complete play-start actions")
        by_id = {project.project_id: project for project in fragment.projects}
        for action in actions:
            owned = tuple(
                project for project in fragment.projects if project.owner == action.actor
            )
            if action.action == "pass":
                model.add(sum(project.declared for project in owned) == 0)
                continue
            project = by_id.get(str(action.project_id))
            if project is None or project.owner != action.actor:
                raise ValueError("play-start action references an unknown CP project")
            model.add(project.declared == 1)

        model_hash = _cp_model_hash(model, authority={
            "stage": "nonplay_project_lifecycle", "rulesHash": self.rules.sha256,
            "prefixHash": prefix.sha256,
        })
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = remaining_seconds()
        status = _solve_cp_sat_with_cancel_v1(solver, model, self.cancel_check)
        if self.cancel_check is not None and self.cancel_check():
            raise ExactNormalizationStopped("cancelled")
        if status != cp_model.OPTIMAL:
            if status == cp_model.UNKNOWN:
                raise ExactNormalizationStopped("unknown", evidence={
                    "sourceStage": "nonplay_project_lifecycle", "modelHash": model_hash,
                    "rulesHash": self.rules.sha256, "prefixHash": prefix.sha256,
                })
            raise RuntimeError(
                "native non-play project lifecycle did not complete: "
                + solver.status_name(status).lower()
            )
        if declaration_only:
            projects = tuple(Project(item.owner, item.kind, item.cards, item.value, item.high_rank,
                item.category_priority, bool(solver.value(item.declared)),
                solver.value(item.declaration_order), False)
                for item in fragment.projects if solver.value(item.selected))
            selected = {(item.owner, item.kind, item.cards, item.value, item.high_rank, item.category_priority)
                        for item in projects}
            if selected != {(item.owner, item.kind, item.cards, item.value, item.high_rank, item.category_priority)
                            for item in structural_projects}:
                raise AssertionError("project declarations disagree with exact-normalization evidence")
            declarations = sorted([item.project_id, solver.value(item.declaration_order)]
                for item in fragment.projects if solver.value(item.declared))
            return projects, declarations, canonical_sha256({"modelHash": model_hash,
                "fragmentHash": fragment.fragment_hash, "declarations": declarations})
        evidence = fragment.extract(solver)
        receipt_selected = {
            (
                project.owner,
                project.kind,
                project.cards,
                project.value,
                project.high_rank,
                project.category_priority,
            )
            for project in structural_projects
        }
        solved_selected = {
            (
                project.owner,
                project.kind,
                project.cards,
                project.value,
                project.high_rank,
                project.category_priority,
            )
            for project in evidence.projects
            if project.selected
        }
        if solved_selected != receipt_selected:
            raise AssertionError(
                "project lifecycle disagrees with exact-normalization evidence"
            )
        return evidence

    def _adjusted_request(self, nonplay_depth: int) -> ScenarioProofRequestV1 | None:
        if self.request.mode == "exact":
            return self.request
        assert self.request.horizon is not None
        unit = str(self.request.horizon["unit"])
        value = int(self.request.horizon["value"])
        remaining = value - nonplay_depth if unit in {"plies", "decisions"} else value
        if remaining <= 0:
            return None
        raw = self.request.to_dict()
        raw.pop("sha256", None)
        raw["horizon"] = {"unit": unit, "value": remaining}
        return ScenarioProofRequestV1.build(raw)

    @staticmethod
    def _final_boundary_id(graph, node):
        return "final-boundary-" + canonical_sha256({
            "scope": graph.scope, "boundaryEvaluator": graph.boundary_evaluator,
            "utilityScale": graph.utility_scale, "utility": node.utility,
        })

    @staticmethod
    def _namespace_graph(graph: ProofGraphV1, namespace: str) -> ProofGraphV1:
        prefix = f"graft-{canonical_sha256({'namespace': namespace})[:16]}:"
        names = {node.node_id: (
            OrToolsNonPlayPlayGraphGraftV1._final_boundary_id(graph, node)
            if graph.scope == "bounded" and node.is_leaf and node.leaf_kind == "boundary"
            else prefix + node.node_id) for node in graph.nodes}
        nodes = {}
        for node in graph.nodes:
            if node.is_leaf:
                nodes[names[node.node_id]] = (
                    ProofGraphNodeV1.leaf(
                        names[node.node_id],
                        int(node.utility),
                        kind=str(node.leaf_kind),
                    )
                )
            else:
                nodes[names[node.node_id]] = (
                    ProofGraphNodeV1.decision(
                        names[node.node_id],
                        str(node.actor_team),
                        tuple(
                            replace(edge, child_node_id=names[edge.child_node_id])
                            for edge in node.edges
                        ),
                        actor_seat=node.actor_seat,
                    )
                )
        return ProofGraphV1.build(
            scope=graph.scope,
            root_node_id=names[graph.root_node_id],
            nodes=tuple(nodes.values()),
            boundary_evaluator=graph.boundary_evaluator,
            utility_scale=graph.utility_scale,
            tactic_context=graph.tactic_context,
        )

    def _namespace_paired_graph(self, graph: ProofGraphV1, namespace: str) -> ProofGraphV1:
        result = self._namespace_graph(graph, namespace)
        if self.project_counterfactual:
            counter = self.project_counterfactual_graphs.get(graph.sha256)
            if counter is None:
                raise ValueError("project continuation omitted its counterfactual graph")
            graph.require_same_legal_graph(counter)
            self.project_counterfactual_graphs[result.sha256] = self._namespace_graph(counter, namespace)
        return result

    @staticmethod
    def _namespace_observations(
        graph: ProofGraphV1,
        namespace: str,
        observations: Mapping[str, str],
        *, private_history: tuple[tuple[str, str], ...] = (),
    ) -> Mapping[str, str]:
        decisions = {node.node_id for node in graph.nodes if not node.is_leaf}
        if set(observations) != decisions:
            raise ValueError(
                "play observations must cover every decision node exactly"
            )
        prefix = f"graft-{canonical_sha256({'namespace': namespace})[:16]}:"
        actors = {node.node_id: node.actor_seat for node in graph.nodes if not node.is_leaf}
        if private_history and any(actor is None for actor in actors.values()):
            raise ValueError("private recall requires a known actor for every continuation decision")
        return {
            prefix + node_id: bind_escalation_recall_v1(observations[node_id],
                tuple(window for actor, window in private_history if actor == actors[node_id]))
            for node_id in sorted(observations)
        }

    def _continue_project(self, prefix, request, nodes, depth, seconds, *, retained):
        if self.project_continuation is None:
            raise ValueError("project frontier requires a continuation compiler")
        seconds = min(seconds, self.deadline - time.perf_counter())
        self._check_budget(seconds=seconds, nodes=nodes, depth=depth,
                           required_depth=1, stage="project_continuation_shared_boundary")
        # Reserve only outcome slots already owned by completed siblings. The
        # returned graph must demonstrate each actual reuse before admission.
        shared = {node.node_id for graph in retained.values()
                  if graph.scope == "bounded" and graph.boundary_evaluator == request.boundary_evaluator
                  for node in graph.nodes if node.is_leaf and node.leaf_kind == "boundary"}
        continuation = self.project_continuation(prefix, request, nodes + len(shared), depth, seconds)
        if self.project_counterfactual:
            child, observations, counter = continuation
            child.require_same_legal_graph(counter)
            self.project_counterfactual_graphs[child.sha256] = counter
        else:
            child, observations = continuation
        child.validate()
        boundaries = [node for node in child.nodes
                      if child.scope == "bounded" and node.is_leaf and node.leaf_kind == "boundary"]
        boundary_ids = {self._final_boundary_id(child, node) for node in boundaries}
        retained_count = self._node_count(child) - len(boundaries) + len(boundary_ids)
        reused = len(boundary_ids & shared)
        self._check_budget(seconds=self.deadline - time.perf_counter(),
                           nodes=nodes - retained_count + reused + 1,
                           depth=depth, required_depth=1, stage="project_continuation_shared_boundary")
        return child, observations

    def _node_count(self, graph):
        return (len(ProofGraphSequencePackV1.retained_node_ids(graph.nodes, graph.root_node_id))
                if self._compact_forced else len(graph.nodes))

    def _replacement_node_count(self, base, replacements):
        nodes = {node.node_id: node for node in base.nodes if node.node_id not in replacements}
        nodes.update((node.node_id, node) for graph in replacements.values() for node in graph.nodes)
        if not self._compact_forced:
            return len(nodes)
        return len(ProofGraphSequencePackV1.retained_node_ids(nodes.values(), base.root_node_id,
            child_aliases={leaf: graph.root_node_id for leaf, graph in replacements.items()}))

    def _replace_leaves(
        self,
        base: ProofGraphV1,
        replacements: Mapping[str, ProofGraphV1],
        *,
        boundary_replacements: set[str] | frozenset[str] = frozenset(),
        _counterfactual_pass: bool = False,
    ) -> ProofGraphV1:
        child_roots = {leaf: graph.root_node_id for leaf, graph in replacements.items()}
        if any(graph.utility_scale != base.utility_scale for graph in replacements.values()):
            raise ValueError("combined proof graphs require identical utility scales")
        if any((graph.scope, graph.boundary_evaluator) != (self.request.mode, self.request.boundary_evaluator)
               for graph in replacements.values()):
            raise ValueError("combined proof graphs require identical proof boundaries")
        nodes: dict[str, ProofGraphNodeV1] = {}
        for node in base.nodes:
            if node.node_id in replacements:
                continue
            if node.is_leaf:
                nodes[node.node_id] = (
                    ProofGraphNodeV1.leaf(
                        node.node_id,
                        0,
                        kind="boundary",
                    )
                    if node.node_id in boundary_replacements
                    else node
                )
                continue
            nodes[node.node_id] = ProofGraphNodeV1.decision(
                node.node_id,
                str(node.actor_team),
                tuple(
                    replace(edge, child_node_id=child_roots.get(edge.child_node_id, edge.child_node_id))
                    for edge in node.edges
                ),
                actor_seat=node.actor_seat,
            )
        for graph in replacements.values():
            for node in graph.nodes:
                existing = nodes.get(node.node_id)
                if existing is not None and existing != node:
                    raise AssertionError("combined graph node identity collision")
                nodes[node.node_id] = node
        scope = "bounded" if self.request.mode == "bounded" else "exact"
        result = ProofGraphV1.build(
            scope=scope,
            root_node_id=base.root_node_id,
            nodes=tuple(nodes.values()),
            utility_scale=base.utility_scale,
            tactic_context=base.tactic_context,
            boundary_evaluator=(
                self.request.boundary_evaluator if scope == "bounded" else None
            ),
        )
        if self.project_counterfactual and not _counterfactual_pass:
            counter_base = self.project_counterfactual_graphs.get(base.sha256)
            if counter_base is None:
                if any(node.is_leaf and node.node_id not in replacements and node.utility != 0 for node in base.nodes):
                    raise ValueError("project continuation has an unpaired scoring terminal")
                counter_base = base
            counters = {}
            for leaf, graph in replacements.items():
                counter = self.project_counterfactual_graphs.get(graph.sha256)
                if counter is None:
                    raise ValueError("project replacement omitted its counterfactual graph")
                graph.require_same_legal_graph(counter)
                counters[leaf] = counter
            paired = self._replace_leaves(counter_base, counters, boundary_replacements=boundary_replacements,
                                          _counterfactual_pass=True)
            result.require_same_legal_graph(paired)
            self.project_counterfactual_graphs[result.sha256] = paired
        return result


__all__ = [
    "NonPlayPlayStartV1",
    "OrToolsNonPlayPlayGraphGraftV1",
    "build_nonplay_play_start_prefix_v1",
    "selected_projects_from_exact_receipt_v1",
]
