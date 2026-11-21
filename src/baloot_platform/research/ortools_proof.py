"""Pure CP-SAT proof over a canonical finite scenario graph.

This module deliberately has no dependency on the mutable game engine, bots,
or the legacy proof services.  Construction supplies an already legal graph;
CP-SAT owns the strategic max/min values and winning-root count.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_CEILING
import json
from typing import Any, Callable, Mapping, Sequence

import ortools
from ortools.sat.python import cp_model

from .scenario_contracts import ScenarioProofRequestV1, canonical_sha256
from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1


PROOF_ENGINE_ID = "ortools_scenario_graph_proof_v1"
PROOF_ENGINE_VERSION = "1.0.0"
LINE_COUNT_BASE = 1_000_000
_TEAMS = frozenset({"north_south", "west_east"})
_SEAT_TEAMS = {
    "north": "north_south",
    "south": "north_south",
    "west": "west_east",
    "east": "west_east",
}
_SEATS = ("north", "west", "south", "east")
_SCOPES = frozenset({"exact", "bounded"})
_POLICY_ID = "fixed_action_map_v1"
_POLICY_VERSION = "1.0.0"
_POLICY_MATERIALIZATION = "seeded_legal_action_order_v1"
_SUPPORTED_CLAIMS = frozenset(
    {
        "outcome_reachable",
        "outcome_forced",
        "winning_root_action_count",
        "selected_action_is_unique_winner",
        "target_team_has_no_salvage",
        "winning_continuation_line_count",
    }
)


def required_line_count_limbs_v1(maximum_count: int) -> int:
    """Return the governed base-1,000,000 limb count without building a model."""

    if type(maximum_count) is not int:
        raise TypeError("maximum line count must be an integer")
    if maximum_count < 0:
        raise ValueError("maximum line count cannot be negative")
    limbs = 1
    capacity = LINE_COUNT_BASE
    while maximum_count >= capacity:
        limbs += 1
        capacity *= LINE_COUNT_BASE
    return limbs


@dataclass(frozen=True, slots=True)
class ProofGraphEdgeV1:
    action_id: int
    action_code: str
    child_node_id: str
    play_trick_index: int | None = None
    completed_trick_index: int | None = None
    completed_trick_winner_seat: str | None = None
    completed_trick_winning_card_id: int | None = None

    @classmethod
    def build(
        cls,
        action_id: int,
        action_code: str,
        child_node_id: str,
        *,
        play_trick_index: int | None = None,
        completed_trick_index: int | None = None,
        completed_trick_winner_seat: str | None = None,
        completed_trick_winning_card_id: int | None = None,
    ) -> "ProofGraphEdgeV1":
        if type(action_id) is not int or action_id < 0:
            raise ValueError("proof graph actionId must be a non-negative integer")
        code = str(action_code).strip()
        child = str(child_node_id).strip()
        if not code or not child:
            raise ValueError("proof graph edge requires an action code and child node")
        if play_trick_index is not None and (
            type(play_trick_index) is not int or not 0 <= play_trick_index <= 7
        ):
            raise ValueError("play trick index is invalid")
        completed = (
            completed_trick_index,
            completed_trick_winner_seat,
            completed_trick_winning_card_id,
        )
        if any(item is None for item in completed) and not all(
            item is None for item in completed
        ):
            raise ValueError("completed trick edge facts must be supplied together")
        if completed_trick_index is not None and (
            type(completed_trick_index) is not int
            or not 0 <= completed_trick_index <= 7
            or str(completed_trick_winner_seat) not in _SEAT_TEAMS
            or type(completed_trick_winning_card_id) is not int
            or not 0 <= completed_trick_winning_card_id <= 31
        ):
            raise ValueError("completed trick edge facts are invalid")
        return cls(
            action_id,
            code,
            child,
            play_trick_index,
            completed_trick_index,
            completed_trick_winner_seat,
            completed_trick_winning_card_id,
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "actionId": self.action_id,
            "actionCode": self.action_code,
            "childNodeId": self.child_node_id,
        }
        if self.play_trick_index is not None:
            value["playTrickIndex"] = self.play_trick_index
        if self.completed_trick_index is not None:
            value["completedTrickIndex"] = self.completed_trick_index
            value["completedTrickWinnerSeat"] = self.completed_trick_winner_seat
            value["completedTrickWinningCardId"] = (
                self.completed_trick_winning_card_id
            )
        return value


@dataclass(frozen=True, slots=True)
class ProofGraphNodeV1:
    node_id: str
    actor_team: str | None
    edges: tuple[ProofGraphEdgeV1, ...]
    leaf_kind: str | None
    utility: int | None
    actor_seat: str | None = None

    @classmethod
    def decision(
        cls,
        node_id: str,
        actor_team: str,
        edges: Sequence[ProofGraphEdgeV1],
        *,
        actor_seat: str | None = None,
    ) -> "ProofGraphNodeV1":
        identity = str(node_id).strip()
        team = str(actor_team).strip()
        ordered = tuple(
            sorted(
                edges,
                key=lambda item: (
                    item.action_id,
                    item.action_code,
                    item.child_node_id,
                ),
            )
        )
        if not identity or team not in _TEAMS or not ordered:
            raise ValueError("proof graph decision requires an id, team, and edges")
        if len({item.action_id for item in ordered}) != len(ordered):
            raise ValueError("proof graph decision action IDs must be unique")
        if len({item.action_code for item in ordered}) != len(ordered):
            raise ValueError("proof graph decision action codes must be unique")
        seat = str(actor_seat).strip() if actor_seat is not None else None
        if seat is not None and _SEAT_TEAMS.get(seat) != team:
            raise ValueError("proof graph actor seat does not match its team")
        return cls(identity, team, ordered, None, None, seat)

    @classmethod
    def leaf(
        cls,
        node_id: str,
        utility: int,
        *,
        kind: str = "terminal",
    ) -> "ProofGraphNodeV1":
        identity = str(node_id).strip()
        leaf_kind = str(kind).strip()
        if not identity or leaf_kind not in {"terminal", "boundary"}:
            raise ValueError("proof graph leaf requires an id and valid kind")
        if type(utility) is not int:
            raise TypeError("proof graph utility must be an integer")
        return cls(identity, None, (), leaf_kind, utility, None)

    @property
    def is_leaf(self) -> bool:
        return self.utility is not None

    def to_dict(self) -> dict[str, Any]:
        if self.is_leaf:
            return {
                "nodeId": self.node_id,
                "leafKind": self.leaf_kind,
                "utility": self.utility,
            }
        value = {
            "nodeId": self.node_id,
            "actorTeam": self.actor_team,
            "edges": [item.to_dict() for item in self.edges],
        }
        if self.actor_seat is not None:
            value["actorSeat"] = self.actor_seat
        return value


@dataclass(frozen=True, slots=True)
class ProofGraphForcedSequenceV1:
    """Lossless forced-action evidence; not a substitute for a qualified graph."""

    nodes: tuple[ProofGraphNodeV1, ...]
    observation_ids: tuple[tuple[str, str], ...]
    sha256: str

    @classmethod
    def build(cls, nodes: Sequence[ProofGraphNodeV1], *,
              observation_ids: Mapping[str, str] | None = None) -> "ProofGraphForcedSequenceV1":
        ordered = tuple(nodes)
        if not ordered or any(not isinstance(node, ProofGraphNodeV1)
                              or node.is_leaf or len(node.edges) != 1 for node in ordered):
            raise ValueError("forced sequence requires nonempty single-action decisions")
        identities = {node.node_id for node in ordered}
        if len(identities) != len(ordered):
            raise ValueError("forced sequence node identities must be unique")
        for index, node in enumerate(ordered):
            if node != ProofGraphNodeV1.decision(node.node_id, node.actor_team, node.edges,
                                                actor_seat=node.actor_seat):
                raise ValueError("forced sequence decision shape is invalid")
            child = node.edges[0].child_node_id
            if index + 1 < len(ordered) and child != ordered[index + 1].node_id:
                raise ValueError("forced sequence actions must form one connected path")
        if ordered[-1].edges[0].child_node_id in identities:
            raise ValueError("forced sequence must not contain a cycle")
        observations = dict(observation_ids or {})
        if observations and (set(observations) != identities or any(
                not isinstance(value, str) or not value.strip() for value in observations.values())):
            raise ValueError("forced sequence observations must cover every decision")
        candidate = cls(ordered, tuple(sorted(observations.items())), "")
        return cls(candidate.nodes, candidate.observation_ids, canonical_sha256(candidate._core()))

    def _core(self) -> dict[str, Any]:
        return {"schemaVersion": 1, "kind": "forced_decision_sequence",
                "nodes": [node.to_dict() for node in self.nodes],
                "observationIds": dict(self.observation_ids), "actionDepth": len(self.nodes)}

    def to_dict(self) -> dict[str, Any]:
        return {**self._core(), "sha256": self.sha256}

    def validate(self) -> None:
        rebuilt = self.build(self.nodes, observation_ids=dict(self.observation_ids))
        if rebuilt != self:
            raise ValueError("forced sequence identity or checksum mismatch")


@dataclass(frozen=True, slots=True)
class ProofGraphV1:
    scope: str
    root_node_id: str
    utility_scale: int
    boundary_evaluator: str | None
    nodes: tuple[ProofGraphNodeV1, ...]
    sha256: str
    tactic_context: Mapping[str, Any] | None = None

    @classmethod
    def build(
        cls,
        *,
        scope: str,
        root_node_id: str,
        nodes: Sequence[ProofGraphNodeV1],
        utility_scale: int = 1,
        boundary_evaluator: str | None = None,
        tactic_context: Mapping[str, Any] | None = None,
    ) -> "ProofGraphV1":
        normalized_scope = str(scope).strip()
        root = str(root_node_id).strip()
        if normalized_scope not in _SCOPES or not root:
            raise ValueError("proof graph scope or root is invalid")
        if type(utility_scale) is not int or utility_scale < 1:
            raise ValueError("proof graph utilityScale must be a positive integer")
        boundary = (
            str(boundary_evaluator).strip()
            if boundary_evaluator is not None
            else None
        )
        if normalized_scope == "exact" and boundary is not None:
            raise ValueError("exact proof graph cannot declare a boundary evaluator")
        if normalized_scope == "bounded" and not boundary:
            raise ValueError("bounded proof graph requires a boundary evaluator")
        ordered = tuple(sorted(nodes, key=lambda item: item.node_id))
        candidate = cls(
            normalized_scope,
            root,
            utility_scale,
            boundary,
            ordered,
            "",
            copy.deepcopy(dict(tactic_context)) if tactic_context is not None else None,
        )
        candidate._validate_structure()
        core = candidate._core()
        return cls(
            normalized_scope,
            root,
            utility_scale,
            boundary,
            ordered,
            canonical_sha256(core),
            copy.deepcopy(dict(tactic_context)) if tactic_context is not None else None,
        )

    def _core(self) -> dict[str, Any]:
        value = {
            "schemaVersion": 1,
            "scope": self.scope,
            "rootNodeId": self.root_node_id,
            "utilityScale": self.utility_scale,
            "boundaryEvaluator": self.boundary_evaluator,
            "nodes": [item.to_dict() for item in self.nodes],
        }
        if self.tactic_context is not None:
            value["tacticContext"] = copy.deepcopy(dict(self.tactic_context))
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProofGraphV1":
        """Read only the canonical, closed graph representation."""
        if not isinstance(value, Mapping) or not isinstance(value.get("nodes"), list):
            raise ValueError("proof graph document must contain nodes")
        if not {"schemaVersion", "scope", "rootNodeId", "utilityScale", "boundaryEvaluator", "sha256"} <= value.keys():
            raise ValueError("proof graph document is missing required fields")
        nodes = []
        for row in value["nodes"]:
            if not isinstance(row, Mapping):
                raise ValueError("proof graph node must be an object")
            if "leafKind" in row:
                if not {"nodeId", "utility"} <= row.keys():
                    raise ValueError("proof graph leaf is missing required fields")
                nodes.append(ProofGraphNodeV1.leaf(row["nodeId"], row["utility"], kind=row["leafKind"]))
            else:
                if not {"nodeId", "actorTeam", "edges"} <= row.keys() or not isinstance(row["edges"], list):
                    raise ValueError("proof graph decision is missing required fields")
                if any(not isinstance(edge, Mapping) or not {"actionId", "actionCode", "childNodeId"} <= edge.keys()
                       for edge in row["edges"]):
                    raise ValueError("proof graph edge is missing required fields")
                edges = tuple(ProofGraphEdgeV1.build(
                    edge["actionId"], edge["actionCode"], edge["childNodeId"],
                    play_trick_index=edge.get("playTrickIndex"),
                    completed_trick_index=edge.get("completedTrickIndex"),
                    completed_trick_winner_seat=edge.get("completedTrickWinnerSeat"),
                    completed_trick_winning_card_id=edge.get("completedTrickWinningCardId"),
                ) for edge in row["edges"])
                nodes.append(ProofGraphNodeV1.decision(row["nodeId"], row["actorTeam"], edges,
                                                     actor_seat=row.get("actorSeat")))
        graph = cls.build(scope=value["scope"], root_node_id=value["rootNodeId"], nodes=nodes,
                          utility_scale=value["utilityScale"], boundary_evaluator=value["boundaryEvaluator"],
                          tactic_context=value.get("tacticContext"))
        if canonical_sha256(graph.to_dict()) != canonical_sha256(value):
            raise ValueError("proof graph document is noncanonical or has an invalid checksum")
        return graph

    def require_same_legal_graph(self, other: "ProofGraphV1") -> None:
        """Require identical decisions and edges, allowing only leaf scores to differ."""
        self.validate()
        if not isinstance(other, ProofGraphV1):
            raise TypeError("project counterfactual requires a proof graph")
        other.validate()
        if (
            (other.scope, other.root_node_id, other.utility_scale,
             other.boundary_evaluator, other.tactic_context)
            != (self.scope, self.root_node_id, self.utility_scale,
                self.boundary_evaluator, self.tactic_context)
            or len(other.nodes) != len(self.nodes)
            or any(replace(node, utility=alternate.utility) != alternate
                   for node, alternate in zip(self.nodes, other.nodes, strict=True))
        ):
            raise ValueError("project counterfactual must preserve the complete legal graph")

    def _validate_structure(self) -> None:
        if self.scope not in _SCOPES or not self.root_node_id.strip():
            raise ValueError("proof graph scope or root is invalid")
        if type(self.utility_scale) is not int or self.utility_scale < 1:
            raise ValueError("proof graph utilityScale must be a positive integer")
        if self.scope == "exact" and self.boundary_evaluator is not None:
            raise ValueError("exact proof graph cannot declare a boundary evaluator")
        if self.scope == "bounded" and not self.boundary_evaluator:
            raise ValueError("bounded proof graph requires a boundary evaluator")
        if not self.nodes:
            raise ValueError("proof graph cannot be empty")
        if self.tactic_context is not None:
            context = dict(self.tactic_context)
            base_tricks = context.get("baseTricks")
            owners = context.get("ownerByCard")
            trump = context.get("trumpSuit")
            if (
                not isinstance(base_tricks, list)
                or len(base_tricks) != 2
                or any(type(item) is not int or not 0 <= item <= 8 for item in base_tricks)
                or not isinstance(owners, list)
                or len(owners) != 32
                or any(type(item) is not int or not -1 <= item <= 3 for item in owners)
                or (trump is not None and (type(trump) is not int or not 0 <= trump <= 3))
            ):
                raise ValueError("proof graph tactic context is invalid")
        if any(not isinstance(item, ProofGraphNodeV1) for item in self.nodes):
            raise TypeError("proof graph nodes must use ProofGraphNodeV1")
        by_id = {item.node_id: item for item in self.nodes}
        if len(by_id) != len(self.nodes):
            raise ValueError("proof graph node IDs must be unique")
        root = by_id.get(self.root_node_id)
        if root is None or root.is_leaf:
            raise ValueError("proof graph root must be a decision node")
        if self.scope == "exact" and any(
            item.leaf_kind == "boundary" for item in self.nodes
        ):
            raise ValueError("exact proof graph cannot contain boundary leaves")
        for node in self.nodes:
            if not node.node_id.strip():
                raise ValueError("proof graph node ID cannot be empty")
            if node.is_leaf:
                if (
                    node.actor_team is not None
                    or node.actor_seat is not None
                    or node.edges
                    or node.leaf_kind not in {"terminal", "boundary"}
                    or type(node.utility) is not int
                ):
                    raise ValueError("proof graph leaf shape is invalid")
                continue
            if (
                node.actor_team not in _TEAMS
                or (
                    node.actor_seat is not None
                    and _SEAT_TEAMS.get(node.actor_seat) != node.actor_team
                )
                or node.leaf_kind is not None
                or node.utility is not None
            ):
                raise ValueError("proof graph decision shape is invalid")
            if not node.edges:
                raise ValueError("proof graph decision must contain edges")
            if any(not isinstance(item, ProofGraphEdgeV1) for item in node.edges):
                raise TypeError("proof graph edges must use ProofGraphEdgeV1")
            if len({item.action_id for item in node.edges}) != len(node.edges):
                raise ValueError("proof graph decision action IDs must be unique")
            if len({item.action_code for item in node.edges}) != len(node.edges):
                raise ValueError("proof graph decision action codes must be unique")
            for edge in node.edges:
                if (
                    type(edge.action_id) is not int
                    or edge.action_id < 0
                    or not edge.action_code.strip()
                    or not edge.child_node_id.strip()
                ):
                    raise ValueError("proof graph edge shape is invalid")
                if edge.child_node_id not in by_id:
                    raise ValueError("proof graph edge references an unknown child")

        visited: set[str] = set()
        active: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in active:
                raise ValueError("proof graph must be acyclic")
            if node_id in visited:
                return
            active.add(node_id)
            for edge in by_id[node_id].edges:
                visit(edge.child_node_id)
            active.remove(node_id)
            visited.add(node_id)

        visit(self.root_node_id)
        if visited != set(by_id):
            raise ValueError("proof graph cannot contain unreachable nodes")

    def validate(self) -> None:
        self._validate_structure()
        if canonical_sha256(self._core()) != self.sha256:
            raise ValueError("proof graph checksum mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {**self._core(), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ProofGraphSequencePackV1:
    """Lossless transport of forced chains, not a standalone proof graph."""

    graph: ProofGraphV1
    sequences: tuple[tuple[str, int, ProofGraphForcedSequenceV1], ...]
    observation_ids: tuple[tuple[str, str], ...]
    source_graph_hash: str
    source_observation_hash: str
    sha256: str

    @staticmethod
    def retained_node_ids(nodes, root_node_id, *, child_aliases=None):
        """Count the same skeleton during construction, before serialization."""
        by_id = {node.node_id: node for node in nodes}
        aliases = child_aliases or {}
        incoming = dict.fromkeys(by_id, 0)
        for node in by_id.values():
            for edge in node.edges:
                child = aliases.get(edge.child_node_id, edge.child_node_id)
                if child not in incoming:
                    raise ValueError("compact graph edge references an unknown child")
                incoming[child] += 1
        return frozenset(node.node_id for node in by_id.values()
                         if node.node_id == root_node_id or len(node.edges) != 1
                         or incoming[node.node_id] != 1)

    @classmethod
    def build(cls, graph: ProofGraphV1, *, observation_ids: Mapping[str, str] | None = None):
        graph.validate()
        observations = dict(observation_ids or {})
        decisions = {node.node_id for node in graph.nodes if not node.is_leaf}
        if observations and (set(observations) != decisions or any(
                not isinstance(value, str) or not value.strip() for value in observations.values())):
            raise ValueError("packed observations must cover every decision")
        nodes = {node.node_id: node for node in graph.nodes}
        removable = nodes.keys() - cls.retained_node_ids(graph.nodes, graph.root_node_id)
        retained, sequences = [], []
        for node in graph.nodes:
            if node.node_id in removable:
                continue
            edges = []
            for edge in node.edges:
                child, chain = edge.child_node_id, []
                while child in removable:
                    chain.append(nodes[child])
                    child = nodes[child].edges[0].child_node_id
                if chain:
                    sequences.append((node.node_id, edge.action_id, ProofGraphForcedSequenceV1.build(
                        chain, observation_ids={item.node_id: observations[item.node_id] for item in chain}
                        if observations else None)))
                edges.append(replace(edge, child_node_id=child))
            retained.append(replace(node, edges=tuple(edges)))
        compact = ProofGraphV1.build(scope=graph.scope, root_node_id=graph.root_node_id,
            nodes=retained, utility_scale=graph.utility_scale, boundary_evaluator=graph.boundary_evaluator,
            tactic_context=graph.tactic_context)
        candidate = cls(compact, tuple(sequences), tuple(sorted(
            (key, value) for key, value in observations.items() if key not in removable)),
            graph.sha256, canonical_sha256(observations), "")
        result = replace(candidate, sha256=canonical_sha256(candidate._core()))
        result.expand()
        return result

    def _core(self):
        return {"schemaVersion": 1, "kind": "proof_graph_sequence_pack", "graph": self.graph.to_dict(),
                "sequences": [{"parentNodeId": parent, "actionId": action, "sequence": sequence.to_dict()}
                              for parent, action, sequence in self.sequences],
                "observationIds": dict(self.observation_ids), "sourceGraphHash": self.source_graph_hash,
                "sourceObservationHash": self.source_observation_hash}

    def to_dict(self):
        return {**self._core(), "sha256": self.sha256}

    def expand(self) -> tuple[ProofGraphV1, Mapping[str, str]]:
        self.graph.validate()
        if canonical_sha256(self._core()) != self.sha256:
            raise ValueError("packed graph checksum mismatch")
        sequences = {(parent, action): sequence for parent, action, sequence in self.sequences}
        if len(sequences) != len(self.sequences):
            raise ValueError("packed sequence attachment must be unique")
        nodes = {node.node_id: node for node in self.graph.nodes}
        observations = dict(self.observation_ids)
        for (parent, action), sequence in sequences.items():
            sequence.validate()
            node = nodes.get(parent)
            edge = next((edge for edge in node.edges if edge.action_id == action), None) if node else None
            if edge is None or edge.child_node_id != sequence.nodes[-1].edges[0].child_node_id:
                raise ValueError("packed sequence endpoint mismatch")
            if any(item.node_id in nodes for item in sequence.nodes):
                raise ValueError("packed sequence node identity collision")
            nodes[parent] = replace(node, edges=tuple(
                replace(item, child_node_id=sequence.nodes[0].node_id) if item.action_id == action else item
                for item in node.edges))
            nodes.update((item.node_id, item) for item in sequence.nodes)
            observations.update(sequence.observation_ids)
        expanded = ProofGraphV1.build(scope=self.graph.scope, root_node_id=self.graph.root_node_id,
            nodes=tuple(nodes.values()), utility_scale=self.graph.utility_scale,
            boundary_evaluator=self.graph.boundary_evaluator, tactic_context=self.graph.tactic_context)
        if expanded.sha256 != self.source_graph_hash or canonical_sha256(observations) != self.source_observation_hash:
            raise ValueError("packed graph differs from source identity")
        return expanded, observations


@dataclass(frozen=True, slots=True)
class NativeProofTacticBindingsV1:
    model: cp_model.CpModel
    root_card_variables: tuple[Any, ...]
    root_actor_variables: tuple[Any, ...]
    root_owner_variables: tuple[Any, ...]
    root_winning_variables: tuple[Any, ...]
    root_value_variables: tuple[Any, ...]
    selected_root_action_index: int | None
    root_actor_is_target_variable: Any
    target_seat_variable: Any | None
    base_trick_variables: tuple[Any, ...]
    principal_trick_winner_seat_variables: tuple[Any, ...]
    principal_trick_winning_card_variables: tuple[Any, ...]
    principal_trick_indexes: tuple[int, ...]
    principal_action_card_variables: tuple[Any, ...]
    principal_action_actor_variables: tuple[Any, ...]
    principal_action_present_variables: tuple[Any, ...]
    principal_action_trick_indexes: tuple[int, ...]
    principal_variation_has_continuation_variable: Any
    first_principal_trick_winner_team_variable: Any | None
    trump_suit_variable: Any | None
    proof_complete_variable: Any
    continuation_line_digits: tuple[Any, ...] = ()
    continuation_line_universe: str | None = None
    root_project_consequence_variables: tuple[Any, ...] = ()
    root_selection_variables: tuple[Any, ...] = ()
    bind_root_raw_payoffs: Callable[..., tuple[Mapping[str, Any], ...]] | None = None

    def with_current_trick(self, trick_index: int, plays: Sequence[tuple[int, int]]) -> "NativeProofTacticBindingsV1":
        """Count a whole PV trick, including its fixed pre-decision cards."""
        if not plays:
            return self
        if (type(trick_index) is not int or not 0 <= trick_index < 8 or len(plays) > 3
                or any(type(card) is not int or not 0 <= card < 32
                       or type(actor) is not int or not 0 <= actor < 4 for card, actor in plays)
                or len({card for card, _ in plays}) != len(plays)
                or not self.principal_action_trick_indexes
                or min(self.principal_action_trick_indexes) != trick_index):
            raise ValueError("current-trick prefix disagrees with native principal path")
        cards = tuple(self.model.new_int_var(card, card, f"principal_prefix_card_{index}")
                      for index, (card, _) in enumerate(plays))
        actors = tuple(self.model.new_int_var(actor, actor, f"principal_prefix_actor_{index}")
                       for index, (_, actor) in enumerate(plays))
        return replace(self,
            principal_action_card_variables=cards + self.principal_action_card_variables,
            principal_action_actor_variables=actors + self.principal_action_actor_variables,
            principal_action_present_variables=(self.model.new_constant(1),) * len(plays)
                                              + self.principal_action_present_variables,
            principal_action_trick_indexes=(trick_index,) * len(plays) + self.principal_action_trick_indexes)

@dataclass(frozen=True, slots=True)
class ProofRootActionV1:
    action_id: int
    action_code: str
    value: int
    satisfies: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "actionId": self.action_id,
            "action": self.action_code,
            "value": self.value,
            "satisfies": self.satisfies,
        }


@dataclass(frozen=True, slots=True)
class PolicyActionChoiceV1:
    node_id: str
    action_id: int

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "PolicyActionChoiceV1":
        if not isinstance(value, Mapping):
            raise TypeError("policy choice must be an object")
        if set(value) != {"nodeId", "actionId"}:
            raise ValueError("policy choice fields are invalid")
        node_id = value.get("nodeId")
        action_id = value.get("actionId")
        if type(node_id) is not str or not node_id.strip():
            raise ValueError("policy choice nodeId must be a non-empty string")
        if type(action_id) is not int or action_id < 0:
            raise ValueError("policy choice actionId must be a non-negative integer")
        return cls(node_id.strip(), action_id)

    def to_dict(self) -> dict[str, Any]:
        return {"nodeId": self.node_id, "actionId": self.action_id}


@dataclass(frozen=True, slots=True)
class DeclarativeActionPolicyV1:
    seat: str
    policy_id: str
    version: str
    seed: int
    choices: tuple[PolicyActionChoiceV1, ...]
    profile_hash: str
    materialized_hash: str
    materialization: str | None

    @classmethod
    def build(
        cls,
        seat: str,
        value: Mapping[str, Any],
    ) -> "DeclarativeActionPolicyV1":
        if seat not in _SEAT_TEAMS:
            raise ValueError("policy seat is invalid")
        if not isinstance(value, Mapping):
            raise TypeError("declarative action policy must be an object")
        expected = {
            "schemaVersion",
            "policyId",
            "version",
            "seed",
            "choices",
            "profileHash",
        }
        template = set(value) == {*expected, "materialization"}
        if set(value) != expected and not template:
            raise ValueError("declarative action policy fields are invalid")
        if type(value.get("schemaVersion")) is not int or value.get("schemaVersion") != 1:
            raise ValueError("declarative action policy schemaVersion must be integer 1")
        if value.get("policyId") != _POLICY_ID or value.get("version") != _POLICY_VERSION:
            raise ValueError("declarative action policy identity is unsupported")
        seed = value.get("seed")
        if type(seed) is not int or seed < 0:
            raise ValueError("declarative action policy seed must be non-negative")
        raw_choices = value.get("choices")
        if not isinstance(raw_choices, list):
            raise TypeError("declarative action policy choices must be a list")
        choices = tuple(
            sorted(
                (PolicyActionChoiceV1.build(item) for item in raw_choices),
                key=lambda item: item.node_id,
            )
        )
        if len({item.node_id for item in choices}) != len(choices):
            raise ValueError("declarative action policy node choices must be unique")
        materialization = value.get("materialization") if template else None
        if template and (
            materialization != _POLICY_MATERIALIZATION or choices
        ):
            raise ValueError("declarative policy template materialization is invalid")
        core = {
            "schemaVersion": 1,
            "seat": seat,
            "policyId": _POLICY_ID,
            "version": _POLICY_VERSION,
            "seed": seed,
            "choices": [item.to_dict() for item in choices],
        }
        if materialization is not None:
            core["materialization"] = materialization
        profile_hash = value.get("profileHash")
        if type(profile_hash) is not str or profile_hash != canonical_sha256(core):
            raise ValueError("declarative action policy profileHash mismatch")
        return cls(
            seat,
            _POLICY_ID,
            _POLICY_VERSION,
            seed,
            choices,
            profile_hash,
            profile_hash,
            materialization,
        )

    def materialize(
        self,
        choices: Sequence[PolicyActionChoiceV1],
    ) -> "DeclarativeActionPolicyV1":
        if self.choices:
            raise ValueError("only an empty declarative policy template can materialize")
        ordered = tuple(sorted(choices, key=lambda item: item.node_id))
        if len({item.node_id for item in ordered}) != len(ordered):
            raise ValueError("materialized policy node choices must be unique")
        core = {
            "schemaVersion": 1,
            "seat": self.seat,
            "policyId": self.policy_id,
            "version": self.version,
            "seed": self.seed,
            "choices": [item.to_dict() for item in ordered],
            "materialization": self.materialization,
        }
        return DeclarativeActionPolicyV1(
            self.seat,
            self.policy_id,
            self.version,
            self.seed,
            ordered,
            self.profile_hash,
            canonical_sha256(core),
            self.materialization,
        )

    def identity_dict(self) -> dict[str, Any]:
        value = {
            "policyId": self.policy_id,
            "version": self.version,
            "seed": self.seed,
            "profileHash": self.profile_hash,
        }
        if self.materialized_hash != self.profile_hash:
            value["materializedHash"] = self.materialized_hash
        if self.materialization is not None:
            value["materialization"] = self.materialization
        return value


def select_declarative_policy_action_v1(
    policy: DeclarativeActionPolicyV1,
    legal_actions: Sequence[ProofGraphEdgeV1],
) -> ProofGraphEdgeV1:
    """Select one legal action using only the governed policy and visible actions."""

    if not isinstance(policy, DeclarativeActionPolicyV1):
        raise TypeError("policy action selection requires a declarative policy")
    if not legal_actions or any(
        not isinstance(item, ProofGraphEdgeV1) for item in legal_actions
    ):
        raise ValueError("policy action selection requires legal graph edges")
    return min(
        legal_actions,
        key=lambda item: (
            canonical_sha256(
                {
                    "schemaVersion": 1,
                    "policyId": policy.policy_id,
                    "version": policy.version,
                    "seed": policy.seed,
                    "actionId": item.action_id,
                    "actionCode": item.action_code,
                }
            ),
            item.action_id,
            item.action_code,
        ),
    )


@dataclass(frozen=True, slots=True)
class ContinuationLineCountV1:
    complete: bool
    line_universe: str
    exact_count: str | None
    lower_bound: str | None
    upper_bound: str | None
    limb_count: int
    limb_limit: int
    solver_status: str | None
    model_hash: str | None
    exhaustion_reason: str | None

    @classmethod
    def exact(
        cls,
        *,
        line_universe: str,
        count: int,
        limb_count: int,
        limb_limit: int,
        solver_status: str,
        model_hash: str,
    ) -> "ContinuationLineCountV1":
        return cls(
            True,
            line_universe,
            str(count),
            None,
            None,
            limb_count,
            limb_limit,
            solver_status,
            model_hash,
            None,
        )

    @classmethod
    def incomplete(
        cls,
        *,
        line_universe: str,
        upper_bound: int | None,
        limb_count: int,
        limb_limit: int,
        solver_status: str | None,
        model_hash: str | None,
        exhaustion_reason: str,
    ) -> "ContinuationLineCountV1":
        return cls(
            False,
            line_universe,
            None,
            "0",
            str(upper_bound) if upper_bound is not None else None,
            limb_count,
            limb_limit,
            solver_status,
            model_hash,
            exhaustion_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "lineUniverse": self.line_universe,
            "exactCount": self.exact_count,
            "lowerBound": self.lower_bound,
            "upperBound": self.upper_bound,
            "limbBase": LINE_COUNT_BASE,
            "limbCount": self.limb_count,
            "limbLimit": self.limb_limit,
            "solverStatus": self.solver_status,
            "modelHash": self.model_hash,
            "exhaustionReason": self.exhaustion_reason,
        }


@dataclass(frozen=True, slots=True)
class OrToolsGraphProofOutcomeV1:
    completion_status: str
    solver_status: str
    claim_satisfied: bool
    selected_action: str | None
    winning_root_action_count: int | None
    root_actions: tuple[ProofRootActionV1, ...]
    principal_variation: tuple[str, ...]
    continuation_lines: ContinuationLineCountV1 | None
    policies: tuple[DeclarativeActionPolicyV1, ...]
    graph_hash: str
    model_hash: str
    request_hash: str
    variable_count: int
    constraint_count: int
    sha256: str
    _model_document_json: str | None
    hidden_worlds: Mapping[str, Any] | None = None
    proof_engine_id: str = PROOF_ENGINE_ID
    proof_engine_version: str = PROOF_ENGINE_VERSION
    native_qualification: Mapping[str, Any] | None = None
    winning_root_action_lower_bound: int | None = None
    winning_root_action_upper_bound: int | None = None

    @classmethod
    def build(
        cls,
        *,
        completion_status: str,
        solver_status: str,
        claim_satisfied: bool,
        selected_action: str | None,
        winning_root_action_count: int | None,
        root_actions: Sequence[ProofRootActionV1],
        principal_variation: Sequence[str],
        continuation_lines: ContinuationLineCountV1 | None,
        policies: Sequence[DeclarativeActionPolicyV1],
        graph_hash: str,
        model_hash: str,
        request_hash: str,
        variable_count: int,
        constraint_count: int,
        model_document: Mapping[str, Any] | None = None,
        hidden_worlds: Mapping[str, Any] | None = None,
        proof_engine_id: str = PROOF_ENGINE_ID,
        proof_engine_version: str = PROOF_ENGINE_VERSION,
        native_qualification: Mapping[str, Any] | None = None,
        winning_root_action_lower_bound: int | None = None,
        winning_root_action_upper_bound: int | None = None,
    ) -> "OrToolsGraphProofOutcomeV1":
        if winning_root_action_count is not None:
            if winning_root_action_lower_bound is None:
                winning_root_action_lower_bound = winning_root_action_count
            if winning_root_action_upper_bound is None:
                winning_root_action_upper_bound = winning_root_action_count
        model_document_json = None
        if model_document is not None:
            if canonical_sha256(model_document) != model_hash:
                raise ValueError("proof model document does not match modelHash")
            model_document_json = json.dumps(
                model_document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        core = {
            "schemaVersion": 1,
            "proofEngine": {
                "id": proof_engine_id,
                "version": proof_engine_version,
                "solverVersion": ortools.__version__,
            },
            "completionStatus": completion_status,
            "solverStatus": solver_status,
            "claimSatisfied": claim_satisfied,
            "selectedAction": selected_action,
            "winningRootActionCount": winning_root_action_count,
            "winningRootActionLowerBound": winning_root_action_lower_bound,
            "winningRootActionUpperBound": winning_root_action_upper_bound,
            "rootActions": [item.to_dict() for item in root_actions],
            "principalVariation": list(principal_variation),
            "continuationLines": (
                continuation_lines.to_dict() if continuation_lines else None
            ),
            "policyIdentities": {
                item.seat: item.identity_dict() for item in policies
            },
            "identities": {
                "graphHash": graph_hash,
                "modelHash": model_hash,
                "proofRequestHash": request_hash,
            },
            "modelSize": {
                "variables": variable_count,
                "constraints": constraint_count,
            },
        }
        if hidden_worlds is not None:
            core["hiddenWorlds"] = copy.deepcopy(dict(hidden_worlds))
        if native_qualification is not None:
            core["nativeQualification"] = copy.deepcopy(
                dict(native_qualification)
            )
        return cls(
            completion_status,
            solver_status,
            claim_satisfied,
            selected_action,
            winning_root_action_count,
            tuple(root_actions),
            tuple(principal_variation),
            continuation_lines,
            tuple(policies),
            graph_hash,
            model_hash,
            request_hash,
            variable_count,
            constraint_count,
            canonical_sha256(core),
            model_document_json,
            copy.deepcopy(dict(hidden_worlds)) if hidden_worlds is not None else None,
            proof_engine_id,
            proof_engine_version,
            copy.deepcopy(dict(native_qualification))
            if native_qualification is not None
            else None,
            winning_root_action_lower_bound,
            winning_root_action_upper_bound,
        )

    def model_document(self) -> dict[str, Any]:
        if self._model_document_json is None:
            raise RuntimeError("proof model document is unavailable")
        return json.loads(self._model_document_json)

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schemaVersion": 1,
            "proofEngine": {
                "id": self.proof_engine_id,
                "version": self.proof_engine_version,
                "solverVersion": ortools.__version__,
            },
            "completionStatus": self.completion_status,
            "solverStatus": self.solver_status,
            "claimSatisfied": self.claim_satisfied,
            "selectedAction": self.selected_action,
            "winningRootActionCount": self.winning_root_action_count,
            "winningRootActionLowerBound": self.winning_root_action_lower_bound,
            "winningRootActionUpperBound": self.winning_root_action_upper_bound,
            "rootActions": [item.to_dict() for item in self.root_actions],
            "principalVariation": list(self.principal_variation),
            "continuationLines": (
                self.continuation_lines.to_dict()
                if self.continuation_lines
                else None
            ),
            "policyIdentities": {
                item.seat: item.identity_dict() for item in self.policies
            },
            "identities": {
                "graphHash": self.graph_hash,
                "modelHash": self.model_hash,
                "proofRequestHash": self.request_hash,
            },
            "modelSize": {
                "variables": self.variable_count,
                "constraints": self.constraint_count,
            },
            "sha256": self.sha256,
        }
        if self.hidden_worlds is not None:
            value["hiddenWorlds"] = copy.deepcopy(dict(self.hidden_worlds))
        if self.native_qualification is not None:
            value["nativeQualification"] = copy.deepcopy(
                dict(self.native_qualification)
            )
        return value


class OrToolsScenarioGraphProofV1:
    """Solve complete-state exact or bounded strategic values with CP-SAT."""

    @staticmethod
    def _solve(
        model: cp_model.CpModel,
        *,
        maximum_seconds: float,
        cancel_check: Callable[[], bool] | None = None,
        maximum_conflicts: int | None = None,
        maximum_deterministic_time: float | None = None,
    ) -> tuple[cp_model.CpSolver, Any]:
        solver = cp_model.CpSolver()
        if cancel_check is not None and cancel_check():
            raise InterruptedError("graph proof cancelled")
        if maximum_seconds <= 0:
            return solver, cp_model.UNKNOWN
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_time_in_seconds = maximum_seconds
        if maximum_deterministic_time is not None:
            solver.parameters.max_deterministic_time = maximum_deterministic_time
        if maximum_conflicts is not None:
            solver.parameters.max_number_of_conflicts = maximum_conflicts
        status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
        if cancel_check is not None and cancel_check():
            raise InterruptedError("graph proof cancelled")
        return solver, status

    @staticmethod
    def _solve_count(
        model: cp_model.CpModel,
        *,
        maximum_seconds: float,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[cp_model.CpSolver, Any]:
        return OrToolsScenarioGraphProofV1._solve(
            model,
            maximum_seconds=maximum_seconds,
            **({"cancel_check": cancel_check} if cancel_check is not None else {}),
        )

    @staticmethod
    def _path_upper_bounds(
        graph: ProofGraphV1,
    ) -> dict[str, int]:
        nodes = {item.node_id: item for item in graph.nodes}
        result: dict[str, int] = {}

        def count(node_id: str) -> int:
            cached = result.get(node_id)
            if cached is not None:
                return cached
            node = nodes[node_id]
            value = (
                1
                if node.is_leaf
                else sum(count(edge.child_node_id) for edge in node.edges)
            )
            result[node_id] = value
            return value

        count(graph.root_node_id)
        return result

    @staticmethod
    def _required_limbs(maximum_count: int) -> int:
        return required_line_count_limbs_v1(maximum_count)

    @staticmethod
    def _policies(
        graph: ProofGraphV1,
        request: ScenarioProofRequestV1,
    ) -> tuple[
        tuple[DeclarativeActionPolicyV1, ...],
        dict[str, ProofGraphEdgeV1],
    ]:
        if request.mode != "policy_conditioned":
            return (), {}
        assert request.policies is not None
        policies = tuple(
            DeclarativeActionPolicyV1.build(
                seat,
                request.policies[seat],
            )
            for seat in sorted(_SEAT_TEAMS)
        )
        nodes = {item.node_id: item for item in graph.nodes}
        expected = {
            item.node_id
            for item in graph.nodes
            if not item.is_leaf and item.node_id != graph.root_node_id
        }
        if all(policy.materialization == _POLICY_MATERIALIZATION for policy in policies):
            by_seat = {item.seat: item for item in policies}
            materialized: list[DeclarativeActionPolicyV1] = []
            for seat, policy in sorted(by_seat.items()):
                choices = []
                for node in sorted(
                    (
                        item
                        for item in graph.nodes
                        if not item.is_leaf
                        and item.node_id != graph.root_node_id
                        and item.actor_seat == seat
                    ),
                    key=lambda item: item.node_id,
                ):
                    edge = select_declarative_policy_action_v1(
                        policy,
                        node.edges,
                    )
                    choices.append(PolicyActionChoiceV1(node.node_id, edge.action_id))
                materialized.append(policy.materialize(choices))
            policies = tuple(materialized)
        selected: dict[str, ProofGraphEdgeV1] = {}
        for policy in policies:
            for choice in policy.choices:
                node = nodes.get(choice.node_id)
                if node is None or node.is_leaf or node.node_id == graph.root_node_id:
                    raise ValueError("policy choice must reference a non-root decision")
                if node.actor_seat != policy.seat:
                    raise ValueError("policy choice does not own its decision node")
                edge = next(
                    (
                        item
                        for item in node.edges
                        if item.action_id == choice.action_id
                    ),
                    None,
                )
                if edge is None:
                    raise ValueError("policy choice actionId is not legal at its node")
                if node.node_id in selected:
                    raise ValueError("policy decision has more than one selected action")
                selected[node.node_id] = edge
        if any(nodes[node_id].actor_seat is None for node_id in expected):
            raise ValueError("policy-conditioned graph decisions require actor seats")
        if set(selected) != expected:
            raise ValueError("policies must select every non-root decision exactly once")
        return policies, selected

    def _count_lines(
        self,
        graph: ProofGraphV1,
        request: ScenarioProofRequestV1,
        *,
        value_solver: cp_model.CpSolver,
        values: Mapping[str, Any],
        selected_edge: ProofGraphEdgeV1,
        threshold: int,
        maximum_seconds: float,
        maximum_limbs: int,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[ContinuationLineCountV1, bool]:
        started = time.perf_counter()
        universe = str(request.line_universe)
        nodes = {item.node_id: item for item in graph.nodes}
        path_upper = self._path_upper_bounds(graph)
        selected_upper = path_upper[selected_edge.child_node_id]
        limb_count = self._required_limbs(selected_upper)
        if limb_count > maximum_limbs:
            return (
                ContinuationLineCountV1.incomplete(
                    line_universe=universe,
                    upper_bound=selected_upper,
                    limb_count=limb_count,
                    limb_limit=maximum_limbs,
                    solver_status=None,
                    model_hash=None,
                    exhaustion_reason="maximum_line_count_limbs",
                ),
                False,
            )

        reachable: set[str] = set()

        def retain(node_id: str) -> None:
            if node_id in reachable:
                return
            reachable.add(node_id)
            for edge in nodes[node_id].edges:
                retain(edge.child_node_id)

        retain(selected_edge.child_node_id)
        counted_nodes = tuple(
            node for node in graph.nodes if node.node_id in reachable
        )
        model = cp_model.CpModel()
        digits = {
            node.node_id: tuple(
                model.new_int_var(
                    0,
                    LINE_COUNT_BASE - 1,
                    f"count_{node_index}_{limb}",
                )
                for limb in range(limb_count)
            )
            for node_index, node in enumerate(counted_nodes)
        }
        for node_index, node in enumerate(counted_nodes):
            if node.is_leaf:
                assert node.utility is not None
                model.add(digits[node.node_id][0] == int(node.utility >= threshold))
                for limb in range(1, limb_count):
                    model.add(digits[node.node_id][limb] == 0)
                continue
            included = list(node.edges)
            if universe == "optimal_strategy":
                node_value = int(value_solver.value(values[node.node_id]))
                included = [
                    edge
                    for edge in included
                    if int(value_solver.value(values[edge.child_node_id]))
                    == node_value
                ]
            if not included:
                raise AssertionError("solved proof node has no counted continuation")
            carry: Any = 0
            for limb in range(limb_count):
                next_carry = model.new_int_var(
                    0,
                    len(included),
                    f"carry_{node_index}_{limb}",
                )
                model.add(
                    sum(digits[edge.child_node_id][limb] for edge in included)
                    + carry
                    == digits[node.node_id][limb]
                    + LINE_COUNT_BASE * next_carry
                )
                carry = next_carry
            model.add(carry == 0)

        model_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "proofEngine": PROOF_ENGINE_ID,
                "proofEngineVersion": PROOF_ENGINE_VERSION,
                "solverVersion": ortools.__version__,
                "graphHash": graph.sha256,
                "proofRequestHash": request.sha256,
                "selectedAction": selected_edge.action_code,
                "lineUniverse": universe,
                "limbBase": LINE_COUNT_BASE,
                "limbCount": limb_count,
                "modelProto": str(model.proto),
            }
        )
        solver, status = self._solve_count(
            model,
            maximum_seconds=maximum_seconds - (time.perf_counter() - started),
            **({"cancel_check": cancel_check} if cancel_check is not None else {}),
        )
        solver_status = solver.status_name(status).lower()
        if status in {cp_model.INFEASIBLE, cp_model.MODEL_INVALID}:
            return (
                ContinuationLineCountV1.incomplete(
                    line_universe=universe,
                    upper_bound=selected_upper,
                    limb_count=limb_count,
                    limb_limit=maximum_limbs,
                    solver_status=solver_status,
                    model_hash=model_hash,
                    exhaustion_reason="line_count_model_error",
                ),
                True,
            )
        if status != cp_model.OPTIMAL:
            return (
                ContinuationLineCountV1.incomplete(
                    line_universe=universe,
                    upper_bound=selected_upper,
                    limb_count=limb_count,
                    limb_limit=maximum_limbs,
                    solver_status=solver_status,
                    model_hash=model_hash,
                    exhaustion_reason="line_count_incomplete",
                ),
                False,
            )
        count = sum(
            int(solver.value(digit)) * (LINE_COUNT_BASE**limb)
            for limb, digit in enumerate(digits[selected_edge.child_node_id])
        )
        return (
            ContinuationLineCountV1.exact(
                line_universe=universe,
                count=count,
                limb_count=limb_count,
                limb_limit=maximum_limbs,
                solver_status=solver_status,
                model_hash=model_hash,
            ),
            False,
        )

    @classmethod
    def _native_line_count_digits(
        cls,
        model: cp_model.CpModel,
        graph: ProofGraphV1,
        request: ScenarioProofRequestV1,
        values: Mapping[str, Any],
        selected_edge: ProofGraphEdgeV1,
        threshold: int,
        maximum_limbs: int,
    ) -> tuple[Any, ...]:
        """Count selected-root action-edge paths in the same proof model."""
        universe = request.line_universe
        if universe not in {"all_legal", "optimal_strategy"}:
            raise ValueError("native continuation-line universe is unsupported")
        nodes = {node.node_id: node for node in graph.nodes}
        upper = cls._path_upper_bounds(graph)[selected_edge.child_node_id]
        limb_count = cls._required_limbs(upper)
        if limb_count > maximum_limbs:
            raise ValueError("native continuation-line count exceeds limb budget")
        digits: dict[str, tuple[Any, ...]] = {}

        def count(node_id: str) -> tuple[Any, ...]:
            if node_id in digits:
                return digits[node_id]
            node = nodes[node_id]
            children = [(edge, count(edge.child_node_id)) for edge in node.edges]
            row = tuple(
                model.new_int_var(0, LINE_COUNT_BASE - 1, f"native_count_{len(digits)}_{limb}")
                for limb in range(limb_count)
            )
            digits[node_id] = row
            if node.is_leaf:
                model.add(row[0] == int(node.utility >= threshold))
                for digit in row[1:]:
                    model.add(digit == 0)
                return row
            included = []
            for edge_index, (edge, child) in enumerate(children):
                if universe == "all_legal":
                    included.append(child)
                    continue
                optimal = model.new_bool_var(f"native_count_opt_{node_id}_{edge_index}")
                model.add(values[edge.child_node_id] == values[node_id]).only_enforce_if(optimal)
                model.add(values[edge.child_node_id] != values[node_id]).only_enforce_if(optimal.Not())
                gated = []
                for limb, digit in enumerate(child):
                    term = model.new_int_var(0, LINE_COUNT_BASE - 1, f"native_count_term_{node_id}_{edge_index}_{limb}")
                    model.add(term == digit).only_enforce_if(optimal)
                    model.add(term == 0).only_enforce_if(optimal.Not())
                    gated.append(term)
                included.append(tuple(gated))
            carry: Any = 0
            for limb in range(limb_count):
                next_carry = model.new_int_var(0, len(included), f"native_count_carry_{node_id}_{limb}")
                model.add(
                    sum(child[limb] for child in included) + carry
                    == row[limb] + LINE_COUNT_BASE * next_carry
                )
                carry = next_carry
            model.add(carry == 0)
            return row

        return count(selected_edge.child_node_id)

    @staticmethod
    def _threshold(request: ScenarioProofRequestV1, scale: int) -> int:
        value = Decimal(str(request.success_threshold)) * Decimal(scale)
        return int(value.to_integral_value(rounding=ROUND_CEILING))

    @staticmethod
    def _claim_literal(
        model: cp_model.CpModel,
        request: ScenarioProofRequestV1,
        root_wins: Sequence[Any],
        winning_count: Any,
        selected_index: int | None,
    ) -> Any:
        claim = model.new_bool_var("proof_claim_satisfied")
        if request.claim == "outcome_reachable":
            model.add(winning_count >= 1).only_enforce_if(claim)
            model.add(winning_count == 0).only_enforce_if(claim.Not())
        elif request.claim == "outcome_forced":
            model.add(winning_count == len(root_wins)).only_enforce_if(claim)
            model.add(winning_count != len(root_wins)).only_enforce_if(claim.Not())
        elif request.claim == "selected_action_is_unique_winner":
            if selected_index is None:
                model.add(claim == 0)
            else:
                count_is_one = model.new_bool_var("proof_winning_count_is_one")
                model.add(winning_count == 1).only_enforce_if(count_is_one)
                model.add(winning_count != 1).only_enforce_if(count_is_one.Not())
                model.add_bool_and(
                    (count_is_one, root_wins[selected_index])
                ).only_enforce_if(claim)
                model.add_bool_or(
                    (count_is_one.Not(), root_wins[selected_index].Not())
                ).only_enforce_if(claim.Not())
        elif request.claim == "target_team_has_no_salvage":
            model.add(winning_count == 0).only_enforce_if(claim)
            model.add(winning_count != 0).only_enforce_if(claim.Not())
        elif request.claim in {
            "winning_root_action_count",
            "winning_continuation_line_count",
        }:
            model.add(claim == 1)
        else:
            raise ValueError("claim is unsupported by the graph proof slice")
        return claim

    def prove(
        self,
        graph: ProofGraphV1 | ProofGraphSequencePackV1,
        request: ScenarioProofRequestV1,
        *,
        maximum_seconds: float = 5.0,
        maximum_conflicts: int | None = None,
        maximum_deterministic_time: float | None = None,
        maximum_line_count_limbs: int = 64,
        cancel_check: Callable[[], bool] | None = None,
        native_action_quality: bool = False,
        native_line_count: bool = False,
        native_root_only: bool = False,
        project_counterfactual_graph: ProofGraphV1 | None = None,
        project_compilation_evidence: Sequence[Mapping[str, Any]] = (),
        tactic_configure: Callable[
            [NativeProofTacticBindingsV1], Mapping[str, Any]
        ]
        | None = None,
    ) -> OrToolsGraphProofOutcomeV1:
        started = time.perf_counter()
        if project_compilation_evidence and project_counterfactual_graph is None:
            raise ValueError("project compilation evidence requires its counterfactual graph")
        if type(native_root_only) is not bool:
            raise TypeError("native_root_only must be boolean")
        if native_root_only and tactic_configure is None:
            raise ValueError("root-only bindings require a callback")
        if maximum_conflicts is not None and (type(maximum_conflicts) is not int or maximum_conflicts < 1):
            raise ValueError("proof conflict limit must be a positive integer")
        if maximum_deterministic_time is not None and (
                not math.isfinite(maximum_deterministic_time) or maximum_deterministic_time <= 0):
            raise ValueError("proof deterministic time bound must be finite and positive")
        if (maximum_conflicts is not None or maximum_deterministic_time is not None) and (
                request.claim == "winning_continuation_line_count"
                or request.line_universe is not None):
            raise ValueError("work-limited graph proof does not support continuation counting")
        if type(native_action_quality) is not bool or type(native_line_count) is not bool:
            raise TypeError("native_action_quality must be boolean")
        if native_action_quality and request.mode not in {"exact", "bounded"}:
            raise ValueError("native complete-state quality requires exact or bounded minimax")
        if cancel_check is not None:
            if not callable(cancel_check):
                raise TypeError("proof cancellation must be callable")
            if cancel_check():
                raise InterruptedError("graph proof cancelled")
        if isinstance(graph, ProofGraphSequencePackV1):
            # Expand before proof so forced actions retain depth, tactics and counts.
            graph, _ = graph.expand()
        graph.validate()
        if (
            request.mode not in {*_SCOPES, "policy_conditioned"}
            or request.visibility != "complete_state"
        ):
            raise ValueError(
                "graph proof supports complete-state exact, bounded, and policy proof"
            )
        if request.claim not in _SUPPORTED_CLAIMS:
            raise ValueError("claim is unsupported by the graph proof slice")
        expected_scope = (
            "exact" if request.mode == "policy_conditioned" else request.mode
        )
        if graph.scope != expected_scope:
            raise ValueError("proof graph scope does not match the request mode")
        if graph.boundary_evaluator != request.boundary_evaluator:
            raise ValueError("proof graph boundary evaluator does not match the request")
        if not math.isfinite(request.success_threshold):
            raise ValueError("proof success threshold must be finite")
        if (type(maximum_seconds) not in {int, float} or maximum_seconds <= 0 or math.isnan(maximum_seconds)
                or (math.isinf(maximum_seconds) and maximum_deterministic_time is None)):
            raise ValueError("proof solve time bound must be positive")
        if (
            type(maximum_line_count_limbs) is not int
            or maximum_line_count_limbs < 1
        ):
            raise ValueError("line-count limb tier must be a positive integer")
        allowed_line_universes = (
            {None, "policy_conditioned"}
            if request.mode == "policy_conditioned"
            else {None, "all_legal", "optimal_strategy"}
        )
        if request.line_universe not in allowed_line_universes:
            raise ValueError("line universe is unsupported by the graph proof slice")
        if native_line_count and (
            tactic_configure is None
            or request.selected_action is None
            or request.line_universe is None
        ):
            raise ValueError("native line count requires a selected action and same-solve bindings")

        nodes = {item.node_id: item for item in graph.nodes}
        root = nodes[graph.root_node_id]
        policies, policy_edges = self._policies(graph, request)
        if (
            request.selected_action is not None
            and request.selected_action
            not in {edge.action_code for edge in root.edges}
        ):
            raise ValueError("proof selectedAction is not a legal root action")
        selected_root_index = next(
            (
                index
                for index, edge in enumerate(root.edges)
                if edge.action_code == request.selected_action
            ),
            None,
        )

        leaves = [item.utility for item in graph.nodes if item.utility is not None]
        lower = min(leaves)
        upper = max(leaves)
        model = cp_model.CpModel()

        def attach_values(source: ProofGraphV1, prefix: str) -> dict[str, Any]:
            utilities = [node.utility for node in source.nodes if node.is_leaf]
            minimum, maximum = min(utilities), max(utilities)
            result = {
                node.node_id: model.new_int_var(minimum, maximum, f"{prefix}_{index}")
                for index, node in enumerate(source.nodes)
            }
            for node in source.nodes:
                value = result[node.node_id]
                if node.is_leaf:
                    model.add(value == node.utility)
                elif node.node_id in policy_edges:
                    model.add(value == result[policy_edges[node.node_id].child_node_id])
                elif node.actor_team == request.target_team:
                    model.add_max_equality(value, [result[edge.child_node_id] for edge in node.edges])
                else:
                    model.add_min_equality(value, [result[edge.child_node_id] for edge in node.edges])
            return result

        values = attach_values(graph, "value")
        project_consequences = []
        if project_counterfactual_graph is not None:
            graph.require_same_legal_graph(project_counterfactual_graph)
            if request.mode != "exact" or request.utility != "awarded_point_margin" or tactic_configure is None:
                raise ValueError("project counterfactual requires exact margin proof with native bindings")
            alternate = project_counterfactual_graph
            counter_values = attach_values(alternate, "project_counterfactual_value")
            counter_leaves = [node.utility for node in alternate.nodes if node.is_leaf]
            for index, edge in enumerate(root.edges):
                delta = model.new_int_var(
                    lower - max(counter_leaves), upper - min(counter_leaves),
                    f"root_project_consequence_{index}",
                )
                model.add(delta == values[edge.child_node_id] - counter_values[edge.child_node_id])
                project_consequences.append(delta)

        threshold = self._threshold(request, graph.utility_scale)
        root_wins = []
        for index, edge in enumerate(root.edges):
            wins = model.new_bool_var(f"root_wins_{index}")
            child_value = values[edge.child_node_id]
            model.add(child_value >= threshold).only_enforce_if(wins)
            model.add(child_value < threshold).only_enforce_if(wins.Not())
            root_wins.append(wins)
        winning_count = model.new_int_var(0, len(root.edges), "winning_root_count")
        model.add(winning_count == sum(root_wins))
        claim_literal = self._claim_literal(
            model,
            request,
            root_wins,
            winning_count,
            selected_root_index,
        )

        tactic_lowerings: Mapping[str, Any] = {}
        native_line_digits: tuple[Any, ...] = ()
        paired_raw_payoff_receipts: list[dict[str, Any]] = []
        if tactic_configure is not None:
            context = graph.tactic_context
            if (context is None and not native_root_only) or root.actor_seat is None:
                raise ValueError("native proof tactic bindings are unavailable")
            if native_line_count and request.line_universe != "policy_conditioned":
                assert selected_root_index is not None
                path_upper = self._path_upper_bounds(graph)[root.edges[selected_root_index].child_node_id]
                limbs = self._required_limbs(path_upper)
                if limbs > maximum_line_count_limbs:
                    refusal = {
                        "schemaVersion": 1,
                        "proofEngine": PROOF_ENGINE_ID,
                        "proofEngineVersion": PROOF_ENGINE_VERSION,
                        "solverVersion": ortools.__version__,
                        "graphHash": graph.sha256,
                        "proofRequestHash": request.sha256,
                        "exhaustionReason": "maximum_line_count_limbs",
                        "requiredLimbs": limbs,
                        "maximumLimbs": maximum_line_count_limbs,
                        "solvePerformed": False,
                        "modelProto": str(model.proto),
                    }
                    return OrToolsGraphProofOutcomeV1.build(
                        completion_status="proof_incomplete", solver_status="unknown",
                        claim_satisfied=False, selected_action=None,
                        winning_root_action_count=None, winning_root_action_lower_bound=0,
                        winning_root_action_upper_bound=len(root.edges),
                        root_actions=(), principal_variation=(),
                        continuation_lines=ContinuationLineCountV1.incomplete(
                            line_universe=str(request.line_universe), upper_bound=path_upper,
                            limb_count=limbs, limb_limit=maximum_line_count_limbs,
                            solver_status=None, model_hash=None,
                            exhaustion_reason="maximum_line_count_limbs",
                        ),
                        policies=policies, graph_hash=graph.sha256,
                        model_hash=canonical_sha256(refusal), request_hash=request.sha256,
                        variable_count=len(model.proto.variables),
                        constraint_count=len(model.proto.constraints), model_document=refusal,
                    )

            def fixed_int(name: str, value: int, lower: int, upper: int) -> Any:
                variable = model.new_int_var(lower, upper, name)
                model.add(variable == value)
                return variable

            def conjunction(name: str, literals: Sequence[Any]) -> Any:
                result = model.new_bool_var(name)
                model.add_bool_and(list(literals)).only_enforce_if(result)
                model.add_bool_or(
                    [literal.Not() for literal in literals]
                ).only_enforce_if(result.Not())
                return result

            def disjunction(name: str, literals: Sequence[Any]) -> Any:
                result = model.new_bool_var(name)
                model.add_bool_or(list(literals)).only_enforce_if(result)
                model.add_bool_and(
                    [literal.Not() for literal in literals]
                ).only_enforce_if(result.Not())
                return result

            def edge_choices(node: ProofGraphNodeV1, name: str) -> tuple[Any, ...]:
                chosen: list[Any] = []
                earlier_equal: list[Any] = []
                for edge_index, edge in enumerate(node.edges):
                    choice = model.new_bool_var(f"{name}_choice_{edge_index}")
                    if node.node_id == root.node_id and selected_root_index is not None:
                        model.add(choice == int(edge_index == selected_root_index))
                    elif node.node_id in policy_edges:
                        model.add(
                            choice
                            == int(
                                policy_edges[node.node_id].action_id
                                == edge.action_id
                            )
                        )
                    else:
                        equal = model.new_bool_var(f"{name}_optimal_{edge_index}")
                        model.add(
                            values[edge.child_node_id] == values[node.node_id]
                        ).only_enforce_if(equal)
                        model.add(
                            values[edge.child_node_id] != values[node.node_id]
                        ).only_enforce_if(equal.Not())
                        first_equal = conjunction(
                            f"{name}_first_{edge_index}",
                            (equal, *(item.Not() for item in earlier_equal)),
                        )
                        model.add(choice == first_equal)
                        earlier_equal.append(equal)
                    chosen.append(choice)
                model.add(sum(chosen) == 1)
                return tuple(chosen)

            root_selection = edge_choices(root, "root_selection")
            # Root-only predicates consume solved root values, not path variables.
            principal_nodes = () if native_root_only else graph.nodes
            active = {
                node.node_id: model.new_bool_var(f"principal_active_{index}")
                for index, node in enumerate(principal_nodes)
            }
            if not native_root_only:
                model.add(active[root.node_id] == 1)
            incoming: dict[str, list[Any]] = {
                node.node_id: [] for node in graph.nodes
            }
            traversals: list[tuple[ProofGraphEdgeV1, Any, str]] = []
            choices_by_node = {}
            for node_index, node in enumerate(principal_nodes):
                if node.is_leaf:
                    continue
                chosen = root_selection if node.node_id == root.node_id else edge_choices(
                    node, f"principal_{node_index}"
                )
                choices_by_node[node.node_id] = chosen
                for edge_index, (edge, choice) in enumerate(zip(node.edges, chosen, strict=True)):
                    traversal = conjunction(
                        f"principal_traverse_{node_index}_{edge_index}",
                        (active[node.node_id], choice),
                    )
                    incoming[edge.child_node_id].append(traversal)
                    traversals.append((edge, traversal, node.node_id))
            for node in principal_nodes:
                if node.node_id == root.node_id:
                    continue
                parents = incoming[node.node_id]
                if not parents:
                    model.add(active[node.node_id] == 0)
                else:
                    model.add(
                        active[node.node_id]
                        == disjunction(f"principal_in_{node.node_id}", parents)
                    )

            winner_variables = []
            winner_card_variables = []
            completed_indexes = sorted(
                {
                    edge.completed_trick_index
                    for edge, _traversal, _parent in traversals
                    if edge.completed_trick_index is not None
                }
            )
            for trick_index in completed_indexes:
                matches = [
                    (edge, traversal)
                    for edge, traversal, _parent in traversals
                    if edge.completed_trick_index == trick_index
                ]
                present = disjunction(
                    f"principal_trick_present_{trick_index}",
                    [traversal for _edge, traversal in matches],
                )
                winner = model.new_int_var(
                    0, 3, f"principal_trick_winner_{trick_index}"
                )
                winner_card = model.new_int_var(
                    0, 31, f"principal_trick_winning_card_{trick_index}"
                )
                for edge, traversal in matches:
                    assert edge.completed_trick_winner_seat is not None
                    assert edge.completed_trick_winning_card_id is not None
                    model.add(
                        winner == _SEATS.index(edge.completed_trick_winner_seat)
                    ).only_enforce_if(traversal)
                    model.add(
                        winner_card == edge.completed_trick_winning_card_id
                    ).only_enforce_if(traversal)
                model.add(winner == 0).only_enforce_if(present.Not())
                model.add(winner_card == 0).only_enforce_if(present.Not())
                winner_variables.append(winner)
                winner_card_variables.append(winner_card)
            after_root = [
                traversal
                for _edge, traversal, parent in traversals
                if parent != root.node_id
            ]
            continuation = (
                None if native_root_only else disjunction(
                    "principal_has_continuation", after_root
                )
            )
            first_winner_team = None
            if winner_variables:
                first_winner_team = model.new_int_var(
                    0, 1, "principal_first_winner_team"
                )
                model.add_modulo_equality(
                    first_winner_team, winner_variables[0], 2
                )
            root_actor = _SEATS.index(root.actor_seat)
            root_actor_variables = tuple(
                fixed_int(f"root_actor_{index}", root_actor, 0, 3)
                for index in range(len(root.edges))
            )
            owners = list(context["ownerByCard"]) if context is not None else None
            root_card_variables = []
            root_owner_variables = []
            for index, edge in enumerate(root.edges):
                if owners is None:
                    continue
                card_id = edge.action_id % 32
                root_card_variables.append(
                    fixed_int(f"root_card_{index}", card_id, 0, 31)
                )
                root_owner_variables.append(
                    fixed_int(f"root_owner_{index}", int(owners[card_id]), -1, 3)
                )
            actor_is_target = model.new_bool_var("root_actor_is_target")
            model.add(
                actor_is_target == int(root.actor_team == request.target_team)
            )
            target_seat = (
                root_actor_variables[0]
                if root.actor_team == request.target_team
                # A team target does not designate an opposing individual seat.
                else fixed_int("root_target_seat_absent", -1, -1, -1)
            )
            base_trick_variables = tuple(
                fixed_int(f"base_tricks_{index}", int(value), 0, 8)
                for index, value in enumerate(context["baseTricks"] if context is not None else ())
            )
            trump_value = context.get("trumpSuit") if context is not None else None
            trump_variable = (
                # Suit 4 means a solved no-trump contract, not missing evidence.
                fixed_int("proof_trump_suit", int(trump_value) if trump_value is not None else 4, 0, 4)
                if context is not None and "trumpSuit" in context
                else None
            )
            proof_complete = model.new_bool_var("proof_complete")
            model.add(proof_complete == 1)
            line_digits: tuple[Any, ...] = ()
            if native_line_count:
                assert selected_root_index is not None
                line_digits = (
                    (root_wins[selected_root_index],)
                    if request.line_universe == "policy_conditioned"
                    else self._native_line_count_digits(
                    model, graph, request, values, root.edges[selected_root_index],
                    threshold, maximum_line_count_limbs,
                    )
                )
            native_line_digits = line_digits
            principal_cards = []
            principal_actors = []
            principal_present = []
            principal_trick_indexes = []
            node_by_id = {node.node_id: node for node in graph.nodes}
            for index, (edge, traversal, parent) in enumerate(traversals):
                parent_node = node_by_id[parent]
                if parent_node.actor_seat is None or edge.play_trick_index is None:
                    continue
                principal_cards.append(
                    fixed_int(f"principal_card_{index}", edge.action_id % 32, 0, 31)
                )
                principal_actors.append(
                    fixed_int(
                        f"principal_actor_{index}",
                        _SEATS.index(parent_node.actor_seat),
                        0,
                        3,
                    )
                )
                principal_present.append(traversal)
                principal_trick_indexes.append(edge.play_trick_index)

            def bind_root_raw_payoffs(card_points, last_trick_bonus, current_trick_cards, *, first_leader=None,
                                      delayed_control=False):
                if (native_root_only or request.mode != "exact" or graph.scope != "exact"
                        or len(card_points) != 32 or any(type(value) is not int or not 0 <= value <= 32767
                            for value in card_points)
                        or type(last_trick_bonus) is not int or not 0 <= last_trick_bonus <= cp_model.INT_MAX // 64
                        or len(current_trick_cards) > 3 or len(set(current_trick_cards)) != len(current_trick_cards)
                        or any(type(card) is not int or not 0 <= card < 32 for card in current_trick_cards)
                        or not completed_indexes or completed_indexes != list(range(completed_indexes[0], 8))
                        or first_leader is not None and (type(first_leader) is not int or not 0 <= first_leader < 4
                            or not current_trick_cards and first_leader != root_actor)
                        or type(delayed_control) is not bool
                        or delayed_control and (first_leader is None or trump_variable is None
                            or any(edge.completed_trick_winning_card_id is None
                                for node in graph.nodes for edge in node.edges
                                if edge.completed_trick_index is not None))
                        or any(node.actor_seat is None or any(edge.play_trick_index is None for edge in node.edges)
                               for node in graph.nodes if not node.is_leaf)):
                    raise ValueError("paired raw payoffs require a complete exact play suffix and native point tables")
                if paired_raw_payoff_receipts:
                    raise ValueError("paired raw payoffs are already bound")
                rows = []
                for root_index, root_edge in enumerate(root.edges):
                    name = f"paired_root_{root_index}"
                    path_active = {node.node_id: model.new_bool_var(f"{name}_active_{index}")
                                   for index, node in enumerate(graph.nodes)}
                    model.add(path_active[root.node_id] == 1)
                    path_incoming = {node.node_id: [] for node in graph.nodes}
                    path_edges = []
                    for node_index, node in enumerate(graph.nodes):
                        if node_index % 128 == 0 and cancel_check is not None and cancel_check():
                            raise InterruptedError("graph proof cancelled")
                        if node.is_leaf:
                            continue
                        for edge_index, edge in enumerate(node.edges):
                            if node.node_id == root.node_id:
                                traversal = model.new_bool_var(f"{name}_root_{edge_index}")
                                model.add(traversal == int(edge_index == root_index))
                            else:
                                traversal = conjunction(f"{name}_traverse_{node_index}_{edge_index}",
                                    (path_active[node.node_id], choices_by_node[node.node_id][edge_index]))
                            path_incoming[edge.child_node_id].append(traversal)
                            path_edges.append((edge, traversal, _SEATS.index(node.actor_seat)))
                    for node in graph.nodes:
                        if node.node_id != root.node_id:
                            model.add(path_active[node.node_id] == disjunction(
                                f"{name}_incoming_{node.node_id}", path_incoming[node.node_id]))
                    row, margins = {}, []
                    for trick_index in completed_indexes:
                        label = f"trick{trick_index + 1}"
                        plays = [(edge, present, actor) for edge, present, actor in path_edges if edge.play_trick_index == trick_index]
                        endings = [(edge, present) for edge, present, _actor in path_edges if edge.completed_trick_index == trick_index]
                        prefix = current_trick_cards if trick_index == completed_indexes[0] else ()
                        model.add(sum(present for _, present, _actor in plays) + len(prefix) == 4)
                        model.add(sum(present for _, present in endings) == 1)
                        bound = 4 * max(card_points) + (last_trick_bonus if trick_index == 7 else 0)
                        points = model.new_int_var(0, bound, f"{name}_{label}_raw_points")
                        model.add(points == sum(card_points[edge.action_id % 32] * present for edge, present, _actor in plays)
                            + sum(card_points[card] for card in prefix) + (last_trick_bonus if trick_index == 7 else 0))
                        winner = model.new_int_var(0, 3, f"{name}_{label}_winner")
                        for edge, present in endings:
                            model.add(winner == _SEATS.index(edge.completed_trick_winner_seat)).only_enforce_if(present)
                        team = model.new_int_var(0, 1, f"{name}_{label}_team")
                        model.add_modulo_equality(team, winner, 2)
                        actor_won = model.new_bool_var(f"{name}_{label}_actor_won")
                        model.add(team == root_actor % 2).only_enforce_if(actor_won)
                        model.add(team != root_actor % 2).only_enforce_if(actor_won.Not())
                        margin = model.new_int_var(-bound, bound, f"{name}_{label}_raw_margin")
                        model.add(margin == points).only_enforce_if(actor_won)
                        model.add(margin == -points).only_enforce_if(actor_won.Not())
                        row.update({label + "RawPoints": points, label + "WinnerSeat": winner, label + "RawMargin": margin})
                        if delayed_control:
                            winning_card = model.new_int_var(0, 31, f"{name}_{label}_winning_card")
                            for edge, present in endings:
                                model.add(winning_card == edge.completed_trick_winning_card_id).only_enforce_if(present)
                            remaining = []
                            for future_index, (edge, present, actor) in enumerate(path_edges):
                                if future_index % 128 == 0 and cancel_check is not None and cancel_check():
                                    raise InterruptedError("graph proof cancelled")
                                if edge.play_trick_index <= trick_index or actor % 2 == root_actor % 2:
                                    continue
                                is_trump = model.new_bool_var(f"{name}_{label}_future_trump_{future_index}")
                                model.add(trump_variable == (edge.action_id % 32) // 8).only_enforce_if(is_trump)
                                model.add(trump_variable != (edge.action_id % 32) // 8).only_enforce_if(is_trump.Not())
                                remaining.append(conjunction(f"{name}_{label}_remaining_trump_{future_index}",
                                    (present, is_trump)))
                            opponent_trumps = model.new_int_var(0, 8, f"{name}_{label}_opponent_trumps_remaining")
                            model.add(opponent_trumps == sum(remaining))
                            row.update({label + "WinningCard": winning_card,
                                        label + "OpponentTrumpsRemaining": opponent_trumps})
                        if first_leader is not None:
                            leader = model.new_int_var(0, 3, f"{name}_{label}_leader")
                            model.add(leader == (first_leader if trick_index == completed_indexes[0]
                                else row[f"trick{trick_index}WinnerSeat"]))
                            lead_card = model.new_int_var(0, 31, f"{name}_{label}_lead_card")
                            if prefix:
                                model.add(lead_card == prefix[0])
                            else:
                                leads = []
                                for play_index, (edge, present, actor) in enumerate(plays):
                                    if play_index % 128 == 0 and cancel_check is not None and cancel_check():
                                        raise InterruptedError("graph proof cancelled")
                                    is_leader = model.new_bool_var(f"{name}_{label}_leader_actor_{play_index}")
                                    model.add(leader == actor).only_enforce_if(is_leader)
                                    model.add(leader != actor).only_enforce_if(is_leader.Not())
                                    leading = conjunction(f"{name}_{label}_lead_{play_index}", (present, is_leader))
                                    model.add(lead_card == edge.action_id % 32).only_enforce_if(leading)
                                    leads.append(leading)
                                model.add(sum(leads) == 1)
                            card_margin = model.new_int_var(-4 * max(card_points), 4 * max(card_points), f"{name}_{label}_card_raw_margin")
                            bonus = last_trick_bonus if trick_index == 7 else 0
                            model.add(card_margin == points - bonus).only_enforce_if(actor_won)
                            model.add(card_margin == -points + bonus).only_enforce_if(actor_won.Not())
                            row.update({label + "LeaderSeat": leader, label + "LeadCard": lead_card,
                                        label + "CardRawMargin": card_margin})
                        margins.append(margin)
                    total_bound = 32 * max(card_points) + last_trick_bonus
                    total = model.new_int_var(-total_bound, total_bound, f"{name}_continuation_raw_margin")
                    model.add(total == sum(margins))
                    immediate = model.new_int_var(-total_bound, total_bound, f"{name}_immediate_raw_margin")
                    model.add(immediate == (margins[0] if len(current_trick_cards) == 3 else 0))
                    row.update(firstTrickRawMargin=margins[0], immediateRawMargin=immediate, continuationRawMargin=total)
                    paired_raw_payoff_receipts.append({"actionId": root_edge.action_id, "action": root_edge.action_code,
                        "scope": "canonical_complete_state_continuation", "unit": "raw_card_points",
                        "cardPointsHash": canonical_sha256(list(card_points)),
                        "currentTrickCards": list(current_trick_cards), "lastTrickBonusRaw": last_trick_bonus,
                        **({"firstLeader": first_leader} if first_leader is not None else {}),
                        **({"delayedControl": True} if delayed_control else {}),
                        "variables": {key: value.name for key, value in row.items()}})
                    rows.append(row)
                return tuple(rows)

            tactic_lowerings = dict(
                tactic_configure(
                    NativeProofTacticBindingsV1(
                        model,
                        tuple(root_card_variables),
                        root_actor_variables,
                        tuple(root_owner_variables),
                        tuple(root_wins),
                        tuple(values[edge.child_node_id] for edge in root.edges),
                        selected_root_index,
                        actor_is_target,
                        target_seat,
                        base_trick_variables,
                        tuple(winner_variables),
                        tuple(winner_card_variables),
                        tuple(completed_indexes),
                        tuple(principal_cards),
                        tuple(principal_actors),
                        tuple(principal_present),
                        tuple(principal_trick_indexes),
                        continuation,
                        first_winner_team,
                        trump_variable,
                        proof_complete,
                        line_digits,
                        request.line_universe if native_line_count else None,
                        tuple(project_consequences),
                        root_selection,
                        bind_root_raw_payoffs,
                    )
                )
            )

        proto_text = str(model.proto)
        model_document = {
            "schemaVersion": 1,
            "proofEngine": PROOF_ENGINE_ID,
            "proofEngineVersion": PROOF_ENGINE_VERSION,
            "solverVersion": ortools.__version__,
            "graphHash": graph.sha256,
            "proofRequestHash": request.sha256,
            "targetTeam": request.target_team,
            "utility": request.utility,
            "proofMode": request.mode,
            "visibility": request.visibility,
            "successThreshold": request.success_threshold,
            "rootActorTeam": root.actor_team,
            "rootActorSeat": root.actor_seat,
            "rootActionVariables": [
                {
                    "actionId": edge.action_id,
                    "action": edge.action_code,
                    "valueVariable": values[edge.child_node_id].name,
                    "winningVariable": root_wins[index].name,
                }
                for index, edge in enumerate(root.edges)
            ],
            "claimVariable": claim_literal.name,
            "claimEvidenceBinding": {
                "kind": request.claim,
                "variable": claim_literal.name,
                "literalIndex": int(claim_literal.index),
                "winningCountVariable": winning_count.name,
            },
            "policyProfileHashes": {
                item.seat: item.profile_hash for item in policies
            },
            "policyMaterializedHashes": {
                item.seat: item.materialized_hash for item in policies
            },
            "modelProto": proto_text,
        }
        if project_counterfactual_graph is not None:
            model_document["projectCounterfactualGraph"] = project_counterfactual_graph.to_dict()
            model_document["rootProjectConsequenceVariables"] = [item.name for item in project_consequences]
            if project_compilation_evidence:
                model_document["projectCompilationEvidence"] = copy.deepcopy(list(project_compilation_evidence))
        if paired_raw_payoff_receipts:
            model_document["pairedRawPayoffVariables"] = paired_raw_payoff_receipts
        if tactic_lowerings:
            model_document["nativeQualificationVariables"] = {
                predicate_id: lowering.literal.name
                for predicate_id, lowering in sorted(tactic_lowerings.items())
            }
            model_document["nativeQualificationLiteralIndices"] = {
                predicate_id: int(lowering.literal.index)
                for predicate_id, lowering in sorted(tactic_lowerings.items())
            }
        model_hash = canonical_sha256(model_document)
        variable_count = len(model.proto.variables)
        constraint_count = len(model.proto.constraints)
        solver, status = self._solve(
            model,
            maximum_seconds=maximum_seconds - (time.perf_counter() - started),
            **({"maximum_conflicts": maximum_conflicts} if maximum_conflicts is not None else {}),
            **({"maximum_deterministic_time": maximum_deterministic_time}
               if maximum_deterministic_time is not None else {}),
            **({"cancel_check": cancel_check} if cancel_check is not None else {}),
        )
        solver_status = solver.status_name(status).lower()
        if status in {cp_model.INFEASIBLE, cp_model.MODEL_INVALID}:
            return OrToolsGraphProofOutcomeV1.build(
                completion_status="model_error",
                solver_status=solver_status,
                claim_satisfied=False,
                selected_action=None,
                winning_root_action_count=None,
                winning_root_action_lower_bound=0,
                winning_root_action_upper_bound=len(root.edges),
                root_actions=(),
                principal_variation=(),
                continuation_lines=None,
                policies=policies,
                graph_hash=graph.sha256,
                model_hash=model_hash,
                request_hash=request.sha256,
                variable_count=variable_count,
                constraint_count=constraint_count,
                model_document=model_document,
            )
        if status != cp_model.OPTIMAL:
            return OrToolsGraphProofOutcomeV1.build(
                completion_status="proof_incomplete",
                solver_status=solver_status,
                claim_satisfied=False,
                selected_action=None,
                winning_root_action_count=None,
                winning_root_action_lower_bound=0,
                winning_root_action_upper_bound=len(root.edges),
                root_actions=(),
                principal_variation=(),
                continuation_lines=None,
                policies=policies,
                graph_hash=graph.sha256,
                model_hash=model_hash,
                request_hash=request.sha256,
                variable_count=variable_count,
                constraint_count=constraint_count,
                model_document=model_document,
            )

        rows = tuple(
            ProofRootActionV1(
                edge.action_id,
                edge.action_code,
                int(solver.value(values[edge.child_node_id])),
                bool(solver.boolean_value(root_wins[index])),
            )
            for index, edge in enumerate(root.edges)
        )
        native_qualification = {
            predicate_id: lowering.extract(solver)
            for predicate_id, lowering in sorted(tactic_lowerings.items())
        }
        action_quality = None
        if native_action_quality:
            # Reuse the native value-variable certificate; never reinsert solved
            # Python values as constants in a second qualification model.
            from .ortools_information_set_compiler import (
                InformationSetCompilationStopped,
                _certify_action_quality_model_v1,
            )

            try:
                action_quality = _certify_action_quality_model_v1(
                    maximum_conflicts=maximum_conflicts,
                    actions=tuple(edge.action_code for edge in root.edges),
                    qualification_model_proto=proto_text,
                    qualification_variable_bindings=tuple(
                        (edge.action_code, values[edge.child_node_id].name, root_wins[index].name)
                        for index, edge in enumerate(root.edges)
                    ),
                    qualification_model_hash=model_hash,
                    action_ids={edge.action_code: edge.action_id for edge in root.edges},
                    actor_relation="target" if root.actor_team == request.target_team else "opponent",
                    selected_action=request.selected_action,
                    deadline=time.monotonic() + maximum_seconds - (time.perf_counter() - started),
                    cancel_check=cancel_check,
                    scope="compiled_complete_state",
                    analysis_scope={"mode": request.mode, "visibility": request.visibility,
                                    "requestHash": request.sha256,
                                    "horizon": request.horizon,
                                    "boundaryEvaluator": request.boundary_evaluator},
                )
                quality_status = "model_error" if action_quality.get("modelStatus") == "quarantined" else None
            except InformationSetCompilationStopped:
                quality_status = "proof_incomplete"
            if quality_status is not None:
                return OrToolsGraphProofOutcomeV1.build(
                    completion_status=quality_status, solver_status=solver_status,
                    claim_satisfied=False, selected_action=None,
                    winning_root_action_count=None, winning_root_action_lower_bound=0,
                    winning_root_action_upper_bound=len(root.edges), root_actions=(),
                    principal_variation=(), continuation_lines=None, policies=policies,
                    graph_hash=graph.sha256, model_hash=model_hash, request_hash=request.sha256,
                    variable_count=variable_count, constraint_count=constraint_count,
                    model_document=model_document,
                )
            model_document = {**model_document,
                "qualificationModelHash": model_hash,
                "actionQuality": {key: value for key, value in action_quality.items() if key != "modelProto"},
                "modelProto": action_quality["modelProto"]}
            model_hash = canonical_sha256(model_document)
            variable_count = action_quality["variableCount"]
            constraint_count = action_quality["constraintCount"]
        selected_code = action_quality["actualAction"] if action_quality is not None else request.selected_action
        if selected_code is None:
            best_value = (
                max(item.value for item in rows)
                if root.actor_team == request.target_team
                else min(item.value for item in rows)
            )
            selected_code = min(
                (item for item in rows if item.value == best_value),
                key=lambda item: item.action_id,
            ).action_code

        selected_edge = next(
            edge for edge in root.edges if edge.action_code == selected_code
        )
        principal = [selected_edge.action_code]
        current = nodes[selected_edge.child_node_id]
        while not current.is_leaf:
            chosen = policy_edges.get(current.node_id)
            if chosen is None:
                current_value = int(solver.value(values[current.node_id]))
                optimal = [
                    edge
                    for edge in current.edges
                    if int(solver.value(values[edge.child_node_id])) == current_value
                ]
                chosen = min(optimal, key=lambda item: item.action_id)
            principal.append(chosen.action_code)
            current = nodes[chosen.child_node_id]

        count = int(solver.value(winning_count))
        claim_satisfied = bool(solver.boolean_value(claim_literal))
        continuation_lines = None
        count_model_error = False
        if native_line_digits:
            native_count = sum(
                int(solver.value(digit)) * (LINE_COUNT_BASE**limb)
                for limb, digit in enumerate(native_line_digits)
            )
            continuation_lines = ContinuationLineCountV1.exact(
                line_universe=str(request.line_universe),
                count=native_count,
                limb_count=len(native_line_digits),
                limb_limit=maximum_line_count_limbs,
                solver_status=solver_status,
                model_hash=model_hash,
            )
        elif request.line_universe == "policy_conditioned":
            selected_row = next(
                item for item in rows if item.action_code == selected_code
            )
            continuation_lines = ContinuationLineCountV1.exact(
                line_universe="policy_conditioned",
                count=int(selected_row.satisfies),
                limb_count=1,
                limb_limit=maximum_line_count_limbs,
                solver_status=solver_status,
                model_hash=model_hash,
            )
        elif request.line_universe is not None:
            continuation_lines, count_model_error = self._count_lines(
                graph,
                request,
                value_solver=solver,
                values=values,
                selected_edge=selected_edge,
                threshold=threshold,
                maximum_seconds=maximum_seconds - (time.perf_counter() - started),
                maximum_limbs=maximum_line_count_limbs,
                cancel_check=cancel_check,
            )
        completion = {
            "exact": "exact_complete",
            "bounded": "bounded_complete",
            "policy_conditioned": "policy_complete",
        }[request.mode]
        if count_model_error:
            completion = "model_error"
            claim_satisfied = False
        elif (
            request.claim == "winning_continuation_line_count"
            and continuation_lines is not None
            and not continuation_lines.complete
        ):
            completion = "proof_incomplete"
            claim_satisfied = False
        elif not claim_satisfied:
            completion = "proof_failed"
        return OrToolsGraphProofOutcomeV1.build(
            completion_status=completion,
            solver_status=solver_status,
            claim_satisfied=claim_satisfied,
            selected_action=selected_code,
            winning_root_action_count=count,
            root_actions=rows,
            principal_variation=principal,
            continuation_lines=continuation_lines,
            policies=policies,
            graph_hash=graph.sha256,
            model_hash=model_hash,
            request_hash=request.sha256,
            variable_count=variable_count,
            constraint_count=constraint_count,
            model_document=model_document,
            native_qualification=native_qualification or None,
        )


__all__ = [
    "ContinuationLineCountV1",
    "DeclarativeActionPolicyV1",
    "LINE_COUNT_BASE",
    "OrToolsGraphProofOutcomeV1",
    "OrToolsScenarioGraphProofV1",
    "ProofGraphEdgeV1",
    "ProofGraphNodeV1",
    "ProofGraphV1",
    "PolicyActionChoiceV1",
    "ProofRootActionV1",
    "select_declarative_policy_action_v1",
]
