"""Compile an exact normalized play decision into a legal proof DAG.

Graph transitions use the native CP-SAT play stage over the published Domain
IR.  The completed finite graph remains the input to CP-SAT proof.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card, FULL_DECK
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.models import Project
from baloot_platform.core.play_ir import PlayDomainIRV1
from baloot_platform.core.project_ir import ProjectDomainIRV1, ProjectDeclarationActionV1
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.core.structural_ir import StructuralDomainIRV1

from .ortools_exact_state import (
    OrToolsNormalizedExactStateV1, _native_project_prefix_projection,
    _extract_native_project_prefix_projection, _NativeProjectPrefixProjectionV1,
    _native_forced_pass_certificate, _effective_prefix_hash,
    _certified_project_followup_rows, _native_project_action_literals,
    _native_baloot_prefix_projection, _native_project_visibility_projection,
)
from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1
from .ortools_infeasible_core import sound_infeasible_assumption_indexes
from .ortools_play_tail import (
    OrToolsPlayTailResultV1,
    OrToolsPlayTailStageV1,
    PlayTailSolveStopped,
    TailPrefixActionV1,
)
from .ortools_proof import ProofGraphEdgeV1, ProofGraphNodeV1, ProofGraphV1, ProofGraphSequencePackV1
from .ortools_project_fragment import attach_project_lifecycle_fragment_v1
from .ortools_scoring_stage import (
    CpSatScoringInputsV1,
    OrToolsScoringFragmentV1,
    attach_scoring_fragment_v1,
    attach_project_counterfactual_scoring_v1,
    extract_scoring_evidence_v1,
)
from .scenario_contracts import ExactTracePrefixV1, ScenarioProofRequestV1, canonical_sha256


COMPILER_ID = "ortools_complete_state_graph_v1"
COMPILER_VERSION = "1.6.0"
INTERNAL_RAW_POINT_MARGIN_UTILITY_V1 = "_internal_raw_point_margin_v1"
_EXACT_MODES = frozenset({"exact", "policy_conditioned"})


def iter_native_path_query_guards_v1(queries, *, maximum_bindings):
    """Reconstruct full guards from legacy rows or append-only native query receipts.

    A null predecessor starts a new refinement. Each yielded map is independent;
    consumers should process it immediately rather than retaining every snapshot.
    This validates evidence encoding, not strategic truth or legal-world coverage.
    """
    if (not isinstance(queries, (list, tuple)) or type(maximum_bindings) is not int
            or maximum_bindings < 1):
        raise ValueError("native query history requires rows and a positive binding cap")
    guards, previous = {}, None
    common = {"schemaVersion", "baseModelHash", "queryModelHash", "threshold", "status", "utility", "sha256"}
    for row in queries:
        if (not isinstance(row, Mapping) or type(row.get("schemaVersion")) is not int
                or row["schemaVersion"] not in {1, 2, 3}
                or row.get("sha256") != canonical_sha256({k: v for k, v in row.items() if k != "sha256"})
                or any(type(row.get(key)) is not str or len(row[key]) != 64
                       or any(char not in "0123456789abcdef" for char in row[key])
                       for key in ("baseModelHash", "queryModelHash"))
                or type(row.get("threshold")) is not int
                or row.get("status") not in {"witness_found", "infeasible"}
                or (type(row.get("utility")) is not int if row.get("status") == "witness_found"
                    else row.get("utility") is not None)):
            raise ValueError("native query receipt integrity or outcome mismatch")
        fields = common | ({"conditionalActions"} if row["schemaVersion"] == 1 else {
            "kind", "previousQueryReceiptHash", "conditionalActionCount", "conditionalActionsAdded"})
        if row["schemaVersion"] == 3:
            fields.add("counterexampleObjective")
            if row.get("counterexampleObjective") != "minimum_utility":
                raise ValueError("native query objective mismatch")
        if set(row) not in (fields, fields | {"projectCoverageHash"}):
            raise ValueError("native query receipt shape mismatch")
        if row["schemaVersion"] == 1:
            guards = {}
            additions = row["conditionalActions"]
        else:
            if row["kind"] != f"native_observed_path_query_v{row['schemaVersion']}":
                raise ValueError("native query receipt kind mismatch")
            if row["previousQueryReceiptHash"] is None:
                guards = {}
            elif (previous is None or row["previousQueryReceiptHash"] != previous["sha256"]
                    or row["baseModelHash"] != previous["baseModelHash"]
                    or row["threshold"] != previous["threshold"]
                    or row.get("counterexampleObjective") != previous.get("counterexampleObjective")
                    or row.get("projectCoverageHash") != previous.get("projectCoverageHash")):
                raise ValueError("native query predecessor or source mismatch")
            additions = row["conditionalActionsAdded"]
        if not isinstance(additions, list) or len(guards) + len(additions) > maximum_bindings:
            raise ValueError("native query binding cap exceeded")
        def action(raw):
            if not isinstance(raw, Mapping) or set(raw) != {"actor", "card", "ekka"} or type(raw["ekka"]) is not bool:
                raise ValueError("native query action shape mismatch")
            parsed = TailPrefixActionV1.build(raw["actor"], raw["card"], ekka=raw["ekka"])
            if parsed.to_dict() != raw:
                raise ValueError("native query action is not canonical")
            return parsed
        for binding in additions:
            if (not isinstance(binding, Mapping) or set(binding) != {"prefix", "action"}
                    or not isinstance(binding["prefix"], list)):
                raise ValueError("native query conditional action shape mismatch")
            prefix = tuple(action(raw) for raw in binding["prefix"])
            if prefix in guards:
                raise ValueError("native query binding cannot be repeated or replaced")
            guards[prefix] = None if binding["action"] is None else action(binding["action"])
        if row["schemaVersion"] >= 2 and (type(row["conditionalActionCount"]) is not int
                or row["conditionalActionCount"] != len(guards)):
            raise ValueError("native query binding count mismatch")
        previous = row
        yield row, dict(guards)


class _CompilationStopped(RuntimeError):
    def __init__(self, reason: str, *, model_error: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.model_error = model_error


@dataclass(frozen=True, slots=True)
class CompleteStateGraphReceiptV1:
    status: str
    exhaustion_reason: str | None
    exact_state_id: str
    request_hash: str
    play_domain_hash: str
    scoring_domain_hash: str
    structural_domain_hash: str
    proof_state_projection_hash: str

    graph_hash: str | None
    model_hash: str
    explored_node_count: int
    explored_edge_count: int
    transposition_edge_count: int
    terminal_node_count: int
    boundary_node_count: int
    maximum_depth_reached: int
    elapsed_seconds: float
    sha256: str

    representation: Mapping[str, Any] | None = None

    @classmethod
    def build(
        cls,
        *,
        status: str,
        exhaustion_reason: str | None,
        exact_state_id: str,
        request_hash: str,
        play_domain_hash: str,
        scoring_domain_hash: str,
        structural_domain_hash: str,
        proof_state_projection_hash: str,
        graph_hash: str | None,
        model_hash: str,
        explored_node_count: int,
        explored_edge_count: int,
        transposition_edge_count: int,
        terminal_node_count: int,
        boundary_node_count: int,
        maximum_depth_reached: int,
        elapsed_seconds: float,
        representation: Mapping[str, Any] | None = None,
    ) -> "CompleteStateGraphReceiptV1":
        evidence = {
            "schemaVersion": 1,
            "compilerId": COMPILER_ID,
            "compilerVersion": COMPILER_VERSION,
            "status": status,
            "exhaustionReason": exhaustion_reason,
            "exactStateId": exact_state_id,
            "requestHash": request_hash,
            "domainIrHashes": {
                "play": play_domain_hash,
                "scoring": scoring_domain_hash,
                "structural": structural_domain_hash,
            },
            "graphHash": graph_hash,
            "proofStateProjectionHash": proof_state_projection_hash,
            "modelHash": model_hash,
            "exploration": {
                "nodes": explored_node_count,
                "edges": explored_edge_count,
                "transpositionEdges": transposition_edge_count,
                "terminalNodes": terminal_node_count,
                "boundaryNodes": boundary_node_count,
                "maximumDepthReached": maximum_depth_reached,
            },
            "elapsedSeconds": elapsed_seconds,
        }
        if representation is not None:
            evidence["representation"] = dict(representation)
        return cls(
            status,
            exhaustion_reason,
            exact_state_id,
            request_hash,
            play_domain_hash,
            scoring_domain_hash,
            structural_domain_hash,
            proof_state_projection_hash,
            graph_hash,
            model_hash,
            explored_node_count,
            explored_edge_count,
            transposition_edge_count,
            terminal_node_count,
            boundary_node_count,
            maximum_depth_reached,
            elapsed_seconds,
            canonical_sha256(evidence),
            dict(representation) if representation is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "compilerId": COMPILER_ID,
            "compilerVersion": COMPILER_VERSION,
            "status": self.status,
            "exhaustionReason": self.exhaustion_reason,
            "exactStateId": self.exact_state_id,
            "requestHash": self.request_hash,
            "domainIrHashes": {
                "play": self.play_domain_hash,
                "scoring": self.scoring_domain_hash,
                "structural": self.structural_domain_hash,
            },
            "graphHash": self.graph_hash,
            "proofStateProjectionHash": self.proof_state_projection_hash,
            "modelHash": self.model_hash,
            "exploration": {
                "nodes": self.explored_node_count,
                "edges": self.explored_edge_count,
                "transpositionEdges": self.transposition_edge_count,
                "terminalNodes": self.terminal_node_count,
                "boundaryNodes": self.boundary_node_count,
                "maximumDepthReached": self.maximum_depth_reached,
            },
            "elapsedSeconds": self.elapsed_seconds,
            "sha256": self.sha256,
            **({"representation": dict(self.representation)} if self.representation is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class CompleteStateGraphCompilationV1:
    graph: ProofGraphV1 | None
    receipt: CompleteStateGraphReceiptV1
    decision_observation_ids: Mapping[str, str] = field(default_factory=dict)
    sequence_pack: ProofGraphSequencePackV1 | None = None
    project_counterfactual_graph: ProofGraphV1 | None = None
    model_document: Mapping[str, Any] | None = None

    @property
    def complete(self) -> bool:
        return self.receipt.status == "complete" and self.graph is not None


@dataclass(frozen=True, slots=True)
class _StartState:
    exact_state_id: str
    rules_hash: str
    cards: tuple[Card, ...]
    ownership: Mapping[Card, Seat]
    contract: Contract
    trump: Suit | None
    first_leader: Seat
    base_prefix: tuple[TailPrefixActionV1, ...]
    locked: bool
    base_raw: tuple[int, int]
    base_tricks: tuple[int, int]
    buyer: Seat
    project_points: tuple[int, int]
    baloot_points: tuple[int, int]
    escalation: str
    last_escalating_team: Team | None
    starting_match_score: tuple[int, int]
    expected_legal_actions: tuple[TailPrefixActionV1, ...]
    proof_state_projection_hash: str
    terminal_project_domain: ProjectDomainIRV1 | None = None
    terminal_project_owners: tuple[int, ...] = ()
    terminal_project_declarations: tuple[tuple[str, int], ...] = ()
    terminal_project_dealer: Seat = Seat.EAST
    terminal_project_second_trick_leader: Seat | None = None
    source_state_hash: str | None = None
    completed_play_history: tuple[TailPrefixActionV1, ...] = ()
    project_disclosure_evidence: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class CompleteStateGraphSeedV1:
    """Fully specified CP-derived input to the existing proof graph compiler."""

    exact_state_id: str
    rules_hash: str
    remaining_hands: tuple[tuple[Card, ...], ...]
    current_trick: tuple[TailPrefixActionV1, ...]
    first_leader: Seat
    next_actor: Seat
    legal_actions: tuple[TailPrefixActionV1, ...]
    contract: Contract
    trump: Suit | None
    buyer: Seat
    locked: bool
    base_raw: tuple[int, int]
    base_tricks: tuple[int, int]
    awarded_project_points: tuple[int, int] | None
    baloot_points: tuple[int, int]
    escalation: str
    last_escalating_team: Team | None
    starting_match_score: tuple[int, int]
    proof_state_projection_hash: str
    pending_projects: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompleteStateGraphSeedV1":
        """Read only the canonical, hash-bound seed emitted by this compiler."""
        try:
            if type(value["schemaVersion"]) is not int or value["schemaVersion"] != 1 or type(value["locked"]) is not bool:
                raise ValueError("invalid proof seed types")
            for field_name in ("baseRaw", "baseTricks", "awardedProjectPoints", "balootPoints", "startingMatchScore"):
                if field_name == "awardedProjectPoints" and value[field_name] is None and value.get("pendingProjects") is not None:
                    continue
                if (not isinstance(value[field_name], list) or len(value[field_name]) != 2
                        or any(type(item) is not int or item < 0 for item in value[field_name])):
                    raise ValueError("invalid proof seed points")
            if any(type(card) is not int for hand in value["remainingHands"].values() for card in hand):
                raise ValueError("invalid proof seed card identities")
            def action(row):
                return TailPrefixActionV1.build(row["actor"], row["card"], ekka=row.get("ekka", False))

            result = cls.build(
                exact_state_id=value["exactStateId"], rules_hash=value["rulesHash"],
                remaining_hands=tuple(tuple(Card(card) for card in value["remainingHands"][seat.name.lower()]) for seat in Seat),
                current_trick=tuple(action(row) for row in value["currentTrick"]["plays"]),
                first_leader=Seat.parse(value["currentTrick"]["leader"]),
                next_actor=Seat.parse(value["nextActor"]),
                legal_actions=tuple(action(row) for row in value["legalActions"]),
                contract=Contract.parse(value["contract"]),
                trump=Suit.parse(value["trump"]) if value["trump"] is not None else None,
                buyer=Seat.parse(value["buyer"]), locked=value["locked"],
                base_raw=tuple(value["baseRaw"]), base_tricks=tuple(value["baseTricks"]),
                awarded_project_points=tuple(value["awardedProjectPoints"]) if value["awardedProjectPoints"] is not None else None,
                baloot_points=tuple(value["balootPoints"]), escalation=value["escalation"],
                last_escalating_team=Team[value["lastEscalatingTeam"].upper()] if value["lastEscalatingTeam"] is not None else None,
                starting_match_score=tuple(value["startingMatchScore"]),
                pending_projects=value.get("pendingProjects"),
            )
            if canonical_sha256(result.to_dict()) != canonical_sha256(dict(value)):
                raise ValueError("noncanonical proof seed")
            return result
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("invalid proof seed") from exc

    @classmethod
    def build(
        cls,
        *,
        exact_state_id: str,
        rules_hash: str,
        remaining_hands: tuple[tuple[Card, ...], ...],
        current_trick: tuple[TailPrefixActionV1, ...],
        first_leader: Seat,
        next_actor: Seat,
        legal_actions: tuple[TailPrefixActionV1, ...],
        contract: Contract,
        trump: Suit | None,
        buyer: Seat,
        locked: bool,
        base_raw: tuple[int, int],
        base_tricks: tuple[int, int],
        awarded_project_points: tuple[int, int] | None,
        baloot_points: tuple[int, int],
        escalation: str,
        last_escalating_team: Team | None,
        starting_match_score: tuple[int, int],
        pending_projects: Mapping[str, Any] | None = None,
    ) -> "CompleteStateGraphSeedV1":
        values = (
            exact_state_id,
            rules_hash,
            tuple(
                tuple(sorted(hand, key=lambda card: card.id))
                for hand in remaining_hands
            ),
            tuple(current_trick),
            first_leader,
            next_actor,
            tuple(sorted(legal_actions, key=_action_sort_key)),
            contract,
            trump,
            buyer,
            locked,
            tuple(base_raw),
            tuple(base_tricks),
            tuple(awarded_project_points) if awarded_project_points is not None else None,
            tuple(baloot_points),
            escalation,
            last_escalating_team,
            tuple(starting_match_score),
        )
        return cls(*values, canonical_sha256(_seed_projection(*values[1:], pending_projects)), pending_projects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "exactStateId": self.exact_state_id,
            **_seed_projection(
                self.rules_hash,
                self.remaining_hands,
                self.current_trick,
                self.first_leader,
                self.next_actor,
                self.legal_actions,
                self.contract,
                self.trump,
                self.buyer,
                self.locked,
                self.base_raw,
                self.base_tricks,
                self.awarded_project_points,
                self.baloot_points,
                self.escalation,
                self.last_escalating_team,
                self.starting_match_score,
                self.pending_projects,
            ),
            "proofStateProjectionHash": self.proof_state_projection_hash,
        }


def _action_sort_key(action: TailPrefixActionV1) -> int:
    return action.card.id + 32 * int(action.ekka)


def _seed_projection(
    rules_hash: str,
    remaining_hands: tuple[tuple[Card, ...], ...],
    current_trick: tuple[TailPrefixActionV1, ...],
    first_leader: Seat,
    next_actor: Seat,
    legal_actions: tuple[TailPrefixActionV1, ...],
    contract: Contract,
    trump: Suit | None,
    buyer: Seat,
    locked: bool,
    base_raw: tuple[int, int],
    base_tricks: tuple[int, int],
    awarded_project_points: tuple[int, int] | None,
    baloot_points: tuple[int, int],
    escalation: str,
    last_escalating_team: Team | None,
    starting_match_score: tuple[int, int],
    pending_projects: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "rulesHash": rules_hash,
        "remainingHands": {
            seat.name.lower(): [card.id for card in remaining_hands[int(seat)]]
            for seat in Seat
        },
        "currentTrick": {
            "leader": first_leader.name.lower(),
            "plays": [action.to_dict() for action in current_trick],
        },
        "nextActor": next_actor.name.lower(),
        "legalActions": [action.to_dict() for action in legal_actions],
        "contract": contract.value,
        "trump": trump.name.lower() if trump is not None else None,
        "buyer": buyer.name.lower(),
        "locked": locked,
        "baseRaw": list(base_raw),
        "baseTricks": list(base_tricks),
        "awardedProjectPoints": list(awarded_project_points) if awarded_project_points is not None else None,
        **({"pendingProjects": dict(pending_projects)} if pending_projects is not None else {}),
        "balootPoints": list(baloot_points),
        "escalation": escalation,
        "lastEscalatingTeam": (
            last_escalating_team.name.lower()
            if last_escalating_team is not None
            else None
        ),
        "startingMatchScore": list(starting_match_score),
    }


@dataclass(frozen=True, slots=True)
class _DerivedPlayState:
    remaining_hands: tuple[tuple[Card, ...], ...]
    current_trick: tuple[TailPrefixActionV1, ...]
    next_actor: Seat | None
    legal_play_actions: tuple[TailPrefixActionV1, ...]
    trick_winners: tuple[Seat, ...]
    trick_raw_points: tuple[int, ...]
    terminal_utility: int | None = None
    boundary_utility: int | None = None
    project_checkpoint: _NativeProjectPrefixProjectionV1 | None = None
    native_play: OrToolsPlayTailResultV1 | None = field(default=None, compare=False, repr=False)
    public_baloot_events: tuple[tuple[int, Seat, str], ...] | None = None
    project_counterfactual_utility: int | None = None
    project_disclosures: tuple[Project, ...] | None = None


@dataclass(frozen=True, slots=True)
class CompleteStateDecisionObservationV1:
    """Observer-safe projection input captured before graph transposition."""

    actor: Seat
    path: tuple[TailPrefixActionV1, ...]
    prefix: tuple[TailPrefixActionV1, ...]
    remaining_hands: tuple[tuple[Card, ...], ...]
    current_trick: tuple[TailPrefixActionV1, ...]
    trick_winners: tuple[Seat, ...]
    trick_raw_points: tuple[int, ...]
    raw: tuple[int, int]
    tricks: tuple[int, int]
    project_checkpoint: _NativeProjectPrefixProjectionV1 | None = None
    native_play: OrToolsPlayTailResultV1 | None = field(default=None, compare=False, repr=False)


CompleteStateDecisionObserverV1 = Callable[[CompleteStateDecisionObservationV1], str]


def _transposition_core_v1(
    state_core: Mapping[str, Any],
    observation_id: str | None,
) -> Mapping[str, Any]:
    """Keep default graph identity stable and separate remembered observations."""

    if observation_id is None:
        return state_core
    return {
        "state": state_core,
        "observerInformationId": observation_id,
    }


@dataclass(frozen=True, slots=True)
class CpSatPolicyPathResultV1:
    """One policy path with hash-bound native play and scoring certificates."""

    play: OrToolsPlayTailResultV1
    terminal_utility: int
    scoring_evidence: Mapping[str, Any]
    model_hash: str


class _CpSatPathOracle:
    """Compile native play paths and bind their terminal scoring evidence."""

    def __init__(
        self,
        start: _StartState,
        *,
        play_stage: OrToolsPlayTailStageV1,
        play_domain: PlayDomainIRV1,
        scoring_domain: ScoringDomainIRV1,
        structural_domain: StructuralDomainIRV1,
        target: Team,
        utility: str,
        utility_encoding: str | None = None,
        play_step_count: int | None = None,
        defer_terminal_scoring: bool = False,
        boundary_evaluator: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
        project_prefix: ExactTracePrefixV1 | None = None,
        project_domain: ProjectDomainIRV1 | None = None,
        project_stop: int = 4,
        project_normalization: OrToolsNormalizedExactStateV1 | None = None,
        project_continuation_actions: tuple[ProjectDeclarationActionV1, ...] | None = None,
        project_counterfactual: bool = False,
    ) -> None:
        if type(project_counterfactual) is not bool:
            raise TypeError("project_counterfactual must be boolean")
        if project_counterfactual and (not play_stage.native_conversion or defer_terminal_scoring
                                       or utility != "awarded_point_margin"):
            raise ValueError("project counterfactual requires native terminal margin scoring")
        if project_continuation_actions is not None and (
            project_prefix is None or not isinstance(project_continuation_actions, tuple)
            or any(not isinstance(action, ProjectDeclarationActionV1) for action in project_continuation_actions)
        ):
            raise ValueError("project continuation requires a prefix and typed action tuple")
        self.project_continuation_actions = project_continuation_actions
        self.start = start
        self.play_stage = play_stage
        self.play_domain = play_domain
        self.scoring_domain = scoring_domain
        self.structural_domain = structural_domain
        self.target = target
        self.utility = utility
        self.utility_encoding = utility_encoding
        self.boundary_evaluator = boundary_evaluator
        self.base_length = len(start.base_prefix)
        self._incremental_policy_batches = (
            defer_terminal_scoring and play_step_count is None
        )
        effective_play_step_count = play_step_count
        if self._incremental_policy_batches:
            effective_play_step_count = min(
                len(start.cards),
                ((self.base_length // 4) + 1) * 4,
            )
        self.build = play_stage.build_model(
            cards=start.cards,
            ownership=start.ownership,
            contract=start.contract,
            trump=start.trump,
            first_leader=start.first_leader,
            play_domain=play_domain,
            structural_domain=structural_domain,
            prefix=start.base_prefix,
            locked=start.locked,
            decision_prefix_length=self.base_length,
            scoring_domain=None,
            play_step_count=effective_play_step_count,
            cancel_check=cancel_check,
        )
        model = self.build.model
        self.project_checkpoints = {}
        self.project_disclosure_bindings = {}
        self.project_disclosure_projects = ()
        self.project_lifecycle = None
        project_certificate_hash = None
        certified_seats = ()
        if project_normalization is not None:
            if project_prefix is None or project_normalization.rules_hash != start.rules_hash:
                raise ValueError("project normalization differs from graph prefix or rules")
            if project_normalization.decision_kind == "project":
                _certified_project_followup_rows(project_normalization, project_prefix)
            elif project_normalization.prefix_hash != _effective_prefix_hash(project_prefix):
                raise ValueError("project normalization differs from graph prefix or rules")
            certificate = _native_forced_pass_certificate(project_normalization)
            if certificate is None:
                raise ValueError("project normalization lacks forced-pass evidence")
            certified_owners = {Card.parse(card): seat for seat in Seat
                for card in project_normalization.state["initialHands"][seat.name.lower()]}
            if certified_owners != dict(start.ownership):
                raise ValueError("project normalization differs from graph ownership")
            state = project_normalization.state
            certified_trump = Suit.parse(state["trumpSuit"]) if state.get("trumpSuit") is not None else None
            if Contract.parse(state["contract"]) != start.contract or certified_trump != start.trump:
                raise ValueError("project normalization differs from graph contract")
            certified_seats = tuple(Seat.parse(seat) for seat in certificate["seats"])
            project_certificate_hash = project_normalization.normalization_receipt.sha256
        if project_prefix is not None:
            if (project_domain is None or len(start.cards) != 32
                    or not 0 <= self.base_length < 4
                    or type(effective_play_step_count) is not int
                    or not 4 <= effective_play_step_count <= len(start.cards)
                    or (effective_play_step_count > 4 and (
                        (boundary_evaluator != "current_raw_point_margin_v1"
                         and not (effective_play_step_count == len(start.cards) and boundary_evaluator is None))
                        or project_stop != 4
                        or project_continuation_actions is None))
                    or len(project_prefix.play_actions) != self.base_length
                    or len(project_prefix.project_actions) != self.base_length + 1
                    or type(project_stop) is not int or not self.base_length < project_stop <= 4):
                raise ValueError("project checkpoints require a first-trick play model")
            dealer = Seat.parse(project_prefix.dealer)
            authored_plays = tuple(TailPrefixActionV1.build(row.get("actor"), row["card"],
                ekka=row.get("ekka") is True) for row in project_prefix.play_actions)
            if authored_plays != start.base_prefix:
                raise ValueError("project checkpoints changed the authored play prefix")
            order = project_domain.declaration_seats(dealer)
            owners = tuple(start.ownership[Card(card)] for card in range(32))
            fragment = attach_project_lifecycle_fragment_v1(model,
                owner_of_card=self.build.initial_owner_variables, fixed_card_owners=owners,
                contract=start.contract, dealer=dealer, trump=start.trump,
                domain=project_domain, name="path_projects", compact_fixed_states=True,
                second_trick_leader=(self.build._extract.winner_seats[0]
                    if project_domain.tie_break_order == "second_trick_reveal" else None))
            self.project_lifecycle = fragment
            pending = order[self.base_length + 1:project_stop]
            if project_continuation_actions is not None:
                if tuple(action.actor for action in project_continuation_actions) != tuple(pending):
                    raise ValueError("project continuation actions disagree with pending actor order")
                for action in project_continuation_actions:
                    model.add_bool_and(_native_project_action_literals(fragment, action))
            elif any(list(project.declared.proto.domain) != [0, 0]
                   for project in fragment.projects if project.owner in pending
                   and project.owner not in certified_seats):
                raise ValueError("project checkpoint sequence includes optional declarations")
            for step in range(self.base_length + 1, max(project_stop, effective_play_step_count)):
                if cancel_check is not None:
                    if cancel_check():
                        raise PlayTailSolveStopped("cancelled")
                raw = project_prefix.to_dict()
                raw.pop("sha256")
                raw["projectActions"].extend(
                    [action.to_dict() for action in project_continuation_actions[:min(step, 4) - len(raw["projectActions"])]]
                    if project_continuation_actions is not None else
                    [{"actor": seat.name.lower(), "action": "pass"}
                     for seat in order[len(raw["projectActions"]):step]])
                self.project_checkpoints[step] = _native_project_prefix_projection(
                    build=self.build, spec=ExactTracePrefixV1.build(raw), fragment=fragment,
                    dealer=dealer, contract=start.contract, trump=start.trump, domain=project_domain,
                    checkpoint_id=f"path_project_{step}", play_prefix_length=step)
        card_index = {
            card.id: index for index, card in enumerate(sorted(start.cards, key=lambda card: card.id))
        }
        ekka_leads = self.build._extract.ekka_leads
        self.binary_ekka_steps = frozenset(trick * len(Seat)
            for trick, flag in enumerate(ekka_leads) if tuple(flag.proto.domain) == (0, 1))
        action_literals: dict[tuple[int, int, bool], Any] = {}
        for step, play_index in enumerate(self.build.selected_play_indexes):
            if step < self.base_length:
                continue
            for card_id, index in card_index.items():
                card_is = model.new_bool_var(f"path_step_{step}_card_{card_id}")
                model.add(play_index == index).only_enforce_if(card_is)
                model.add(play_index != index).only_enforce_if(card_is.Not())
                if step % 4:
                    action_literals[step, card_id, False] = card_is
                    continue
                ekka = ekka_leads[step // 4]
                normal = model.new_bool_var(f"path_step_{step}_card_{card_id}_normal")
                model.add_bool_and([card_is, ekka.Not()]).only_enforce_if(normal)
                model.add_bool_or([card_is.Not(), ekka]).only_enforce_if(normal.Not())
                declared = model.new_bool_var(f"path_step_{step}_card_{card_id}_ekka")
                model.add_bool_and([card_is, ekka]).only_enforce_if(declared)
                model.add_bool_or([card_is.Not(), ekka.Not()]).only_enforce_if(
                    declared.Not()
                )
                action_literals[step, card_id, False] = normal
                action_literals[step, card_id, True] = declared
        self.action_literals = action_literals
        self.boundary_utilities: tuple[cp_model.IntVar, ...] = ()
        if boundary_evaluator is not None:
            if boundary_evaluator != "current_raw_point_margin_v1":
                raise ValueError("native path boundary evaluator is unsupported")
            maximum = scoring_domain.raw_card_totals[start.contract.effective.value]
            margins = [model.new_int_var(-maximum, maximum, "boundary_raw_margin_0")]
            model.add(margins[0] == start.base_raw[int(target)] - start.base_raw[int(target.other)])
            for index, (winner, points) in enumerate(zip(
                self.build.winner_team_variables, self.build.trick_point_variables, strict=True,
            )):
                target_won = model.new_bool_var(f"boundary_target_won_{index}")
                model.add(winner == int(target)).only_enforce_if(target_won)
                model.add(winner != int(target)).only_enforce_if(target_won.Not())
                margin = model.new_int_var(-maximum, maximum, f"boundary_raw_margin_{index + 1}")
                model.add(margin == margins[-1] + points).only_enforce_if(target_won)
                model.add(margin == margins[-1] - points).only_enforce_if(target_won.Not())
                margins.append(margin)
            self.boundary_utilities = tuple(margins)
        self.scoring_fragment: OrToolsScoringFragmentV1 | None = None
        self.terminal_utility_variable: cp_model.IntVar | None = None
        self.terminal_project_fragment = None
        self.public_baloot_binding = None
        if start.terminal_project_domain is not None:
            self.terminal_project_fragment = self._attach_terminal_projects(model)
            historic = tuple(SimpleNamespace(actor=model.new_constant(int(action.actor)),
                card=model.new_constant(action.card.id)) for action in start.completed_play_history)
            slots = (*historic, *self.build.play_state_context.play_slots[len(historic):])
            self.public_baloot_binding = _native_baloot_prefix_projection(
                model=model, fragment=self.terminal_project_fragment, play_slot=slots.__getitem__,
                prefix_length=len(slots), contract=start.contract, trump=start.trump,
                domain=start.terminal_project_domain)
        if start.project_disclosure_evidence:
            if not play_stage.native_conversion or project_domain is None:
                raise ValueError("project disclosure requires its native domain")
            projects, literals = [], []
            for row in start.project_disclosure_evidence:
                project = Project(Seat.parse(row["owner"]), row["kind"],
                    tuple(Card(card) for card in row["cards"]), row["value"], row["highRank"],
                    row["categoryPriority"], row["declared"],
                    row["declarationOrder"] if row["declared"] else 999, row["awarded"])
                projects.append(project)
                literals.append(SimpleNamespace(project_id=row["projectId"], owner=project.owner,
                    eligible=model.new_constant(1), declared=model.new_constant(int(row["declared"])),
                    declaration_order=model.new_constant(row["declarationOrder"] or 0),
                    comparison_winner=model.new_constant(int(row["comparisonWinner"])),
                    awarded=model.new_constant(int(row["awarded"]))))
            self.project_disclosure_projects = tuple(projects)
            fragment = SimpleNamespace(projects=tuple(literals))
            history_length = len(start.completed_play_history)
            # Tail slots keep absolute round indexes; completed tricks occupy their earlier slots.
            slots = self.build.play_state_context.play_slots
            last = min(len(self.build.selected_play_indexes), 8 - history_length)
            for length in range(self.base_length, last + 1):
                prefix_length = history_length + length
                def owner_seen(owner, begin, end, label):
                    matches = []
                    for step in range(begin, min(end, prefix_length)):
                        match = model.new_bool_var(f"{label}_{step}")
                        model.add(slots[step].actor == int(owner)).only_enforce_if(match)
                        model.add(slots[step].actor != int(owner)).only_enforce_if(match.Not())
                        matches.append(match)
                    seen = model.new_bool_var(label)
                    model.add_max_equality(seen, matches or [0])
                    return seen
                visibility = _native_project_visibility_projection(model=model, fragment=fragment,
                    seen_first={seat: model.new_constant(1) for seat in Seat}, owner_seen=owner_seen,
                    prefix_length=prefix_length, declaration_seats=tuple(Seat),
                    name=f"path_project_disclosure_{length}", reveal_scope=project_domain.reveal_scope)
                self.project_disclosure_bindings[length] = tuple(
                    visibility[5][row["projectId"]] for row in start.project_disclosure_evidence)
        if (
            not defer_terminal_scoring
            and (
                effective_play_step_count is None
                or effective_play_step_count == len(start.cards)
            )
        ):
            self.scoring_fragment, self.terminal_utility_variable = (
                self._attach_terminal_scoring(
                    scoring_domain=scoring_domain,
                    target=target,
                    utility=utility,
                    utility_encoding=utility_encoding,
                )
            )
        # The request hash content-addresses this immutable prefix model.  Keep
        # one serialization for both nested identities; model_hash previously
        # serialized the same multi-megabyte proto twice.
        self.project_counterfactual_utility_variable = None
        if project_counterfactual:
            if self.scoring_fragment is None:
                raise ValueError("project counterfactual requires a complete play model")
            counterfactual = attach_project_counterfactual_scoring_v1(self.scoring_fragment, scoring_domain)
            maximum = max(tuple(variable.proto.domain)[-1] for variable in counterfactual.awarded_points)
            self.project_counterfactual_utility_variable = model.new_int_var(
                -maximum, maximum, "project_counterfactual_terminal_margin")
            model.add(self.project_counterfactual_utility_variable ==
                      counterfactual.awarded_points[int(target)] - counterfactual.awarded_points[int(target.other)])
        base_model_proto = str(model.proto)
        play_model_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "requestHash": self.build.request_hash,
                "cpModelProtoHash": hashlib.sha256(
                    base_model_proto.encode("utf-8")
                ).hexdigest(),
            }
        )
        self.model_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "playModelHash": play_model_hash,
                "scoringDomainHash": scoring_domain.sha256,
                "targetTeam": target.name.lower(),
                "utility": utility,
                "cpModelProto": base_model_proto,
                **({"utilityEncoding": utility_encoding} if utility_encoding is not None else {}),
                **({"projectCertificateHash": project_certificate_hash} if project_certificate_hash is not None else {}),
                **({"projectContinuationActions": [action.to_dict() for action in project_continuation_actions]}
                   if project_continuation_actions is not None else {}),
            }
        )
        self._base_model_hash = self.model_hash
        self.results: dict[
            tuple[TailPrefixActionV1, ...], _DerivedPlayState
        ] | None = None
        self._enumerated_witnesses: dict[
            tuple[TailPrefixActionV1, ...], OrToolsPlayTailResultV1
        ] | None = None
        self._solved_witnesses: dict[int, tuple[OrToolsPlayTailResultV1, cp_model.CpSolver]] = {}

    def _attach_terminal_projects(self, model):
        start = self.start
        fragment = attach_project_lifecycle_fragment_v1(model,
            owner_of_card=tuple(
                model.new_int_var(owner, owner, f"proof_terminal_owner_{card}")
                for card, owner in enumerate(start.terminal_project_owners)
            ),
            fixed_card_owners=start.terminal_project_owners,
            contract=start.contract, dealer=start.terminal_project_dealer,
            trump=start.trump, domain=start.terminal_project_domain, name="proof_terminal_projects",
            second_trick_leader=(self.build._extract.winner_seats[0]
                if start.terminal_project_domain.tie_break_order == "second_trick_reveal"
                and not sum(start.base_tricks) else start.terminal_project_second_trick_leader))
        declarations = dict(start.terminal_project_declarations)
        if not declarations.keys() <= {project.project_id for project in fragment.projects}:
            raise ValueError("terminal declaration references an unknown project")
        for project in fragment.projects:
            model.add(project.declared == int(project.project_id in declarations))
            if project.project_id in declarations:
                model.add(project.declaration_order == declarations[project.project_id])
        return fragment

    def _read_public_baloot_events(self, solver, prefix_length):
        if self.public_baloot_binding is None:
            return None
        auto, declared, _ = self.public_baloot_binding
        history = self.start.completed_play_history
        events = []
        for step in range(len(history) + prefix_length):
            kind = "included" if solver.value(auto[step]) else "declared" if solver.value(declared[step]) else None
            if kind is not None:
                actor = history[step].actor if step < len(history) else Seat(
                    solver.value(self.build._extract.actors[step - len(history)]))
                events.append((step, actor, kind))
        return tuple(events)

    def _read_project_disclosures(self, solver, prefix_length):
        bindings = getattr(self, "project_disclosure_bindings", {})
        if not bindings:
            return None
        return tuple(replace(project, revealed=bool(solver.value(revealed)))
            for project, revealed in zip(self.project_disclosure_projects,
                bindings[min(prefix_length, max(bindings))], strict=True))

    def _attach_terminal_scoring(
        self,
        *,
        scoring_domain: ScoringDomainIRV1,
        target: Team,
        utility: str,
        utility_encoding: str | None = None,
        model: cp_model.CpModel | None = None,
        winner_team_variables: tuple[cp_model.IntVar, ...] | None = None,
        trick_point_variables: tuple[cp_model.IntVar, ...] | None = None,
        play_request_hash: str | None = None,
    ) -> tuple[OrToolsScoringFragmentV1, cp_model.IntVar]:
        model = self.build.model if model is None else model
        start = self.start
        winners = (
            self.build.winner_team_variables
            if winner_team_variables is None
            else winner_team_variables
        )
        trick_points = (
            self.build.trick_point_variables
            if trick_point_variables is None
            else trick_point_variables
        )
        if len(winners) != len(trick_points):
            raise ValueError("terminal scoring winner and point counts disagree")
        if len(winners) + sum(start.base_tricks) != 8:
            raise ValueError("terminal scoring requires exactly eight trick winners")

        def fixed(name: str, value: int, upper: int) -> cp_model.IntVar:
            variable = model.new_int_var(0, max(value, upper), name)
            model.add(variable == value)
            return variable

        def equals(variable: cp_model.IntVar, value: int, name: str) -> cp_model.IntVar:
            literal = model.new_bool_var(name)
            model.add(variable == value).only_enforce_if(literal)
            model.add(variable != value).only_enforce_if(literal.Not())
            return literal

        terminal_baloot = tuple(
            fixed(f"proof_score_baloot_{team}", value, value)
            for team, value in enumerate(start.baloot_points)
        )
        if start.terminal_project_domain is not None:
            fragment = (self.terminal_project_fragment if model is self.build.model
                        else self._attach_terminal_projects(model))
            terminal_baloot = tuple(
                model.new_int_var(0, fragment.baloot_value, f"proof_terminal_baloot_{team}")
                for team in Team
            )
            for team in Team:
                model.add(terminal_baloot[int(team)] == fragment.baloot_value * fragment.baloot_team[int(team)])
        if self.project_lifecycle is not None:
            if model is not self.build.model:
                raise ValueError("interleaved terminal scoring requires its owning play model")
            terminal_baloot = tuple(model.new_int_var(0, self.project_lifecycle.baloot_value,
                f"proof_interleaved_baloot_{int(team)}") for team in Team)
            for team in Team:
                model.add(terminal_baloot[int(team)] == self.project_lifecycle.baloot_value
                    * self.project_lifecycle.baloot_team[int(team)])

        raw_maximum = scoring_domain.raw_card_totals[start.contract.effective.value]
        raw_variables = []
        for team in Team:
            contributions = []
            for trick, (winner, points) in enumerate(
                zip(
                    winners,
                    trick_points,
                    strict=True,
                )
            ):
                won = equals(
                    winner,
                    int(team),
                    f"proof_terminal_trick_{trick}_team_{int(team)}",
                )
                contribution = model.new_int_var(
                    0, 128, f"proof_terminal_raw_{trick}_{int(team)}"
                )
                model.add(contribution == points).only_enforce_if(won)
                model.add(contribution == 0).only_enforce_if(won.Not())
                contributions.append(contribution)
            last_won = equals(
                winners[-1],
                int(team),
                f"proof_terminal_last_winner_{int(team)}",
            )
            raw = model.new_int_var(
                0, raw_maximum, f"proof_terminal_raw_team_{int(team)}"
            )
            model.add(
                raw
                == start.base_raw[int(team)]
                + sum(contributions)
                + scoring_domain.last_trick_bonus_raw * last_won
            )
            raw_variables.append(raw)

        base_winners = tuple(
            fixed(f"proof_base_winner_{index}", int(team), 1)
            for team in Team
            for index in range(start.base_tricks[int(team)])
        )
        source_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "stage": "complete_state_terminal_scoring",
                "exactStateId": start.exact_state_id,
                "proofStateProjectionHash": start.proof_state_projection_hash,
                "playRequestHash": play_request_hash or self.build.request_hash,
                "scoringDomainHash": scoring_domain.sha256,
                "targetTeam": target.name.lower(),
                "utility": utility,
                **({"utilityEncoding": utility_encoding} if utility_encoding is not None else {}),
            }
        )
        fragment = attach_scoring_fragment_v1(
            model,
            scoring_domain,
            CpSatScoringInputsV1(
                contract=fixed(
                    "proof_score_contract",
                    0 if start.contract.effective == Contract.SUN else 1,
                    1,
                ),
                buyer_seat=fixed("proof_score_buyer", int(start.buyer), 3),
                escalation=fixed(
                    "proof_score_escalation",
                    {
                        "normal": 0,
                        "double": 1,
                        "triple": 2,
                        "four": 3,
                        "gahwa": 4,
                    }[start.escalation],
                    4,
                ),
                last_escalating_team=fixed(
                    "proof_score_last_team",
                    2
                    if start.last_escalating_team is None
                    else int(start.last_escalating_team),
                    2,
                ),
                raw_points=(raw_variables[0], raw_variables[1]),
                project_points=self.project_lifecycle.combined_awarded_value if self.project_lifecycle is not None else
                fragment.combined_awarded_value if start.terminal_project_domain is not None else tuple(
                    fixed(f"proof_score_projects_{team}", value, value)
                    for team, value in enumerate(start.project_points)
                ),
                baloot_points=terminal_baloot,
                trick_winner_teams=(
                    *base_winners,
                    *winners,
                ),
                starting_match_score=tuple(
                    fixed(
                        f"proof_score_match_{team}",
                        value,
                        scoring_domain.match_target,
                    )
                    for team, value in enumerate(start.starting_match_score)
                ),
            ),
            source_hash=source_hash,
            native_conversion=self.play_stage.native_conversion,
        )
        if utility == INTERNAL_RAW_POINT_MARGIN_UTILITY_V1:
            result = model.new_int_var(
                -raw_maximum, raw_maximum, "proof_terminal_utility"
            )
            model.add(
                result
                == raw_variables[int(target)]
                - raw_variables[int(target.other)]
            )
        elif utility == "awarded_point_margin":
            maximum = max(
                int(fragment.awarded_points[0].proto.domain[1]),
                int(fragment.awarded_points[1].proto.domain[1]),
            )
            result = model.new_int_var(
                -maximum, maximum, "proof_terminal_utility"
            )
            model.add(
                result
                == fragment.awarded_points[int(target)]
                - fragment.awarded_points[int(target.other)]
            )
        elif utility in {"contract_success", "round_win"}:
            won = equals(
                fragment.round_winner,
                int(target),
                "proof_terminal_target_round_win",
            )
            loss = 0 if utility == "contract_success" and utility_encoding is None else -1
            result = model.new_int_var(loss, 1, "proof_terminal_utility")
            model.add(result == (1 - loss) * won + loss)
        elif utility == "match_win":
            result = model.new_int_var(-1, 1, "proof_terminal_utility")
            model.add_allowed_assignments(
                [fragment.match_winner, result],
                [
                    (int(target), 1),
                    (int(target.other), -1),
                    (2, 0),
                ],
            )
        elif utility == "kabout":
            target_won = equals(
                fragment.round_winner,
                int(target),
                "proof_terminal_target_kabout_winner",
            )
            achieved = model.new_bool_var("proof_terminal_target_kabout")
            model.add_bool_and([fragment.kabout, target_won]).only_enforce_if(
                achieved
            )
            model.add_bool_or(
                [fragment.kabout.Not(), target_won.Not()]
            ).only_enforce_if(achieved.Not())
            result = model.new_int_var(-1, 1, "proof_terminal_utility")
            model.add(result == 2 * achieved - 1)
        else:
            raise ValueError("proof utility is unsupported by graph compilation")
        return fragment, result

    def solve_policy_path(
        self,
        policies: Mapping[Seat, Any],
        *,
        maximum_seconds: float,
        cancelled: Callable[[], bool] | None,
        _terminal_required: bool = True,
    ) -> CpSatPolicyPathResultV1 | OrToolsPlayTailResultV1:
        """Constrain one seeded policy action at every CP-SAT play decision."""

        if set(policies) != set(Seat):
            raise ValueError("policy play solve requires every seat policy")
        self._solved_witnesses.clear()
        model = self.build.model
        extract = self.build._extract
        pool = extract.pool
        play_indexes = self.build.selected_play_indexes

        def equals(variable: Any, value: int, name: str) -> Any:
            result = model.new_bool_var(name)
            model.add(variable == value).only_enforce_if(result)
            model.add(variable != value).only_enforce_if(result.Not())
            return result

        def conjunction(literals: list[Any], name: str) -> Any:
            if not literals:
                return model.new_constant(1)
            result = model.new_bool_var(name)
            model.add_bool_and(literals).only_enforce_if(result)
            model.add_bool_or([item.Not() for item in literals]).only_enforce_if(
                result.Not()
            )
            return result

        played_before: dict[tuple[int, int], Any] = {}
        step_assumptions: dict[int, Any] = {}

        def attach_policy_step(step: int) -> None:
            step_assumption = model.new_bool_var(f"assume_policy_step_{step}")
            model.add_assumption(step_assumption)
            step_assumptions[step] = step_assumption
            availability: dict[tuple[int, bool], Any] = {}
            for index, card in enumerate(pool):
                normal = extract.legal_flags_by_step[step][index]
                availability[card.id, False] = normal
                if (
                    step % 4
                    or not self.play_domain.ekka_enabled
                    or self.start.contract.effective != Contract.HOKUM
                    or card.suit == self.start.trump
                ):
                    continue
                higher = [
                    other_index
                    for other_index, other in enumerate(pool)
                    if other.suit == card.suit
                    and self.play_domain.sun_strength[int(other.rank)]
                    > self.play_domain.sun_strength[int(card.rank)]
                ]
                prior = []
                for other_index in higher:
                    key = (step, other_index)
                    if key not in played_before:
                        flags = [
                            equals(
                                play_indexes[prior_step],
                                other_index,
                                f"policy_step_{step}_higher_{other_index}_prior_{prior_step}",
                            )
                            for prior_step in range(step)
                        ]
                        seen = model.new_bool_var(
                            f"policy_step_{step}_higher_{other_index}_played"
                        )
                        if flags:
                            model.add_max_equality(seen, flags)
                        else:
                            model.add(seen == 0)
                        played_before[key] = seen
                    prior.append(played_before[key])
                eligible = conjunction(
                    prior,
                    f"policy_step_{step}_card_{card.id}_ekka_eligible",
                )
                availability[card.id, True] = conjunction(
                    [normal, eligible],
                    f"policy_step_{step}_card_{card.id}_ekka_available",
                )

            for seat in Seat:
                actor_is = equals(
                    extract.actors[step],
                    int(seat),
                    f"policy_step_{step}_actor_{int(seat)}",
                )
                policy = policies[seat]
                ordered = sorted(
                    availability,
                    key=lambda item: (
                        canonical_sha256(
                            {
                                "schemaVersion": 1,
                                "policyId": policy.policy_id,
                                "version": policy.version,
                                "seed": policy.seed,
                                "actionId": item[0] + 32 * int(item[1]),
                                "actionCode": Card(item[0]).code
                                + ("+ekka" if item[1] else ""),
                            }
                        ),
                        item[0] + 32 * int(item[1]),
                        Card(item[0]).code + ("+ekka" if item[1] else ""),
                    ),
                )
                selected_rank = model.new_int_var(
                    0,
                    len(ordered) - 1,
                    f"policy_step_{step}_actor_{int(seat)}_selected_rank",
                )
                model.add(
                    selected_rank
                    == sum(
                        index
                        * self.action_literals[step, action[0], action[1]]
                        for index, action in enumerate(ordered)
                    )
                )
                available_ranks = []
                for index, action in enumerate(ordered):
                    rank = model.new_int_var(
                        0,
                        len(ordered),
                        f"policy_step_{step}_actor_{int(seat)}_rank_{index}",
                    )
                    model.add(rank == index).only_enforce_if(
                        availability[action]
                    )
                    model.add(rank == len(ordered)).only_enforce_if(
                        availability[action].Not()
                    )
                    available_ranks.append(rank)
                minimum_available_rank = model.new_int_var(
                    0,
                    len(ordered),
                    f"policy_step_{step}_actor_{int(seat)}_minimum_rank",
                )
                model.add_min_equality(
                    minimum_available_rank,
                    available_ranks,
                )
                model.add(selected_rank == minimum_available_rank).only_enforce_if(
                    [step_assumption, actor_is]
                )

        started = time.monotonic()
        policy_path_model_hashes: list[str] = []

        def current_model_hash() -> str:
            return canonical_sha256(
                {
                    "schemaVersion": 1,
                    "baseModelHash": self._base_model_hash,
                    "policyIds": [
                        {
                            "seat": seat.name.lower(),
                            "policyId": policies[seat].policy_id,
                            "version": policies[seat].version,
                            "seed": policies[seat].seed,
                        }
                        for seat in Seat
                    ],
                    "policyPathModelHashes": policy_path_model_hashes,
                    "cpModelProto": str(model.proto),
                }
            )

        def solve_current(seconds: float) -> tuple[cp_model.CpSolver, int]:
            active = cp_model.CpSolver()
            active.parameters.num_search_workers = 1
            active.parameters.random_seed = 0
            active.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
            active.parameters.max_time_in_seconds = max(0.001, seconds)
            return active, _solve_cp_sat_with_cancel_v1(active, model, cancelled)

        def require_solution(
            active: cp_model.CpSolver,
            status: int,
            *,
            phase: str,
        ) -> None:
            if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
                return
            core = (
                set(sound_infeasible_assumption_indexes(
                    active, model.proto.assumptions
                ))
                if status == cp_model.INFEASIBLE
                else set()
            )
            failed_steps = [
                step
                for step, literal in step_assumptions.items()
                if literal.index in core
            ]
            raise PlayTailSolveStopped(
                active.status_name(status).lower(),
                evidence={
                    "policyStepCore": failed_steps,
                    "policyPhase": phase,
                },
            )

        if self.scoring_fragment is None or self.terminal_utility_variable is None:
            if not _terminal_required:
                for step in range(self.base_length, len(play_indexes)):
                    attach_policy_step(step)
                remaining = maximum_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise PlayTailSolveStopped(
                        "maximum_seconds",
                        evidence={
                            "policyStepCore": [],
                            "policyPhase": "policy_path",
                        },
                    )
                self.model_hash = current_model_hash()
                solver, status = solve_current(remaining)
                require_solution(solver, status, phase="policy_path")
                return self.build._extract(solver, status, self.model_hash)

            selected_path: tuple[TailPrefixActionV1, ...] = ()
            terminal_play: OrToolsPlayTailResultV1 | None = None
            first_trick_end = ((self.base_length // 4) + 1) * 4
            terminal_play_count = (
                len(self.start.cards)
                if self._incremental_policy_batches
                else len(play_indexes)
            )
            for batch_end in range(first_trick_end, terminal_play_count + 1, 4):
                remaining = maximum_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise PlayTailSolveStopped(
                        "maximum_seconds",
                        evidence={
                            "policyStepCore": [],
                            "policyPhase": "policy_path",
                        },
                    )
                if self._incremental_policy_batches and batch_end == first_trick_end:
                    partial_start = self.start
                    partial = self
                else:
                    partial_start = replace(
                        self.start,
                        base_prefix=(
                            *self.start.base_prefix,
                            *selected_path,
                        ),
                    )
                    partial = _CpSatPathOracle(
                        partial_start,
                        play_stage=self.play_stage,
                        play_domain=self.play_domain,
                        scoring_domain=self.scoring_domain,
                        structural_domain=self.structural_domain,
                        target=self.target,
                        utility=self.utility,
                        utility_encoding=self.utility_encoding,
                        play_step_count=batch_end,
                        defer_terminal_scoring=True,
                    )
                try:
                    partial_result = partial.solve_policy_path(
                        policies,
                        maximum_seconds=remaining,
                        cancelled=cancelled,
                        _terminal_required=False,
                    )
                except PlayTailSolveStopped as exc:
                    evidence = dict(exc.partial_receipt.get("evidence") or {})
                    evidence.update(
                        {
                            "policyBatchEnd": batch_end,
                            "policyPrefix": [
                                action.to_dict()
                                for action in partial_start.base_prefix
                            ],
                        }
                    )
                    raise PlayTailSolveStopped(
                        exc.solver_status,
                        evidence=evidence,
                    ) from exc
                if not isinstance(partial_result, OrToolsPlayTailResultV1):
                    raise TypeError("policy prefix solve returned terminal evidence")
                selected_path = tuple(
                    partial_result.plays[self.base_length : batch_end]
                )
                terminal_play = partial_result
                policy_path_model_hashes.append(partial.model_hash)

            if terminal_play is None:
                raise ValueError("policy path did not produce a terminal play result")
            scoring_model = cp_model.CpModel()

            def fixed_score_input(name: str, value: int, upper: int) -> cp_model.IntVar:
                variable = scoring_model.new_int_var(0, max(value, upper), name)
                scoring_model.add(variable == value)
                return variable

            winner_teams = []
            for trick, winner in enumerate(terminal_play.trick_winners):
                winner_seat = fixed_score_input(
                    f"policy_score_trick_{trick}_winner_seat",
                    int(winner),
                    3,
                )
                winner_team = scoring_model.new_int_var(
                    0,
                    1,
                    f"policy_score_trick_{trick}_winner_team",
                )
                scoring_model.add_allowed_assignments(
                    [winner_seat, winner_team],
                    [(int(seat), int(seat.team)) for seat in Seat],
                )
                winner_teams.append(winner_team)
            point_variables = tuple(
                fixed_score_input(
                    f"policy_score_trick_{trick}_raw",
                    points,
                    128,
                )
                for trick, points in enumerate(terminal_play.trick_raw_points)
            )
            terminal_play_binding_hash = canonical_sha256(
                {
                    "schemaVersion": 1,
                    "stage": "policy_terminal_play_binding",
                    "policyPathModelHashes": policy_path_model_hashes,
                    "playModelHash": terminal_play.model_hash,
                    "playEvidenceHash": terminal_play.evidence_hash,
                    "plays": [action.to_dict() for action in terminal_play.plays],
                    "trickWinners": [
                        winner.name.lower() for winner in terminal_play.trick_winners
                    ],
                    "trickRawPoints": list(terminal_play.trick_raw_points),
                }
            )
            scoring_fragment, terminal_utility_variable = self._attach_terminal_scoring(
                scoring_domain=self.scoring_domain,
                target=self.target,
                utility=self.utility,
                utility_encoding=self.utility_encoding,
                model=scoring_model,
                winner_team_variables=tuple(winner_teams),
                trick_point_variables=point_variables,
                play_request_hash=terminal_play_binding_hash,
            )
            remaining = maximum_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise PlayTailSolveStopped(
                    "maximum_seconds",
                    evidence={
                        "policyStepCore": [],
                        "policyPhase": "terminal_scoring",
                    },
                )
            score_solver = cp_model.CpSolver()
            score_solver.parameters.num_search_workers = 1
            score_solver.parameters.random_seed = 0
            score_solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
            score_solver.parameters.max_time_in_seconds = max(0.001, remaining)
            score_status = _solve_cp_sat_with_cancel_v1(
                score_solver,
                scoring_model,
                cancelled,
            )
            require_solution(score_solver, score_status, phase="terminal_scoring")
            self.model_hash = canonical_sha256(
                {
                    "schemaVersion": 1,
                    "baseModelHash": self._base_model_hash,
                    "policyPathModelHashes": policy_path_model_hashes,
                    "terminalPlayBindingHash": terminal_play_binding_hash,
                    "scoringDomainHash": self.scoring_domain.sha256,
                    "targetTeam": self.target.name.lower(),
                    "utility": self.utility,
                    "scoringCpModelProto": str(scoring_model.proto),
                }
            )
            return CpSatPolicyPathResultV1(
                terminal_play.at_prefix(self.base_length),
                int(score_solver.value(terminal_utility_variable)),
                extract_scoring_evidence_v1(
                    score_solver,
                    scoring_fragment,
                    self.scoring_domain,
                    contract=self.start.contract,
                ),
                self.model_hash,
            )
        else:
            for step in range(self.base_length, len(play_indexes)):
                attach_policy_step(step)
            self.model_hash = current_model_hash()
            solver, status = solve_current(maximum_seconds)
            require_solution(solver, status, phase="policy_path_with_scoring")
        if self.scoring_fragment is None or self.terminal_utility_variable is None:
            raise ValueError("policy path requires native terminal scoring")
        return CpSatPolicyPathResultV1(
            self.build._extract(solver, status, self.model_hash),
            int(solver.value(self.terminal_utility_variable)),
            extract_scoring_evidence_v1(
                solver,
                self.scoring_fragment,
                self.scoring_domain,
                contract=self.start.contract,
            ),
            self.model_hash,
        )

    def _project(
        self,
        result: Any,
        prefix_length: int,
        terminal_utility: int | None = None,
        boundary_utility: int | None = None,
        project_checkpoint: _NativeProjectPrefixProjectionV1 | None = None,
        public_baloot_events: tuple[tuple[int, Seat, str], ...] | None = None,
        project_counterfactual_utility: int | None = None,
        project_disclosures: tuple[Project, ...] | None = None,
    ) -> _DerivedPlayState:
        completed_tricks = prefix_length // 4
        hands, trick, actor, _, actions = result._prefix_fields(prefix_length)
        return _DerivedPlayState(
            hands,
            trick,
            actor,
            actions,
            result.trick_winners[:completed_tricks],
            result.trick_raw_points[:completed_tricks],
            terminal_utility
            if prefix_length == len(self.start.cards)
            else None,
            boundary_utility,
            project_checkpoint,
            result if project_checkpoint is not None else None,
            public_baloot_events,
            project_counterfactual_utility if prefix_length == len(self.start.cards) else None,
            project_disclosures,
        )

    def enumerate_paths(
        self,
        *,
        maximum_nodes: int,
        maximum_solutions: int | None = None,
        maximum_seconds: float,
        cancelled: Callable[[], bool] | None,
        root_action: str | None = None,
        maximum_conflicts: int | None = None,
    ) -> str | None:
        if maximum_solutions is not None and (
            type(maximum_solutions) is not int or maximum_solutions < 1
        ):
            raise ValueError("maximum path solutions must be a positive integer")
        self.results = None
        self._enumerated_witnesses = None
        deadline = time.monotonic() + maximum_seconds

        def stop_reason() -> str | None:
            if cancelled is not None and cancelled():
                return "cancelled"
            if time.monotonic() >= deadline:
                return "maximum_seconds"
            return None

        if reason := stop_reason():
            return reason
        build = self.build
        model = build.model.clone()
        model.clear_assumptions()
        if root_action is not None:
            action = next((item for item in self.start.expected_legal_actions
                           if item.card.code + ("+ekka" if item.ekka else "") == root_action), None)
            if action is None:
                raise ValueError("enumerated root action is not legal")
            literal = self.action_literals.get((self.base_length, action.card.id, action.ekka))
            if literal is None:
                raise ValueError("enumerated root action is unavailable")
            model.add_assumption(model.get_bool_var_from_proto_index(literal.index))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.enumerate_all_solutions = True
        if maximum_conflicts is not None:
            solver.parameters.max_number_of_conflicts = maximum_conflicts
        full_results: dict[
            tuple[TailPrefixActionV1, ...], tuple[Any, int | None, tuple[int, ...], Any]
        ] = {}
        prefix_keys: set[tuple[TailPrefixActionV1, ...]] = {()}
        prefix_ceiling = maximum_nodes * max(1, len(self.start.cards))
        stopped_reason: str | None = None
        counterfactual_values: dict[tuple[TailPrefixActionV1, ...], int] = {}
        checkpoints: dict[
            tuple[TailPrefixActionV1, ...], _NativeProjectPrefixProjectionV1
        ] = {}
        disclosures = {}

        class Collector(cp_model.CpSolverSolutionCallback):
            # The callback owns one live response; avoid rechecking it per scalar.
            value = cp_model.CpSolverSolutionCallback.Value
            boolean_value = cp_model.CpSolverSolutionCallback.BooleanValue

            def status_name(self, _status: int) -> str:
                return "feasible"

            def on_solution_callback(inner) -> None:
                nonlocal stopped_reason
                if not inner.has_response():
                    raise RuntimeError("native path callback lacks a solution response")
                if reason := stop_reason():
                    stopped_reason = reason
                    inner.stop_search()
                    return
                full = build._extract(inner, cp_model.FEASIBLE, self.model_hash)
                path = tuple(full.plays[self.base_length :])
                for length, binding in self.project_checkpoints.items():
                    key = path[:length - self.base_length]
                    _, checkpoint = _extract_native_project_prefix_projection(
                        binding=binding, solver=inner, model_hash=self.model_hash,
                        prefix_length=length, uniqueness_model_hash=None,
                        uniqueness_solver_status="not_requested")
                    prior = checkpoints.setdefault(key, checkpoint)
                    if prior != checkpoint:
                        raise ValueError("native project model produced inconsistent prefix evidence")
                for length in getattr(self, "project_disclosure_bindings", {}):
                    snapshot = self._read_project_disclosures(inner, length)
                    if disclosures.setdefault(length, snapshot) != snapshot:
                        raise ValueError("native project disclosure produced inconsistent prefix evidence")
                utility = (
                    int(inner.value(self.terminal_utility_variable))
                    if self.terminal_utility_variable is not None
                    else None
                )
                boundary_values = tuple(int(inner.value(variable)) for variable in self.boundary_utilities)
                public_baloot = self._read_public_baloot_events(inner, len(full.plays))
                counter_variable = getattr(self, "project_counterfactual_utility_variable", None)
                if counter_variable is not None:
                    counter_value = int(inner.value(counter_variable))
                    if counterfactual_values.setdefault(path, counter_value) != counter_value:
                        raise ValueError("native project counterfactual produced inconsistent utility")
                if path in full_results:
                    if full_results[path][1:] != (utility, boundary_values, public_baloot):
                        raise ValueError("native play model produced inconsistent utility evidence")
                    return
                full_results[path] = (full, utility, boundary_values, public_baloot)
                if (
                    maximum_solutions is not None
                    and len(full_results) > maximum_solutions
                ):
                    stopped_reason = "maximum_solutions"
                    inner.stop_search()
                    return
                prefix_keys.update(path[:depth] for depth in range(1, len(path) + 1))
                if len(prefix_keys) > prefix_ceiling:
                    stopped_reason = "maximum_nodes"
                    inner.stop_search()

        if reason := stop_reason():
            return reason
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "maximum_seconds"
        solver.parameters.max_time_in_seconds = remaining
        status = _solve_cp_sat_with_cancel_v1(
            solver,
            model,
            cancelled,
            Collector(),
        )
        if stopped_reason is not None:
            return stopped_reason
        if reason := stop_reason():
            return reason
        if status != cp_model.OPTIMAL:
            return (
                "cancelled"
                if cancelled is not None and cancelled()
                else "maximum_seconds"
                if status in {cp_model.FEASIBLE, cp_model.UNKNOWN}
                else solver.status_name(status).lower()
            )
        if not full_results:
            return "infeasible"
        results: dict[tuple[TailPrefixActionV1, ...], _DerivedPlayState] = {}
        for path, (full, utility, boundary_values, public_baloot) in full_results.items():
            for depth in range(len(path) + 1):
                if reason := stop_reason():
                    return reason
                prefix = path[:depth]
                projected = (full.at_prefix(self.base_length + depth)
                             if checkpoints.get(prefix) is not None else full)
                derived = self._project(
                    projected,
                    self.base_length + depth,
                    utility if depth == len(path) else None,
                    boundary_values[(self.base_length + depth) // 4] if boundary_values else None,
                    checkpoints.get(prefix),
                    None if public_baloot is None else tuple(event for event in public_baloot
                        if event[0] < len(self.start.completed_play_history) + self.base_length + depth),
                    counterfactual_values.get(path) if depth == len(path) else None,
                    disclosures[min(self.base_length + depth, max(disclosures))] if disclosures else None,
                )
                prior = results.setdefault(prefix, derived)
                if prior != derived:
                    raise ValueError(
                        "native play model produced inconsistent prefix evidence"
                    )
        if reason := stop_reason():
            return reason
        self.results = results
        self._enumerated_witnesses = {
            path: full for path, (full, _, _, _) in full_results.items()
        }
        return None

    def _query_utility(self):
        utility = self.terminal_utility_variable
        if utility is None:
            utility = self.boundary_utilities[-1] if self.boundary_utilities else None
        if utility is None:
            raise ValueError("adverse path requires a native utility variable")
        return utility

    def query_utility_bounds(self) -> tuple[int, int]:
        domain = tuple(self._query_utility().proto.domain)
        return int(domain[0]), int(domain[-1])

    def with_native_play_policy(self, *, team: Team, policy_id: str | None = None, play_policies=None):
        """Bind a registered visible policy on a clone, never the legal universe."""
        from .ortools_play_tail import native_observer_safe_policy_identity_v1

        if (policy_id is None) == (play_policies is None):
            raise ValueError("path policy requires exactly one policy authority")
        identity = (native_observer_safe_policy_identity_v1(policy_id, team)
            if play_policies is None else {"team": team.name.lower(), "playPolicies": play_policies})
        owned = copy.copy(self)
        owned.build = replace(self.build, model=self.build.model.clone())
        owned.results = owned._enumerated_witnesses = None
        owned._solved_witnesses = {}
        if play_policies is None:
            owned.build.constrain_native_observer_safe_policy(team=team, policy_id=policy_id,
                start_step=self.base_length)
        else:
            owned.build.constrain_declarative_play_policies(team=team, policies=play_policies,
                start_step=self.base_length)
        owned.model_hash = canonical_sha256({"schemaVersion": 1,
            "kind": "native_visible_path_policy_model_v1" if play_policies is None else "native_ranked_path_policy_model_v1",
            "sourceModelHash": self.model_hash,
            "policy": identity, "cpModelProtoHash": hashlib.sha256(
                str(owned.build.model.proto).encode("utf-8")).hexdigest()})
        owned._base_model_hash = owned.model_hash
        return owned

    def find_path_below(
        self,
        threshold: int,
        *,
        path: tuple[TailPrefixActionV1, ...] = (),
        conditional_actions: Mapping[tuple[TailPrefixActionV1, ...], TailPrefixActionV1 | None],
        maximum_seconds: float,
        cancelled: Callable[[], bool] | None,
        maximum_conflicts: int | None = None,
        maximize_utility: bool = False,
        feasibility_only: bool = False,
    ) -> tuple[OrToolsPlayTailResultV1 | None, int | None, str]:
        """Optimize a path below a threshold, not a strategic proof.

        Conditional actions bind concrete prefixes; None excludes that prefix.
        An empty excluded prefix rules out this entire branch. The caller must certify
        observation coverage/nonanticipativity before using this query in proof.
        None means this exact constrained query is infeasible, not the scenario.
        Maximization with a threshold above the utility domain proves an upper
        relaxation bound; it does not assume cooperative player behavior.
        Feasibility-only mode returns any matching witness, not an extremal
        value. Only INFEASIBLE returns None; UNKNOWN remains incomplete.
        """
        if type(threshold) is not int:
            raise TypeError("path threshold must be an integer")
        if type(maximize_utility) is not bool:
            raise TypeError("path objective selection must be boolean")
        if type(feasibility_only) is not bool:
            raise TypeError("path feasibility selection must be boolean")
        if feasibility_only and maximize_utility:
            raise ValueError("feasibility query cannot maximize utility")
        if maximum_conflicts is not None and (type(maximum_conflicts) is not int or maximum_conflicts < 1):
            raise ValueError("path conflict limit must be a positive integer")
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds):
            raise ValueError("path query time limit must be finite")
        if not isinstance(path, tuple) or not isinstance(conditional_actions, Mapping):
            raise TypeError("path query requires a tuple prefix and conditional action mapping")
        if any(not isinstance(prefix, tuple) for prefix in conditional_actions):
            raise TypeError("conditional action prefixes must be tuples")
        deadline = time.monotonic() + maximum_seconds
        model_hash = None
        phase = "model_clone"

        def check_stop() -> None:
            reason = "cancelled" if cancelled is not None and cancelled() else (
                "maximum_seconds" if time.monotonic() >= deadline else None)
            if reason is not None:
                raise PlayTailSolveStopped(reason, evidence=({
                    "queryModelHash": model_hash, "baseModelHash": self.model_hash,
                    "objective": "feasibility" if feasibility_only else "maximum" if maximize_utility else "minimum",
                    "queryPhase": phase} if model_hash is not None else {"queryPhase": phase}))

        check_stop()
        utility = self._query_utility()
        model = self.build.model.clone()
        phase = "query_constraints"
        model.clear_assumptions()
        model.clear_objective()
        literals = {}

        def literal(step: int, action: TailPrefixActionV1):
            if not isinstance(action, TailPrefixActionV1):
                raise TypeError("path queries require typed play actions")
            if self.start.ownership.get(action.card) != action.actor:
                raise ValueError("path action actor disagrees with card ownership")
            value = self.action_literals.get((step, action.card.id, action.ekka))
            if value is None:
                raise ValueError("path query contains an unavailable action")
            variable = literals.get(value.index)
            if variable is None:
                variable = literals[value.index] = model.get_bool_var_from_proto_index(value.index)
            return variable

        model.add_assumptions(literal(self.base_length + offset, action)
                              for offset, action in enumerate(path))
        for prefix, action in sorted(conditional_actions.items(), key=lambda item:
                tuple((int(step.actor), step.card.id, step.ekka) for step in item[0])):
            check_stop()
            guards = [literal(self.base_length + offset, step)
                      for offset, step in enumerate(prefix)]
            model.add_bool_or([guard.Not() for guard in guards] +
                              ([literal(self.base_length + len(prefix), action)] if action is not None else []))
        model.add(model.get_int_var_from_proto_index(utility.index) < threshold)
        if not feasibility_only:
            (model.maximize if maximize_utility else model.minimize)(
                model.get_int_var_from_proto_index(utility.index))
        model_hash = canonical_sha256({
            "kind": "native_adverse_path_feasibility_v1" if feasibility_only else "native_adverse_path_v2",
            "baseModelHash": self.model_hash,
            "threshold": threshold,
            "path": [action.to_dict() for action in path],
            "conditionalActions": [
                {
                    "prefix": [step.to_dict() for step in prefix],
                    "action": action.to_dict() if action is not None else None,
                }
                for prefix, action in sorted(
                    conditional_actions.items(),
                    key=lambda item: tuple(
                        (int(step.actor), step.card.id, step.ekka)
                        for step in item[0]
                    ),
                )
            ],
            "objective": "feasibility_v1" if feasibility_only else "maximize_utility_v1" if maximize_utility else "minimize_utility_v1",
        })
        check_stop()
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_time_in_seconds = deadline - time.monotonic()
        if maximum_conflicts is not None:
            solver.parameters.max_number_of_conflicts = maximum_conflicts
        check_stop()
        phase = "solver"
        status = _solve_cp_sat_with_cancel_v1(solver, model, cancelled)
        check_stop()
        if status == cp_model.INFEASIBLE:
            return None, None, model_hash
        if status != cp_model.OPTIMAL and not (feasibility_only and status == cp_model.FEASIBLE):
            raise PlayTailSolveStopped(solver.status_name(status).lower(), evidence={
                "queryModelHash": model_hash, "baseModelHash": self.model_hash,
                "objective": "feasibility" if feasibility_only else "maximum" if maximize_utility else "minimum",
                "queryPhase": phase})
        phase = "witness_extraction"
        result = self.build._extract(solver, status, model_hash)
        solved_utility = int(solver.value(utility))
        check_stop()
        # A constrained-query witness also satisfies the base model. Reuse its
        # solver assignment for prefix/project reads, preserving query provenance
        # on the returned evidence and base provenance on the cached projection.
        self._solved_witnesses[len(path)] = (
            replace(result, model_hash=self.model_hash, sha256=canonical_sha256({
                "schemaVersion": 1, "modelHash": self.model_hash,
                "evidenceHash": result.evidence_hash, "solverStatus": result.solver_status,
            })), solver)
        check_stop()
        return result, solved_utility, model_hash

    def solve(
        self,
        path: tuple[TailPrefixActionV1, ...],
        *,
        maximum_seconds: float,
        cancelled: Callable[[], bool] | None,
        maximum_conflicts: int | None = None,
    ) -> tuple[_DerivedPlayState, tuple[TailPrefixActionV1, ...]]:
        if cancelled is not None and cancelled():
            raise PlayTailSolveStopped("cancelled")
        if maximum_seconds <= 0:
            raise PlayTailSolveStopped("maximum_seconds")
        prefix = (*self.start.base_prefix, *path)
        if self.results is not None:
            result = self.results.get(path)
            if result is not None:
                return result, prefix
        witness = next((item for item in self._solved_witnesses.values()
                        if item[0].model_hash == self.model_hash
                        and item[0].plays[:len(prefix)] == prefix), None)
        if witness is not None:
            full, solver = witness
        else:
            deadline = time.monotonic() + maximum_seconds
            assumptions = []
            for offset, action in enumerate(path, start=self.base_length):
                literal = self.action_literals.get((offset, action.card.id, action.ekka))
                if literal is None:
                    raise ValueError("graph path contains an unavailable play action")
                assumptions.append(literal)
            model = self.build.model
            if self.play_stage.native_conversion and assumptions:
                # Fixed paths need no unsat core. Let presolve eliminate their
                # history instead of keeping it live as search assumptions.
                model = model.clone()
                model.clear_assumptions()
                model.add_bool_and(model.get_bool_var_from_proto_index(item.index)
                                   for item in assumptions)
            else:
                model.clear_assumptions()
                model.add_assumptions(assumptions)
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = 0
            solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PlayTailSolveStopped("maximum_seconds")
            solver.parameters.max_time_in_seconds = remaining
            if maximum_conflicts is not None:
                solver.parameters.max_number_of_conflicts = maximum_conflicts
            status = _solve_cp_sat_with_cancel_v1(solver, model, cancelled)
            if cancelled is not None and cancelled():
                raise PlayTailSolveStopped("cancelled")
            if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
                raise PlayTailSolveStopped(solver.status_name(status).lower())
            full = self.build._extract(solver, status, self.model_hash)
            # One witness per play depth, bounded by the cards in this model.
            # Keep ancestor continuations available when DFS returns to siblings.
            self._solved_witnesses[len(path)] = full, solver
        result = full.at_prefix(len(prefix))
        if result.plays[: len(prefix)] != prefix:
            raise ValueError("native play transition changed the fixed graph path")
        checkpoint = None
        binding = self.project_checkpoints.get(len(prefix))
        if binding is not None:
            _, checkpoint = _extract_native_project_prefix_projection(
                binding=binding, solver=solver, model_hash=self.model_hash,
                prefix_length=len(prefix), uniqueness_model_hash=None,
                uniqueness_solver_status="not_requested")
        return (
            self._project(
                result,
                len(prefix),
                int(solver.value(self.terminal_utility_variable))
                if self.terminal_utility_variable is not None
                and len(prefix) == len(self.start.cards)
                else None,
                int(solver.value(self.boundary_utilities[len(prefix) // 4]))
                if self.boundary_utilities else None,
                checkpoint,
                self._read_public_baloot_events(solver, len(prefix)),
                int(solver.value(self.project_counterfactual_utility_variable))
                if getattr(self, "project_counterfactual_utility_variable", None) is not None
                and len(prefix) == len(self.start.cards) else None,
                self._read_project_disclosures(solver, len(prefix)),
            ),
            prefix,
        )


class OrToolsCompleteStateGraphCompilerV1:
    """Enumerate a legal complete-state DAG with governed resource bounds."""

    def __init__(
        self,
        *,
        play_domain: PlayDomainIRV1,
        scoring_domain: ScoringDomainIRV1,
        structural_domain: StructuralDomainIRV1,
        project_domain: ProjectDomainIRV1 | None = None,
        native_conversion: bool = False,
    ) -> None:
        if not isinstance(play_domain, PlayDomainIRV1):
            raise TypeError("graph compiler requires PlayDomainIRV1")
        if not isinstance(scoring_domain, ScoringDomainIRV1):
            raise TypeError("graph compiler requires ScoringDomainIRV1")
        if not isinstance(structural_domain, StructuralDomainIRV1):
            raise TypeError("graph compiler requires StructuralDomainIRV1")
        if project_domain is not None and not isinstance(
            project_domain, ProjectDomainIRV1
        ):
            raise TypeError("graph compiler project domain must be ProjectDomainIRV1")
        play_domain.require_integrity(structural_domain)
        structural_domain.require_integrity()
        self.play_domain = play_domain
        self.scoring_domain = scoring_domain
        self.structural_domain = structural_domain
        self.project_domain = project_domain
        self.play_stage = OrToolsPlayTailStageV1(native_conversion=native_conversion)

    def _parse_start(
        self,
        source: OrToolsNormalizedExactStateV1 | CompleteStateGraphSeedV1,
        *,
        expected_rules_hash: str,
        request: ScenarioProofRequestV1 | None = None,
        project_receipt: Mapping[str, Any] | None = None,
    ) -> _StartState:
        if isinstance(source, CompleteStateGraphSeedV1):
            if project_receipt is not None:
                raise ValueError("native project receipt requires an exact state")
            return self._parse_seed_start(
                source,
                expected_rules_hash=expected_rules_hash,
            )
        return self._parse_exact_start(
            source,
            expected_rules_hash=expected_rules_hash,
            request=request,
            project_receipt=project_receipt,
        )

    def proof_state_projection_hash(
        self,
        source: OrToolsNormalizedExactStateV1 | CompleteStateGraphSeedV1,
        *,
        expected_rules_hash: str,
    ) -> str:
        """Validate a graph input and return its canonical proof-state identity."""

        return self._parse_start(
            source,
            expected_rules_hash=expected_rules_hash,
        ).proof_state_projection_hash

    def _parse_exact_start(
        self,
        exact: OrToolsNormalizedExactStateV1,
        *,
        expected_rules_hash: str,
        request: ScenarioProofRequestV1 | None = None,
        project_receipt: Mapping[str, Any] | None = None,
    ) -> _StartState:
        if not isinstance(exact, OrToolsNormalizedExactStateV1):
            raise TypeError(
                "graph compilation requires a normalized exact state or CP-derived seed"
            )
        if exact.rules_hash != expected_rules_hash:
            raise ValueError(
                "exact state rules hash does not match the selected bundle"
            )
        if canonical_sha256(exact.state) != exact.state_hash:
            raise ValueError("exact state document checksum mismatch")
        if exact.decision_kind != "play" or exact.phase != "playing":
            raise ValueError("complete-state graph requires an active play decision")
        state = exact.state
        if str(state.get("rulesHash") or "") != expected_rules_hash:
            raise ValueError("exact state document rules hash mismatch")
        current = state.get("currentTrick")
        if not isinstance(current, Mapping):
            raise ValueError("play decision requires a current trick")
        hands = state.get("hands")
        if not isinstance(hands, Mapping):
            raise ValueError("play decision requires exact remaining hands")
        remaining_hands: list[tuple[Card, ...]] = []
        seen: set[Card] = set()
        for seat in Seat:
            raw_hand = hands.get(seat.name.lower())
            if not isinstance(raw_hand, list):
                raise ValueError("exact remaining hands must contain every seat")
            hand: list[Card] = []
            for raw_card in raw_hand:
                card = Card.parse(raw_card)
                if card in seen:
                    raise ValueError("exact play state contains duplicate cards")
                seen.add(card)
                hand.append(card)
            remaining_hands.append(tuple(hand))
        base_prefix: list[TailPrefixActionV1] = []
        raw_plays = current.get("plays")
        if not isinstance(raw_plays, list):
            raise ValueError("current trick plays must be a list")
        for item in raw_plays:
            if not isinstance(item, Mapping):
                raise ValueError("current trick play must be an object")
            action = TailPrefixActionV1.build(
                item.get("seat"),
                item.get("card"),
                ekka=item.get("ekka") is True,
            )
            if action.card in seen:
                raise ValueError("played card remains in an exact hand")
            seen.add(action.card)
            base_prefix.append(action)
        contract = Contract.parse(state.get("contract"))
        trump_raw = state.get("trumpSuit")
        trump = Suit.parse(trump_raw) if trump_raw is not None else None
        if contract.effective == Contract.HOKUM and trump is None:
            raise ValueError("Hokum play proof requires trump")
        if contract.effective != Contract.HOKUM and trump is not None:
            raise ValueError("Sun play proof cannot carry trump")
        completed = state.get("completedTricks")
        if not isinstance(completed, list):
            raise ValueError("completed tricks must be a list")
        base_raw = [0, 0]
        base_tricks = [0, 0]
        for trick in completed:
            if not isinstance(trick, Mapping):
                raise ValueError("completed trick must be an object")
            team = Seat.parse(trick.get("winner")).team
            points = trick.get("rawPoints")
            if type(points) is not int or points < 0:
                raise ValueError("completed trick rawPoints must be non-negative")
            base_raw[int(team)] += points
            base_tricks[int(team)] += 1
        projects = state.get("projects")
        winning_team_raw = state.get("projectWinningTeam")
        if not isinstance(projects, list):
            raise ValueError("exact play projects must be a list")
        project_points = [0, 0]
        # A single raw-point play frontier cannot consume a later declaration
        # or project award. Longer/exact continuations still require resolution.
        raw_play_frontier = (
            request is not None and request.mode == "bounded"
            and request.boundary_evaluator == "current_raw_point_margin_v1"
            and request.horizon is not None
            and request.horizon["value"] == 1
            and (request.horizon["unit"] in {"plies", "decisions"}
                 or request.horizon["unit"] == "tricks" and len(raw_plays) == len(Seat) - 1)
        )
        declared_owners = {
            event.get("owner") for event in state.get("events", ())
            if isinstance(event, Mapping)
            and event.get("type") in {"project_passed", "project_declared"}
        }
        if (
            not raw_play_frontier and current.get("index") == 0
            and 0 < len(declared_owners) < len(Seat)
        ):
            raise ValueError(
                "play proof requires resolved project comparison and interleaved declaration support"
            )
        if projects:
            passed_owners = {event.get("owner") for event in state.get("events", ())
                if isinstance(event, Mapping) and event.get("type") == "project_passed"}
            all_passed = (len(passed_owners) == len(Seat) and int(current.get("index", 0)) >= 1
                and not any(isinstance(event, Mapping) and event.get("type") == "project_declared"
                    for event in state.get("events", ())))
            if winning_team_raw is None and not raw_play_frontier and not all_passed:
                raise ValueError("play proof requires resolved project comparison")
            winning_team = Team[str(winning_team_raw).strip().upper()] if winning_team_raw is not None else None
            scoring_projects = 0
            for project in projects:
                if not isinstance(project, Mapping):
                    raise ValueError("exact project evidence must be an object")
                if type(project.get("scoring")) is not bool:
                    raise ValueError("project scoring evidence must be boolean")
                if project.get("scoring") is True:
                    if winning_team is None:
                        raise ValueError("awarded project requires a comparison winner")
                    value = project.get("value")
                    if type(value) is not int or value < 0:
                        raise ValueError("project scoring value must be non-negative")
                    project_points[int(winning_team)] += value
                    scoring_projects += 1
            if scoring_projects == 0 and not raw_play_frontier and not all_passed:
                raise ValueError("resolved project comparison lacks an awarded project")
        baloot_teams = state.get("balootTeams")
        if not isinstance(baloot_teams, list):
            raise ValueError("balootTeams must be a list")
        if baloot_teams and self.project_domain is None:
            raise ValueError("Baloot-valued play proof requires Project Domain IR")
        baloot_points = [0, 0]
        if self.project_domain is not None:
            for team_name in baloot_teams:
                team = Team[str(team_name).strip().upper()]
                baloot_points[int(team)] = self.project_domain.baloot_value
        project_disclosure_evidence = ()
        if (self.play_stage.native_conversion and self.project_domain is not None
                and int(current.get("index", 0)) == 1
                and any(row.get("scoring") is True and row.get("revealed") is not True
                        for row in state["projects"])):
            if project_receipt is None:
                exact.require_integrity()
                evidence = exact.normalization_receipt.payload.get("derivedProjects") or {}
                rows = evidence.get("projects")
            else:
                receipt_core = {key: value for key, value in project_receipt.items()
                                if key != "sha256"}
                if (project_receipt.get("worldStateHash") != exact.state_hash
                        or project_receipt.get("sha256") != canonical_sha256(receipt_core)
                        or project_receipt.get("solverStatus") != "optimal"):
                    raise ValueError("native project disclosure receipt differs from its world")
                rows = project_receipt.get("selectedProjects")
            if not isinstance(rows, list):
                raise ValueError("native project disclosure evidence is missing")
            selected = [row for row in rows if row.get("selected") is True]
            if not selected or any(row.get("eligible") is not True or not isinstance(row.get("cards"), list)
                                   or not row["cards"] for row in selected):
                raise ValueError("native project disclosure requires selected native card evidence")
            order = self.project_domain.declaration_seats(Seat.parse(state["dealer"]))
            project_disclosure_evidence = tuple(copy.deepcopy(row) for row in sorted(selected,
                key=lambda row: (order.index(Seat.parse(row["owner"])), -row["value"],
                                 -row["categoryPriority"], -row["highRank"])))
        terminal_project_context = None
        if (self.play_stage.native_conversion and self.project_domain is not None
                and contract.effective == Contract.HOKUM
                and int(current.get("index", 0)) >= 1):
            receipt = exact.normalization_receipt.to_dict()
            projects = receipt.get("derivedProjects")
            play = receipt.get("derivedPlay")
            if isinstance(projects, Mapping) and isinstance(play, Mapping):
                owners = [None] * 32
                for seat_name, hand in play["initialHands"].items():
                    for raw_card in hand:
                        card_id = Card.parse(raw_card).id
                        if owners[card_id] is not None:
                            raise ValueError("terminal project ownership contains duplicate cards")
                        owners[card_id] = int(Seat.parse(seat_name))
                if any(owner is None for owner in owners):
                    raise ValueError("terminal project ownership requires the complete deal")
                declarations = tuple(sorted(
                    (row["projectId"], int(row["declarationOrder"]))
                    for row in projects["projects"] if row["declared"]
                ))
                terminal_project_context = (tuple(owners), declarations, Seat.parse(state["dealer"]))
        if contract.effective == Contract.HOKUM and trump is not None:
            raw_trick_frontier = (request is not None and request.mode == "bounded"
                and request.boundary_evaluator == "current_raw_point_margin_v1"
                and request.horizon is not None and request.horizon["unit"] == "tricks"
                and int(current.get("index", 0)) + request.horizon["value"] < len(FULL_DECK) // len(Seat))
            for seat in Seat:
                remaining_ranks = {
                    card.rank
                    for card in remaining_hands[int(seat)]
                    if card.suit == trump
                }
                if (not raw_play_frontier and not raw_trick_frontier and terminal_project_context is None
                        and {5, 6}.issubset({int(rank) for rank in remaining_ranks})):
                    raise ValueError(
                        "unresolved future Baloot requires Project Domain IR"
                    )
        escalation = str(state.get("escalation") or "")
        ScoringDomainIRV1.multiplier(escalation)
        history = state.get("escalationHistory")
        if not isinstance(history, list):
            raise ValueError("escalationHistory must be a list")
        last_team = None
        if history:
            final = history[-1]
            if not isinstance(final, Mapping):
                raise ValueError("escalation history item must be an object")
            last_team = Team[str(final.get("team")).strip().upper()]
        match_score = state.get("matchScore")
        if (
            not isinstance(match_score, list)
            or len(match_score) != 2
            or any(type(item) is not int or item < 0 for item in match_score)
        ):
            raise ValueError("matchScore must contain two non-negative integers")
        first_leader = Seat.parse(current.get("leader"))
        if exact.next_actor != str(state.get("currentPlayer") or ""):
            raise ValueError("exact state next actor disagrees with its document")
        locked = state.get("locked")
        if type(locked) is not bool:
            raise ValueError("locked must be a boolean")
        legal_actions: list[TailPrefixActionV1] = []
        for item in exact.legal_actions:
            if not isinstance(item, Mapping):
                raise ValueError("exact legal action must be an object")
            legal_actions.append(
                TailPrefixActionV1.build(
                    exact.next_actor,
                    item.get("card") or item.get("code"),
                    ekka=item.get("ekka") is True,
                )
            )
        seed = CompleteStateGraphSeedV1.build(
            exact_state_id=exact.exact_state_id,
            rules_hash=exact.rules_hash,
            remaining_hands=tuple(remaining_hands),
            current_trick=tuple(base_prefix),
            first_leader=first_leader,
            next_actor=Seat.parse(exact.next_actor),
            legal_actions=tuple(legal_actions),
            contract=contract,
            trump=trump,
            buyer=Seat.parse(state.get("buyer")),
            locked=locked,
            base_raw=(base_raw[0], base_raw[1]),
            base_tricks=(base_tricks[0], base_tricks[1]),
            awarded_project_points=(project_points[0], project_points[1]),
            baloot_points=(baloot_points[0], baloot_points[1]),
            escalation=escalation,
            last_escalating_team=last_team,
            starting_match_score=(match_score[0], match_score[1]),
        )
        start = self._parse_seed_start(seed, expected_rules_hash=expected_rules_hash)
        if terminal_project_context is not None:
            owners, declarations, dealer = terminal_project_context
            history = tuple(TailPrefixActionV1.build(row["seat"], row["card"], ekka=row.get("ekka", False))
                for trick in state["completedTricks"] for row in trick["plays"])
            if len(history) != 4 * sum(start.base_tricks):
                raise ValueError("native public project history must cover completed tricks")
            start = replace(
                start, terminal_project_domain=self.project_domain,
                terminal_project_owners=owners, terminal_project_declarations=declarations,
                terminal_project_dealer=dealer,
                terminal_project_second_trick_leader=(
                    Seat.parse(state["completedTricks"][0]["winner"])
                    if self.project_domain.tie_break_order == "second_trick_reveal"
                    and state.get("completedTricks") else None),
                completed_play_history=history,
                proof_state_projection_hash=canonical_sha256({
                    "seedProjectionHash": start.proof_state_projection_hash,
                    "projectDomainHash": self.project_domain.sha256,
                    "owners": owners, "declarations": declarations, "dealer": int(dealer),
                    "completedPlayHistory": [action.to_dict() for action in history],
                }),
            )
        if project_disclosure_evidence:
            history = tuple(TailPrefixActionV1.build(row["seat"], row["card"], ekka=row.get("ekka", False))
                for trick in state["completedTricks"] for row in trick["plays"])
            start = replace(start, project_disclosure_evidence=project_disclosure_evidence,
                completed_play_history=history,
                proof_state_projection_hash=canonical_sha256({
                    "sourceProjectionHash": start.proof_state_projection_hash,
                    "projectDisclosureEvidence": project_disclosure_evidence}))
        return replace(start, source_state_hash=exact.state_hash)

    def _parse_seed_start(
        self,
        seed: CompleteStateGraphSeedV1,
        *,
        expected_rules_hash: str,
    ) -> _StartState:
        if not isinstance(seed.exact_state_id, str) or not seed.exact_state_id:
            raise ValueError("proof seed exact state ID is required")
        if seed.rules_hash != expected_rules_hash:
            raise ValueError("proof seed rules hash does not match the selected bundle")
        if len(seed.remaining_hands) != len(Seat):
            raise ValueError("proof seed must contain every remaining hand")
        if not isinstance(seed.contract, Contract):
            raise TypeError("proof seed contract must be a Contract")
        if seed.trump is not None and not isinstance(seed.trump, Suit):
            raise TypeError("proof seed trump must be a Suit or None")
        if not all(
            isinstance(item, Seat)
            for item in (seed.first_leader, seed.next_actor, seed.buyer)
        ):
            raise TypeError("proof seed actors must be Seat values")
        if seed.last_escalating_team is not None and not isinstance(
            seed.last_escalating_team,
            Team,
        ):
            raise TypeError("proof seed last escalating team must be a Team or None")
        if type(seed.locked) is not bool:
            raise ValueError("proof seed locked must be a boolean")
        if seed.contract.effective == Contract.HOKUM and seed.trump is None:
            raise ValueError("Hokum play proof requires trump")
        if seed.contract.effective != Contract.HOKUM and seed.trump is not None:
            raise ValueError("Sun play proof cannot carry trump")
        for name, pair in (
            ("base raw", seed.base_raw),
            ("base tricks", seed.base_tricks),
            ("awarded project points", seed.awarded_project_points),
            ("Baloot points", seed.baloot_points),
            ("starting match score", seed.starting_match_score),
        ):
            if name == "awarded project points" and pair is None and seed.pending_projects is not None:
                continue
            if pair is None or len(pair) != 2 or any(
                type(item) is not int or item < 0 for item in pair
            ):
                raise ValueError(
                    f"proof seed {name} must contain two non-negative integers"
                )
        ScoringDomainIRV1.multiplier(seed.escalation)
        if len(seed.current_trick) > 3:
            raise ValueError(
                "proof seed current trick must contain at most three plays"
            )
        if not seed.legal_actions:
            raise ValueError("proof seed requires legal root actions")
        for action in (*seed.current_trick, *seed.legal_actions):
            if not isinstance(action, TailPrefixActionV1):
                raise TypeError("proof seed actions must be play actions")
        if len({(action.card, action.ekka) for action in seed.legal_actions}) != len(
            seed.legal_actions
        ):
            raise ValueError("proof seed legal actions must be unique")
        expected_projection_hash = canonical_sha256(
            _seed_projection(
                seed.rules_hash,
                seed.remaining_hands,
                seed.current_trick,
                seed.first_leader,
                seed.next_actor,
                seed.legal_actions,
                seed.contract,
                seed.trump,
                seed.buyer,
                seed.locked,
                seed.base_raw,
                seed.base_tricks,
                seed.awarded_project_points,
                seed.baloot_points,
                seed.escalation,
                seed.last_escalating_team,
                seed.starting_match_score,
                seed.pending_projects,
            )
        )
        if seed.proof_state_projection_hash != expected_projection_hash:
            raise ValueError("proof seed projection checksum mismatch")
        ownership: dict[Card, Seat] = {}
        for seat in Seat:
            for card in seed.remaining_hands[int(seat)]:
                if not isinstance(card, Card):
                    raise TypeError("proof seed hands must contain Card values")
                if card in ownership:
                    raise ValueError("proof seed contains duplicate cards")
                ownership[card] = seat
        for index, action in enumerate(seed.current_trick):
            expected_actor = self.structural_domain.offset(seed.first_leader, index)
            if action.actor != expected_actor:
                raise ValueError("proof seed current trick actor order is illegal")
            if action.card in ownership:
                raise ValueError("proof seed played card remains in a hand")
            ownership[action.card] = action.actor
        cards = tuple(sorted(ownership, key=lambda card: card.id))
        if not 4 <= len(cards) <= 32 or len(cards) % 4:
            raise ValueError(
                "play proof must begin with one through eight tricks"
            )
        counts = [sum(owner == seat for owner in ownership.values()) for seat in Seat]
        if len(set(counts)) != 1:
            raise ValueError("play proof ownership must be balanced at trick start")
        if sum(seed.base_tricks) != 8 - len(cards) // 4:
            raise ValueError(
                "proof seed base trick count disagrees with remaining cards"
            )
        expected_actor = self.structural_domain.offset(
            seed.first_leader,
            len(seed.current_trick),
        )
        if seed.next_actor != expected_actor:
            raise ValueError("proof seed next actor disagrees with its current trick")
        if any(action.actor != seed.next_actor for action in seed.legal_actions):
            raise ValueError("proof seed legal action actor disagrees with next actor")
        start = _StartState(
            seed.exact_state_id,
            seed.rules_hash,
            cards,
            ownership,
            seed.contract,
            seed.trump,
            seed.first_leader,
            seed.current_trick,
            seed.locked,
            seed.base_raw,
            seed.base_tricks,
            seed.buyer,
            seed.awarded_project_points,
            seed.baloot_points,
            seed.escalation,
            seed.last_escalating_team,
            seed.starting_match_score,
            seed.legal_actions,
            seed.proof_state_projection_hash,
        )
        if seed.pending_projects is not None:
            pending = seed.pending_projects
            if (self.project_domain is None or not self.play_stage.native_conversion
                    or self.project_domain.tie_break_order != "second_trick_reveal"
                    or seed.awarded_project_points is not None or sum(seed.base_tricks) != 0
                    or len(cards) != 32 or set(pending) != {"dealer", "declarations"}
                    or not isinstance(pending["declarations"], list)):
                raise ValueError("pending projects require a native first-trick project context")
            declarations = pending["declarations"]
            if (any(not isinstance(row, list) or len(row) != 2 or type(row[0]) is not str
                    or type(row[1]) is not int or row[1] < 1 for row in declarations)
                    or len({row[0] for row in declarations}) != len(declarations)):
                raise ValueError("pending project declarations are invalid")
            start = replace(start, terminal_project_domain=self.project_domain,
                terminal_project_owners=tuple(int(ownership[Card(card)]) for card in range(32)),
                terminal_project_declarations=tuple((row[0], row[1]) for row in declarations),
                terminal_project_dealer=Seat.parse(pending["dealer"]))
        elif seed.awarded_project_points is None:
            raise ValueError("unresolved project points require pending declarations")
        return start

    def _derive_path(
        self,
        start: _StartState,
        path: tuple[TailPrefixActionV1, ...],
        *,
        maximum_seconds: float,
        cancelled: Callable[[], bool] | None,
    ) -> tuple[_DerivedPlayState, tuple[TailPrefixActionV1, ...]]:
        """Solve one graph path through the native CP-SAT play transition stage."""

        prefix = (*start.base_prefix, *path)
        result = self.play_stage.solve(
            cards=start.cards,
            ownership=start.ownership,
            contract=start.contract,
            trump=start.trump,
            first_leader=start.first_leader,
            play_domain=self.play_domain,
            structural_domain=self.structural_domain,
            scoring_domain=(
                self.scoring_domain if len(start.cards) == 32 else None
            ),
            prefix=prefix,
            locked=start.locked,
            decision_prefix_length=len(prefix),
            maximum_time_seconds=max(0.001, maximum_seconds),
            cancel_check=cancelled,
        )
        if result.plays[: len(prefix)] != prefix:
            raise ValueError("native play transition changed the fixed graph path")
        completed_tricks = len(prefix) // 4

        return (
            _DerivedPlayState(
                result.remaining_hands,
                result.current_trick,
                result.next_actor,
                result.legal_play_actions,
                result.trick_winners[:completed_tricks],
                result.trick_raw_points[:completed_tricks],
            ),
            prefix,
        )

    def _path_oracle(
        self,
        start: _StartState,
        *,
        target: Team,
        utility: str,
        utility_encoding: str | None = None,
        play_step_count: int | None = None,
        defer_terminal_scoring: bool = False,
        boundary_evaluator: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
        project_prefix: ExactTracePrefixV1 | None = None,
        project_stop: int = 4,
        project_normalization: OrToolsNormalizedExactStateV1 | None = None,
        project_continuation_actions: tuple[ProjectDeclarationActionV1, ...] | None = None,
        project_counterfactual: bool = False,
    ) -> _CpSatPathOracle:
        return _CpSatPathOracle(
            start,
            play_stage=self.play_stage,
            play_domain=self.play_domain,
            scoring_domain=self.scoring_domain,
            structural_domain=self.structural_domain,
            target=target,
            utility=utility,
            utility_encoding=utility_encoding,
            play_step_count=play_step_count,
            defer_terminal_scoring=defer_terminal_scoring,
            boundary_evaluator=boundary_evaluator,
            cancel_check=cancel_check,
            project_prefix=project_prefix,
            project_domain=self.project_domain,
            project_stop=project_stop,
            project_normalization=project_normalization,
            project_continuation_actions=project_continuation_actions,
            project_counterfactual=project_counterfactual,
        )

    @staticmethod
    def _local_score(
        start: _StartState,
        result: _DerivedPlayState,
        prefix_length: int,
        scoring: ScoringDomainIRV1,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        complete_tricks = prefix_length // 4
        raw = list(start.base_raw)
        tricks = list(start.base_tricks)
        for index in range(complete_tricks):
            team = result.trick_winners[index].team
            raw[int(team)] += result.trick_raw_points[index]
            tricks[int(team)] += 1
        if prefix_length == len(start.cards):
            final_team = result.trick_winners[-1].team
            raw[int(final_team)] += scoring.last_trick_bonus_raw
        return (raw[0], raw[1]), (tricks[0], tricks[1])

    @staticmethod
    def _terminal_utility(
        result: _DerivedPlayState,
    ) -> int:
        if result.terminal_utility is None:
            raise _CompilationStopped(
                "terminal_scoring_variable_missing",
                model_error=True,
            )
        return result.terminal_utility

    @staticmethod
    def _boundary_reached(
        request: ScenarioProofRequestV1,
        *,
        depth: int,
        prefix_length: int,
        base_prefix_length: int,
        terminal: bool,
    ) -> bool:
        if request.mode != "bounded" or request.horizon is None or terminal:
            return False
        unit = str(request.horizon["unit"])
        value = int(request.horizon["value"])
        if unit in {"plies", "decisions"}:
            return depth >= value
        if unit == "tricks":
            completed = prefix_length // 4 - base_prefix_length // 4
            return completed >= value
        if unit == "round_end":
            return False
        raise ValueError("bounded horizon unit is unsupported")

    @staticmethod
    def _bounded_play_step_count(request: ScenarioProofRequestV1, start: _StartState) -> int | None:
        if request.mode != "bounded" or request.horizon is None:
            return None
        unit, value = request.horizon["unit"], int(request.horizon["value"])
        if unit in {"plies", "decisions"}:
            required_steps = len(start.base_prefix) + value
        elif unit == "tricks":
            required_steps = (len(start.base_prefix) // 4 + value) * 4
        elif unit == "round_end":
            required_steps = len(start.cards)
        else:
            raise ValueError("bounded horizon unit is unsupported")
        return min(len(start.cards), max(4, ((required_steps + 3) // 4) * 4))

    def compile(
        self,
        source: OrToolsNormalizedExactStateV1 | CompleteStateGraphSeedV1,
        request: ScenarioProofRequestV1,
        *,
        expected_rules_hash: str,
        maximum_nodes: int,
        maximum_depth: int,
        maximum_seconds: float,
        maximum_conflicts: int | None = None,
        cancelled: Callable[[], bool] | None = None,
        decision_observer: CompleteStateDecisionObserverV1 | None = None,
        _project_prefix: ExactTracePrefixV1 | None = None,
        _project_normalization: OrToolsNormalizedExactStateV1 | None = None,
        _merge_final_boundary: bool = False,
        _compact_forced: bool = False,
        project_counterfactual: bool = False,
    ) -> CompleteStateGraphCompilationV1:
        if type(project_counterfactual) is not bool:
            raise TypeError("project_counterfactual must be boolean")
        if project_counterfactual and (request.mode != "exact" or request.utility != "awarded_point_margin"
                                       or not self.play_stage.native_conversion):
            raise ValueError("project counterfactual requires exact native margin proof")
        if request.visibility != "complete_state" or request.mode not in {
            *_EXACT_MODES,
            "bounded",
        }:
            raise ValueError("graph compiler supports complete-state proof only")
        if maximum_conflicts is not None and (type(maximum_conflicts) is not int or maximum_conflicts < 1):
            raise ValueError("graph conflict limit must be a positive integer")
        if type(maximum_nodes) is not int or maximum_nodes < 1:
            raise ValueError("maximum_nodes must be a positive integer")
        if type(_compact_forced) is not bool:
            raise TypeError("compact forced construction must be bool")
        if type(_merge_final_boundary) is not bool or (_merge_final_boundary and request.mode != "bounded"):
            raise ValueError("final boundary sharing requires bounded proof")
        if type(maximum_depth) is not int or maximum_depth < 1:
            raise ValueError("maximum_depth must be a positive integer")
        if type(maximum_seconds) not in {int, float} or maximum_seconds <= 0:
            raise ValueError("maximum_seconds must be positive")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if decision_observer is not None and not callable(decision_observer):
            raise TypeError("decision_observer must be callable")
        started = time.monotonic()
        deadline = started + float(maximum_seconds)
        start = self._parse_start(source, expected_rules_hash=expected_rules_hash, request=request)
        play_step_count = self._bounded_play_step_count(request, start)
        if _project_normalization is not None and _project_prefix is None:
            raise ValueError("project normalization requires its exact prefix")
        if _project_prefix is not None and (
            not self.play_stage.native_conversion or request.mode != "bounded" or play_step_count != 4
            or request.horizon is None or request.horizon["unit"] not in {"plies", "decisions"}
        ):
            raise ValueError("project checkpoints require native first-trick bounded proof")
        target = Team[str(request.target_team).upper()]
        path_oracle = None
        nodes: dict[str, ProofGraphNodeV1] = {}
        counterfactual_leaves: dict[str, int] = {}
        state_nodes: dict[str, str] = {}
        retained_states: set[str] = set()
        decision_observation_ids: dict[str, str] = {}
        edge_count = 0
        terminal_count = 0
        boundary_count = 0
        transpositions = 0
        maximum_reached = 0

        def stop_check(depth: int) -> None:
            nonlocal maximum_reached
            maximum_reached = max(maximum_reached, depth)
            if cancelled is not None and cancelled():
                raise _CompilationStopped("cancelled")
            if time.monotonic() >= deadline:
                raise _CompilationStopped("maximum_seconds")

        def solve(path: tuple[TailPrefixActionV1, ...], depth: int):
            stop_check(depth)
            try:
                return path_oracle.solve(
                    path,
                    maximum_seconds=deadline - time.monotonic(),
                    cancelled=cancelled,
                    **({"maximum_conflicts": maximum_conflicts} if maximum_conflicts is not None else {}),
                )
            except PlayTailSolveStopped as exc:
                reason = str(exc.solver_status)
                raise _CompilationStopped(
                    reason,
                    model_error=reason == "infeasible",
                ) from exc
            except ValueError as exc:
                raise _CompilationStopped(
                    "transition_model_error",
                    model_error=True,
                ) from exc

        def visit(
            path: tuple[TailPrefixActionV1, ...], depth: int
        ) -> tuple[
            str,
            OrToolsPlayTailResultV1,
            tuple[TailPrefixActionV1, ...],
        ]:
            nonlocal edge_count, terminal_count, boundary_count, transpositions
            result, prefix = solve(path, depth)
            terminal = len(prefix) == len(start.cards)
            raw, tricks = self._local_score(
                start,
                result,
                len(prefix),
                self.scoring_domain,
            )
            boundary = self._boundary_reached(
                request,
                depth=depth,
                prefix_length=len(prefix),
                base_prefix_length=len(start.base_prefix),
                terminal=terminal,
            )
            if terminal:
                utility = self._terminal_utility(
                    result,
                )
                state_core: Mapping[str, Any] = {
                    "kind": "terminal",
                    "utility": utility,
                }
                if project_counterfactual:
                    if result.project_counterfactual_utility is None:
                        raise _CompilationStopped("missing_project_counterfactual", model_error=True)
                    state_core = {**state_core, "projectCounterfactualUtility": result.project_counterfactual_utility}
            else:
                state_core = {
                    "kind": "boundary" if boundary else "decision",
                    "remainingHands": [
                        [card.id for card in hand] for hand in result.remaining_hands
                    ],
                    "currentTrick": [item.to_dict() for item in result.current_trick],
                    "nextActor": (
                        result.next_actor.name.lower()
                        if result.next_actor is not None
                        else None
                    ),
                    "raw": list(raw),
                    "tricks": list(tricks),
                }
                if (start.terminal_project_domain is not None
                        and start.terminal_project_domain.tie_break_order == "second_trick_reveal"
                        and not sum(start.base_tricks) and result.trick_winners):
                    state_core["projectSecondTrickLeader"] = int(result.trick_winners[0])
                if boundary:
                    state_core = {
                        **state_core,
                        "utility": raw[int(target)] - raw[int(target.other)],
                    }
                    if _merge_final_boundary:
                        state_core = {"kind": "boundary", "utility": state_core["utility"]}
            observation_id = None
            if not terminal and not boundary and decision_observer is not None:
                if result.next_actor is None:
                    raise _CompilationStopped(
                        "decision_observer_actor_missing",
                        model_error=True,
                    )
                observation_id = decision_observer(
                    CompleteStateDecisionObservationV1(
                        actor=result.next_actor,
                        path=path,
                        prefix=prefix,
                        remaining_hands=result.remaining_hands,
                        current_trick=result.current_trick,
                        trick_winners=result.trick_winners,
                        trick_raw_points=result.trick_raw_points,
                        raw=raw,
                        tricks=tricks,
                        project_checkpoint=result.project_checkpoint,
                        native_play=result.native_play,
                    )
                )
                if not isinstance(observation_id, str) or not observation_id:
                    raise _CompilationStopped(
                        "decision_observer_invalid",
                        model_error=True,
                    )
            transposition_core = _transposition_core_v1(
                state_core, observation_id
            )
            state_hash = canonical_sha256(transposition_core)
            existing = state_nodes.get(state_hash)
            if existing is not None:
                # Shared forced states remain real DAG nodes: contracting them
                # into two incoming chains would duplicate their identity.
                if _compact_forced and state_hash not in retained_states:
                    if len(retained_states) >= maximum_nodes:
                        raise _CompilationStopped("maximum_nodes")
                    retained_states.add(state_hash)
                transpositions += 1
                return existing, result, prefix
            retain = not _compact_forced or depth == 0 or terminal or boundary or len(result.legal_play_actions) != 1
            if retain:
                if (len(retained_states) if _compact_forced else len(state_nodes)) >= maximum_nodes:
                    raise _CompilationStopped("maximum_nodes")
                if _compact_forced:
                    retained_states.add(state_hash)
            # Forced evidence is bounded too, in original action-depth units.
            if _compact_forced and len(state_nodes) >= maximum_nodes * (maximum_depth + 1):
                raise _CompilationStopped("maximum_nodes")
            node_id = f"node-{state_hash}"
            state_nodes[state_hash] = node_id
            if terminal:
                nodes[node_id] = ProofGraphNodeV1.leaf(
                    node_id,
                    int(state_core["utility"]),
                )
                if project_counterfactual:
                    counterfactual_leaves[node_id] = result.project_counterfactual_utility
                terminal_count += 1
                return node_id, result, prefix
            if boundary:
                nodes[node_id] = ProofGraphNodeV1.leaf(
                    node_id,
                    int(state_core["utility"]),
                    kind="boundary",
                )
                boundary_count += 1
                return node_id, result, prefix
            if depth >= maximum_depth:
                raise _CompilationStopped("maximum_depth")
            if result.next_actor is None or not result.legal_play_actions:
                raise _CompilationStopped("missing_legal_actions", model_error=True)
            if depth == 0:
                expected_root_actions = {
                    (f"{item.card.code}+ekka" if item.ekka else item.card.code)
                    for item in start.expected_legal_actions
                }
                actual_root_actions = {
                    (f"{item.card.code}+ekka" if item.ekka else item.card.code)
                    for item in result.legal_play_actions
                }
                if expected_root_actions != actual_root_actions:
                    raise _CompilationStopped(
                        "root_legal_action_mismatch",
                        model_error=True,
                    )
            edges = []
            preferred = None
            if self.play_stage.native_conversion and path_oracle.results is None:
                preferred = next((full.plays[len(prefix)]
                    for full, _solver in path_oracle._solved_witnesses.values()
                    if full.model_hash == path_oracle.model_hash and len(full.plays) > len(prefix)
                    and full.plays[:len(prefix)] == prefix), None)
            for action in sorted(
                result.legal_play_actions,
                key=lambda item: (item != preferred, item.card.id + 32 * int(item.ekka)),
            ):
                child, child_result, child_prefix = visit(
                    (*path, action), depth + 1
                )
                action_id = action.card.id + 32 * int(action.ekka)
                code = f"{action.card.code}+ekka" if action.ekka else action.card.code
                completed_index = None
                completed_winner = None
                completed_winning_card_id = None
                if len(child_result.trick_winners) > len(result.trick_winners):
                    completed_index = sum(start.base_tricks) + len(
                        result.trick_winners
                    )
                    completed_winner = child_result.trick_winners[-1].name.lower()
                    completed_winning_card_id = next(
                        item.card.id
                        for item in child_prefix[-4:]
                        if item.actor == child_result.trick_winners[-1]
                    )
                edges.append(
                    ProofGraphEdgeV1.build(
                        action_id,
                        code,
                        child,
                        play_trick_index=(
                            sum(start.base_tricks)
                            + (len(start.base_prefix) + depth) // 4
                        ),
                        completed_trick_index=completed_index,
                        completed_trick_winner_seat=completed_winner,
                        completed_trick_winning_card_id=(
                            completed_winning_card_id
                        ),
                    )
                )
                edge_count += 1
            nodes[node_id] = ProofGraphNodeV1.decision(
                node_id,
                result.next_actor.team.name.lower(),
                edges,
                actor_seat=result.next_actor.name.lower(),
            )
            if observation_id is not None:
                decision_observation_ids[node_id] = observation_id
            return node_id, result, prefix

        graph = None
        counterfactual_graph = None
        sequence_pack = None
        status = "complete"
        reason = None
        try:
            stop_check(0)
            def construction_cancelled() -> bool:
                stop_check(0)
                return False

            path_oracle = self._path_oracle(
                start, target=target, utility=request.utility,
                utility_encoding=request.utility_encoding,
                play_step_count=play_step_count, cancel_check=construction_cancelled,
                **({"project_counterfactual": True} if project_counterfactual else {}),
                **({"project_prefix": _project_prefix,
                    "project_normalization": _project_normalization,
                    "project_stop": len(start.base_prefix) + int(request.horizon["value"])}
                   if _project_prefix is not None else {}),
            )
            stop_check(0)
            if request.mode in _EXACT_MODES:
                if len(start.cards) - len(start.base_prefix) > maximum_depth:
                    raise _CompilationStopped("maximum_depth")
            played = {action.card for action in start.base_prefix}
            # Each per-seat card ordering determines at most one play line;
            # each future trick lead adds at most one binary Ekka choice.
            line_upper_bound = math.prod(math.factorial(sum(
                card not in played and owner == seat for card, owner in start.ownership.items())) for seat in Seat)
            ekka_steps = ({step for step in path_oracle.binary_ekka_steps
                if step >= len(start.base_prefix)}
                if self.play_stage.native_conversion else {
                    step for step in range(len(start.base_prefix), len(start.cards)) if step % len(Seat) == 0})
            line_upper_bound *= 2 ** len(ekka_steps)
            if path_oracle.project_checkpoints or (request.mode in _EXACT_MODES and (
                    not self.play_stage.native_conversion or line_upper_bound <= maximum_nodes)):
                enumeration_reason = path_oracle.enumerate_paths(
                    maximum_nodes=maximum_nodes,
                    maximum_seconds=max(0.001, deadline - time.monotonic()),
                    cancelled=cancelled,
                    **({"maximum_conflicts": maximum_conflicts} if maximum_conflicts is not None else {}),
                )
                if enumeration_reason is not None:
                    raise _CompilationStopped(
                        enumeration_reason,
                        model_error=enumeration_reason == "infeasible",
                    )
            root_id, _root_result, _root_prefix = visit((), 0)
            graph_scope = "bounded" if request.mode == "bounded" else "exact"
            owner_by_card = [-1] * 32
            for card, seat in start.ownership.items():
                owner_by_card[card.id] = int(seat)
            for action in start.base_prefix:
                owner_by_card[action.card.id] = int(action.actor)
            graph = ProofGraphV1.build(
                scope=graph_scope,
                root_node_id=root_id,
                nodes=tuple(nodes.values()),
                boundary_evaluator=(
                    request.boundary_evaluator if graph_scope == "bounded" else None
                ),
                tactic_context={
                    "baseTricks": list(start.base_tricks),
                    "trumpSuit": int(start.trump) if start.trump is not None else None,
                    "ownerByCard": owner_by_card,
                },
            )
            if _compact_forced:
                stop_check(0)
                sequence_pack = ProofGraphSequencePackV1.build(
                    graph, observation_ids=decision_observation_ids)
                if len(sequence_pack.graph.nodes) != len(retained_states):
                    raise _CompilationStopped("compact_node_accounting_mismatch", model_error=True)
                stop_check(0)
            if project_counterfactual:
                stop_check(0)
                counterfactual_graph = ProofGraphV1.build(
                    scope=graph.scope, root_node_id=graph.root_node_id, utility_scale=graph.utility_scale,
                    nodes=[replace(node, utility=counterfactual_leaves[node.node_id]) if node.is_leaf else node
                           for node in graph.nodes], tactic_context=graph.tactic_context,
                )
                stop_check(0)
        except _CompilationStopped as exc:
            status = "model_error" if exc.model_error else "incomplete"
            reason = exc.reason
            graph = None
            counterfactual_graph = None
            sequence_pack = None
        model_document = {
                "schemaVersion": 1,
                "compilerId": COMPILER_ID,
                "compilerVersion": COMPILER_VERSION,
                "exactStateId": start.exact_state_id,
                "proofStateProjectionHash": start.proof_state_projection_hash,
                "requestHash": request.sha256,
                "domainIrHashes": {
                    "play": self.play_domain.sha256,
                    "scoring": self.scoring_domain.sha256,
                    "structural": self.structural_domain.sha256,
                    "project": (
                        self.project_domain.sha256
                        if self.project_domain is not None
                        else None
                    ),
                },
                "transitionAuthority": "ortools_play_tail_v1",
                "playScoringModelHash": path_oracle.model_hash if path_oracle is not None else None,
                **({"projectCounterfactualGraphHash": counterfactual_graph.sha256 if counterfactual_graph else None}
                   if project_counterfactual else {}),
                **({"representation": "forced_sequence_pack_v1"} if _compact_forced else {}),
            }
        model_hash = canonical_sha256(model_document)
        elapsed = time.monotonic() - started
        receipt = CompleteStateGraphReceiptV1.build(
            status=status,
            exhaustion_reason=reason,
            exact_state_id=start.exact_state_id,
            request_hash=request.sha256,
            play_domain_hash=self.play_domain.sha256,
            scoring_domain_hash=self.scoring_domain.sha256,
            structural_domain_hash=self.structural_domain.sha256,
            proof_state_projection_hash=start.proof_state_projection_hash,
            graph_hash=graph.sha256 if graph is not None else None,
            model_hash=model_hash,
            explored_node_count=len(state_nodes),
            explored_edge_count=edge_count,
            transposition_edge_count=transpositions,
            terminal_node_count=terminal_count,
            boundary_node_count=boundary_count,
            maximum_depth_reached=maximum_reached,
            elapsed_seconds=elapsed,
            representation=({
                "kind": "forced_sequence_pack_v1",
                "nodeBudgetUnit": "retained_graph_nodes",
                "depthUnit": "original_actions",
                "retainedNodes": len(retained_states),
                "forcedRecords": len(state_nodes) - len(retained_states),
                "maximumEvidenceRecords": maximum_nodes * (maximum_depth + 1),
                "packHash": sequence_pack.sha256 if sequence_pack is not None else None,
            } if _compact_forced else None),
        )
        return CompleteStateGraphCompilationV1(
            graph,
            receipt,
            dict(sorted(decision_observation_ids.items())),
            sequence_pack,
            counterfactual_graph,
            model_document if project_counterfactual else None,
        )


__all__ = [
    "COMPILER_ID",
    "COMPILER_VERSION",
    "INTERNAL_RAW_POINT_MARGIN_UTILITY_V1",
    "CpSatPolicyPathResultV1",
    "CompleteStateDecisionObservationV1",
    "CompleteStateDecisionObserverV1",
    "CompleteStateGraphCompilationV1",
    "CompleteStateGraphReceiptV1",
    "CompleteStateGraphSeedV1",
    "OrToolsCompleteStateGraphCompilerV1",
]
