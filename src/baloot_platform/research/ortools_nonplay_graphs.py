"""Canonical non-play decision graphs for the CP-SAT Research proof service.

The compilers below only consume immutable Domain IR operations.  They do not
replay a mutable round and do not select actions with runtime policies.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64
from dataclasses import dataclass
import itertools
import math
import threading
import time
import zlib
from typing import Any, Callable, Mapping, Sequence
from ortools.sat.python import cp_model
from ortools.sat import cp_model_pb2
from google.protobuf import text_format
from google.protobuf.message import DecodeError

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.enums import Contract, Rank, Seat, Suit, Team
from baloot_platform.core.escalation_ir import EscalationDomainIRV1, EscalationStateV1
from baloot_platform.core.models import Project
from baloot_platform.core.project_ir import (
    ProjectDeclarationActionV1,
    ProjectDeclarationStateV1,
    ProjectDomainIRV1,
)

from .ortools_proof import ProofGraphEdgeV1, ProofGraphNodeV1, ProofGraphV1
from .ortools_escalation_stage import OrToolsEscalationStageV1
from .ortools_scenario_model import (
    OrToolsAuctionActionV1,
    OrToolsNativeAuctionFragmentV1,
    attach_auction_path_v1,
    attach_auction_legal_actions_v1,
    _advance_auction,
    _auction_actor,
    _auction_legal_choices,
    _auction_order,
    _initial_auction_state,
)
from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1
from .scenario_contracts import canonical_sha256


@dataclass(frozen=True, slots=True)
class NonPlayGraphBudgetV1:
    maximum_nodes: int = 50_000
    maximum_depth: int = 64

    def __post_init__(self) -> None:
        if type(self.maximum_nodes) is not int or self.maximum_nodes < 2:
            raise ValueError("non-play graph maximum_nodes must be an integer >= 2")
        if type(self.maximum_depth) is not int or self.maximum_depth < 1:
            raise ValueError("non-play graph maximum_depth must be a positive integer")


class NonPlayGraphCompilationStoppedV1(RuntimeError):
    def __init__(self, reason: str, *, nodes: int, depth: int) -> None:
        self.reason = reason
        self.nodes = nodes
        self.depth = depth
        super().__init__(f"non-play graph compilation stopped: {reason}")


def encode_native_model_receipt_v1(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Compact wire representation; preserve the original semantic receipt hash."""
    core = {key: value for key, value in receipt.items() if key != "sha256"}
    if receipt.get("sha256") != canonical_sha256(core) or "modelEncoding" in core:
        raise ValueError("native auction receipt checksum or encoding is invalid")
    identity = dict(core["modelIdentity"])
    for key in ("model", "boundaryUniquenessModel"):
        if key in identity:
            model = cp_model.CpModel()
            if not model.proto.parse_text_format(identity[key]):
                raise ValueError("native auction model text is invalid")
            identity[key] = base64.b64encode(zlib.compress(identity[key].encode("utf-8"))).decode("ascii")
    core.update(modelIdentity=identity, modelEncoding="cp_sat_text_zlib_base64_v1", sourceReceiptHash=receipt["sha256"])
    return {**core, "sha256": canonical_sha256(core)}


def decode_native_model_receipt_v1(
    receipt: Mapping[str, Any], *, maximum_model_bytes: int,
) -> dict[str, Any]:
    """Restore and verify exact solver text under an explicit read budget."""
    if type(maximum_model_bytes) is not int or maximum_model_bytes <= 0:
        raise ValueError("native model read budget must be a positive integer")
    core = {key: value for key, value in receipt.items() if key != "sha256"}
    if receipt.get("sha256") != canonical_sha256(core) or core.get("modelEncoding") not in {
        "cp_sat_proto_base64_v1", "cp_sat_proto_zlib_base64_v1",
        "cp_sat_text_zlib_base64_v1",
    }:
        raise ValueError("native auction encoded receipt checksum or encoding is invalid")
    original_hash = core.pop("sourceReceiptHash")
    encoding = core.pop("modelEncoding")
    identity = dict(core["modelIdentity"])
    remaining = maximum_model_bytes
    for key in ("model", "boundaryUniquenessModel"):
        if key in identity:
            encoded = identity[key]
            if not isinstance(encoded, str) or len(encoded) > 4 * ((remaining + 2) // 3):
                raise ValueError("native auction model exceeds its read budget")
            binary = base64.b64decode(encoded, validate=True)
            if encoding in {"cp_sat_proto_zlib_base64_v1", "cp_sat_text_zlib_base64_v1"}:
                decoder = zlib.decompressobj()
                try:
                    binary = decoder.decompress(binary, remaining + 1)
                except zlib.error as exc:
                    raise ValueError("native auction model compressed protobuf is invalid") from exc
                if len(binary) > remaining or decoder.unconsumed_tail:
                    raise ValueError("native auction model exceeds its read budget")
                if not decoder.eof or decoder.unused_data:
                    raise ValueError("native auction model compressed protobuf is invalid")
            if len(binary) > remaining:
                raise ValueError("native auction model exceeds its read budget")
            try:
                model_text = (
                    binary.decode("utf-8") if encoding == "cp_sat_text_zlib_base64_v1"
                    else text_format.MessageToString(cp_model_pb2.CpModelProto.FromString(binary))
                )
            except (DecodeError, UnicodeDecodeError) as exc:
                raise ValueError("native auction model protobuf is invalid") from exc
            model = cp_model.CpModel()
            if not model.proto.parse_text_format(model_text):
                raise ValueError("native auction model could not be restored")
            identity[key] = model_text if encoding == "cp_sat_text_zlib_base64_v1" else str(model.proto)
            remaining -= len(identity[key].encode("utf-8"))
            if remaining < 0:
                raise ValueError("native auction model exceeds its read budget")
    core["modelIdentity"] = identity
    if canonical_sha256(identity) != core["modelHash"] or canonical_sha256(core) != original_hash:
        raise ValueError("native auction restored model identity mismatch")
    return {**core, "sha256": original_hash}


# Historical callers retain their names and identical wire/identity semantics.
encode_native_auction_model_receipt_v1 = encode_native_model_receipt_v1
decode_native_auction_model_receipt_v1 = decode_native_model_receipt_v1


@dataclass(frozen=True, slots=True)
class AuctionGraphChoiceV1:
    action: str
    trump: Suit | None = None

    @classmethod
    def build(
        cls,
        action: str,
        trump: Suit | str | int | None = None,
    ) -> "AuctionGraphChoiceV1":
        normalized = str(action).strip().lower()
        if not normalized:
            raise ValueError("auction graph action cannot be empty")
        return cls(normalized, Suit.parse(trump) if trump is not None else None)


@dataclass(frozen=True, slots=True)
class AuctionGraphStateV1:
    actions: tuple[OrToolsAuctionActionV1, ...]
    round_index: int
    actor: Seat | None
    candidate_contract: Contract | None
    candidate_buyer: Seat | None
    candidate_trump: Suit | None
    terminal: bool
    all_passed: bool


@dataclass(frozen=True, slots=True)
class AuctionGraphTerminalV1:
    actions: tuple[OrToolsAuctionActionV1, ...]
    buyer: Seat | None
    declarer: Seat | None
    contract: Contract | None
    trump: Suit | None
    purchase_round: int | None
    all_passed: bool


@dataclass(frozen=True, slots=True)
class _NativeAuctionGraphStateV1:
    records: tuple[OrToolsAuctionActionV1, ...]
    candidate: OrToolsAuctionActionV1 | None
    round_index: int
    terminal: bool
    all_passed: bool
    contract: Contract | None
    buyer: Seat | None
    trump: Suit | None
    actor: Seat | None
    legal_choices: tuple[tuple[str, Suit | None], ...]
    model_hash: str
    variables: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class EscalationGraphActionV1:
    actor: Seat
    action: str
    mode: str | None = None

    @classmethod
    def build(
        cls,
        actor: Seat | str | int,
        action: str,
        mode: str | None = None,
    ) -> "EscalationGraphActionV1":
        normalized_action = str(action).strip().lower()
        normalized_mode = str(mode).strip().lower() if mode is not None else None
        if not normalized_action:
            raise ValueError("escalation graph action cannot be empty")
        if normalized_mode not in {None, "open", "locked"}:
            raise ValueError("escalation graph mode must be open or locked")
        return cls(Seat.parse(actor), normalized_action, normalized_mode)


@dataclass(frozen=True, slots=True)
class EscalationGraphStateV1:
    actions: tuple[EscalationGraphActionV1, ...]
    state: EscalationStateV1
    eligible_actors: tuple[Seat, ...]


@dataclass(frozen=True, slots=True)
class EscalationGraphTerminalV1:
    actions: tuple[EscalationGraphActionV1, ...]
    state: EscalationStateV1
    declined: bool
    decision_depth: int | None = None
    private_history: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectDeclarationTerminalV1:
    actor: Seat
    action_code: str
    projects: tuple[Project, ...]
    announced: Project | None
    actions: tuple[ProjectDeclarationActionV1, ...]


@dataclass(frozen=True, slots=True)
class NonPlayDecisionObservationV1:
    """Semantic public decision state supplied to an observer projector."""

    node_id: str
    stage: str
    actor: Seat
    public_state: Mapping[str, Any]
    own_private_windows: tuple[str, ...] = ()


def bind_escalation_recall_v1(observation_hash: str, own_windows: Sequence[str]) -> str:
    """Remember only the acting player's own private declines, never other seats'."""
    if not own_windows:
        return observation_hash
    if any(type(value) is not str or len(value) != 64
           or any(char not in "0123456789abcdef" for char in value) for value in own_windows):
        raise ValueError("private recall requires canonical announcement-window hashes")
    return canonical_sha256({"schemaVersion": 1, "kind": "escalation_private_recall_v1",
        "observationHash": observation_hash, "ownDeclinedWindows": list(own_windows)})


NonPlayDecisionObserverV1 = Callable[[NonPlayDecisionObservationV1], str]


def normalize_escalation_recall_v1(value: Mapping[str, Sequence[str]] | None) -> dict[str, tuple[str, ...]]:
    """Copy internal recall; callers must derive it from native transition receipts."""
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(type(seat) is not str or seat not in
            {item.name.lower() for item in Seat} for seat in value):
        raise ValueError("private recall requires canonical seat keys")
    result = {}
    for seat, windows in sorted(value.items()):
        if not isinstance(windows, (list, tuple)):
            raise ValueError("private recall requires a sequence of window hashes")
        bind_escalation_recall_v1("", windows)
        if len(set(windows)) != len(windows):
            raise ValueError("an actor cannot decline twice in the same window")
        if windows:
            result[seat] = tuple(windows)
    return result


class _GraphBuilderV1:
    def __init__(
        self,
        budget: NonPlayGraphBudgetV1,
        cancel_check: Callable[[], bool] | None,
    ) -> None:
        self.budget = budget
        self.cancel_check = cancel_check
        self.nodes: dict[str, ProofGraphNodeV1] = {}
        self._lock = threading.Lock()

    def check(self, depth: int) -> None:
        if self.cancel_check is not None and self.cancel_check():
            raise NonPlayGraphCompilationStoppedV1(
                "cancelled", nodes=len(self.nodes), depth=depth
            )
        if depth > self.budget.maximum_depth:
            raise NonPlayGraphCompilationStoppedV1(
                "maximum_depth", nodes=len(self.nodes), depth=depth
            )

    def add(self, node: ProofGraphNodeV1, *, depth: int) -> str:
        self.check(depth)
        with self._lock:
            existing = self.nodes.get(node.node_id)
            if existing is not None:
                if existing != node:
                    raise AssertionError("canonical graph node identity collision")
                return node.node_id
            if len(self.nodes) >= self.budget.maximum_nodes:
                raise NonPlayGraphCompilationStoppedV1(
                    "maximum_nodes", nodes=len(self.nodes), depth=depth
                )
            self.nodes[node.node_id] = node
        return node.node_id


def _strict_utility(value: int) -> int:
    if type(value) is not int:
        raise TypeError("non-play graph utility hook must return an integer")
    return value


def _observe_decision(
    observer: NonPlayDecisionObserverV1 | None,
    *,
    node_id: str,
    stage: str,
    actor: Seat | None,
    public_state: Mapping[str, Any],
    own_private_windows: tuple[str, ...] = (),
) -> None:
    if observer is None:
        return
    if actor is None:
        raise ValueError(f"{stage} observer projection requires one acting seat")
    observation_id = observer(
        NonPlayDecisionObservationV1(node_id, stage, actor, public_state, own_private_windows)
    )
    if not isinstance(observation_id, str) or not observation_id:
        raise ValueError(f"{stage} observer projection returned an invalid hash")


def _graph_options(
    *,
    scope: str,
    horizon: int | None,
    boundary_evaluator: str | None,
    boundary_utility: Callable[[Any], int] | None,
    native_boundary: bool = False,
) -> tuple[str, int | None, str | None]:
    normalized_scope = str(scope).strip().lower()
    if normalized_scope == "exact":
        if (
            horizon is not None
            or boundary_evaluator is not None
            or boundary_utility is not None
            or native_boundary
        ):
            raise ValueError("exact non-play graph cannot define a proof boundary")
        return normalized_scope, None, None
    if normalized_scope != "bounded":
        raise ValueError("non-play graph scope must be exact or bounded")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("bounded non-play graph requires a positive integer horizon")
    boundary = str(boundary_evaluator or "").strip()
    if not boundary or (boundary_utility is None and not native_boundary):
        raise ValueError(
            "bounded non-play graph requires its evaluator and utility hook"
        )
    return normalized_scope, horizon, boundary


def _node_id(stage: str, kind: str, value: Mapping[str, Any]) -> str:
    return f"{stage}:{kind}:{canonical_sha256(value)[:24]}"


def _team_name(team: Team) -> str:
    return team.name.lower()


def _auction_action_core(action: OrToolsAuctionActionV1) -> dict[str, Any]:
    return {
        "round": action.round_index,
        "actor": action.actor.name.lower(),
        "action": action.action,
        "trump": action.trump.name.lower() if action.trump is not None else None,
    }


def _auction_state_view(state: Any, *, dealer: Seat) -> AuctionGraphStateV1:
    actor = (state.actor if isinstance(state, _NativeAuctionGraphStateV1)
             else None if state.terminal else _auction_actor(state, _auction_order(dealer)))
    candidate = state.candidate
    return AuctionGraphStateV1(
        tuple(state.records),
        state.round_index,
        actor,
        candidate.contract if candidate is not None else None,
        candidate.buyer if candidate is not None else None,
        candidate.trump if candidate is not None else None,
        state.terminal,
        state.all_passed,
    )


def _auction_state_core(state: Any) -> dict[str, Any]:
    if isinstance(state, _NativeAuctionGraphStateV1):
        return {"nativeModelHash": state.model_hash, "variables": dict(state.variables),
                "records": [_auction_action_core(item) for item in state.records]}
    return {
        "round": state.round_index,
        "cursor": state.cursor,
        "records": [_auction_action_core(item) for item in state.records],
        "candidate": (
            _auction_action_core(state.candidate)
            if state.candidate is not None
            else None
        ),
        "recall": [seat.name.lower() for seat in state.recall_queue],
        "recallPosition": state.recall_position,
        "terminal": state.terminal,
        "allPassed": state.all_passed,
    }


def compile_auction_decision_graph_v1(
    domain: AuctionDomainIRV1,
    *,
    dealer: Seat | str | int,
    purchase_suit: Suit | str | int,
    purchase_rank: Rank | str | int | None = None,
    terminal_utility: Callable[[AuctionGraphTerminalV1], int],
    prefix: Sequence[AuctionGraphChoiceV1] = (),
    scope: str = "exact",
    horizon: int | None = None,
    boundary_evaluator: str | None = None,
    boundary_utility: Callable[[AuctionGraphStateV1], int] | None = None,
    budget: NonPlayGraphBudgetV1 = NonPlayGraphBudgetV1(),
    cancel_check: Callable[[], bool] | None = None,
    continuation_action_selector: Callable[
        [Seat, str, Sequence[tuple[int, str]]], int
    ]
    | None = None,
    root_action_code: str | None = None,
    root_worker_count: int = 1,
    decision_observer: NonPlayDecisionObserverV1 | None = None,
    native_cp_sat: bool = False,
    maximum_seconds: float | None = None,
    native_boundary_value: Callable[
        [cp_model.CpModel, Mapping[str, cp_model.IntVar], Sequence[Mapping[str, cp_model.IntVar]]],
        cp_model.IntVar,
    ] | None = None,
    native_model_receipts: list[dict[str, Any]] | None = None,
) -> ProofGraphV1:
    """Compile one auction decision and every legal continuation."""

    if not isinstance(domain, AuctionDomainIRV1):
        raise TypeError("auction graph requires AuctionDomainIRV1")
    if not isinstance(budget, NonPlayGraphBudgetV1):
        raise TypeError("auction graph requires NonPlayGraphBudgetV1")
    if type(root_worker_count) is not int or root_worker_count < 1:
        raise ValueError("auction graph root_worker_count must be positive")
    if native_boundary_value is not None and (
        not native_cp_sat or not callable(native_boundary_value) or boundary_utility is not None
    ):
        raise ValueError("native auction boundary requires one native constraint producer")
    if native_model_receipts is not None and (not native_cp_sat or not isinstance(native_model_receipts, list)):
        raise ValueError("native auction model receipts require a native receipt list")
    normalized_scope, bounded_horizon, boundary = _graph_options(
        scope=scope,
        horizon=horizon,
        boundary_evaluator=boundary_evaluator,
        boundary_utility=boundary_utility,
        native_boundary=native_boundary_value is not None,
    )
    parsed_dealer = Seat.parse(dealer)
    parsed_purchase = Suit.parse(purchase_suit)
    if isinstance(purchase_rank, bool):
        raise TypeError("purchase_rank must be a rank")
    if purchase_rank is None and domain.exposed_ace_hokum_sun_overcall == "dealer_right_only":
        raise ValueError("rank-aware auction graph requires purchase_rank")
    parsed_rank = Rank.parse(purchase_rank) if purchase_rank is not None else None
    if type(native_cp_sat) is not bool:
        raise TypeError("native_cp_sat must be boolean")
    if native_cp_sat and (type(maximum_seconds) not in {int, float}
                          or not math.isfinite(maximum_seconds) or maximum_seconds <= 0):
        raise ValueError("native auction graph requires a positive finite time budget")
    if native_cp_sat and root_worker_count != 1:
        raise ValueError("native auction graph requires one solver worker")
    builder = _GraphBuilderV1(budget, cancel_check)
    deadline = time.monotonic() + maximum_seconds if native_cp_sat else None
    native_states: dict[tuple[AuctionGraphChoiceV1, ...], _NativeAuctionGraphStateV1] = {}

    def native_state(actions: tuple[AuctionGraphChoiceV1, ...]) -> _NativeAuctionGraphStateV1:
        builder.check(max(0, len(actions) - len(prefix)))
        if actions in native_states:
            return native_states[actions]
        if len(actions) > 8:
            raise ValueError("auction prefix exceeds the native action universe")
        if domain.require_preselected:
            raise ValueError("preselected auction has no player bid decision")
        model = cp_model.CpModel()
        states, choices = attach_auction_path_v1(
            model, domain, dealer=model.new_constant(int(parsed_dealer)),
            purchase_suit=model.new_constant(int(parsed_purchase)),
            purchase_rank=model.new_constant(int(parsed_rank) if parsed_rank is not None else 0),
            maximum_actions=min(8, len(actions) + 1),
        )
        for index, item in enumerate(actions):
            trump = item.trump
            if item.action == "hokum" and trump is None and index < len(Seat):
                trump = parsed_purchase
            code = f"hokum:{trump.name.lower()}" if item.action == "hokum" and trump is not None else item.action
            if code not in choices[index] or code == "closed" or (item.action != "hokum" and item.trump is not None):
                raise ValueError("native auction prefix contains an invalid action")
            model.add(choices[index][code] == 1)
        current = dict(states[len(actions)])
        if (native_boundary_value is not None and bounded_horizon is not None
                and len(actions) - len(prefix) >= bounded_horizon):
            value = native_boundary_value(model, current, choices)
            if not isinstance(value, cp_model.IntVar):
                raise TypeError("native auction boundary must return a solver variable")
            current["boundaryValue"] = value
        if len(actions) < 8:
            _, legal = attach_auction_legal_actions_v1(
                model, domain, round_index=current["round"], position=current["position"],
                in_recall=current["inRecall"], terminal=current["terminal"],
                candidate_contract=current["contract"], purchase_rank=model.new_constant(int(parsed_rank) if parsed_rank is not None else 0),
                purchase_suit=model.new_constant(int(parsed_purchase)),
                prior_nonpass_count=current["nonpass"], name="graph_root_legal",
            )
        else:
            legal = {}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise NonPlayGraphCompilationStoppedV1("maximum_seconds", nodes=len(builder.nodes), depth=len(actions))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = remaining
        model_identity = {"domain": domain.sha256, "model": str(model.proto)}

        def retain_model(construction_status, derived_values, uniqueness_status=None):
            if native_model_receipts is None:
                return
            core = {
                "schemaVersion": 1, "kind": "native_auction_model_receipt_v1",
                "modelIdentity": dict(model_identity), "modelHash": canonical_sha256(model_identity),
                "dealer": parsed_dealer.name.lower(), "purchaseSuit": parsed_purchase.name.lower(),
                "purchaseRank": parsed_rank.name.lower() if parsed_rank is not None else None,
                "actions": [{"action": item.action, "trumpSuit": item.trump.name.lower() if item.trump is not None else None}
                            for item in actions],
                "variableBindings": {key: value.index for key, value in current.items()},
                "derivedValues": dict(derived_values),
                "constructionStatus": construction_status,
                "boundaryUniquenessStatus": uniqueness_status,
            }
            native_model_receipts.append({**core, "sha256": canonical_sha256(core)})

        status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
        if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            retain_model(solver.status_name(status).lower(), {})
            if status == cp_model.INFEASIBLE:
                raise ValueError("native auction prefix is infeasible")
            raise NonPlayGraphCompilationStoppedV1(
                "cancelled" if cancel_check is not None and cancel_check() else "native_transition_incomplete",
                nodes=len(builder.nodes), depth=len(actions))
        values = {key: solver.value(value) for key, value in current.items()}
        if "boundaryValue" in current:
            uniqueness = model.clone()
            uniqueness.add(
                uniqueness.get_int_var_from_proto_index(current["boundaryValue"].index) != values["boundaryValue"]
            ).with_name("auction_boundary_value_uniqueness")
            model_identity["boundaryUniquenessModel"] = str(uniqueness.proto)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                retain_model(solver.status_name(status).lower(), values, "not_started_budget_exhausted")
                raise NonPlayGraphCompilationStoppedV1("maximum_seconds", nodes=len(builder.nodes), depth=len(actions))
            check = cp_model.CpSolver()
            check.parameters.num_search_workers = 1
            check.parameters.max_time_in_seconds = remaining
            unique_status = _solve_cp_sat_with_cancel_v1(check, uniqueness, cancel_check)
            retain_model(solver.status_name(status).lower(), values, check.status_name(unique_status).lower())
            if unique_status in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
                raise ValueError("native auction boundary value is not uniquely derived")
            if unique_status != cp_model.INFEASIBLE:
                raise NonPlayGraphCompilationStoppedV1(
                    "cancelled" if cancel_check is not None and cancel_check() else "native_boundary_incomplete",
                    nodes=len(builder.nodes), depth=len(actions))
        else:
            retain_model(solver.status_name(status).lower(), values)
        records = OrToolsNativeAuctionFragmentV1(states, choices, len(actions), {}, {}, domain.sha256).extract_actions(solver, length=len(actions))
        candidate = next((row for row in reversed(records) if row.contract is not None), None)
        terminal = bool(values["terminal"])
        result = _NativeAuctionGraphStateV1(
            records, candidate, values["round"], terminal, bool(values["allPassed"]),
            candidate.contract if terminal and candidate else None,
            Seat(values["buyer"]) if terminal and values["buyer"] >= 0 else None,
            Suit(values["trump"]) if terminal and values["trump"] < 4 else None,
            None if terminal else Seat(values["actor"]),
            tuple((code.split(":")[0], Suit.parse(code.split(":")[1]) if ":" in code else None)
                  for code, literal in sorted(legal.items()) if solver.boolean_value(literal)),
            canonical_sha256(model_identity), values,
        )
        native_states[actions] = result
        return result

    for item in prefix:
        if not isinstance(item, AuctionGraphChoiceV1):
            raise TypeError("auction graph prefix must use AuctionGraphChoiceV1")
    state = native_state(tuple(prefix)) if native_cp_sat else _initial_auction_state(
        domain, dealer=parsed_dealer, purchase_suit=parsed_purchase)
    for item in (() if native_cp_sat else prefix):
        state = _advance_auction(
            domain,
            state,
            dealer=parsed_dealer,
            purchase_suit=parsed_purchase,
            purchase_rank=parsed_rank,
            action=item.action,
            trump=item.trump,
        )
    if state.terminal:
        raise ValueError("auction graph prefix must stop at a decision")

    def expand(current: Any, depth: int) -> str:
        builder.check(depth)
        core = {
            "stage": "auction",
            "domain": domain.sha256,
            "dealer": parsed_dealer.name.lower(),
            "purchaseSuit": parsed_purchase.name.lower(),
            **({"purchaseRank": parsed_rank.name.lower()}
               if domain.exposed_ace_hokum_sun_overcall == "dealer_right_only" and parsed_rank is not None else {}),
            "state": _auction_state_core(current),
        }
        if native_boundary_value is not None and bounded_horizon is not None and depth >= bounded_horizon:
            # An auction terminal is not a round terminal. At the requested
            # boundary, retain the solved evaluator value and bounded label.
            return builder.add(ProofGraphNodeV1.leaf(
                _node_id("auction", "boundary", core),
                current.variables["boundaryValue"], kind="boundary",
            ), depth=depth)
        if current.terminal:
            candidate = current.candidate
            if current.buyer is not None and (candidate is None or candidate.declarer is None):
                raise ValueError("purchased auction terminal lacks a declarer")
            terminal = AuctionGraphTerminalV1(
                tuple(current.records),
                current.buyer,
                candidate.declarer if candidate is not None else None,
                current.contract,
                current.trump,
                candidate.round_index if candidate is not None else None,
                current.all_passed,
            )
            return builder.add(
                ProofGraphNodeV1.leaf(
                    _node_id("auction", "terminal", core),
                    _strict_utility(terminal_utility(terminal)),
                ),
                depth=depth,
            )
        if bounded_horizon is not None and depth >= bounded_horizon:
            assert boundary_utility is not None
            return builder.add(
                ProofGraphNodeV1.leaf(
                    _node_id("auction", "boundary", core),
                    _strict_utility(
                        boundary_utility(
                            _auction_state_view(current, dealer=parsed_dealer)
                        )
                    ),
                    kind="boundary",
                ),
                depth=depth,
            )
        actor = current.actor if native_cp_sat else _auction_actor(current, _auction_order(parsed_dealer))
        choices = sorted(
            current.legal_choices if native_cp_sat else _auction_legal_choices(
                domain,
                current,
                dealer=parsed_dealer,
                purchase_suit=parsed_purchase,
                purchase_rank=parsed_rank,
            ),
            key=lambda item: (
                item[0],
                item[1].name.lower() if item[1] is not None else "",
            ),
        )
        node_id = _node_id("auction", "decision", core)
        _observe_decision(
            decision_observer,
            node_id=node_id,
            stage="bid",
            actor=actor,
            public_state=core,
        )
        choice_rows = [
            (
                action_id,
                action,
                trump,
                (
                    f"hokum:{trump.name.lower()}"
                    if action == "hokum" and trump is not None
                    else action
                ),
            )
            for action_id, (action, trump) in enumerate(choices)
        ]
        if depth == 0 and root_action_code is not None:
            choice_rows = [item for item in choice_rows if item[3] == root_action_code]
            if len(choice_rows) != 1:
                raise ValueError("auction witness selected an illegal root action")
        if continuation_action_selector is not None and depth > 0:
            selected_id = continuation_action_selector(
                actor,
                node_id,
                tuple((action_id, code) for action_id, _a, _t, code in choice_rows),
            )
            choice_rows = [
                item for item in choice_rows if item[0] == selected_id
            ]
            if len(choice_rows) != 1:
                raise ValueError("auction continuation policy selected an illegal action")
        def build_edge(row: tuple[int, str, Suit | None, str]) -> ProofGraphEdgeV1:
            action_id, action, trump, code = row
            child = native_state(tuple(AuctionGraphChoiceV1(row.action, row.trump) for row in current.records)
                                 + (AuctionGraphChoiceV1(action, trump),)) if native_cp_sat else _advance_auction(
                domain,
                current,
                dealer=parsed_dealer,
                purchase_suit=parsed_purchase,
                purchase_rank=parsed_rank,
                action=action,
                trump=trump,
            )
            return ProofGraphEdgeV1.build(
                action_id, code, expand(child, depth + 1)
            )
        if depth == 0 and root_worker_count > 1 and len(choice_rows) > 1:
            with ThreadPoolExecutor(
                max_workers=min(root_worker_count, len(choice_rows)),
                thread_name_prefix="balootx-auction-root",
            ) as pool:
                edges = list(pool.map(build_edge, choice_rows))
        else:
            edges = [build_edge(row) for row in choice_rows]
        return builder.add(
            ProofGraphNodeV1.decision(
                node_id,
                _team_name(actor.team),
                edges,
                actor_seat=actor.name.lower(),
            ),
            depth=depth,
        )

    root = expand(state, 0)
    return ProofGraphV1.build(
        scope=normalized_scope,
        root_node_id=root,
        nodes=tuple(builder.nodes.values()),
        boundary_evaluator=boundary,
    )


def _escalation_choices(
    domain: EscalationDomainIRV1,
    *,
    state: EscalationStateV1,
    declarer: Seat,
    contract: Contract,
    match_score: tuple[int, int],
    actors: Sequence[Seat] | None = None,
) -> tuple[tuple[EscalationGraphActionV1, EscalationStateV1], ...]:
    decisions = domain.legal_decisions(
        state,
        declarer=declarer,
        contract=contract,
        match_score=match_score,
        actors=actors,
        include_closure=False,
    )
    return tuple(
        (
            EscalationGraphActionV1(
                decision.actor,
                decision.action,
                (
                    decision.resolved_mode
                    if decision.action in {"double", "four"}
                    else None
                ),
            ),
            decision.state,
        )
        for decision in decisions
        if decision.actor is not None
    )


def _escalation_state_core(
    *,
    state: EscalationStateV1,
    actions: Sequence[EscalationGraphActionV1],
    actors: Sequence[Seat],
) -> dict[str, Any]:
    return {
        "state": {
            "level": state.level,
            "locked": state.locked,
            "doubleSeat": (
                state.double_seat.name.lower()
                if state.double_seat is not None
                else None
            ),
        },
        "actions": [
            {
                "actor": item.actor.name.lower(),
                "action": item.action,
                "mode": item.mode,
            }
            for item in actions
        ],
        "eligibleActors": [item.name.lower() for item in actors],
    }


def _escalation_window_key(actions: Sequence[EscalationGraphActionV1]) -> str:
    return canonical_sha256({"actions": [
        [item.actor.name.lower(), item.action, item.mode] for item in actions
    ]})


def enumerate_escalation_prior_histories_v1(
    domain: EscalationDomainIRV1, *, declarer: Seat | str | int,
    contract: Contract | str, match_score: tuple[int, int], observer_seat: Seat | str | int,
    prefix: Sequence[EscalationGraphActionV1], own_declined_windows: Sequence[str] = (),
    closed: bool = False, maximum_worlds: int, maximum_seconds: float,
    budget: NonPlayGraphBudgetV1, cancel_check: Callable[[], bool] | None = None,
    declared_maximum_seconds: float | None = None,
) -> dict[str, Any]:
    """Certify prior decline-membership worlds, conditioning only on own recall.

    Private action ordering within a past window is not an observation in this
    model; these are recall worlds, not continuation-line counts.
    """
    if not isinstance(domain, EscalationDomainIRV1) or not isinstance(budget, NonPlayGraphBudgetV1):
        raise TypeError("prior history enumeration requires a domain and graph budget")
    if type(closed) is not bool or type(maximum_worlds) is not int or maximum_worlds < 1:
        raise ValueError("prior history enumeration requires a boolean closure and positive world limit")
    if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("history time limit must be positive and finite")
    declared = maximum_seconds if declared_maximum_seconds is None else declared_maximum_seconds
    if type(declared) not in {int, float} or not math.isfinite(declared) or declared < maximum_seconds:
        raise ValueError("declared history time limit must cover its runtime allowance")
    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("history cancellation check must be callable")
    if any(type(seat) not in {Seat, str, int} for seat in (observer_seat, declarer)):
        raise TypeError("history actors must be seats")
    observer = Seat.parse(observer_seat).name.lower()
    own = normalize_escalation_recall_v1({observer: own_declined_windows}).get(observer, ())
    history = tuple(prefix)
    if any(not isinstance(action, EscalationGraphActionV1) or action.action == domain.closure_action for action in history):
        raise ValueError("prior history prefix must contain typed public announcements")
    keys = [_escalation_window_key(history[:index]) for index in range(len(history) + int(closed))]
    if any(key not in keys for key in own) or list(own) != sorted(own, key=keys.index):
        raise ValueError("own recall must reference chronological observed windows")
    stage = OrToolsEscalationStageV1(domain, declarer=Seat.parse(declarer),
        contract=Contract.parse(contract), match_score=match_score, native_cp_sat=True)
    deadline = time.monotonic() + maximum_seconds
    evidence, dimensions, worlds = {}, [], []
    count, solves, reason, infeasible = 1, 0, None, False

    def allowance(cost):
        nonlocal solves, reason
        cancelled = cancel_check is not None and cancel_check()
        seconds = min(maximum_seconds, deadline - time.monotonic())
        reason = ("cancelled" if cancelled else
                  "maximum_seconds" if seconds <= 0 else
                  "maximum_nodes" if solves + cost > budget.maximum_nodes else None)
        if reason is not None:
            return None
        solves += cost
        return seconds

    def retain(result):
        nonlocal reason
        evidence[canonical_sha256(result)] = result
        if result["status"] not in {"satisfied", "infeasible"}:
            reason = result.get("stopReason", "native_history_incomplete")
        if reason is None:
            allowance(0)
        return result["status"] == "satisfied"

    if len(history) > budget.maximum_depth:
        reason = "maximum_depth"
    for index, key in enumerate(keys):
        if reason is not None:
            break
        seconds = allowance(1)
        if seconds is None:
            break
        actions = tuple((item.actor, item.action, item.mode) for item in history[:index])
        try:
            base = stage.solve(required_actions=actions, required_actions_complete=True,
                opportunity_non_announcements=(), maximum_seconds=seconds, cancel_check=cancel_check).to_dict()
        except InterruptedError:
            reason = "cancelled"
            break
        if not retain(base):
            infeasible = base["status"] == "infeasible"
            break
        eligible = base["opportunityState"]["remainingEligibleSeats"]
        announcing = history[index].actor.name.lower() if index < len(history) else None
        other = [seat for seat in eligible if seat not in {observer, announcing}]
        forced = [observer] if key in own else []
        subsets = ([tuple(other)] if announcing is None else
                   itertools.chain.from_iterable(itertools.combinations(other, size) for size in range(len(other) + 1)))
        choices = []
        for subset in subsets:
            declines = tuple(sorted((*subset, *forced)))
            seconds = allowance(1 if announcing is None else 2)
            if seconds is None:
                break
            try:
                if announcing is None:
                    result = stage.solve(required_actions=actions, required_actions_complete=True,
                        opportunity_non_announcements=declines, required_terminal=True,
                        maximum_seconds=seconds, cancel_check=cancel_check).to_dict()
                else:
                    action = history[index]
                    result = stage.transition_opportunity(required_actions=actions, non_announcements=declines,
                        actor=announcing, action_code=action.action + (":" + action.mode if action.mode else ""),
                        maximum_seconds=seconds, declared_maximum_seconds=declared, cancel_check=cancel_check)
            except InterruptedError:
                reason = "cancelled"
                break
            if retain(result):
                choices.append(declines)
            if reason is not None:
                break
        if reason is not None:
            break
        if not choices:
            infeasible = True
            break
        dimensions.append((key, choices))
        count *= len(choices)
        if count > maximum_worlds:
            reason = "maximum_worlds"
            break
    if reason is None and not infeasible:
        for values in itertools.product(*(choices for _key, choices in dimensions)):
            if allowance(0) is None:
                break
            if len(history) + sum(len(value) for value in values) > budget.maximum_depth:
                reason = "maximum_depth"
                break
            recall = {}
            for (key, _choices), seats in zip(dimensions, values, strict=True):
                for seat in seats:
                    recall.setdefault(seat, []).append(key)
            worlds.append({"worldId": canonical_sha256(recall), "privateRecall": recall})
    if reason is not None or infeasible:
        worlds.clear()
    core = {"schemaVersion": 1, "kind": "native_escalation_prior_history_coverage_v1",
        "domainIrHash": domain.sha256, "declarer": Seat.parse(declarer).name.lower(),
        "contract": Contract.parse(contract).value, "matchScore": list(match_score),
        "prefixHash": _escalation_window_key(history), "observerSeat": observer,
        "ownDeclinedWindows": list(own), "closed": closed,
        "status": "incomplete" if reason is not None else "infeasible" if infeasible else "satisfied",
        "complete": reason is None, "stopReason": reason,
        "worldCount": len(worlds) if reason is None else None,
        "nativeSolvesReserved": solves,
        "limits": {"maximumWorlds": maximum_worlds, "maximumNativeSolves": budget.maximum_nodes,
            "maximumDepth": budget.maximum_depth, "maximumSeconds": declared},
        "evidence": dict(sorted(evidence.items())), "worlds": sorted(worlds, key=lambda row: row["worldId"])}
    return {**core, "sha256": canonical_sha256(core)}


def enumerate_escalation_opportunity_orders_v1(
    domain: EscalationDomainIRV1, *, declarer: Seat | str | int,
    contract: Contract | str, match_score: tuple[int, int],
    maximum_worlds: int, maximum_seconds: float,
    budget: NonPlayGraphBudgetV1,
    hidden_world_count: int = 1,
    acting_seat: Seat | str | int | None = None,
    declared_maximum_seconds: float | None = None,
    include_private_root_histories: bool = False,
    maximum_announcement_depth: int | None = None,
    prefix: Sequence[EscalationGraphActionV1] = (),
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Enumerate contingent orders for every native-reachable announcement window.

    Incomplete discovery returns receipts, never a silently truncated world set.
    Order worlds are not hidden-card worlds and do not certify round utilities.
    """
    if type(maximum_worlds) is not int or maximum_worlds < 1:
        raise ValueError("maximum_worlds must be a positive integer")
    if maximum_announcement_depth is not None and (
            type(maximum_announcement_depth) is not int or maximum_announcement_depth < 0):
        raise ValueError("maximum_announcement_depth must be a nonnegative integer")
    if type(hidden_world_count) is not int or hidden_world_count < 1:
        raise ValueError("hidden_world_count must be a positive integer")
    if (type(maximum_seconds) not in {int, float} or maximum_seconds <= 0
            or not math.isfinite(maximum_seconds)):
        raise ValueError("maximum_seconds must be positive and finite")
    declared_seconds = maximum_seconds if declared_maximum_seconds is None else declared_maximum_seconds
    if (type(declared_seconds) not in {int, float} or not math.isfinite(declared_seconds)
            or declared_seconds < maximum_seconds):
        raise ValueError("declared wall limit must be finite and cover the runtime allowance")
    if not isinstance(budget, NonPlayGraphBudgetV1):
        raise TypeError("opportunity enumeration requires NonPlayGraphBudgetV1")
    history = tuple(prefix)
    root_actor = Seat.parse(acting_seat) if acting_seat is not None else None
    if type(include_private_root_histories) is not bool or (include_private_root_histories and root_actor is None):
        raise ValueError("private root histories require a known acting seat and a boolean scope")
    if any(not isinstance(item, EscalationGraphActionV1) for item in history):
        raise TypeError("escalation prefix must use EscalationGraphActionV1")
    stage = OrToolsEscalationStageV1(domain, declarer=Seat.parse(declarer),
        contract=Contract.parse(contract), match_score=match_score, native_cp_sat=True)
    deadline = time.monotonic() + maximum_seconds
    windows = {}
    dimensions = {}
    count = 1
    reason = None

    def check():
        nonlocal reason
        if cancel_check is not None and cancel_check():
            reason = "cancelled"
        elif time.monotonic() >= deadline:
            reason = "maximum_seconds"
        return reason is None

    def visit(actions, depth):
        nonlocal count, reason
        if not check():
            return
        if depth > budget.maximum_depth:
            reason = "maximum_depth"
            return
        if len(windows) >= budget.maximum_nodes:
            reason = "maximum_nodes"
            return
        key = _escalation_window_key(actions)
        try:
            result = stage.solve(required_actions=tuple((a.actor, a.action, a.mode) for a in actions),
                required_actions_complete=True, opportunity_non_announcements=(),
                maximum_seconds=max(deadline - time.monotonic(), 1e-9),
                cancel_check=cancel_check)
        except InterruptedError:
            reason = "cancelled"
            return
        windows[key] = result.to_dict()
        if not check():
            return
        if result.status != "satisfied":
            reason = (("invalid_prefix" if depth == 0 else "native_transition_infeasible")
                      if result.status == "infeasible" else "native_transition_incomplete")
            return
        actors = result.observation.eligible_seats
        if depth == 0 and root_actor is not None and root_actor not in actors:
            reason = "ineligible_root_actor"
            return
        if len(actors) > 1:
            dimensions[key] = tuple(order for order in itertools.permutations(actor.name.lower() for actor in actors)
                if depth != 0 or root_actor is None or include_private_root_histories or order[0] == root_actor.name.lower())
            count *= len(dimensions[key])
            if count * hidden_world_count > maximum_worlds:
                reason = "maximum_order_worlds"
                return
        if maximum_announcement_depth is not None and depth >= maximum_announcement_depth:
            return
        for choice in result.observation.legal_next_choices:
            visit((*actions, EscalationGraphActionV1(choice.actor, choice.action, choice.mode)), depth + 1)
            if reason is not None:
                return

    if hidden_world_count > maximum_worlds:
        reason = "maximum_combined_worlds"
    else:
        visit(history, 0)
    worlds = []
    keys = sorted(dimensions)
    if reason is None:
        for values in itertools.product(*(dimensions[key] for key in keys)):
            if not check():
                worlds.clear()
                break
            orders = {key: list(value) for key, value in zip(keys, values, strict=True)}
            if include_private_root_histories:
                root_order = orders.get(_escalation_window_key(history), [root_actor.name.lower()])
                private = root_order[:root_order.index(root_actor.name.lower())]
                identity = {"opportunityOrders": orders, "priorNonAnnouncingSeats": private}
                worlds.append({"worldId": canonical_sha256(identity), **identity})
            else:
                worlds.append({"worldId": canonical_sha256(orders), "opportunityOrders": orders})
    core = {
        "schemaVersion": 1, "kind": "native_escalation_order_coverage_v1",
        "domainIrHash": domain.sha256, "declarer": Seat.parse(declarer).name.lower(),
        "contract": Contract.parse(contract).value, "matchScore": list(match_score),
        "prefixHash": _escalation_window_key(history),
        "limits": {"maximumWorlds": maximum_worlds, "maximumSeconds": declared_seconds,
                   "maximumNodes": budget.maximum_nodes, "maximumDepth": budget.maximum_depth},
        "hiddenWorldCount": hidden_world_count,
        "actingSeat": root_actor.name.lower() if root_actor is not None else None,
        **({"rootHistoryScope": "actor_visible_private_histories"} if include_private_root_histories else {}),
        "complete": reason is None, "stopReason": reason,
        "orderWorldCount": count if reason is None else None,
        "combinedWorldCount": count * hidden_world_count if reason is None else None,
        "orderWorldCountLowerBound": count if dimensions and reason != "invalid_prefix" else (1 if reason is None else 0),
        "windows": dict(sorted(windows.items())), "worlds": worlds,
    }
    if maximum_announcement_depth is not None:
        core["announcementDepthScope"] = maximum_announcement_depth
    return {**core, "sha256": canonical_sha256(core)}


def compile_escalation_decision_graph_v1(
    domain: EscalationDomainIRV1,
    *,
    declarer: Seat | str | int,
    contract: Contract | str,
    match_score: tuple[int, int],
    terminal_utility: Callable[[EscalationGraphTerminalV1], int],
    acting_seat: Seat | str | int | None = None,
    prefix: Sequence[EscalationGraphActionV1] = (),
    scope: str = "exact",
    horizon: int | None = None,
    boundary_evaluator: str | None = None,
    boundary_utility: Callable[[EscalationGraphStateV1], int] | None = None,
    budget: NonPlayGraphBudgetV1 = NonPlayGraphBudgetV1(),
    cancel_check: Callable[[], bool] | None = None,
    continuation_action_selector: Callable[
        [Seat, str, Sequence[tuple[int, str]]], int
    ]
    | None = None,
    root_action_code: str | None = None,
    root_worker_count: int = 1,
    decision_observer: NonPlayDecisionObserverV1 | None = None,
    native_cp_sat: bool = False,
    opportunity_orders: Mapping[str, Sequence[Seat | str | int]] | None = None,
    opportunity_order: Sequence[Seat | str | int] | None = None,
    prior_non_announcing_seats: Sequence[Seat | str | int] = (),
    prior_window_recall: Mapping[str, Sequence[str]] | None = None,
) -> ProofGraphV1:
    """Compile escalation; supplied opportunity orders describe one order world.

    Multi-actor window keys hash {"actions": [[seat, action, mode], ...]}.
    An exact graph is exact only for those supplied orders, not all order worlds.
    """

    if not isinstance(domain, EscalationDomainIRV1):
        raise TypeError("escalation graph requires EscalationDomainIRV1")
    if not isinstance(budget, NonPlayGraphBudgetV1):
        raise TypeError("escalation graph requires NonPlayGraphBudgetV1")
    if type(root_worker_count) is not int or root_worker_count < 1:
        raise ValueError("escalation graph root_worker_count must be positive")
    if (
        not isinstance(match_score, tuple)
        or len(match_score) != 2
        or any(type(value) is not int or value < 0 for value in match_score)
    ):
        raise ValueError("escalation graph match score must contain two integers")
    normalized_scope, bounded_horizon, boundary = _graph_options(
        scope=scope,
        horizon=horizon,
        boundary_evaluator=boundary_evaluator,
        boundary_utility=boundary_utility,
    )
    parsed_declarer = Seat.parse(declarer)
    parsed_contract = Contract.parse(contract)
    if type(native_cp_sat) is not bool:
        raise TypeError("native_cp_sat must be boolean")
    if native_cp_sat and root_worker_count != 1:
        raise ValueError("native escalation graph requires one root worker")
    orders = None
    frozen_order = None
    if opportunity_order is not None:
        if not native_cp_sat or opportunity_orders is not None:
            raise ValueError("frozen opportunity order requires native mode and no per-window orders")
        if (not isinstance(opportunity_order, Sequence) or isinstance(opportunity_order, (str, bytes))
                or any(type(value) not in {Seat, str, int} for value in opportunity_order)):
            raise ValueError("frozen opportunity order must list every seat exactly once")
        frozen_order = tuple(Seat.parse(value) for value in opportunity_order)
        if len(frozen_order) != len(Seat) or len(set(frozen_order)) != len(Seat):
            raise ValueError("frozen opportunity order must list every seat exactly once")
        orders = {}
    if opportunity_orders is not None:
        if not native_cp_sat or not isinstance(opportunity_orders, Mapping):
            raise ValueError("opportunity orders require native escalation and a mapping")
        orders = {}
        for key, values in opportunity_orders.items():
            if (type(key) is not str or len(key) != 64
                    or any(char not in "0123456789abcdef" for char in key)):
                raise ValueError("opportunity window key must be a canonical SHA-256")
            if (not isinstance(values, Sequence) or isinstance(values, (str, bytes))
                    or any(type(value) not in {Seat, str, int} for value in values)):
                raise TypeError("opportunity order must be a sequence of seats")
            parsed = tuple(Seat.parse(value) for value in values)
            if not parsed or len(set(parsed)) != len(parsed):
                raise ValueError("opportunity order must contain distinct seats")
            orders[key] = parsed
    native_stage = OrToolsEscalationStageV1(
        domain, declarer=parsed_declarer, contract=parsed_contract, match_score=match_score,
        native_cp_sat=True,
    ) if native_cp_sat else None
    native_results = {}
    builder = _GraphBuilderV1(budget, cancel_check)
    if (not isinstance(prior_non_announcing_seats, Sequence)
            or isinstance(prior_non_announcing_seats, (str, bytes))
            or any(type(value) not in {Seat, str, int} for value in prior_non_announcing_seats)):
        raise TypeError("private escalation history must be a sequence of seats")
    prior_declines = tuple(Seat.parse(value) for value in prior_non_announcing_seats)
    if prior_declines and (orders is None or len(set(prior_declines)) != len(prior_declines)):
        raise ValueError("private escalation history requires distinct seats and native opportunity orders")

    def native_result(actions, closed=False, non_announcing=()):
        key = (actions, closed, non_announcing)
        if key not in native_results:
            declines = native_result(actions).observation.eligible_seats if closed else non_announcing
            try:
                result = native_stage.solve(
                    required_actions=tuple((item.actor, item.action, item.mode) for item in actions),
                    required_actions_complete=True,
                    opportunity_non_announcements=declines,
                    cancel_check=cancel_check,
                )
            except InterruptedError as exc:
                raise NonPlayGraphCompilationStoppedV1(
                    "cancelled", nodes=len(builder.nodes), depth=0
                ) from exc
            if result.status == "incomplete":
                raise NonPlayGraphCompilationStoppedV1(
                    "native_transition_incomplete", nodes=len(builder.nodes), depth=0
                )
            if result.status != "satisfied" or result.observation is None:
                raise ValueError(f"native escalation transition unavailable: {result.status}")
            native_results[key] = result
        return native_results[key]

    def native_state(actions):
        observed = native_result(actions).observation
        return EscalationStateV1(observed.level, observed.mode == "locked", observed.double_seat)

    def choices_for(current, actions, actors=None):
        if not native_cp_sat:
            return _escalation_choices(domain, state=current, declarer=parsed_declarer,
                contract=parsed_contract, match_score=match_score, actors=actors)
        rows = []
        for choice in native_result(actions).observation.legal_next_choices:
            if actors is not None and choice.actor not in actors:
                continue
            action = EscalationGraphActionV1(choice.actor, choice.action, choice.mode)
            rows.append((action, native_state((*actions, action))))
        return tuple(rows)
    root_actors = (
        (Seat.parse(acting_seat),)
        if acting_seat is not None
        else (() if native_cp_sat else domain.opening_actors(parsed_declarer))
    )
    history = tuple(prefix)
    if any(not isinstance(item, EscalationGraphActionV1) for item in history):
        raise TypeError("escalation prefix must use EscalationGraphActionV1")
    if native_cp_sat:
        state = native_state(history)
    else:
        state = domain.initial_state()
        for item in history:
            state = domain.transition(
                state,
                declarer=parsed_declarer,
                contract=parsed_contract,
                match_score=match_score,
                actor=item.actor,
                action=item.action,
                mode=item.mode,
            ).state

    if native_cp_sat:
        if acting_seat is None:
            root_actors = native_result(history).observation.eligible_seats
        if not root_actors:
            raise ValueError("escalation decision root requires at least two legal actions")

    def terminal_node(
        current: EscalationStateV1,
        actions: tuple[EscalationGraphActionV1, ...],
        *,
        declined: bool,
        depth: int,
        private_history: tuple[tuple[str, str], ...] = (),
    ) -> str:
        core = {
            "stage": "escalation",
            "domain": domain.sha256,
            **_escalation_state_core(state=current, actions=actions, actors=()),
            "declined": declined,
            **({"privateHistory": private_history} if private_history else {}),
            **({"decisionDepth": depth} if orders is not None else {}),
            **({"nativeTransitionEvidence": native_result(actions, closed=declined).evidence_hash} if native_cp_sat else {}),
        }
        return builder.add(
            ProofGraphNodeV1.leaf(
                _node_id("escalation", "terminal", core),
                _strict_utility(
                    terminal_utility(
                        EscalationGraphTerminalV1(actions, current, declined,
                            decision_depth=depth if orders is not None else None,
                            private_history=private_history)
                    )
                ),
            ),
            depth=depth,
        )

    def expand(
        current: EscalationStateV1,
        actions: tuple[EscalationGraphActionV1, ...],
        actors: tuple[Seat, ...],
        depth: int,
        non_announcing: tuple[Seat, ...] = (),
        private_history: tuple[tuple[str, str], ...] = (),
    ) -> str:
        builder.check(depth)
        public_actors = actors
        window_order = None
        if orders is not None:
            public_actors = native_result(actions).observation.eligible_seats
            window_key = _escalation_window_key(actions)
            window_order = (tuple(actor for actor in frozen_order if actor in public_actors)
                if frozen_order is not None else
                orders.get(window_key, public_actors if len(public_actors) == 1 else None))
            if window_order is None or set(window_order) != set(public_actors):
                raise ValueError("opportunity order must cover exactly the eligible window actors")
            if tuple(non_announcing) != window_order[:len(non_announcing)]:
                raise ValueError("private declines must be the leading opportunity order prefix")
            opportunity = native_result(actions, non_announcing=non_announcing)
            remaining = {Seat.parse(value) for value in opportunity.opportunity_state["remainingEligibleSeats"]}
            if not remaining:
                if depth == 0:
                    raise ValueError("escalation root has no remaining opportunity")
                return terminal_node(current, actions, declined=True, depth=depth, private_history=private_history)
            actors = (next(actor for actor in window_order if actor in remaining),)
            if depth == 0 and acting_seat is not None and actors[0] != Seat.parse(acting_seat):
                raise ValueError("acting seat disagrees with opportunity order")
        core = {
            "stage": "escalation",
            "domain": domain.sha256,
            "declarer": parsed_declarer.name.lower(),
            "contract": parsed_contract.value,
            "matchScore": list(match_score),
            **_escalation_state_core(state=current, actions=actions, actors=public_actors),
            **({"nativeTransitionEvidence": native_result(actions).evidence_hash} if native_cp_sat else {}),
        }
        public_core = dict(core)
        if orders is not None:
            core.update({"opportunityOrder": [actor.name.lower() for actor in window_order],
                         "decisionDepth": depth,
                         "nonAnnouncingSeats": [actor.name.lower() for actor in non_announcing],
                         "opportunityEvidence": opportunity.evidence_hash})
            if private_history:
                core["privateHistory"] = private_history
        if bounded_horizon is not None and depth >= bounded_horizon:
            assert boundary_utility is not None
            return builder.add(
                ProofGraphNodeV1.leaf(
                    _node_id("escalation", "boundary", core),
                    _strict_utility(
                        boundary_utility(
                            EscalationGraphStateV1(actions, current, actors)
                        )
                    ),
                    kind="boundary",
                ),
                depth=depth,
            )
        choices = choices_for(current, actions, actors)
        teams = {item.actor.team for item, _next in choices}
        if len(teams) > 1:
            raise AssertionError(
                "escalation Domain IR exposed opposing actors together"
            )
        actor_team = next(iter(teams), actors[0].team)
        actor_seat = actors[0] if len(actors) == 1 else None
        descriptors: list[
            tuple[str, EscalationGraphActionV1 | None, EscalationStateV1]
        ] = []
        can_close = (native_result(actions, closed=True).observation.terminal
                     if native_cp_sat else domain.can_close(current))
        if can_close:
            descriptors.append(
                (domain.closure_action, None, current)
            )
        for action, next_state in choices:
            base = (
                action.action
                if action.mode is None
                else f"{action.action}:{action.mode}"
            )
            code = base if len(actors) == 1 else f"{action.actor.name.lower()}:{base}"
            descriptors.append((code, action, next_state))
        ordered = list(enumerate(sorted(descriptors, key=lambda item: item[0])))
        if depth == 0 and len(ordered) < (1 if native_cp_sat else 2):
            raise ValueError(
                "escalation decision root requires at least two legal actions"
            )
        if depth == 0 and root_action_code is not None:
            ordered = [item for item in ordered if item[1][0] == root_action_code]
            if len(ordered) != 1:
                raise ValueError("escalation witness selected an illegal root action")
        if (
            len(ordered) == 1
            and ordered[0][1][1] is None
            and not (depth == 0 and root_action_code is not None)
            and not (depth == 0 and native_cp_sat)
        ):
            return terminal_node(
                current,
                actions,
                declined=True,
                depth=depth + 1,
                private_history=private_history,
            )
        node_id = _node_id("escalation", "decision", core)
        _observe_decision(
            decision_observer,
            node_id=node_id,
            stage="escalation",
            actor=actor_seat,
            public_state=public_core,
            own_private_windows=tuple(window for seat, window in private_history if seat == actor_seat.name.lower()),
        )
        if continuation_action_selector is not None and depth > 0:
            if actor_seat is None:
                raise ValueError(
                    "escalation continuation policy requires one acting seat"
                )
            selected_id = continuation_action_selector(
                actor_seat,
                node_id,
                tuple((index, item[0]) for index, item in ordered),
            )
            ordered = [item for item in ordered if item[0] == selected_id]
            if len(ordered) != 1:
                raise ValueError(
                    "escalation continuation policy selected an illegal action"
                )

        def build_edge(
            row: tuple[
                int,
                tuple[str, EscalationGraphActionV1 | None, EscalationStateV1],
            ],
        ) -> ProofGraphEdgeV1:
            index, (code, action, next_state) = row
            if action is None:
                if orders is not None:
                    child = expand(current, actions, public_actors, depth + 1,
                                   (*non_announcing, actor_seat),
                                   (*private_history, (actor_seat.name.lower(), _escalation_window_key(actions))))
                else:
                    child = terminal_node(
                        current, actions, declined=True, depth=depth + 1
                    )
            else:
                next_actions = (*actions, action)
                following = choices_for(next_state, next_actions)
                if following:
                    next_actors = tuple(
                        sorted({item.actor for item, _state in following}, key=int)
                    )
                    child = expand(
                        next_state, next_actions, next_actors, depth + 1,
                        private_history=private_history,
                    )
                else:
                    child = terminal_node(
                        next_state,
                        next_actions,
                        declined=False,
                        depth=depth + 1,
                        private_history=private_history,
                    )
            return ProofGraphEdgeV1.build(index, code, child)

        if depth == 0 and root_worker_count > 1 and len(ordered) > 1:
            with ThreadPoolExecutor(
                max_workers=min(root_worker_count, len(ordered)),
                thread_name_prefix="balootx-escalation-root",
            ) as pool:
                edges = list(pool.map(build_edge, ordered))
        else:
            edges = [build_edge(row) for row in ordered]
        return builder.add(
            ProofGraphNodeV1.decision(
                node_id,
                _team_name(actor_team),
                edges,
                actor_seat=(
                    actor_seat.name.lower() if actor_seat is not None else None
                ),
            ),
            depth=depth,
        )

    recall = normalize_escalation_recall_v1(prior_window_recall)
    if recall:
        if orders is None:
            raise ValueError("prior-window recall requires native opportunity orders")
        windows = {_escalation_window_key(history[:index]): index for index in range(len(history))}
        for remembered in recall.values():
            if any(window not in windows for window in remembered):
                raise ValueError("recall must reference an earlier public announcement window")
            if list(remembered) != sorted(remembered, key=windows.__getitem__):
                raise ValueError("prior-window recall must be chronological")
    private_history = tuple((seat, window) for seat, remembered in recall.items() for window in remembered)
    root = expand(state, history, root_actors, 0, prior_declines,
                  (*private_history, *((seat.name.lower(), _escalation_window_key(history)) for seat in prior_declines)))
    return ProofGraphV1.build(
        scope=normalized_scope,
        root_node_id=root,
        nodes=tuple(builder.nodes.values()),
        boundary_evaluator=boundary,
    )


def project_graph_observation_rows_v1(projects: Sequence[Project]) -> list[dict[str, Any]]:
    """Serialize graph project evidence without evaluating declaration rules."""
    return [
        {
            "owner": item.owner.name.lower(), "kind": item.kind,
            "cards": [card.id for card in item.cards], "value": item.value,
            "highRank": item.high_rank, "categoryPriority": item.category_priority,
            "declared": item.declared, "declarationOrder": item.declaration_order,
            "scoring": item.scoring,
        }
        for item in projects
    ]


def compile_project_declaration_graph_v1(
    domain: ProjectDomainIRV1,
    *,
    projects: Sequence[Project],
    actor: Seat | str | int,
    dealer: Seat | str | int | None = None,
    prefix: Sequence[ProjectDeclarationActionV1 | Mapping[str, Any] | str] = (),
    trump_suit: Suit | str | int | None,
    terminal_utility: Callable[[ProjectDeclarationTerminalV1], int],
    scope: str = "exact",
    horizon: int | None = None,
    boundary_evaluator: str | None = None,
    boundary_utility: Callable[[ProjectDeclarationTerminalV1], int] | None = None,
    budget: NonPlayGraphBudgetV1 = NonPlayGraphBudgetV1(),
    cancel_check: Callable[[], bool] | None = None,
    continuation_action_selector: Callable[
        [Seat, str, Sequence[tuple[int, str]]], int
    ]
    | None = None,
    root_action_code: str | None = None,
    decision_observer: NonPlayDecisionObserverV1 | None = None,
) -> ProofGraphV1:
    """Compile a final first-trick declaration decision from Project Domain IR."""

    if not isinstance(domain, ProjectDomainIRV1):
        raise TypeError("project graph requires ProjectDomainIRV1")
    if not isinstance(budget, NonPlayGraphBudgetV1):
        raise TypeError("project graph requires NonPlayGraphBudgetV1")
    normalized_scope, _bounded_horizon, boundary = _graph_options(
        scope=scope,
        horizon=horizon,
        boundary_evaluator=boundary_evaluator,
        boundary_utility=boundary_utility,
    )
    if any(not isinstance(item, Project) for item in projects):
        raise TypeError("project graph requires canonical Project values")
    parsed_actor = Seat.parse(actor)
    parsed_dealer = (
        Seat.parse(dealer) if dealer is not None else parsed_actor.previous()
    )
    parsed_trump = Suit.parse(trump_suit) if trump_suit is not None else None
    state = domain.declaration_state(tuple(projects), dealer=parsed_dealer)
    for action in prefix:
        state, _announced = domain.apply_declaration_action(
            state,
            action,
            trump_suit=parsed_trump,
        )
    if state.current_actor != parsed_actor:
        raise ValueError("project graph actor disagrees with its declaration prefix")
    legal = domain.legal_declaration_actions(state, trump_suit=parsed_trump)
    if not legal:
        raise ValueError("project graph requires an active declaration decision")
    builder = _GraphBuilderV1(budget, cancel_check)
    core = {
        "stage": "project",
        "domain": domain.sha256,
        "declarationMode": domain.declaration_mode,
        "dealer": parsed_dealer.name.lower(),
        "actor": parsed_actor.name.lower(),
        "prefix": [item.to_dict() for item in state.actions],
        "projects": project_graph_observation_rows_v1(projects),
    }
    def terminal_node(
        resolved: ProjectDeclarationStateV1,
        *,
        root_actor: Seat,
        root_code: str,
        announced: Project | None,
        depth: int,
        at_boundary: bool = False,
    ) -> str:
        outcome = ProjectDeclarationTerminalV1(
            root_actor,
            root_code,
            tuple(resolved.projects),
            announced,
            tuple(resolved.actions),
        )
        terminal_id = _node_id(
            "project",
            "terminal",
            {**core, "actions": [item.to_dict() for item in resolved.actions]},
        )
        utility = boundary_utility if at_boundary else terminal_utility
        assert utility is not None
        return builder.add(
            ProofGraphNodeV1.leaf(
                terminal_id,
                _strict_utility(utility(outcome)),
                kind="boundary" if at_boundary else "terminal",
            ),
            depth=depth,
        )

    def continue_policy(
        current: ProjectDeclarationStateV1,
        *,
        root_actor: Seat,
        root_code: str,
        announced: Project | None,
        depth: int,
    ) -> str:
        if current.complete:
            return terminal_node(
                current,
                root_actor=root_actor,
                root_code=root_code,
                announced=announced,
                depth=depth,
            )
        if continuation_action_selector is None:
            raise ValueError(
                "project graph requires later declaration/play interleaving support"
            )
        actor = current.current_actor
        assert actor is not None
        choices = domain.legal_declaration_actions(current, trump_suit=parsed_trump)
        node_id = _node_id(
            "project",
            "decision",
            {**core, "actions": [item.to_dict() for item in current.actions]},
        )
        _observe_decision(
            decision_observer,
            node_id=node_id,
            stage="project",
            actor=actor,
            public_state={**core, "actions": [item.to_dict() for item in current.actions]},
        )
        selected_id = continuation_action_selector(
            actor,
            node_id,
            tuple((index, item.code) for index, item in enumerate(choices)),
        )
        selected = [
            (index, item)
            for index, item in enumerate(choices)
            if index == selected_id
        ]
        if len(selected) != 1:
            raise ValueError("project continuation policy selected an illegal action")
        action_id, action = selected[0]
        updated, selected_project = domain.apply_declaration_action(
            current,
            action,
            trump_suit=parsed_trump,
        )
        child = continue_policy(
            updated,
            root_actor=root_actor,
            root_code=root_code,
            announced=announced or selected_project,
            depth=depth + 1,
        )
        return builder.add(
            ProofGraphNodeV1.decision(
                node_id,
                _team_name(actor.team),
                (ProofGraphEdgeV1.build(action_id, action.code, child),),
                actor_seat=actor.name.lower(),
            ),
            depth=depth,
        )

    edges = []
    for action_id, action in enumerate(legal):
        if root_action_code is not None and action.code != root_action_code:
            continue
        updated, announced = domain.apply_declaration_action(
            state,
            action,
            trump_suit=parsed_trump,
        )
        at_boundary = normalized_scope == "bounded" and _bounded_horizon == 1
        if at_boundary:
            child = terminal_node(
                updated,
                root_actor=parsed_actor,
                root_code=action.code,
                announced=announced,
                depth=1,
                at_boundary=True,
            )
        else:
            child = continue_policy(
                updated,
                root_actor=parsed_actor,
                root_code=action.code,
                announced=announced,
                depth=1,
            )
        edges.append(
            ProofGraphEdgeV1.build(action_id, action.code, child)
        )
    if root_action_code is not None and len(edges) != 1:
        raise ValueError("project witness selected an illegal root action")
    root_id = _node_id("project", "decision", core)
    _observe_decision(
        decision_observer,
        node_id=root_id,
        stage="project",
        actor=parsed_actor,
        public_state=core,
    )
    builder.add(
        ProofGraphNodeV1.decision(
            root_id,
            _team_name(parsed_actor.team),
            tuple(edges),
            actor_seat=parsed_actor.name.lower(),
        ),
        depth=0,
    )
    return ProofGraphV1.build(
        scope=normalized_scope,
        root_node_id=root_id,
        nodes=tuple(builder.nodes.values()),
        boundary_evaluator=boundary,
    )


__all__ = [
    "AuctionGraphChoiceV1",
    "AuctionGraphStateV1",
    "AuctionGraphTerminalV1",
    "EscalationGraphActionV1",
    "EscalationGraphStateV1",
    "EscalationGraphTerminalV1",
    "NonPlayGraphBudgetV1",
    "NonPlayGraphCompilationStoppedV1",
    "NonPlayDecisionObservationV1",
    "NonPlayDecisionObserverV1",
    "ProjectDeclarationTerminalV1",
    "compile_auction_decision_graph_v1",
    "compile_escalation_decision_graph_v1",
    "enumerate_escalation_opportunity_orders_v1",
    "compile_project_declaration_graph_v1",
]
