"""Engine-free CP-SAT normalization for Research V4 exact trace prefixes."""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass, field as dataclass_field, replace
from importlib.metadata import version
from typing import Any, Callable, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1, DealPacketV1
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.enums import Contract, Rank, Seat, Suit, Team
from baloot_platform.core.exact_state_view import (
    build_canonical_observer_information_v1,
)
from baloot_platform.core.models import LegalPlayActionV1, Project
from baloot_platform.core.project_ir import (
    ProjectDeclarationActionV1,
    ProjectDomainIRV1,
    project_id_v1,
)
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.project_visibility import project_rank_visibility_policy
from baloot_platform.core.scoring_ir import ScoringDomainIRV1

from .ortools_escalation_stage import (
    OrToolsEscalationObservationV1,
    OrToolsEscalationStageV1,
)
from .ortools_play_tail import (
    OrToolsPlayTailBuildV1,
    OrToolsPlayTailResultV1,
    OrToolsPlayTailStageV1,
    PlayTailSolveStopped,
    TailActionConstraintV1,
    TailPrefixActionV1,
)
from .ortools_project_fragment import (
    OrToolsProjectLifecycleFragmentV1,
    attach_project_lifecycle_fragment_v1,
)
from .ortools_project_model import (
    OrToolsProjectLifecycleModelV1,
    OrToolsProjectLifecycleResultV1,
    ProjectLifecycleEvidenceV1,
)
from .ortools_scoring_stage import (
    CpSatScoringInputsV1,
    OrToolsScoringFragmentV1,
    OrToolsScoringResultV1,
    attach_scoring_fragment_v1,
    extract_scoring_evidence_v1,
    scoring_contract_id_v1,
    scoring_level_id_v1,
)
from .ortools_scenario_model import (
    OrToolsAuctionActionV1,
    OrToolsScenarioAuctionFragmentV1,
    OrToolsScenarioAuctionObservationV1,
    OrToolsScenarioDealSolutionV1,
    OrToolsScenarioModelV1,
)
from .scenario_contracts import ExactTracePrefixV1, canonical_sha256
from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1


_BACKEND_ID_V2 = "ortools_cp_sat_v2"
_BACKEND_ID_V3 = "ortools_cp_sat_v3"
_NORMALIZER_ID = "ortools_exact_trace_prefix_v1"
_ORTOOLS_VERSION = version("ortools")
_CONTRACT_IDS = {
    Contract.SUN: 0,
    Contract.HOKUM: 1,
    Contract.ASHKAL: 2,
}
_NO_TRUMP = len(Suit)


class ProjectDecisionNotStrategicError(ValueError):
    """An optional declaration point has no genuine pass/declare choice."""

    code = "project_decision_not_strategic"

    def __init__(self) -> None:
        super().__init__(self.code)


class ExactNormalizationStopped(RuntimeError):
    """Resource exhaustion is incomplete normalization, not invalid input."""

    def __init__(self, reason: str, *, evidence: Mapping[str, Any] | None = None) -> None:
        core = {
            "schemaVersion": 1, "status": "incomplete", "stage": "exactNormalization",
            "exhaustionReason": reason, "evidence": dict(evidence or {}),
        }
        self.partial_receipt = {**core, "sha256": canonical_sha256(core)}
        super().__init__(
            "exact trace normalization cancelled" if reason == "cancelled"
            else "exact trace normalization was not completed within budget"
        )


def _normalization_clock(
    maximum_seconds: float, cancel_check: Callable[[], bool] | None,
) -> Callable[[], float]:
    if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("exact normalization maximum_seconds must be finite and positive")
    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("exact normalization cancellation check must be callable")
    deadline = time.monotonic() + maximum_seconds

    def remaining() -> float:
        if cancel_check is not None and cancel_check():
            raise ExactNormalizationStopped("cancelled")
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise ExactNormalizationStopped("maximum_seconds")
        # Floating-point deadline arithmetic can round the remainder above its cap.
        return min(maximum_seconds, seconds)

    return remaining


def _cp_model_hash(model: cp_model.CpModel, *, authority: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schemaVersion": 1,
            "authority": dict(authority),
            "cpModelProtoHash": hashlib.sha256(
                str(model.proto).encode("utf-8")
            ).hexdigest(),
        }
    )


def _deterministic_solver(*, seed: int, maximum_seconds: float) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed & 0x7FFFFFFF
    solver.parameters.max_time_in_seconds = float(maximum_seconds)
    return solver


def _bool_or(
    model: cp_model.CpModel,
    values: Sequence[cp_model.IntVar],
    name: str,
) -> cp_model.IntVar:
    if not values:
        return model.new_constant(0)
    result = model.new_bool_var(name)
    model.add_max_equality(result, list(values))
    return result


def _bool_and(
    model: cp_model.CpModel,
    values: Sequence[cp_model.IntVar],
    name: str,
) -> cp_model.IntVar:
    if not values:
        return model.new_constant(1)
    # Fold a Boolean contradiction, not domain truth: x AND NOT x is false.
    indexes = {value.index for value in values}
    if any(-index - 1 in indexes for index in indexes):
        return model.new_constant(0)
    result = model.new_bool_var(name)
    model.add_min_equality(result, list(values))
    return result


def _equals(
    model: cp_model.CpModel,
    variable: cp_model.IntVar,
    value: int,
    name: str,
) -> cp_model.IntVar:
    result = model.new_bool_var(name)
    model.add(variable == value).only_enforce_if(result)
    model.add(variable != value).only_enforce_if(result.Not())
    return result


@dataclass(frozen=True, slots=True)
class _NativeRoleProjectionV1:
    caller: Seat
    buyer: Seat
    contract_owner: Seat
    purchase_recipient: Seat
    declarer: Seat
    first_leader: Seat
    model_hash: str
    evidence_hash: str
    uniqueness_model_hash: str
    uniqueness_solver_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "caller": self.caller.name.lower(),
            "buyer": self.buyer.name.lower(),
            "contractOwner": self.contract_owner.name.lower(),
            "purchaseRecipient": self.purchase_recipient.name.lower(),
            "declarer": self.declarer.name.lower(),
            "firstLeader": self.first_leader.name.lower(),
            "modelHash": self.model_hash,
            "evidenceHash": self.evidence_hash,
            "uniquenessModelHash": self.uniqueness_model_hash,
            "uniquenessSolverStatus": self.uniqueness_solver_status,
        }


def _native_role_projection(
    *,
    rules: RuleBook,
    dealer: Seat,
    contract: Contract,
    caller: Seat,
    buyer: Seat,
    purchase_recipient: Seat,
    declarer: Seat,
    seed: int,
    maximum_seconds: float,
    remaining_seconds: Callable[[], float] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> _NativeRoleProjectionV1:
    """Bind purchased-round roles and first leader in one unique CP-SAT view."""

    remaining = remaining_seconds or _normalization_clock(maximum_seconds, cancel_check)
    remaining()
    model = cp_model.CpModel()
    dealer_var = model.new_int_var(0, 3, "role_dealer")
    caller_var = model.new_int_var(0, 3, "role_caller")
    buyer_var = model.new_int_var(0, 3, "role_buyer")
    recipient_var = model.new_int_var(0, 3, "role_purchase_recipient")
    owner_var = model.new_int_var(0, 3, "role_contract_owner")
    declarer_var = model.new_int_var(0, 3, "role_declarer")
    leader_var = model.new_int_var(0, 3, "role_first_leader")
    model.add(dealer_var == int(dealer))
    model.add(caller_var == int(caller))
    model.add(buyer_var == int(buyer))
    model.add(recipient_var == int(purchase_recipient))
    model.add(declarer_var == int(declarer))
    model.add(owner_var == buyer_var)
    auction_domain = AuctionDomainIRV1.from_rules(rules.data)
    if contract == Contract.ASHKAL:
        if auction_domain.ashkal_contract_owner == "bidder":
            model.add(buyer_var == caller_var)
        else:
            model.add(buyer_var == recipient_var)
        if auction_domain.ashkal_taker == "bidder_partner":
            model.add_allowed_assignments(
                [caller_var, recipient_var],
                [(int(seat), int(seat.partner)) for seat in Seat],
            )
        else:
            model.add(recipient_var == caller_var)
        model.add(declarer_var == buyer_var)
    else:
        model.add(caller_var == buyer_var)
        model.add(recipient_var == caller_var)
        model.add(declarer_var == buyer_var)

    play_domain = rules.play_domain.require_integrity(rules.structural_domain)
    leader_rows: list[tuple[int, int, int]] = []
    for possible_dealer in Seat:
        for possible_buyer in Seat:
            if play_domain.first_leader == "dealer_right":
                leader = rules.structural_domain.dealer_right(possible_dealer)
            elif play_domain.first_leader == "buyer":
                leader = possible_buyer
            elif play_domain.first_leader == "configured":
                leader = play_domain.configured_first_leader
            else:
                raise ValueError(
                    "active first-leader rule lacks native CP-SAT exact binding"
                )
            leader_rows.append((int(possible_dealer), int(possible_buyer), int(leader)))
    model.add_allowed_assignments(
        [dealer_var, buyer_var, leader_var],
        leader_rows,
    )
    authority = {
        "stage": "exact_role_projection",
        "rulesHash": rules.sha256,
        "structuralDomainIrHash": rules.structural_domain.sha256,
        "playDomainIrHash": play_domain.sha256,
        "auctionDomainIrHash": auction_domain.sha256,
        "dealer": dealer.name.lower(),
        "contract": contract.value,
        "caller": caller.name.lower(),
        "buyer": buyer.name.lower(),
        "purchaseRecipient": purchase_recipient.name.lower(),
        "declarer": declarer.name.lower(),
    }
    model_hash = _cp_model_hash(model, authority=authority)
    solver = _deterministic_solver(seed=seed, maximum_seconds=remaining())
    status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
    remaining()
    if status != cp_model.OPTIMAL:
        if status == cp_model.INFEASIBLE:
            raise AssertionError("native CP-SAT exact role projection is infeasible")
        raise ExactNormalizationStopped(
            solver.status_name(status).lower(), evidence={"sourceStage": "role_projection"}
        )
    variables = (
        caller_var,
        buyer_var,
        owner_var,
        recipient_var,
        declarer_var,
        leader_var,
    )
    values = tuple(int(solver.value(item)) for item in variables)
    evidence_core = {
        "schemaVersion": 1,
        "modelHash": model_hash,
        "values": list(values),
    }
    model.add_forbidden_assignments(list(variables), [list(values)])
    uniqueness_model_hash = _cp_model_hash(
        model,
        authority={**authority, "purpose": "semantic_uniqueness"},
    )
    uniqueness_solver = _deterministic_solver(
        seed=seed,
        maximum_seconds=remaining(),
    )
    uniqueness_status = _solve_cp_sat_with_cancel_v1(uniqueness_solver, model, cancel_check)
    remaining()
    if uniqueness_status == cp_model.OPTIMAL:
        raise AssertionError("exact role projection is semantically ambiguous")
    if uniqueness_status != cp_model.INFEASIBLE:
        raise ExactNormalizationStopped(
            uniqueness_solver.status_name(uniqueness_status).lower(),
            evidence={"sourceStage": "role_semantic_uniqueness",
                      "modelHash": uniqueness_model_hash, "sourceModelHash": model_hash},
        )
    return _NativeRoleProjectionV1(
        caller=Seat(values[0]),
        buyer=Seat(values[1]),
        contract_owner=Seat(values[2]),
        purchase_recipient=Seat(values[3]),
        declarer=Seat(values[4]),
        first_leader=Seat(values[5]),
        model_hash=model_hash,
        evidence_hash=canonical_sha256(evidence_core),
        uniqueness_model_hash=uniqueness_model_hash,
        uniqueness_solver_status=uniqueness_solver.status_name(
            uniqueness_status
        ).lower(),
    )


def _mapping_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ValueError(f"exact-state assertion path does not exist: {path}")
        current = current[component]
    return current


def _effective_prefix_hash(spec: ExactTracePrefixV1) -> str:
    """Hash solver-effective input, excluding optional validation assertions."""

    core = {
            "schemaVersion": 1,
            "deckOrder": list(spec.deck_order),
            "dealer": spec.dealer,
            "roundNumber": spec.round_number,
            "matchScore": list(spec.match_score),
            "auctionActions": [
                {
                    "action": str(action["action"]).strip(),
                    "trumpSuit": (
                        Suit.parse(str(action["trumpSuit"])).name.lower()
                        if action.get("trumpSuit") is not None
                        else None
                    ),
                }
                for action in spec.auction_actions
            ],
            "escalationActions": [
                {
                    "actor": Seat.parse(str(action["actor"])).name.lower(),
                    "action": str(action["action"]).strip(),
                    "mode": (
                        str(action.get("mode") or action.get("requestedMode"))
                        if action.get("mode") is not None
                        or action.get("requestedMode") is not None
                        else None
                    ),
                }
                for action in spec.escalation_actions
            ],
            "playActions": [
                {
                    "card": Card.parse(action["card"]).id,
                    "ekka": action.get("ekka") is True,
                }
                for action in spec.play_actions
            ],
        }
    if spec.project_actions_present:
        core["projectActions"] = [dict(action) for action in spec.project_actions]
    return canonical_sha256(core)


def _card_rows(
    deck: Sequence[int], positions: Sequence[int]
) -> list[dict[str, object]]:
    return [Card(deck[position]).to_dict() for position in positions]


def _hand_rows(
    deck: Sequence[int], positions_by_seat: Sequence[Sequence[int]]
) -> dict[str, list[dict[str, object]]]:
    return {
        seat.name.lower(): [
            Card(card_id).to_dict()
            for card_id in sorted(
                deck[position] for position in positions_by_seat[int(seat)]
            )
        ]
        for seat in Seat
    }


def _deal_row(
    deck: Sequence[int], packet: DealPacketV1, sequence: int
) -> dict[str, Any]:
    cards = _card_rows(deck, packet.positions)
    return {
        "sequence": sequence,
        "stage": packet.stage,
        "packet": packet.packet,
        "seat": packet.seat.name.lower(),
        "count": len(cards),
        "faceUp": False,
        "cards": cards,
    }


def _bid_rows(
    actions: Sequence[OrToolsAuctionActionV1],
) -> list[dict[str, Any]]:
    turns = {1: 0, 2: 0}
    rows: list[dict[str, Any]] = []
    for item in actions:
        turn = turns[item.round_index]
        turns[item.round_index] += 1
        row = {
                "round": item.round_index,
                "turn": turn,
                "seat": item.actor.name.lower(),
                "action": item.action,
                "contract": item.contract.value if item.contract is not None else None,
                "trumpSuit": (
                    item.trump.name.lower() if item.trump is not None else None
                ),
                "bidder": (
                    item.caller.name.lower()
                    if item.caller is not None
                    else None
                ),
                "buyer": (item.buyer.name.lower() if item.buyer is not None else None),
            }
        if (
            item.purchase_recipient is not None
            and item.purchase_recipient != item.buyer
        ):
            row["purchaseRecipient"] = item.purchase_recipient.name.lower()
        rows.append(row)
    return rows


def _legal_bid_rows(
    observation: OrToolsScenarioAuctionObservationV1,
    rules: RuleBook,
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    domain = AuctionDomainIRV1.from_rules(rules.data)
    governed_order = (
        domain.first_round_allowed
        if observation.current_round == 1
        else domain.second_round_allowed
    )
    legal_choices = set(observation.legal_next_choices)
    for action in governed_order:
        if action not in observation.legal_next_actions:
            continue
        if action != "hokum":
            rows.append({"code": action, "action": action, "trumpSuit": None})
            continue
        for suit in Suit:
            choice = f"hokum:{suit.name.lower()}"
            if choice in legal_choices:
                rows.append(
                    {
                        "code": choice,
                        "action": "hokum",
                        "trumpSuit": suit.name.lower(),
                    }
                )
    return tuple(rows)


def _validate_supported_slice(spec: ExactTracePrefixV1, rules: RuleBook) -> None:
    target = int(rules.data["scoring"]["match"]["target"])
    if max(spec.match_score) >= target and spec.match_score[0] != spec.match_score[1]:
        raise ValueError("exact trace matchScore belongs to a completed match")


def _apply_escalation(
    state: dict[str, Any],
    observation: OrToolsEscalationObservationV1,
) -> None:
    history: list[dict[str, Any]] = []
    events = state["events"]
    for item in observation.actions:
        record = {
            "seat": item.actor.name.lower(),
            "team": item.actor.team.name.lower(),
            "action": item.action,
            "mode": item.resolved_mode,
        }
        history.append(record)
        events.append(
            {
                "sequence": len(events),
                "type": "escalation_announced",
                **record,
            }
        )
    state["escalation"] = observation.level
    state["escalationHistory"] = history
    state["locked"] = observation.mode == "locked"


def _legal_escalation_rows(
    observation: OrToolsEscalationObservationV1,
    *,
    actor: Seat,
    domain: EscalationDomainIRV1,
    include_actor: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    choice_actions = {choice.action for choice in observation.legal_next_choices}
    if (
        domain.closure_action in observation.legal_next_actions
        and domain.closure_action not in choice_actions
    ):
        rows.append(
            {
                "code": domain.closure_action,
                **({"actor": actor.name.lower()} if include_actor else {}),
                "action": domain.closure_action,
                "mode": None,
            }
        )
    choices = {
        (choice.action, choice.mode)
        for choice in observation.legal_next_choices
        if choice.actor == actor
    }
    for action in domain.supported_actions:
        modes: tuple[str | None, ...] = (
            ("open", "locked") if action in {"double", "four"} else (None,)
        )
        for mode in modes:
            if (action, mode) not in choices:
                continue
            code = action if mode is None else f"{action}:{mode}"
            rows.append(
                {
                    "code": code,
                    **({"actor": actor.name.lower()} if include_actor else {}),
                    "action": action,
                    "mode": mode,
                }
            )
    return tuple(rows)


def _derive_play_prefix(
    spec: ExactTracePrefixV1,
    *,
    first_leader: Seat,
    contract: Contract,
    trump: Suit | None,
    rules: RuleBook,
) -> tuple[TailPrefixActionV1, ...]:
    actions: list[TailPrefixActionV1] = []
    leader = first_leader
    trick: list[TailPrefixActionV1] = []
    for index, raw in enumerate(spec.play_actions):
        expected_actor = rules.structural_domain.offset(leader, index % 4)
        asserted_actor = raw.get("actor")
        if (
            asserted_actor is not None
            and Seat.parse(str(asserted_actor)) != expected_actor
        ):
            raise ValueError(f"exact play action {index} actor mismatch")
        action = TailPrefixActionV1.build(
            expected_actor,
            raw["card"],
            ekka=raw.get("ekka") is True,
        )
        actions.append(action)
        trick.append(action)
        if len(trick) != 4:
            continue
        led_suit = trick[0].card.suit
        eligible = trick
        if (
            contract.effective == Contract.HOKUM
            and trump is not None
            and any(item.card.suit == trump for item in trick)
        ):
            eligible = [item for item in trick if item.card.suit == trump]
        else:
            eligible = [item for item in trick if item.card.suit == led_suit]
        leader = max(
            eligible,
            key=lambda item: rules.play_domain.strength(
                item.card,
                contract,
                trump,
            ),
        ).actor
        trick = []
    return tuple(actions)


@dataclass(frozen=True, slots=True)
class _NativePlayNormalizationV1:
    result: OrToolsPlayTailResultV1
    legal_play_actions: tuple[TailPrefixActionV1, ...]
    semantic_vector_hash: str
    uniqueness_model_hash: str
    uniqueness_solver_status: str
    project_result: OrToolsProjectLifecycleResultV1
    project_projection: "_NativeProjectPrefixProjectionV1"
    legal_project_actions: tuple[ProjectDeclarationActionV1, ...]
    scoring_result: OrToolsScoringResultV1 | None
    construction_hint_model_hash: str | None
    construction_hint_solution_hash: str | None
    construction_hint_status: str
    play_template: tuple[str, str, OrToolsPlayTailBuildV1, OrToolsProjectLifecycleFragmentV1] | None
    witness_raw_margins: tuple[int, int] | None = None
    forced_project_passes: tuple[Seat, ...] = ()
    conditional_forced_project_passes: tuple[Seat, ...] | None = None

    def evidence_hash(self) -> str:
        return canonical_sha256({
            "normalization": self.to_dict(), "play": self.result.to_dict(),
            "projects": self.project_result.to_dict(),
            "projection": self.project_projection.to_dict(),
            "legalProjects": [action.to_dict() for action in self.legal_project_actions],
            "scoring": self.scoring_result.to_dict() if self.scoring_result is not None else None,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "legalPlayActions": [
                action.to_dict() for action in self.legal_play_actions
            ],
            "semanticVectorHash": self.semantic_vector_hash,
            "uniquenessModelHash": self.uniqueness_model_hash,
            "uniquenessSolverStatus": self.uniqueness_solver_status,
            "constructionHintModelHash": self.construction_hint_model_hash,
            "constructionHintSolutionHash": self.construction_hint_solution_hash,
            "constructionHintStatus": self.construction_hint_status,
            "forcedProjectPasses": {
                "modelHash": self.result.model_hash,
                "seats": [seat.name.lower() for seat in self.forced_project_passes],
            },
            **({"conditionalForcedProjectPasses": {
                "modelHash": self.result.model_hash,
                "seats": [seat.name.lower() for seat in self.conditional_forced_project_passes],
                "keepAllFeasibleSolutionsInPresolve": True,
                "feasibilityOnly": True,
            }} if self.conditional_forced_project_passes is not None else {}),
            **({"witnessRawMargins": {
                "modelHash": self.result.model_hash,
                "variables": ["exact_witness_raw_margin_0", "exact_witness_raw_margin_1"],
                "values": list(self.witness_raw_margins),
            }} if self.witness_raw_margins is not None else {}),
        }


def _native_forced_pass_certificate(normalized):
    normalized.require_integrity()
    receipt = normalized.normalization_receipt.payload
    native = receipt.get("nativePlayNormalization") or {}
    certificate = native.get("conditionalForcedProjectPasses")
    if certificate is not None:
        if (certificate.get("keepAllFeasibleSolutionsInPresolve") is not True
                or certificate.get("feasibilityOnly") is not True):
            raise ValueError("conditional project certificate lacks feasibility preservation")
    else:
        certificate = native.get("forcedProjectPasses")
    if certificate is not None and certificate["modelHash"] != receipt["derivedPlay"]["modelHash"]:
        raise ValueError("forced project certificate ownership mismatch")
    return copy.deepcopy(certificate)


def _certified_project_followup_rows(normalized, prefix):
    """Read parent-certified play choices; never manufacture a successor state."""
    normalized.require_integrity()
    if (normalized.decision_kind != "project" or not prefix.project_actions_present
            or len(prefix.project_actions) != len(prefix.play_actions) + 1
            or not 0 <= len(prefix.play_actions) < 4):
        raise ValueError("project follow-up requires one certified declaration")
    parent = prefix.to_dict()
    parent.pop("sha256")
    chosen = parent["projectActions"].pop()
    if normalized.prefix_hash != _effective_prefix_hash(ExactTracePrefixV1.build(parent)):
        raise ValueError("project follow-up parent differs from declaration prefix")
    action = ProjectDeclarationActionV1.build(chosen, actor=normalized.next_actor)
    if not any(ProjectDeclarationActionV1.build(row["code"], actor=normalized.next_actor) == action
               for row in normalized.legal_actions):
        raise ValueError("project follow-up has an uncertified declaration")
    native = normalized.normalization_receipt.payload.get("nativePlayNormalization") or {}
    legal = native.get("legalPlayActions", ())
    if any(Seat.parse(row["actor"]).name.lower() != normalized.next_actor for row in legal):
        raise ValueError("project follow-up play belongs to another actor")
    return tuple({**row, "code": LegalPlayActionV1(Card.parse(row["card"]),
        row.get("ekka") is True).code} for row in legal)


def _native_fixed_play_prefix(
    *,
    spec: ExactTracePrefixV1,
    hands: Mapping[Seat, Sequence[Card]],
    contract: Contract,
    trump: Suit | None,
    first_leader: Seat,
    dealer: Seat,
    buyer: Seat,
    escalation: str,
    last_escalating_team: Team | None,
    starting_match_score: Sequence[int],
    locked: bool,
    rules: RuleBook,
    seed: int,
    maximum_seconds: float,
    remaining_seconds: Callable[[], float] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    play_template: tuple[str, str, OrToolsPlayTailBuildV1, OrToolsProjectLifecycleFragmentV1] | None = None,
    first_trick_templates: dict[int, tuple[str, str, OrToolsPlayTailBuildV1, OrToolsProjectLifecycleFragmentV1]] | None = None,
    paired_pass_cache: dict[str, tuple[str, _NativePlayNormalizationV1]] | None = None,
) -> _NativePlayNormalizationV1:
    """Solve play, projects, and terminal score in one native CP-SAT model."""

    remaining = remaining_seconds or _normalization_clock(maximum_seconds, cancel_check)
    remaining()
    paired_pass_cache = paired_pass_cache if paired_pass_cache is not None else {}
    prefix_length = len(spec.play_actions)
    if prefix_length > 32:
        raise ValueError("exact play prefix cannot exceed 32 actions")
    fixed_prefix = (
        tuple(
            TailPrefixActionV1(
                Seat.parse(str(raw["actor"])),
                Card.parse(raw["card"]),
                raw.get("ekka") is True,
            )
            for raw in spec.play_actions
        )
        if all(raw.get("actor") is not None for raw in spec.play_actions)
        else ()
    )
    action_constraints = (
        ()
        if fixed_prefix
        else tuple(
            TailActionConstraintV1.build(
                step=index,
                cards=(raw["card"],),
            )
            for index, raw in enumerate(spec.play_actions)
        )
    )
    ekka_leads = [False] * 8
    for index, raw in enumerate(spec.play_actions):
        declared = raw.get("ekka") is True
        if declared and index % 4 != 0:
            raise ValueError("Ekka may only be declared on a trick lead")
        if index % 4 == 0:
            ekka_leads[index // 4] = declared
    ownership = {card: seat for seat, hand in hands.items() for card in hand}
    # The play-tail model compiles complete tricks.  At a completed-trick
    # decision boundary it must include the following trick so next-actor and
    # legal-action variables exist; partial prefixes only finish their trick.
    play_step_count = min(32, max(4, ((prefix_length + 4) // 4) * 4))
    reusable_prefix = spec.project_actions_present and prefix_length < 4
    template_key = canonical_sha256({
        "rules": rules.sha256, "owners": [int(ownership[Card(i)]) for i in range(32)],
        "contract": contract.value, "trump": int(trump) if trump is not None else None,
        "leader": int(first_leader), "dealer": int(dealer), "locked": locked,
        "plays": [] if reusable_prefix else [dict(action) for action in spec.play_actions],
        "prefixLength": prefix_length, "ekka": ekka_leads,
    })
    cache_key = canonical_sha256({"prefix": spec.sha256, "template": template_key})
    build_arguments = dict(
        cards=tuple(Card(card_id) for card_id in range(32)),
        ownership=ownership,
        contract=contract,
        trump=trump,
        first_leader=first_leader,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        prefix=() if reusable_prefix else fixed_prefix,
        tail=(),
        ekka_leads=tuple(ekka_leads),
        locked=locked,
        action_constraints=() if reusable_prefix else action_constraints,
        decision_prefix_length=prefix_length,
        play_step_count=play_step_count,
        cancel_check=cancel_check,
    )
    if reusable_prefix and play_template is None and first_trick_templates is not None:
        play_template = first_trick_templates.get(prefix_length)
    reuse_template = play_template is not None and play_template[0] == template_key
    if reuse_template:
        if play_template[2].model_hash != play_template[1]:
            raise ValueError("native play template integrity mismatch")
        cached = paired_pass_cache.get(cache_key)
        if cached is not None:
            digest, value = cached
            if digest != value.evidence_hash():
                raise ValueError("native paired-pass certificate integrity mismatch")
            remaining()
            return value
        build = replace(play_template[2], model=play_template[2].model.clone())
        project_fragment = play_template[3]
    else:
        try:
            build = OrToolsPlayTailStageV1(native_conversion=True).build_model(**build_arguments)
        except PlayTailSolveStopped as exc:
            if exc.solver_status == "infeasible":
                raise
            raise ExactNormalizationStopped(exc.solver_status, evidence={
                "sourceStage": "exact_play_prefix", "causeReceipt": exc.partial_receipt,
            }) from exc
        project_fragment = attach_project_lifecycle_fragment_v1(
            build.model, owner_of_card=build.initial_owner_variables,
            fixed_card_owners=tuple(ownership[Card(card_id)] for card_id in range(32)),
            contract=contract, dealer=dealer, trump=trump,
            domain=rules.project_domain, name="exact_projects", compact_fixed_states=True,
            second_trick_leader=build._extract.winner_seats[0],
        )
    retained_template = (
        (play_template if reuse_template else
         (template_key, build.model_hash, replace(build, model=build.model.clone()), project_fragment))
        if reusable_prefix else None
    )
    if reusable_prefix:
        if first_trick_templates is not None:
            first_trick_templates[prefix_length] = retained_template
        # Bind branch inputs only on the isolated solve model, never the frame.
        for step, raw in enumerate(spec.play_actions):
            slot = build.play_state_context.play_slots[step]
            build.model.add(slot.card == Card.parse(raw["card"]).id)
            if raw.get("actor") is not None:
                build.model.add(slot.actor == int(Seat.parse(str(raw["actor"]))))
    cached = paired_pass_cache.get(cache_key)
    if cached is not None:
        digest, value = cached
        if digest != value.evidence_hash():
            raise ValueError("native paired-pass certificate integrity mismatch")
        remaining()
        return value
    declaration_order = rules.project_domain.declaration_seats(dealer)
    forced_pass_actor = declaration_order[prefix_length] if reusable_prefix else None
    # Only fixed CP domains permit pairing; optional declarations retain their
    # normal proof path. Both checkpoints enter the same uniqueness solve.
    forced_project_passes = tuple(
        seat for seat in declaration_order if reusable_prefix and all(
            list(project.declared.proto.domain) == [0, 0]
            for project in project_fragment.projects if project.owner == seat
        )
    )
    forced_pass = forced_pass_actor in forced_project_passes
    if forced_pass and len(spec.project_actions) == prefix_length + 1:
        action = ProjectDeclarationActionV1.build(spec.project_actions[-1], actor=forced_pass_actor)
        if action.action == "pass":
            root = spec.to_dict()
            root.pop("sha256", None)
            root["projectActions"] = root["projectActions"][:-1]
            root = ExactTracePrefixV1.build(root)
            _native_fixed_play_prefix(
                spec=root, hands=hands, contract=contract, trump=trump,
                first_leader=first_leader, dealer=dealer, buyer=buyer,
                escalation=escalation, last_escalating_team=last_escalating_team,
                starting_match_score=starting_match_score, locked=locked, rules=rules,
                seed=int(_effective_prefix_hash(root)[:16], 16) & 0x7FFFFFFF,
                maximum_seconds=remaining(), remaining_seconds=remaining, cancel_check=cancel_check,
                play_template=retained_template, first_trick_templates=first_trick_templates,
                paired_pass_cache=paired_pass_cache,
            )
            canonical_child = root.to_dict()
            canonical_child.pop("sha256", None)
            canonical_child["projectActions"].append(action.to_dict())
            canonical_child = ExactTracePrefixV1.build(canonical_child)
            child_key = canonical_sha256({"prefix": canonical_child.sha256, "template": template_key})
            return paired_pass_cache[child_key][1]
    hint_model_hash = None
    hint_solution_hash = None
    hint_status = "unavailable"
    for step, raw in enumerate(spec.play_actions):
        build.model.add_hint(
            build.selected_play_indexes[step],
            Card.parse(raw["card"]).id,
        )
    if prefix_length == 32 and fixed_prefix:
        # No continuation variables need a hint. The joint solve and semantic
        # uniqueness solve below still verify the complete supplied trace.
        hint_status = "not_required_complete_prefix"
    else:
        hint_status = "joint_solution_continuation"
    project_binding = _native_project_prefix_projection(
        build=build,
        spec=spec,
        fragment=project_fragment,
        dealer=dealer,
        contract=contract,
        trump=trump,
        domain=rules.project_domain,
    )
    following_binding = None
    following_key = None
    if forced_pass and len(spec.project_actions) == prefix_length:
        following = spec.to_dict()
        following.pop("sha256", None)
        following["projectActions"].append({"actor": forced_pass_actor.name.lower(), "action": "pass"})
        following = ExactTracePrefixV1.build(following)
        following_key = canonical_sha256({"prefix": following.sha256, "template": template_key})
        # The actor's declarations are fixed zero in CP. Advancing only its
        # declaration marker therefore leaves every projected variable equal.
        # Alias those variables; retain the distinct authored action metadata.
        following_binding = replace(project_binding,
            declaration_actions=(*project_binding.declaration_actions,
                                 ProjectDeclarationActionV1(forced_pass_actor, "pass")),
            reached_action_count=prefix_length + 1, project_phase_open=False)
    scoring_domain = ScoringDomainIRV1.from_rules(rules.data)
    scoring_fragment: OrToolsScoringFragmentV1 | None = None
    if prefix_length == 32:
        if len(build.raw_team_variables) != 2 or len(build.winner_team_variables) != 8:
            raise AssertionError("complete native play lacks live scoring variables")

        def fixed(name: str, lower: int, upper: int, value: int) -> cp_model.IntVar:
            variable = build.model.new_int_var(lower, upper, name)
            build.model.add(variable == value)
            return variable

        baloot_points = tuple(
            build.model.new_int_var(
                0,
                rules.project_domain.baloot_value,
                f"exact_baloot_points_{int(team)}",
            )
            for team in Team
        )
        for team in Team:
            build.model.add(
                baloot_points[int(team)]
                == rules.project_domain.baloot_value
                * project_binding.team_baloot[int(team)]
            )
        maximum_match_score = max(
            scoring_domain.match_target,
            *(int(value) for value in starting_match_score),
        )
        scoring_inputs = CpSatScoringInputsV1(
            contract=fixed(
                "exact_score_contract",
                0,
                1,
                scoring_contract_id_v1(contract),
            ),
            buyer_seat=fixed("exact_score_buyer", 0, 3, int(buyer)),
            escalation=fixed(
                "exact_score_escalation",
                0,
                4,
                scoring_level_id_v1(escalation),
            ),
            last_escalating_team=fixed(
                "exact_score_last_escalating_team",
                0,
                2,
                2 if last_escalating_team is None else int(last_escalating_team),
            ),
            raw_points=(build.raw_team_variables[0], build.raw_team_variables[1]),
            project_points=(
                project_binding.fragment.combined_awarded_value[0],
                project_binding.fragment.combined_awarded_value[1],
            ),
            baloot_points=(baloot_points[0], baloot_points[1]),
            trick_winner_teams=tuple(build.winner_team_variables),
            starting_match_score=tuple(
                fixed(
                    f"exact_score_starting_match_{int(team)}",
                    0,
                    maximum_match_score,
                    int(starting_match_score[int(team)]),
                )
                for team in Team
            ),
        )
        scoring_fragment = attach_scoring_fragment_v1(
            build.model,
            scoring_domain,
            scoring_inputs,
            native_conversion=True,
            source_hash=canonical_sha256(
                {
                    "schemaVersion": 1,
                    "stage": "exact_play_project_scoring",
                    "rulesHash": rules.sha256,
                    "prefixHash": spec.sha256,
                    "projectFragmentHash": project_binding.fragment.fragment_hash,
                }
            ),
        )
    witness_margins = []
    if prefix_length == 3:
        points = build.trick_point_variables[0]
        winner = build.winner_team_variables[0]
        maximum_points = points.proto.domain[len(points.proto.domain) - 1]
        for team in Team:
            margin = build.model.new_int_var(
                -maximum_points, maximum_points,
                f"exact_witness_raw_margin_{int(team)}")
            won = winner if int(team) else winner.Not()
            build.model.add(margin == points).only_enforce_if(won)
            build.model.add(margin == -points).only_enforce_if(won.Not())
            witness_margins.append(margin)
    solver = _deterministic_solver(seed=seed, maximum_seconds=remaining())
    if reusable_prefix:
        if build.model.has_objective():
            raise ValueError("conditional project evidence requires a feasibility model")
        solver.parameters.keep_all_feasible_solutions_in_presolve = True
        solver.parameters.fill_tightened_domains_in_response = True
    try:
        result = build.solve(
            maximum_time_seconds=remaining(),
            solver=solver,
            cancel_check=cancel_check,
        )
    except PlayTailSolveStopped as exc:
        if exc.solver_status == "infeasible":
            raise
        raise ExactNormalizationStopped(exc.solver_status, evidence={
            "sourceStage": "exact_play_prefix", "causeReceipt": exc.partial_receipt,
        }) from exc
    remaining()
    if result.solver_status != "optimal":
        raise ExactNormalizationStopped(
            result.solver_status, evidence={"sourceStage": "exact_play_prefix"}
        )
    conditional_forced_passes = None
    if reusable_prefix:
        tightened = solver.response_proto.tightened_variables
        if len(tightened) != len(build.model.proto.variables):
            raise ValueError("conditional project evidence lacks native tightened domains")
        conditional_forced_passes = tuple(seat for seat in declaration_order if all(
            list(tightened[project.declared.index].domain) == [0, 0]
            for project in project_fragment.projects if project.owner == seat))
    for index, raw in enumerate(spec.play_actions):
        solved = result.plays[index]
        if solved.card != Card.parse(raw["card"]):
            raise AssertionError("native exact play solve changed an authored card")
        if solved.ekka != (raw.get("ekka") is True):
            raise AssertionError("native exact play solve changed an Ekka declaration")
        asserted_actor = raw.get("actor")
        if asserted_actor is not None and solved.actor != Seat.parse(
            str(asserted_actor)
        ):
            raise ValueError(f"exact play action {index} actor mismatch")

    context = build.play_state_context
    semantic_variables: list[cp_model.IntVar] = [
        *build.initial_owner_variables,
    ]
    for step in range(prefix_length):
        slot = context.play_slots[step]
        semantic_variables.extend((slot.card, slot.actor, slot.ekka, slot.present))
    completed_count = prefix_length // 4
    semantic_variables.extend(
        context.completed_trick_winners[index] for index in range(completed_count)
    )
    semantic_variables.extend(
        (
            context.next_actor,
            context.current_trick_size,
            context.current_trick_active,
            context.current_trick_leader,
            context.current_trick_suit_led,
            context.phase,
        )
    )
    for slot in context.current_trick_slots:
        semantic_variables.extend((slot.card, slot.actor, slot.ekka, slot.present))
    semantic_variables.extend(context.remaining_hand_sizes[seat] for seat in Seat)
    semantic_variables.extend(
        context.legal_card_literals[card_id] for card_id in range(32)
    )
    semantic_variables.extend(
        context.legal_action_literals[(card_id, ekka)]
        for card_id in range(32)
        for ekka in (False, True)
    )
    semantic_variables.extend(context.raw_points[team] for team in Team)
    semantic_variables.extend(context.game_points[team] for team in Team)
    semantic_variables.extend(project_binding.semantic_variables)
    semantic_variables.extend(witness_margins)
    if following_binding is not None:
        semantic_variables.extend(following_binding.semantic_variables)
    if scoring_fragment is not None:
        semantic_variables.extend(
            (
                scoring_fragment.counting_team,
                *scoring_fragment.card_game_points,
                *scoring_fragment.total_before_award,
                scoring_fragment.round_winner,
                scoring_fragment.multiplier,
                scoring_fragment.kabout,
                scoring_fragment.gahwa,
                *scoring_fragment.awarded_points,
                *scoring_fragment.final_match_score,
                scoring_fragment.match_winner,
            )
        )
    semantic_variables = list(
        dict.fromkeys(
            # A folded literal and its complement share one underlying value.
            item.index if item.index >= 0 else -item.index - 1
            for item in semantic_variables
        )
    )
    semantic_variables = [
        build.model.get_int_var_from_proto_index(index)
        for index in semantic_variables
    ]
    semantic_values = [int(solver.value(item)) for item in semantic_variables]
    legal_play_actions = (
        tuple(
            TailPrefixActionV1(
                result.next_actor,
                Card(card_id),
                ekka,
            )
            for card_id in range(32)
            for ekka in (False, True)
            if solver.boolean_value(context.legal_action_literals[(card_id, ekka)])
        )
        if result.next_actor is not None
        else ()
    )
    semantic_vector_hash = canonical_sha256(
        {
            "schemaVersion": 1,
            "playModelHash": result.model_hash,
            "values": semantic_values,
        }
    )
    legal_project_actions = _native_project_legal_actions(
        model=build.model,
        binding=project_binding,
        solved=solver,
        seed=seed,
        maximum_seconds=remaining(),
        remaining_seconds=remaining,
        cancel_check=cancel_check,
    )
    scoring_result = None
    if scoring_fragment is not None:
        scoring_evidence = extract_scoring_evidence_v1(
            solver,
            scoring_fragment,
            scoring_domain,
            contract=contract,
        )
        scoring_result = OrToolsScoringResultV1(
            "satisfied",
            "optimal",
            scoring_evidence,
            scoring_domain.sha256,
            result.model_hash,
            str(scoring_evidence["sha256"]),
            len(build.model.proto.variables),
            len(build.model.proto.constraints),
            int(solver.num_branches),
            int(solver.num_conflicts),
            scoring_fragment.outcome_context,
        )
    # Only the temporary suffix is fixed: the joint CP solution already proved
    # this legal witness. Prefix semantic variables remain subject to uniqueness.
    for step, action in enumerate(result.plays):
        if step >= prefix_length:
            build.model.add(build.selected_play_indexes[step] == action.card.id)
    build.model.add_forbidden_assignments(
        semantic_variables,
        [semantic_values],
    )
    uniqueness_model_hash = _cp_model_hash(
        build.model,
        authority={
            "stage": "exact_play_semantic_uniqueness",
            "rulesHash": rules.sha256,
            "sourceModelHash": result.model_hash,
            "prefixLength": prefix_length,
        },
    )
    uniqueness_solver = _deterministic_solver(
        seed=seed,
        maximum_seconds=remaining(),
    )
    uniqueness_status = _solve_cp_sat_with_cancel_v1(uniqueness_solver, build.model, cancel_check)
    remaining()
    if uniqueness_status == cp_model.OPTIMAL:
        raise AssertionError("exact play normalization is semantically ambiguous")
    if uniqueness_status != cp_model.INFEASIBLE:
        raise ExactNormalizationStopped(
            uniqueness_solver.status_name(uniqueness_status).lower(),
            evidence={"sourceStage": "play_semantic_uniqueness",
                      "modelHash": uniqueness_model_hash, "sourceModelHash": result.model_hash},
        )
    uniqueness_solver_status = uniqueness_solver.status_name(
        uniqueness_status
    ).lower()
    project_result, project_projection = _extract_native_project_prefix_projection(
        binding=project_binding,
        solver=solver,
        model_hash=result.model_hash,
        prefix_length=prefix_length,
        uniqueness_model_hash=uniqueness_model_hash,
        uniqueness_solver_status=uniqueness_solver_status,
    )
    normalized = _NativePlayNormalizationV1(
        result,
        legal_play_actions,
        semantic_vector_hash,
        uniqueness_model_hash,
        uniqueness_solver_status,
        project_result,
        project_projection,
        legal_project_actions,
        scoring_result,
        hint_model_hash,
        hint_solution_hash,
        hint_status,
        retained_template if len(spec.project_actions) == prefix_length else None,
        tuple(int(solver.value(value)) for value in witness_margins) if witness_margins else None,
        forced_project_passes,
        conditional_forced_passes,
    )
    if following_binding is not None:
        following_projects, following_projection = _extract_native_project_prefix_projection(
            binding=following_binding, solver=solver, model_hash=result.model_hash,
            prefix_length=prefix_length, uniqueness_model_hash=uniqueness_model_hash,
            uniqueness_solver_status=uniqueness_solver_status,
        )
        following_result = replace(normalized, project_result=following_projects,
            project_projection=following_projection, legal_project_actions=(), play_template=None)
        # Retain only the immediately following pass, never a campaign cache.
        paired_pass_cache.clear()
        paired_pass_cache[following_key] = (following_result.evidence_hash(), following_result)
    return normalized


def _selected_projects(
    result: OrToolsProjectLifecycleResultV1,
    *,
    dealer: Seat,
) -> list[Project]:
    first = dealer.next()
    declaration_seats = (first, first.next(), first.next().next(), dealer)
    selected = [item for item in result.projects if item.selected]
    selected.sort(
        key=lambda item: (
            declaration_seats.index(item.owner),
            -item.value,
            -item.category_priority,
            -item.high_rank,
        )
    )
    return [
        Project(
            item.owner,
            item.kind,
            item.cards,
            item.value,
            item.high_rank,
            item.category_priority,
        )
        for item in selected
    ]


@dataclass(frozen=True, slots=True)
class _NativeProjectPrefixProjectionV1:
    projects: tuple[Project, ...]
    comparison_winner_ids: frozenset[str]
    winning_team: Team | None
    baloot_teams: frozenset[Team]
    baloot_events: tuple[str | None, ...]
    lifecycle_by_id: Mapping[str, Mapping[str, bool]]
    model_hash: str
    evidence_hash: str
    uniqueness_model_hash: str | None
    uniqueness_solver_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "projects": [project.to_dict() for project in self.projects],
            "comparisonWinnerIds": sorted(self.comparison_winner_ids),
            "winningTeam": (
                self.winning_team.name.lower()
                if self.winning_team is not None
                else None
            ),
            "balootTeams": [
                team.name.lower() for team in sorted(self.baloot_teams, key=int)
            ],
            "balootEvents": list(self.baloot_events),
            "lifecycleById": {
                key: dict(value) for key, value in sorted(self.lifecycle_by_id.items())
            },
            "modelHash": self.model_hash,
            "evidenceHash": self.evidence_hash,
            "uniquenessModelHash": self.uniqueness_model_hash,
            "uniquenessSolverStatus": self.uniqueness_solver_status,
        }


@dataclass(frozen=True, slots=True)
class _NativeProjectPrefixBindingV1:
    fragment: OrToolsProjectLifecycleFragmentV1
    declaration_actions: tuple[ProjectDeclarationActionV1, ...]
    declaration_seats: tuple[Seat, ...]
    reached_action_count: int
    project_phase_open: bool
    declared_now: Mapping[str, cp_model.IntVar]
    declaration_order_now: Mapping[str, cp_model.IntVar]
    comparison_now: Mapping[str, cp_model.IntVar]
    awarded_now: Mapping[str, cp_model.IntVar]
    revealed_now: Mapping[str, cp_model.IntVar]
    winning_team: cp_model.IntVar
    auto_events: tuple[cp_model.IntVar, ...]
    declared_events: tuple[cp_model.IntVar, ...]
    team_baloot: tuple[cp_model.IntVar, cp_model.IntVar]
    combined_awarded_value: tuple[cp_model.IntVar, cp_model.IntVar]
    semantic_variables: tuple[cp_model.IntVar, ...]
    declaration_guard: cp_model.IntVar | None = None


def _native_project_action_literals(fragment, action):
    """Bind a typed action to native declaration variables, without deciding truth."""
    if not isinstance(action, ProjectDeclarationActionV1):
        raise TypeError("native project binding requires a typed declaration action")
    owned = tuple(project for project in fragment.projects if project.owner == action.actor)
    if action.action == "pass":
        return tuple(project.declared.Not() for project in owned)
    if action.action == "declare":
        project = next((project for project in owned if project.project_id == action.project_id), None)
        if project is not None:
            return (project.declared,)
    raise ValueError("project declaration action references an unknown project or action")


def _native_project_visibility_projection(
    *, model, fragment, seen_first, owner_seen, prefix_length, declaration_seats,
    name="exact_project", reveal_scope="scoring_team",
):
    """Shared native declaration/comparison/revelation timing for a play prefix."""
    seen_second = {seat: owner_seen(seat, 4, 8, f"{name}_seen_second_{int(seat)}") for seat in Seat}
    compared = model.new_bool_var(f"{name}_prefix_compared")
    model.add(compared == int(prefix_length >= 4))
    declared_now, declaration_order_now, comparison_now, awarded_now, revealed_now = {}, {}, {}, {}, {}
    for index, project in enumerate(fragment.projects):
        if list(project.eligible.proto.domain) == [0, 0]:
            for projection in (declared_now, declaration_order_now, comparison_now, awarded_now, revealed_now):
                projection[project.project_id] = project.eligible
            continue
        declared_now[project.project_id] = _bool_and(model,
            [project.declared, seen_first[project.owner]], f"{name}_declared_now_{index}")
        current_order = model.new_int_var(0, len(declaration_seats), f"{name}_declaration_order_now_{index}")
        model.add(current_order == project.declaration_order).only_enforce_if(declared_now[project.project_id])
        model.add(current_order == 0).only_enforce_if(declared_now[project.project_id].Not())
        declaration_order_now[project.project_id] = current_order
        comparison_now[project.project_id] = _bool_and(model,
            [project.comparison_winner, compared], f"{name}_comparison_now_{index}")
        awarded_now[project.project_id] = _bool_and(model,
            [project.awarded, compared], f"{name}_awarded_now_{index}")
        reveal_source = comparison_now if reveal_scope == "comparison_winner" else awarded_now
        revealed_now[project.project_id] = _bool_and(model,
            [reveal_source[project.project_id], seen_second[project.owner]], f"{name}_revealed_now_{index}")
    return compared, declared_now, declaration_order_now, comparison_now, awarded_now, revealed_now


def _native_project_prefix_projection(
    *,
    build: OrToolsPlayTailBuildV1,
    spec: ExactTracePrefixV1,
    fragment: OrToolsProjectLifecycleFragmentV1,
    dealer: Seat,
    contract: Contract,
    trump: Suit | None,
    domain: ProjectDomainIRV1,
    checkpoint_id: str | None = None,
    declaration_guard: cp_model.IntVar | None = None,
    play_prefix_length: int | None = None,
) -> _NativeProjectPrefixBindingV1:
    """Attach project lifecycle and public-prefix timing to the live play model.

    A guarded checkpoint binds its authored declarations only while its guard
    is active. Callers must activate that guard before extracting branch evidence.
    """

    model = build.model
    prefix_length = len(spec.play_actions) if play_prefix_length is None else play_prefix_length
    if (type(prefix_length) is not int or prefix_length < len(spec.play_actions)
            or prefix_length > len(build.selected_play_indexes)
            or (play_prefix_length is not None and checkpoint_id is None)):
        raise ValueError("project checkpoint requires a named compiled play prefix")
    if declaration_guard is not None:
        if (checkpoint_id is None or not spec.project_actions_present
                or not isinstance(declaration_guard, cp_model.IntVar)
                or declaration_guard.model_proto is not model.proto
                or list(declaration_guard.proto.domain) != [0, 1]):
            raise ValueError("conditional declaration requires a named checkpoint and same-model Boolean guard")
    if checkpoint_id is not None:
        if type(checkpoint_id) is not str or not checkpoint_id.isidentifier():
            raise ValueError("project checkpoint ID must be an identifier")
        if any(variable.name.startswith(checkpoint_id + ":") for variable in model.proto.variables):
            raise ValueError("project checkpoint ID already exists")
    first_variable = len(model.proto.variables)
    declaration_seats = tuple(domain.declaration_seats(dealer))
    if len(spec.project_actions) > len(declaration_seats):
        raise ValueError("project declaration action prefix is too long")
    declaration_actions = tuple(
        ProjectDeclarationActionV1.build(
            raw,
            actor=declaration_seats[index],
        )
        for index, raw in enumerate(spec.project_actions)
    )
    for action in declaration_actions:
        constraint = model.add_bool_and(_native_project_action_literals(fragment, action))
        if declaration_guard is not None:
            constraint.only_enforce_if(declaration_guard)

    # Legacy exact traces carry public declaration rows rather than typed
    # project actions. Once play has begun those rows describe the complete
    # declaration result, so bind them directly to the same lifecycle model.
    if not spec.project_actions_present and spec.play_actions:
        matched_ids: set[str] = set()
        for row in spec.project_declarations:
            matches = tuple(
                project
                for project in fragment.projects
                if project.owner == Seat.parse(str(row["owner"]))
                and project.kind == str(row["kind"])
                and project.value == int(row["value"])
                and project.high_rank == int(row["highRank"])
            )
            if not matches:
                raise ValueError(
                    "exact-state project declaration references an unknown project"
                )
            model.add(sum(project.declared for project in matches) == 1)
            for project in matches:
                model.add(
                    project.declaration_order == int(row["declarationOrder"])
                ).only_enforce_if(project.declared)
                matched_ids.add(project.project_id)

    play_slots = build.play_state_context.play_slots

    def play_slot(step: int):
        # Exact normalization starts from a full deal even when only a prefix
        # of its play variables is compiled. Slots retain absolute round indexes.
        return play_slots[step]

    def owner_seen(owner: Seat, start: int, stop: int, label: str) -> cp_model.IntVar:
        return _bool_or(
            model,
            [
                _equals(
                    model,
                    play_slot(step).actor,
                    int(owner),
                    f"{label}_{step}",
                )
                for step in range(start, min(stop, prefix_length))
            ],
            f"{label}_any",
        )

    if spec.project_actions_present:
        acted = set(declaration_seats[: len(declaration_actions)])
        seen_first = {}
        for seat in Seat:
            literal = model.new_bool_var(f"exact_project_seen_first_{int(seat)}")
            model.add(literal == int(seat in acted))
            seen_first[seat] = literal
    else:
        seen_first = {
            seat: owner_seen(
                seat,
                0,
                4,
                f"exact_project_seen_first_{int(seat)}",
            )
            for seat in Seat
        }
    if not spec.project_actions_present and spec.play_actions:
        for project in fragment.projects:
            if project.project_id not in matched_ids:
                model.add(project.declared == 0).only_enforce_if(
                    seen_first[project.owner]
                )
    (compared, declared_now, declaration_order_now, comparison_now,
     awarded_now, revealed_now) = _native_project_visibility_projection(
        model=model, fragment=fragment, seen_first=seen_first, owner_seen=owner_seen,
        prefix_length=prefix_length, declaration_seats=declaration_seats,
        reveal_scope=domain.reveal_scope)

    winner_sentinel = len(Team)
    winning_team = model.new_int_var(
        0,
        winner_sentinel,
        "exact_project_prefix_winner",
    )
    model.add_allowed_assignments(
        [
            compared,
            fragment.winning_team[0],
            fragment.winning_team[1],
            winning_team,
        ],
        [
            (0, 0, 0, winner_sentinel),
            (0, 1, 0, winner_sentinel),
            (0, 0, 1, winner_sentinel),
            (1, 0, 0, winner_sentinel),
            (1, 1, 0, int(Team.NORTH_SOUTH)),
            (1, 0, 1, int(Team.WEST_EAST)),
        ],
    )

    auto_events, declared_events, team_baloot = _native_baloot_prefix_projection(
        model=model, fragment=fragment, play_slot=play_slot, prefix_length=prefix_length,
        contract=contract, trump=trump, domain=domain,
    )

    combined_awarded_value = tuple(
        model.new_int_var(0, int(list(total.proto.domain)[-1]), f"exact_project_prefix_points_{int(team)}")
        for team, total in zip(Team, fragment.combined_awarded_value, strict=True)
    )
    for current, total in zip(combined_awarded_value, fragment.combined_awarded_value, strict=True):
        model.add(current == total).only_enforce_if(compared)
        model.add(current == 0).only_enforce_if(compared.Not())
    semantic_variables: list[cp_model.IntVar] = [compared, winning_team, *combined_awarded_value]
    for project in fragment.projects:
        semantic_variables.extend(
            (
                project.eligible,
                project.selected,
                project.declarable,
                declared_now[project.project_id],
                declaration_order_now[project.project_id],
                comparison_now[project.project_id],
                awarded_now[project.project_id],
                revealed_now[project.project_id],
            )
        )
    semantic_variables.extend(auto_events)
    semantic_variables.extend(declared_events)
    semantic_variables.extend(team_baloot)
    if checkpoint_id is not None:
        for index in range(first_variable, len(model.proto.variables)):
            variable = model.proto.variables[index]
            if variable.name:
                variable.name = checkpoint_id + ":" + variable.name
    return _NativeProjectPrefixBindingV1(
        fragment,
        declaration_actions,
        declaration_seats,
        len(declaration_actions),
        (
            prefix_length < len(declaration_seats)
            and len(declaration_actions) == prefix_length
        ),
        declared_now,
        declaration_order_now,
        comparison_now,
        awarded_now,
        revealed_now,
        winning_team,
        tuple(auto_events),
        tuple(declared_events),
        (team_baloot[0], team_baloot[1]),
        (combined_awarded_value[0], combined_awarded_value[1]),
        tuple(semantic_variables),
        declaration_guard,
    )


def _native_baloot_prefix_projection(
    *, model: cp_model.CpModel, fragment: OrToolsProjectLifecycleFragmentV1,
    play_slot: Callable[[int], Any], prefix_length: int,
    contract: Contract, trump: Suit | None, domain: ProjectDomainIRV1,
) -> tuple[tuple[cp_model.IntVar, ...], tuple[cp_model.IntVar, ...], tuple[cp_model.IntVar, ...]]:
    """Compile public Baloot timing from solved project and play variables."""
    actor_matches: dict[tuple[int, Seat], cp_model.IntVar] = {}
    card_matches: dict[tuple[int, int], cp_model.IntVar] = {}
    for step in range(prefix_length):
        for seat in Seat:
            actor_matches[(step, seat)] = _equals(
                model,
                play_slot(step).actor,
                int(seat),
                f"exact_baloot_actor_{step}_{int(seat)}",
            )
        if trump is not None:
            for rank in (Rank.KING, Rank.QUEEN):
                card_id = Card.of(trump, rank).id
                card_matches[(step, card_id)] = _equals(
                    model,
                    play_slot(step).card,
                    card_id,
                    f"exact_baloot_card_{step}_{card_id}",
                )

    included_by_seat: dict[Seat, cp_model.IntVar] = {}
    for seat in Seat:
        pair = (
            {
                Card.of(trump, Rank.KING).id,
                Card.of(trump, Rank.QUEEN).id,
            }
            if contract.effective == Contract.HOKUM and trump is not None
            else set()
        )
        included = tuple(
            project.awarded
            for project in fragment.projects
            if project.owner == seat
            and pair.issubset({card.id for card in project.cards})
            and (
                project.kind in {"siri", "fifty"}
                or (
                    project.kind == "hundred_sequence"
                    and domain.baloot_hundred_inclusion == "automatic"
                )
            )
        )
        included_by_seat[seat] = _bool_or(
            model,
            included,
            f"exact_baloot_included_{int(seat)}",
        )

    auto_trigger: list[cp_model.IntVar] = []
    card_trigger: list[cp_model.IntVar] = []
    seen_king = {seat: model.new_constant(0) for seat in Seat}
    seen_queen = dict(seen_king)
    for step in range(prefix_length):
        auto_by_seat: list[cp_model.IntVar] = []
        card_by_seat: list[cp_model.IntVar] = []
        for seat in Seat:
            eligible = fragment.baloot_seat[int(seat)]
            auto_by_seat.append(
                _bool_and(
                    model,
                    [
                        eligible,
                        included_by_seat[seat],
                        actor_matches[(step, seat)],
                    ],
                    f"exact_baloot_auto_{step}_{int(seat)}",
                )
                if 4 <= step < 8
                else _bool_and(
                    model,
                    [eligible, eligible.Not()],
                    f"exact_baloot_auto_disabled_{step}_{int(seat)}",
                )
            )
            if contract.effective != Contract.HOKUM or trump is None:
                card_by_seat.append(
                    _bool_and(
                        model,
                        [eligible, eligible.Not()],
                        f"exact_baloot_card_disabled_{step}_{int(seat)}",
                    )
                )
                continue
            king = Card.of(trump, Rank.KING).id
            queen = Card.of(trump, Rank.QUEEN).id
            prior_king, prior_queen = seen_king[seat], seen_queen[seat]
            current_king = _bool_and(
                model,
                [
                    actor_matches[(step, seat)],
                    card_matches[(step, king)],
                ],
                f"exact_baloot_current_king_{step}_{int(seat)}",
            )
            current_queen = _bool_and(
                model,
                [
                    actor_matches[(step, seat)],
                    card_matches[(step, queen)],
                ],
                f"exact_baloot_current_queen_{step}_{int(seat)}",
            )
            second_pair_card = _bool_or(
                model,
                [
                    _bool_and(
                        model,
                        [current_king, prior_queen],
                        f"exact_baloot_king_second_{step}_{int(seat)}",
                    ),
                    _bool_and(
                        model,
                        [current_queen, prior_king],
                        f"exact_baloot_queen_second_{step}_{int(seat)}",
                    ),
                ],
                f"exact_baloot_second_pair_{step}_{int(seat)}",
            )
            card_by_seat.append(
                _bool_and(
                    model,
                    [eligible, second_pair_card],
                    f"exact_baloot_card_trigger_{step}_{int(seat)}",
                )
            )
            seen_king[seat] = _bool_or(
                model, [prior_king, current_king], f"exact_baloot_seen_king_{step}_{int(seat)}",
            )
            seen_queen[seat] = _bool_or(
                model, [prior_queen, current_queen], f"exact_baloot_seen_queen_{step}_{int(seat)}",
            )
        auto_trigger.append(
            _bool_or(model, auto_by_seat, f"exact_baloot_auto_trigger_{step}")
        )
        card_trigger.append(
            _bool_or(model, card_by_seat, f"exact_baloot_card_trigger_{step}")
        )

    auto_events: list[cp_model.IntVar] = []
    declared_events: list[cp_model.IntVar] = []
    events_by_team: dict[Team, list[cp_model.IntVar]] = {team: [] for team in Team}
    for step in range(prefix_length):
        actor_team_literals = []
        for team in Team:
            actor_team_literals.append(
                _bool_or(
                    model,
                    [
                        actor_matches[(step, seat)]
                        for seat in Seat
                        if seat.team == team
                    ],
                    f"exact_baloot_actor_team_{step}_{int(team)}",
                )
            )
        team_events: list[cp_model.IntVar] = []
        for team in Team:
            prior_event = _bool_or(
                model,
                events_by_team[team],
                f"exact_baloot_prior_team_{step}_{int(team)}",
            )
            auto_event = _bool_and(
                model,
                [
                    auto_trigger[step],
                    actor_team_literals[int(team)],
                    prior_event.Not(),
                ],
                f"exact_baloot_auto_event_{step}_{int(team)}",
            )
            declared_event = _bool_and(
                model,
                [
                    card_trigger[step],
                    auto_trigger[step].Not(),
                    actor_team_literals[int(team)],
                    prior_event.Not(),
                ],
                f"exact_baloot_declared_event_{step}_{int(team)}",
            )
            events_by_team[team].extend((auto_event, declared_event))
            team_events.extend((auto_event, declared_event))
        auto_events.append(
            _bool_or(
                model,
                [team_events[index] for index in range(0, len(team_events), 2)],
                f"exact_baloot_auto_event_{step}",
            )
        )
        declared_events.append(
            _bool_or(
                model,
                [team_events[index] for index in range(1, len(team_events), 2)],
                f"exact_baloot_declared_event_{step}",
            )
        )
    team_baloot = tuple(
        _bool_or(model, events_by_team[team], f"exact_baloot_team_{int(team)}")
        for team in Team
    )

    return tuple(auto_events), tuple(declared_events), team_baloot


def _extract_native_project_prefix_projection(
    *,
    binding: _NativeProjectPrefixBindingV1,
    solver: cp_model.CpSolver,
    model_hash: str,
    prefix_length: int,
    uniqueness_model_hash: str | None,
    uniqueness_solver_status: str,
) -> tuple[OrToolsProjectLifecycleResultV1, _NativeProjectPrefixProjectionV1]:
    if binding.declaration_guard is not None and not solver.boolean_value(binding.declaration_guard):
        raise ValueError("cannot extract an inactive declaration checkpoint")
    if prefix_length != len(binding.auto_events):
        raise ValueError("project checkpoint evidence does not cover the requested prefix")
    lifecycle_by_id: dict[str, Mapping[str, bool]] = {}
    evidence_rows: list[ProjectLifecycleEvidenceV1] = []
    state_projects: list[Project] = []
    comparison_ids: set[str] = set()
    for project in binding.fragment.projects:
        if not solver.boolean_value(project.eligible):
            continue
        project_id = project.project_id
        lifecycle = {
            "eligible": True,
            "selected": bool(solver.boolean_value(project.selected)),
            "declared": bool(solver.boolean_value(binding.declared_now[project_id])),
            "comparison_winner": bool(
                solver.boolean_value(binding.comparison_now[project_id])
            ),
            "awarded": bool(solver.boolean_value(binding.awarded_now[project_id])),
            "revealed": bool(solver.boolean_value(binding.revealed_now[project_id])),
        }
        lifecycle_by_id[project_id] = lifecycle
        declaration_order = (
            int(solver.value(binding.declaration_order_now[project_id]))
            if lifecycle["declared"]
            else None
        )
        evidence_rows.append(
            ProjectLifecycleEvidenceV1(
                project_id,
                project.owner,
                project.kind,
                project.cards,
                project.value,
                project.high_rank,
                project.category_priority,
                True,
                lifecycle["selected"],
                bool(solver.boolean_value(project.declarable)),
                lifecycle["declared"],
                declaration_order,
                lifecycle["comparison_winner"],
                lifecycle["awarded"],
            )
        )
        if lifecycle["comparison_winner"]:
            comparison_ids.add(project_id)
        if lifecycle["selected"]:
            state_projects.append(
                Project(
                    project.owner,
                    project.kind,
                    project.cards,
                    project.value,
                    project.high_rank,
                    project.category_priority,
                    lifecycle["declared"],
                    declaration_order if declaration_order is not None else 999,
                    lifecycle["awarded"],
                    lifecycle["revealed"],
                )
            )

    state_projects.sort(
        key=lambda item: (
            binding.declaration_seats.index(item.owner),
            -item.value,
            -item.category_priority,
            -item.high_rank,
        )
    )
    winner_value = int(solver.value(binding.winning_team))
    winning_team = None if winner_value == len(Team) else Team(winner_value)
    baloot_seats = frozenset(
        seat
        for seat in Seat
        if solver.boolean_value(binding.fragment.baloot_seat[int(seat)])
    )
    fragment_baloot_teams = frozenset(
        team
        for team in Team
        if solver.boolean_value(binding.fragment.baloot_team[int(team)])
    )
    reached_baloot_teams = frozenset(
        team for team in Team if solver.boolean_value(binding.team_baloot[int(team)])
    )
    combined_awarded_value = tuple(
        int(solver.value(total)) for total in binding.combined_awarded_value
    )
    baloot_points = tuple(
        binding.fragment.baloot_value * int(team in fragment_baloot_teams)
        for team in Team
    )
    evidence_core = {
        "schemaVersion": 1,
        "modelHash": model_hash,
        "fragmentHash": binding.fragment.fragment_hash,
        "prefixLength": prefix_length,
        "projects": [row.to_dict() for row in evidence_rows],
        "winningTeam": (
            winning_team.name.lower() if winning_team is not None else None
        ),
        "combinedAwardedValue": list(combined_awarded_value),
        "balootTeams": [
            team.name.lower() for team in sorted(reached_baloot_teams, key=int)
        ],
        "declarationActions": [
            action.to_dict() for action in binding.declaration_actions
        ],
    }
    evidence_hash = canonical_sha256(evidence_core)
    project_result = OrToolsProjectLifecycleResultV1(
        tuple(evidence_rows),
        winning_team,
        len(evidence_rows),
        sum(row.selected for row in evidence_rows),
        sum(row.declared for row in evidence_rows),
        sum(row.awarded for row in evidence_rows),
        (int(combined_awarded_value[0]), int(combined_awarded_value[1])),
        baloot_seats,
        fragment_baloot_teams,
        (int(baloot_points[0]), int(baloot_points[1])),
        model_hash,
        evidence_hash,
        "optimal",
        int(solver.num_conflicts),
        int(solver.num_branches),
        binding.declaration_actions,
    )
    baloot_events = tuple(
        "included"
        if solver.boolean_value(binding.auto_events[step])
        else "declared"
        if solver.boolean_value(binding.declared_events[step])
        else None
        for step in range(prefix_length)
    )
    projection_core = {
        "schemaVersion": 1,
        "modelHash": model_hash,
        "fragmentHash": binding.fragment.fragment_hash,
        "projectEvidenceHash": evidence_hash,
        "lifecycleById": {
            key: dict(value) for key, value in sorted(lifecycle_by_id.items())
        },
        "balootEvents": list(baloot_events),
    }
    projection = _NativeProjectPrefixProjectionV1(
        tuple(state_projects),
        frozenset(comparison_ids),
        winning_team,
        reached_baloot_teams,
        baloot_events,
        lifecycle_by_id,
        model_hash,
        canonical_sha256(projection_core),
        uniqueness_model_hash,
        uniqueness_solver_status,
    )
    return project_result, projection


def _native_project_legal_actions(
    *,
    model: cp_model.CpModel,
    binding: _NativeProjectPrefixBindingV1,
    solved: cp_model.CpSolver,
    seed: int,
    maximum_seconds: float,
    remaining_seconds: Callable[[], float] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[ProjectDeclarationActionV1, ...]:
    remaining = remaining_seconds or _normalization_clock(maximum_seconds, cancel_check)
    remaining()
    if binding.declaration_guard is not None and not solved.boolean_value(binding.declaration_guard):
        raise ValueError("cannot extract an inactive declaration checkpoint")
    if (
        not binding.project_phase_open
        or binding.reached_action_count >= len(binding.declaration_seats)
    ):
        return ()
    actor = binding.declaration_seats[binding.reached_action_count]
    owned = tuple(
        project for project in binding.fragment.projects if project.owner == actor
    )
    probe_model = None
    inherited_assumptions = tuple(model.proto.assumptions)

    def feasible(action: ProjectDeclarationActionV1) -> bool:
        nonlocal probe_model
        remaining()
        literals = _native_project_action_literals(binding.fragment, action)
        # The primary CP assignment already certifies its own action. Only
        # counterfactual declarations need another feasibility solve.
        if all(solved.boolean_value(literal) for literal in literals):
            return True
        if probe_model is None:
            probe_model = model.clone()
        clone = probe_model
        clone.clear_assumptions()
        clone.proto.assumptions.extend(inherited_assumptions)
        clone.proto.assumptions.extend(literal.index for literal in literals)
        checker = _deterministic_solver(
            seed=seed,
            maximum_seconds=remaining(),
        )
        status = _solve_cp_sat_with_cancel_v1(checker, clone, cancel_check)
        remaining()
        if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            return True
        if status == cp_model.INFEASIBLE:
            return False
        raise ExactNormalizationStopped(checker.status_name(status).lower(), evidence={
            "sourceStage": "project_legal_actions", "actor": actor.name.lower(),
            "declaredVariable": None if action.action == "pass" else literals[0].name,
            "modelHash": _cp_model_hash(clone, authority={"stage": "project_legal_actions"}),
        })

    actions: list[ProjectDeclarationActionV1] = []
    pass_action = ProjectDeclarationActionV1(actor, "pass")
    if feasible(pass_action):
        actions.append(pass_action)
    for project in owned:
        if not solved.boolean_value(project.declarable):
            continue
        action = ProjectDeclarationActionV1(actor, "declare", project.project_id)
        if feasible(action):
            actions.append(action)
    if not actions:
        raise ValueError("project declaration prefix has no legal continuation")
    return tuple(actions)

def _verify_native_project_assertions(
    result: OrToolsProjectLifecycleResultV1,
    projection: _NativeProjectPrefixProjectionV1,
    assertions: Sequence[Mapping[str, Any]],
) -> int:
    for assertion in assertions:
        expected = dict(assertion)
        lifecycle = str(expected.pop("state", "eligible"))

        def matches(project) -> bool:
            values = {
                "owner": project.owner.name.lower(),
                "team": project.team.name.lower(),
                "kind": project.kind,
                "value": project.value,
                "highRank": project.high_rank,
            }
            if any(values.get(key) != value for key, value in expected.items()):
                return False
            return bool(projection.lifecycle_by_id[project.project_id][lifecycle])

        if not any(matches(project) for project in result.projects):
            raise ValueError(
                "exact-state project assertion does not match CP-SAT projects"
            )
    return len(assertions)


def _legal_play_rows(
    actions: Sequence[TailPrefixActionV1],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "actionId": action.card.id + (32 if action.ekka else 0),
            "code": f"{action.card.code}+ekka" if action.ekka else action.card.code,
            "card": action.card.to_dict(),
            "ekka": action.ekka,
        }
        for action in actions
    )


def _native_boundary_legal_play_actions(
    state: Mapping[str, Any],
    *,
    rules: RuleBook,
    contract: Contract,
    trump: Suit | None,
    maximum_seconds: float,
    remaining_seconds: Callable[[], float] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[TailPrefixActionV1, ...]:
    """Re-solve the exact decision boundary over only its live cards.

    Full-round normalization may solve additional continuation variables.  The
    canonical root legal set must instead come from the same remaining-card
    transition model used by strategic proof.
    """

    remaining = remaining_seconds or _normalization_clock(maximum_seconds, cancel_check)
    remaining()
    current = state.get("currentTrick")
    hands = state.get("hands")
    if not isinstance(current, Mapping) or not isinstance(hands, Mapping):
        raise ValueError("native play boundary requires current trick and hands")
    ownership: dict[Card, Seat] = {}
    for seat in Seat:
        rows = hands.get(seat.name.lower())
        if not isinstance(rows, list):
            raise ValueError("native play boundary requires every remaining hand")
        for row in rows:
            ownership[Card.parse(row)] = seat
    prefix = tuple(
        TailPrefixActionV1.build(
            item.get("actor") or item.get("seat"),
            item.get("card"),
            ekka=item.get("ekka") is True,
        )
        for item in current.get("plays", ())
    )
    for action in prefix:
        ownership[action.card] = action.actor
    try:
        result = OrToolsPlayTailStageV1().solve(
            cards=tuple(sorted(ownership, key=lambda card: card.id)),
            ownership=ownership,
            contract=contract,
            trump=trump,
            first_leader=Seat.parse(current.get("leader")),
            play_domain=rules.play_domain,
            structural_domain=rules.structural_domain,
            prefix=prefix,
            locked=bool(state.get("locked")),
            decision_prefix_length=len(prefix),
            # Legality is local to the current trick; do not construct an unused round suffix.
            play_step_count=len(Seat),
            maximum_time_seconds=remaining(),
            cancel_check=cancel_check,
        )
    except PlayTailSolveStopped as exc:
        if exc.solver_status == "infeasible":
            raise
        raise ExactNormalizationStopped(exc.solver_status, evidence={
            "sourceStage": "boundary_legal_actions", "causeReceipt": exc.partial_receipt,
        }) from exc
    remaining()
    if result.solver_status != "optimal":
        raise ExactNormalizationStopped(result.solver_status, evidence={"sourceStage": "boundary_legal_actions"})
    return result.legal_play_actions


def _project_declarations(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = ("owner", "kind", "value", "highRank", "declarationOrder")
    return [
        {key: event[key] for key in keys}
        for event in events
        if event.get("type") == "project_declared"
    ]


def _verify_project_assertions(
    projects: Sequence[Project],
    assertions: Sequence[Mapping[str, Any]],
    *,
    domain,
    trump: Suit | None,
    second_trick_leader: Seat | None = None,
) -> int:
    comparison_winner = domain.comparison_winner(
        projects,
        trump_suit=trump,
        second_trick_leader=second_trick_leader,
    ) if any(item.get("state") == "comparison_winner" for item in assertions) else None
    for assertion in assertions:
        expected = dict(assertion)
        lifecycle = str(expected.pop("state", "eligible"))

        def matches(project: Project) -> bool:
            values = {
                "owner": project.owner.name.lower(),
                "team": project.team.name.lower(),
                "kind": project.kind,
                "value": project.value,
                "highRank": project.high_rank,
            }
            if any(values.get(key) != value for key, value in expected.items()):
                return False
            return {
                "eligible": True,
                "declared": project.declared,
                "comparison_winner": project == comparison_winner,
                "awarded": project.scoring,
            }[lifecycle]

        if not any(matches(project) for project in projects):
            raise ValueError(
                "exact-state project assertion does not match CP-SAT projects"
            )
    return len(assertions)


def _replay_play_prefix(
    state: dict[str, Any],
    *,
    play: OrToolsPlayTailResultV1,
    projects: list[Project],
    contract: Contract,
    trump: Suit | None,
    rules: RuleBook,
) -> tuple[list[Project], frozenset[Team]]:
    project_domain = rules.project_domain
    state["phase"] = "playing"
    state["projects"] = [project.to_dict() for project in projects]
    events: list[dict[str, Any]] = state["events"]

    def emit(event_type: str, **data: Any) -> None:
        events.append({"sequence": len(events), "type": event_type, **data})

    emit("project_holdings_detected", count=len(projects))
    first_leader = Seat.parse(state["firstLeader"])
    emit("play_started", firstLeader=first_leader.name.lower())
    completed: list[dict[str, Any]] = []
    current: dict[str, Any] | None = {
        "index": 0,
        "leader": first_leader.name.lower(),
        "plays": [],
        "winner": None,
        "rawPoints": 0,
    }
    baloot_teams: set[Team] = set()
    initial_hands = {seat: set(play.initial_hands[int(seat)]) for seat in Seat}
    played_before: list[TailPrefixActionV1] = []
    project_winner: Team | None = None
    for step, action in enumerate(play.plays[: play.prefix_length]):
        trick_index, position = divmod(step, 4)
        if current is None:
            raise AssertionError("play prefix continued after a terminal trick")
        if project_domain.should_announce(trick_index):
            projects, announced = project_domain.announce_for_turn(
                projects,
                action.actor,
                trump_suit=trump,
            )
            if announced is not None:
                emit(
                    "project_declared",
                    owner=announced.owner.name.lower(),
                    kind=announced.kind,
                    value=announced.value,
                    highRank=announced.high_rank,
                    declarationOrder=announced.declaration_order,
                )
        if project_domain.should_reveal(trick_index):
            projects, revealed = project_domain.reveal_for_turn(
                projects,
                action.actor,
                trump_suit=trump,
                second_trick_leader=Seat.parse(completed[0]["winner"]) if completed else None,
            )
            if revealed:
                if (
                    project_domain.baloot_enabled
                    and contract == Contract.HOKUM
                    and trump is not None
                ):
                    included, _blocked = project_domain.baloot_project_state(
                        projects,
                        owner=action.actor,
                        trump_suit=trump,
                    )
                    if included and action.actor.team not in baloot_teams:
                        baloot_teams.add(action.actor.team)
                        emit(
                            "baloot_included_in_project",
                            owner=action.actor.name.lower(),
                            team=action.actor.team.name.lower(),
                        )
                emit(
                    "projects_revealed",
                    owner=action.actor.name.lower(),
                    projects=[project.to_dict() for project in revealed],
                )
        baloot = False
        if (
            project_domain.baloot_enabled
            and contract == Contract.HOKUM
            and trump is not None
            and action.card.suit == trump
            and action.card.rank in {Rank.KING, Rank.QUEEN}
        ):
            other_rank = Rank.QUEEN if action.card.rank == Rank.KING else Rank.KING
            other = Card.of(trump, other_rank)
            already_played = any(
                item.actor == action.actor and item.card == other
                for item in played_before
            )
            included, blocked = project_domain.baloot_project_state(
                projects,
                owner=action.actor,
                trump_suit=trump,
            )
            baloot = (
                other in initial_hands[action.actor]
                and already_played
                and not blocked
                and action.actor.team not in baloot_teams
            )
            if baloot:
                baloot_teams.add(action.actor.team)
                emit(
                    "baloot_declared",
                    seat=action.actor.name.lower(),
                    team=action.actor.team.name.lower(),
                )
        legal_cards = play.legal_actions_by_step[step]
        row = {
            "seat": action.actor.name.lower(),
            "card": action.card.to_dict(),
            "trickIndex": trick_index,
            "position": position,
            "legalCards": [card.to_dict() for card in legal_cards],
            "ekka": action.ekka,
            "baloot": baloot,
        }
        current["plays"].append(row)
        emit("card_played", **row)
        played_before.append(action)
        if (
            trump is not None
            and action.card.suit == trump
            and current["plays"][0]["card"]["suit"] != trump.name.lower()
        ):
            state["trumpBroken"] = True
        if position != 3:
            continue
        winner = play.trick_winners[trick_index]
        raw_points = play.trick_raw_points[trick_index]
        current["winner"] = winner.name.lower()
        current["rawPoints"] = raw_points
        completed.append(current)
        emit(
            "trick_completed",
            trickIndex=trick_index,
            winner=winner.name.lower(),
            rawPoints=raw_points,
        )
        if project_domain.should_compare(trick_index) and projects:
            projects, project_winner = project_domain.score_team(
                projects,
                trump_suit=trump,
                second_trick_leader=play.trick_winners[0],
            )
            emit(
                "projects_compared",
                winningTeam=(
                    project_winner.name.lower() if project_winner is not None else None
                ),
                declarations=[
                    project.public_declaration()
                    for project in projects
                    if project.declared
                ],
            )
        if trick_index == 1 and projects:
            emit(
                "projects_collected",
                owners=sorted(
                    {
                        project.owner.name.lower()
                        for project in projects
                        if project.revealed
                    }
                ),
            )
        current = (
            None
            if step + 1 == 32
            else {
                "index": trick_index + 1,
                "leader": winner.name.lower(),
                "plays": [],
                "winner": None,
                "rawPoints": 0,
            }
        )
    state["hands"] = {
        seat.name.lower(): [card.to_dict() for card in play.remaining_hands[int(seat)]]
        for seat in Seat
    }
    state["currentPlayer"] = (
        play.next_actor.name.lower() if play.next_actor is not None else None
    )
    state["currentTrick"] = current
    state["completedTricks"] = completed
    state["projects"] = [project.to_dict() for project in projects]
    state["projectWinningTeam"] = (
        project_winner.name.lower() if project_winner is not None else None
    )
    state["balootTeams"] = [team.name.lower() for team in sorted(baloot_teams, key=int)]
    return projects, frozenset(baloot_teams)


def _render_native_play_prefix(
    state: dict[str, Any],
    *,
    play: OrToolsPlayTailResultV1,
    projection: _NativeProjectPrefixProjectionV1,
    trump: Suit | None,
    project_actions: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[Project], frozenset[Team]]:
    """Extract the public state solely from solved play/project variables."""

    if len(projection.baloot_events) != play.prefix_length:
        raise ValueError("native project evidence does not cover the play prefix")
    projects = list(projection.projects)
    state["phase"] = "playing"
    state["projects"] = [project.to_dict() for project in projects]
    events: list[dict[str, Any]] = state["events"]

    def emit(event_type: str, **data: Any) -> None:
        events.append({"sequence": len(events), "type": event_type, **data})

    emit("project_holdings_detected", count=len(projects))
    first_leader = Seat.parse(state["firstLeader"])
    emit("play_started", firstLeader=first_leader.name.lower())
    completed: list[dict[str, Any]] = []
    current: dict[str, Any] | None = {
        "index": 0,
        "leader": first_leader.name.lower(),
        "plays": [],
        "winner": None,
        "rawPoints": 0,
    }
    declared_owners: set[Seat] = set()
    revealed_owners: set[Seat] = set()

    def emit_project_action(raw: Mapping[str, Any]) -> None:
        action = ProjectDeclarationActionV1.build(raw)
        declared_owners.add(action.actor)
        if action.action == "pass":
            emit("project_passed", owner=action.actor.name.lower())
            return
        announced = next(
            (
                project
                for project in projects
                if project.owner == action.actor
                and project_id_v1(project) == action.project_id
                and project.declared
            ),
            None,
        )
        if announced is None:
            raise AssertionError("solved project declaration action is missing")
        emit(
            "project_declared",
            owner=announced.owner.name.lower(),
            kind=announced.kind,
            value=announced.value,
            highRank=announced.high_rank,
            declarationOrder=announced.declaration_order,
        )

    for step, action in enumerate(play.plays[: play.prefix_length]):
        trick_index, position = divmod(step, 4)
        if current is None:
            raise AssertionError("native play prefix continued after a terminal trick")
        if trick_index == 0 and project_actions is not None:
            emit_project_action(project_actions[step])
        elif trick_index == 0 and action.actor not in declared_owners:
            announced = next(
                (
                    project
                    for project in projects
                    if project.owner == action.actor and project.declared
                ),
                None,
            )
            if announced is not None:
                declared_owners.add(action.actor)
                emit(
                    "project_declared",
                    owner=announced.owner.name.lower(),
                    kind=announced.kind,
                    value=announced.value,
                    highRank=announced.high_rank,
                    declarationOrder=announced.declaration_order,
                )
        if trick_index == 1 and action.actor not in revealed_owners:
            revealed = tuple(
                project
                for project in projects
                if project.owner == action.actor
                and project.scoring
                and project.revealed
            )
            if revealed:
                revealed_owners.add(action.actor)
                if projection.baloot_events[step] == "included":
                    emit(
                        "baloot_included_in_project",
                        owner=action.actor.name.lower(),
                        team=action.actor.team.name.lower(),
                    )
                emit(
                    "projects_revealed",
                    owner=action.actor.name.lower(),
                    projects=[project.to_dict() for project in revealed],
                )
        baloot = projection.baloot_events[step] == "declared"
        if baloot:
            emit(
                "baloot_declared",
                seat=action.actor.name.lower(),
                team=action.actor.team.name.lower(),
            )
        legal_cards = play.legal_actions_by_step[step]
        row = {
            "seat": action.actor.name.lower(),
            "card": action.card.to_dict(),
            "trickIndex": trick_index,
            "position": position,
            "legalCards": [card.to_dict() for card in legal_cards],
            "ekka": action.ekka,
            "baloot": baloot,
        }
        current["plays"].append(row)
        emit("card_played", **row)
        if (
            trump is not None
            and action.card.suit == trump
            and current["plays"][0]["card"]["suit"] != trump.name.lower()
        ):
            state["trumpBroken"] = True
        if position != 3:
            continue
        winner = play.trick_winners[trick_index]
        raw_points = play.trick_raw_points[trick_index]
        current["winner"] = winner.name.lower()
        current["rawPoints"] = raw_points
        completed.append(current)
        emit(
            "trick_completed",
            trickIndex=trick_index,
            winner=winner.name.lower(),
            rawPoints=raw_points,
        )
        if trick_index == 0 and projects:
            emit(
                "projects_compared",
                winningTeam=(
                    projection.winning_team.name.lower()
                    if projection.winning_team is not None
                    else None
                ),
                declarations=[
                    project.public_declaration()
                    for project in projects
                    if project.declared
                ],
            )
        if trick_index == 1 and projects:
            emit(
                "projects_collected",
                owners=sorted(
                    {
                        project.owner.name.lower()
                        for project in projects
                        if project.revealed
                    }
                ),
            )
        current = (
            None
            if step + 1 == 32
            else {
                "index": trick_index + 1,
                "leader": winner.name.lower(),
                "plays": [],
                "winner": None,
                "rawPoints": 0,
            }
        )
    if project_actions is not None:
        for raw in project_actions[min(play.prefix_length, 4):]:
            emit_project_action(raw)
    state["hands"] = {
        seat.name.lower(): [card.to_dict() for card in play.remaining_hands[int(seat)]]
        for seat in Seat
    }
    state["currentPlayer"] = (
        play.next_actor.name.lower() if play.next_actor is not None else None
    )
    state["currentTrick"] = current
    state["completedTricks"] = completed
    state["projects"] = [project.to_dict() for project in projects]
    state["projectWinningTeam"] = (
        projection.winning_team.name.lower()
        if projection.winning_team is not None
        else None
    )
    state["balootTeams"] = [
        team.name.lower() for team in sorted(projection.baloot_teams, key=int)
    ]
    return projects, projection.baloot_teams




def _score_terminal_state(
    state: dict[str, Any],
    *,
    play: OrToolsPlayTailResultV1,
    projects: Sequence[Project],
    baloot_teams: frozenset[Team],
    contract: Contract,
    declarer: Seat,
    scoring: ScoringDomainIRV1,
    rules: RuleBook,
    native_cp_sat: bool = False,
    native_result: OrToolsScoringResultV1 | None = None,
) -> OrToolsScoringResultV1 | None:
    if play.prefix_length != 32 or len(state["completedTricks"]) != 8:
        return None
    if play.raw_team_points is None or play.card_game_points is None:
        raise AssertionError("complete CP-SAT play lacks scoring evidence")
    raw = tuple(int(value) for value in play.raw_team_points)
    escalation = str(state["escalation"])
    history = state["escalationHistory"]
    if native_cp_sat:
        if native_result is None:
            raise ValueError("native exact scoring requires a same-model binding")
        result = native_result
        if result.status != "satisfied" or result.score is None:
            raise RuntimeError(
                f"native CP-SAT exact scoring did not complete: {result.solver_status}"
            )
        native = result.score
        score = {
            "rawCardPoints": list(native["rawPoints"]),
            "cardGamePoints": list(native["cardGamePoints"]),
            "projectPoints": list(native["projectPoints"]),
            "balootPoints": list(native["balootPoints"]),
            "totalBeforeAward": list(native["totalBeforeAward"]),
            "winner": native["roundWinner"],
            "awardedPoints": list(native["gamePoints"]),
            "countingTeam": native["countingTeam"],
            "multiplier": native["multiplier"],
            "kabout": native["kabout"],
            "gahwa": native["gahwa"],
            "matchWinner": native["matchWinner"],
            "explanation": [
                f"counting_team={native['countingTeam']}",
                f"card_game_points={list(native['cardGamePoints'])}",
                f"projects={list(native['projectPoints'])}",
                f"baloot={list(native['balootPoints'])}",
            ],
        }
        state["roundScore"] = score
        state["matchScore"] = list(native["finalMatchScore"])
        state["matchWinner"] = native["matchWinner"]
        state["phase"] = "scored"
        state["events"].append(
            {
                "sequence": len(state["events"]),
                "type": "round_scored",
                "score": copy.deepcopy(score),
                "matchScore": list(native["finalMatchScore"]),
                "matchWinner": native["matchWinner"],
            }
        )
        return result
    counting = scoring.resolve_counting_team(
        declarer_team=declarer.team,
        escalation=escalation,
        last_escalating_team=(
            Team[str(history[-1]["team"]).upper()] if history else None
        ),
    )
    total_cards = scoring.card_game_point_totals[contract.effective.value]
    converted = scoring.convert_raw(raw[int(counting)], contract)
    card_game = [0, 0]
    card_game[int(counting)] = converted
    card_game[int(counting.other)] = total_cards - converted
    if tuple(card_game) != play.card_game_points:
        raise AssertionError("CP-SAT card-game scoring disagrees with Domain IR")
    project_points = rules.project_domain.project_points(projects)
    baloot_points = tuple(
        rules.project_domain.baloot_value * int(team in baloot_teams) for team in Team
    )
    totals = [
        card_game[index] + project_points[index] + baloot_points[index]
        for index in range(2)
    ]
    if totals[0] > totals[1]:
        winner = Team.NORTH_SOUTH
    elif totals[1] > totals[0]:
        winner = Team.WEST_EAST
    else:
        winner = scoring.tie_winner(
            contract,
            raw[int(counting)],
            counting,
            declarer.team,
        )
    tricks_won = [0, 0]
    for trick in state["completedTricks"]:
        tricks_won[int(Seat.parse(trick["winner"]).team)] += 1
    kabout = scoring.kabout_enabled and 0 in tricks_won
    if kabout and scoring.kabout_winner == "trick_sweeper":
        winner = Team.NORTH_SOUTH if tricks_won[0] else Team.WEST_EAST
    gahwa = escalation == "gahwa"
    awarded = scoring.round_award(
        contract=contract,
        winner=winner,
        declarer_team=declarer.team,
        escalation=escalation,
        total_before_award=totals,
        project_points=project_points,
        baloot_points=baloot_points,
        kabout=kabout,
    )
    final_match_score, match_winner = scoring.apply_match_award(
        starting_score=state["matchScore"],
        awarded_points=awarded,
        round_winner=winner,
        gahwa=gahwa,
    )
    match_score = list(final_match_score)
    score = {
        "rawCardPoints": list(raw),
        "cardGamePoints": card_game,
        "projectPoints": list(project_points),
        "balootPoints": list(baloot_points),
        "totalBeforeAward": totals,
        "winner": winner.name.lower(),
        "awardedPoints": list(awarded),
        "countingTeam": counting.name.lower(),
        "multiplier": scoring.multiplier(escalation),
        "kabout": kabout,
        "gahwa": gahwa,
        "matchWinner": (
            match_winner.name.lower() if match_winner is not None else None
        ),
        "explanation": [
            f"counting_team={counting.name.lower()}",
            f"card_game_points={card_game}",
            f"projects={list(project_points)}",
            f"baloot={list(baloot_points)}",
        ],
    }
    state["roundScore"] = score
    state["matchScore"] = match_score
    state["matchWinner"] = (
        match_winner.name.lower() if match_winner is not None else None
    )
    state["phase"] = "scored"
    state["events"].append(
        {
            "sequence": len(state["events"]),
            "type": "round_scored",
            "score": copy.deepcopy(score),
            "matchScore": match_score,
            "matchWinner": state["matchWinner"],
        }
    )
    return None


def _initial_deal_trace(deck, layout, purchase_card):
    trace = [_deal_row(deck, packet, sequence)
             for sequence, packet in enumerate(layout.initial_packets)]
    trace.append({"sequence": len(trace), "stage": "purchase", "packet": 0,
                  "seat": None, "count": 1, "faceUp": True,
                  "cards": [purchase_card.to_dict()]})
    return trace


def _initial_deal_events(spec, prefix_hash, layout, purchase_card):
    seed = int(prefix_hash[:16], 16)
    dealer = Seat.parse(spec.dealer).name.lower()
    events = []
    def emit(kind, **data):
        events.append({"sequence": len(events), "type": kind, **data})
    emit("round_created", dealer=dealer, seed=seed, roundNumber=spec.round_number,
         matchScore=list(spec.match_score))
    emit("deck_shuffled", algorithm="governed_preordered_v1", seed=seed, dealer=dealer,
         cutter=None, cutPosition=None, deckCommitment=hashlib.sha256(bytes(spec.deck_order)).hexdigest())
    for packet in layout.initial_packets:
        emit("cards_dealt", stage=packet.stage, packet=packet.packet,
             seat=packet.seat.name.lower(), count=len(packet.positions))
    emit("purchase_card_exposed", card=purchase_card.to_dict())
    emit("initial_deal_complete", firstBidder=layout.initial_packets[0].seat.name.lower(), cutter=None)
    return events


def _build_state(
    *,
    rules: RuleBook,
    spec: ExactTracePrefixV1,
    prefix_hash: str,
    solution: OrToolsScenarioDealSolutionV1,
    native_roles: _NativeRoleProjectionV1 | None = None,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...], str | None, str]:
    if solution.deck_order is None or solution.purchase_card is None:
        raise AssertionError("satisfied exact-input solve lacks a deal")
    if solution.auction is None:
        raise AssertionError("satisfied exact-input solve lacks auction evidence")
    deck = solution.deck_order
    auction = solution.auction
    dealer = Seat.parse(spec.dealer)
    initial_layout = DealLayoutV1.build(
        rules.structural_domain,
        dealer=dealer,
    )
    seed = int(prefix_hash[:16], 16)
    initial_hands = _hand_rows(deck, initial_layout.initial_positions)
    zero_hands = {seat.name.lower(): [] for seat in Seat}
    deal_trace = _initial_deal_trace(deck, initial_layout, solution.purchase_card)

    events = _initial_deal_events(spec, prefix_hash, initial_layout, solution.purchase_card)

    def emit(event_type: str, **data: Any) -> None:
        events.append({"sequence": len(events), "type": event_type, **data})

    commitment = hashlib.sha256(bytes(deck)).hexdigest()

    bid_rows = _bid_rows(auction.actions)
    buyer = auction.buyer
    contract = auction.contract
    trump = auction.trump
    first_leader: Seat | None = None
    if auction.prefix_terminal:
        if (
            buyer is None
            or contract is None
            or auction.purchase_round is None
            or auction.caller is None
            or auction.prefix_purchase_recipient is None
            or auction.declarer is None
        ):
            raise AssertionError("terminal auction prefix lacks its derived result")
        caller = native_roles.caller if native_roles else auction.caller
        contract_owner = native_roles.contract_owner if native_roles else buyer
        purchase_recipient = (
            native_roles.purchase_recipient
            if native_roles
            else auction.prefix_purchase_recipient
        )
        declarer = native_roles.declarer if native_roles else auction.declarer
        if native_roles is not None and (
            caller,
            buyer,
            purchase_recipient,
            declarer,
        ) != (
            auction.caller,
            auction.buyer,
            auction.prefix_purchase_recipient,
            auction.declarer,
        ):
            raise AssertionError("native exact roles differ from auction semantics")
        final_layout = DealLayoutV1.build(
            rules.structural_domain,
            dealer=dealer,
            buyer=purchase_recipient,
        )
        if final_layout.final_hand_positions is None:
            raise AssertionError("terminal deal lacks final hand positions")
        if native_roles is not None:
            if solution.owner_of_card is None:
                raise ValueError(
                    "native exact deal lacks solved final ownership variables"
                )
            solved_hands = {
                seat: tuple(
                    Card(card_id)
                    for card_id, owner in enumerate(solution.owner_of_card)
                    if owner == seat
                )
                for seat in Seat
            }
            if any(
                len(solved_hands[seat]) != rules.structural_domain.cards_per_player
                for seat in Seat
            ):
                raise AssertionError(
                    "native exact deal did not solve one complete hand per seat"
                )
            hands = {
                seat.name.lower(): [card.to_dict() for card in solved_hands[seat]]
                for seat in Seat
            }
        else:
            hands = _hand_rows(deck, final_layout.final_hand_positions)
        initial_hand_rows = copy.deepcopy(hands)
        for row in bid_rows:
            emit("bid_made", **row)
        winning_bidder = next(
            row["bidder"] for row in reversed(bid_rows) if row["bidder"] is not None
        )
        if winning_bidder != caller.name.lower():
            raise AssertionError("exact auction caller differs from its bid trace")
        emit(
            "contract_purchased",
            winningBidder=caller.name.lower(),
            buyer=buyer.name.lower(),
            contractOwner=contract_owner.name.lower(),
            purchaseRecipient=purchase_recipient.name.lower(),
            declarer=declarer.name.lower(),
            contract=contract.value,
            trumpSuit=trump.name.lower() if trump is not None else None,
            purchaseCard=solution.purchase_card.to_dict(),
        )
        for packet in final_layout.final_packets:
            deal_trace.append(_deal_row(deck, packet, len(deal_trace)))
            emit(
                "cards_dealt",
                stage=packet.stage,
                packet=packet.packet,
                seat=packet.seat.name.lower(),
                count=len(packet.positions),
            )
        first_leader = (
            native_roles.first_leader
            if native_roles is not None
            else rules.play_domain.resolve_first_leader(
                dealer=dealer,
                buyer=buyer,
                structural_domain=rules.structural_domain,
            )
        )
        emit(
            "final_deal_complete",
            buyer=buyer.name.lower(),
            contractOwner=contract_owner.name.lower(),
            purchaseRecipient=purchase_recipient.name.lower(),
            firstLeader=first_leader.name.lower(),
        )
        phase = "final_dealt"
        legal_actions: tuple[Mapping[str, Any], ...] = ()
        next_actor = None
        bidding_history = bid_rows
    else:
        hands = initial_hands
        initial_hand_rows = zero_hands
        phase = "bidding"
        legal_actions = _legal_bid_rows(auction, rules)
        next_actor = (
            auction.current_actor.name.lower()
            if auction.current_actor is not None
            else None
        )
        bidding_history = []

    winning_bidder = next(
        (row["bidder"] for row in reversed(bid_rows) if row["bidder"] is not None),
        None,
    )
    state: dict[str, Any] = {
        "rulesHash": rules.sha256,
        "seed": seed,
        "roundNumber": spec.round_number,
        "dealer": dealer.name.lower(),
        "phase": phase,
        "matchScore": list(spec.match_score),
        "shuffle": {
            "algorithm": "governed_preordered_v1",
            "seed": seed,
            "dealer": dealer.name.lower(),
            "cutter": None,
            "cutPosition": None,
            "deckCommitment": commitment,
            "deckOrder": list(deck),
        },
        "dealTrace": deal_trace,
        "purchaseCard": solution.purchase_card.to_dict(),
        "hands": hands,
        "initialHands": initial_hand_rows,
        "biddingHistory": bidding_history,
        "contract": contract.value if contract is not None else None,
        "winningBidder": winning_bidder if auction.prefix_terminal else None,
        "buyer": buyer.name.lower() if buyer is not None else None,
        "contractOwner": (
            native_roles.contract_owner.name.lower()
            if native_roles is not None
            else buyer.name.lower()
            if buyer is not None
            else None
        ),
        "purchaseRecipient": (
            native_roles.purchase_recipient.name.lower()
            if native_roles is not None
            else purchase_recipient.name.lower()
            if auction.prefix_terminal
            else None
        ),
        "declarer": (
            native_roles.declarer.name.lower()
            if native_roles is not None
            else declarer.name.lower()
            if auction.prefix_terminal
            else None
        ),
        "trumpSuit": trump.name.lower() if trump is not None else None,
        "escalation": "normal",
        "escalationHistory": [],
        "locked": False,
        "firstLeader": first_leader.name.lower() if first_leader is not None else None,
        "currentPlayer": first_leader.name.lower()
        if first_leader is not None
        else None,
        "currentTrick": None,
        "completedTricks": [],
        "projects": [],
        "projectWinningTeam": None,
        "balootTeams": [],
        "trumpBroken": False,
        "roundScore": None,
        "matchWinner": None,
        "annulledBy": None,
        "annulReason": None,
        "events": events,
    }
    rank_visibility = project_rank_visibility_policy(rules)
    if rank_visibility is not None:
        state["projectRankVisibility"] = rank_visibility
    if not auction.prefix_terminal:
        state["bidding"] = {
            "round": auction.current_round,
            "actor": next_actor,
            "history": bid_rows,
        }
    return state, legal_actions, next_actor, phase


@dataclass(frozen=True, slots=True)
class OrToolsExactNormalizationReceiptV1:
    payload: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return str(self.payload["sha256"])

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.payload))


def _exact_state_identity(*, prefix_hash, rules_hash, state, phase, decision_kind, next_actor, legal_actions):
    state_hash = canonical_sha256(state)
    event_hash = canonical_sha256(state["events"])
    digest = canonical_sha256({
        "schemaVersion": 1, "prefixHash": prefix_hash, "rulesHash": rules_hash,
        "stateHash": state_hash, "phase": phase, "decisionKind": decision_kind,
        "nextActor": next_actor, "legalActions": [dict(item) for item in legal_actions],
        "eventChainHash": event_hash,
    })
    return state_hash, event_hash, digest


@dataclass(frozen=True, slots=True)
class OrToolsNormalizedExactStateV1:
    exact_state_id: str
    prefix_hash: str
    rules_hash: str
    state_hash: str
    phase: str
    decision_kind: str
    next_actor: str | None
    legal_actions: tuple[Mapping[str, Any], ...]
    state: Mapping[str, Any]
    project_declarations_verified: int
    project_assertions_verified: int
    event_chain_hash: str
    sha256: str
    normalization_receipt: OrToolsExactNormalizationReceiptV1
    _play_template: tuple[str, str, OrToolsPlayTailBuildV1, OrToolsProjectLifecycleFragmentV1] | None = dataclass_field(default=None, repr=False, compare=False)

    def require_integrity(self) -> None:
        state_hash, event_hash, digest = _exact_state_identity(
            prefix_hash=self.prefix_hash, rules_hash=self.rules_hash, state=self.state,
            phase=self.phase, decision_kind=self.decision_kind, next_actor=self.next_actor,
            legal_actions=self.legal_actions,
        )
        if (state_hash, event_hash, digest, f"STATE-{digest[:20].upper()}") != (
            self.state_hash, self.event_chain_hash, self.sha256, self.exact_state_id,
        ):
            raise ValueError("exact normalization state identity mismatch")
        receipt = dict(self.normalization_receipt.payload)
        receipt_hash = receipt.pop("sha256", None)
        if receipt_hash != canonical_sha256(receipt) or any(
            receipt.get(key) != value for key, value in {
                "exactStateId": self.exact_state_id, "stateHash": self.state_hash,
                "prefixHash": self.prefix_hash, "rulesHash": self.rules_hash,
            }.items()
        ):
            raise ValueError("exact normalization receipt identity mismatch")

    def recorded_escalation_history(self) -> dict[str, Any] | None:
        """Read owned private history without exposing it as actor information."""
        history = copy.deepcopy(self.normalization_receipt.payload.get("derivedEscalationOpportunities"))
        if self.state.get("escalationOpportunityHistoryHash") is not None and history is None:
            raise ValueError("recorded escalation proof history is missing")
        if history is not None and (
            self.decision_kind != "escalation" or history.get("status") != "satisfied"
            or history.get("nextActor") != self.next_actor
            or history.get("sha256") != canonical_sha256({key: value for key, value in history.items() if key != "sha256"})
            or self.state.get("escalationOpportunityHistoryHash") != canonical_sha256({
                "events": history["events"], "order": history["opportunityOrder"]})
        ):
            raise ValueError("recorded escalation proof history disagrees with normalized state")
        return history

    def to_dict(self) -> dict[str, Any]:
        """Preserve the existing neutral exact-state wire shape."""

        return {
            "schemaVersion": 1,
            "exactStateId": self.exact_state_id,
            "prefixHash": self.prefix_hash,
            "rulesHash": self.rules_hash,
            "stateHash": self.state_hash,
            "phase": self.phase,
            "decisionKind": self.decision_kind,
            "nextActor": self.next_actor,
            "legalActions": [copy.deepcopy(dict(item)) for item in self.legal_actions],
            "state": copy.deepcopy(dict(self.state)),
            "projectDeclarationsVerified": self.project_declarations_verified,
            "projectAssertionsVerified": self.project_assertions_verified,
            "eventChainHash": self.event_chain_hash,
            "sha256": self.sha256,
        }


class OrToolsExactStateNormalizerV1:
    """Normalize a fixed legal trace through engine-free CP-SAT stage owners."""

    def __init__(self, rules: RuleBook, *, native_cp_sat: bool = False):
        if not isinstance(rules, RuleBook):
            raise TypeError("OR-Tools exact normalization requires RuleBook")
        if type(native_cp_sat) is not bool:
            raise TypeError("native_cp_sat must be bool")
        self.rules = rules
        self.native_cp_sat = native_cp_sat
        self._deal_prefix_certificate = None
        self._terminal_escalation_certificate = None
        self._paired_pass_cache: dict[str, tuple[str, _NativePlayNormalizationV1]] = {}
        # Four protocol positions, not a growing cache of searched states.
        self._first_trick_templates: dict[int, tuple[str, str, OrToolsPlayTailBuildV1, OrToolsProjectLifecycleFragmentV1]] = {}

    def _normalize_all_pass(self, spec, decision_kind, remaining, cancel_check):
        from .ortools_scenario_model import attach_auction_path_v1, OrToolsNativeAuctionFragmentV1

        if decision_kind != "bid" or any((spec.escalation_actions, spec.project_actions,
                spec.project_declarations, spec.project_assertions, spec.play_actions)):
            raise ValueError("all-pass auction cannot contain post-purchase actions")
        if not self.rules.structural_domain.all_pass_redeal:
            raise ValueError("rules do not support all-pass redeal")
        dealer = Seat.parse(spec.dealer)
        layout = DealLayoutV1.build(self.rules.structural_domain, dealer=dealer)
        purchase = Card(spec.deck_order[layout.purchase_position])
        domain = AuctionDomainIRV1.from_rules(self.rules.data)
        model = cp_model.CpModel()
        states, choices = attach_auction_path_v1(model, domain,
            dealer=model.new_constant(int(dealer)), purchase_suit=model.new_constant(int(purchase.suit)),
            purchase_rank=model.new_constant(int(purchase.rank)), maximum_actions=8)
        for choice in choices:
            model.add(choice["pass"] == 1)
        model.add(states[-1]["allPassed"] == 1)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = remaining()
        model_hash = canonical_sha256(str(model.proto))
        status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
        if status == cp_model.INFEASIBLE:
            raise ValueError("exact all-pass auction is infeasible")
        if status != cp_model.OPTIMAL:
            raise ExactNormalizationStopped("cancelled" if cancel_check and cancel_check() else "maximum_seconds")
        records = OrToolsNativeAuctionFragmentV1(states, choices, 8, {}, {}, domain.sha256).extract_actions(solver, length=8)
        for supplied, record in zip(spec.auction_actions, records, strict=True):
            if supplied.get("actor") is not None and Seat.parse(supplied["actor"]) != record.actor:
                raise ValueError("all-pass auction action has the wrong actor")
            if supplied.get("trumpSuit") is not None:
                raise ValueError("pass cannot select a trump suit")
        variables = [value for state in states for value in state.values()]
        values = [solver.value(value) for value in variables]
        model.add_forbidden_assignments(variables, [values])
        uniqueness_model_hash = canonical_sha256(str(model.proto))
        solver.parameters.max_time_in_seconds = remaining()
        unique = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
        if unique != cp_model.INFEASIBLE:
            raise ExactNormalizationStopped("all_pass_uniqueness_incomplete")
        prefix_hash = _effective_prefix_hash(spec)
        events = _initial_deal_events(spec, prefix_hash, layout, purchase)
        bid_rows = _bid_rows(records)
        for row in bid_rows:
            events.append({"sequence": len(events), "type": "bid_made", **row})
        events.append({"sequence": len(events), "type": "all_pass_redeal",
                       "nextDealer": self.rules.structural_domain.dealer_right(dealer).name.lower()})
        state = {"rulesHash": self.rules.sha256, "phase": "complete", "dealer": dealer.name.lower(),
            "seed": int(prefix_hash[:16], 16),
            "shuffle": {key: value for key, value in events[1].items()
                        if key not in {"sequence", "type"}},
            "dealTrace": _initial_deal_trace(spec.deck_order, layout, purchase),
            "roundNumber": spec.round_number, "matchScore": list(spec.match_score),
            "contract": None, "buyer": None, "declarer": None, "trumpSuit": None,
            "currentPlayer": None, "currentTrick": None, "completedTricks": [],
            "purchaseCard": purchase.to_dict(), "hands": _hand_rows(spec.deck_order, layout.initial_positions),
            "initialHands": _hand_rows(spec.deck_order, layout.initial_positions),
            "biddingHistory": bid_rows,
            "winningBidder": None, "contractOwner": None, "purchaseRecipient": None,
            "escalation": "normal", "escalationHistory": [], "locked": False,
            "firstLeader": None, "projects": [], "projectWinningTeam": None,
            "balootTeams": [], "trumpBroken": False, "roundScore": None,
            "matchWinner": None, "annulledBy": None, "annulReason": None,
            "auctionTerminal": True, "allPassed": True, "events": events}
        state["shuffle"]["deckOrder"] = list(spec.deck_order)
        for path, expected in spec.state_assertions.items():
            if canonical_sha256(_mapping_path(state, str(path))) != canonical_sha256(expected):
                raise ValueError(f"exact-state assertion mismatch at {path}")
        state_hash, event_hash, digest = _exact_state_identity(
            prefix_hash=prefix_hash, rules_hash=self.rules.sha256, state=state,
            phase="complete", decision_kind="bid", next_actor=None, legal_actions=(),
        )
        exact_id = f"STATE-{digest[:20].upper()}"
        receipt = {"schemaVersion": 1, "backendId": _BACKEND_ID_V3, "normalizerId": _NORMALIZER_ID,
            "solverVersion": _ORTOOLS_VERSION, "rulesHash": self.rules.sha256,
            "prefixHash": prefix_hash, "modelHash": model_hash, "solverStatus": "optimal",
            "uniquenessStatus": "proven_unique", "uniquenessSolverStatus": "infeasible",
            "uniquenessModelHash": uniqueness_model_hash,
            "structuralDomainIrHash": self.rules.structural_domain.sha256,
            "semanticSolutionVectorHash": canonical_sha256(values),
            "solutionHash": canonical_sha256({"deckOrder": list(spec.deck_order), "auctionStateVector": values}),
            "uniquenessStages": {"dealAuction": "cp_sat_second_solve_infeasible"},
            "variableCount": len(model.proto.variables), "constraintCount": len(model.proto.constraints) - 1,
            "auctionDomainIrHash": domain.sha256, "derivedAuction": {"prefixTerminal": True,
                "allPassed": True, "actions": [record.to_dict() for record in records]},
            "exactStateId": exact_id, "stateHash": state_hash}
        remaining()
        return OrToolsNormalizedExactStateV1(exact_id, prefix_hash, self.rules.sha256, state_hash,
            "complete", "bid", None, (), state, 0, 0, event_hash, digest,
            OrToolsExactNormalizationReceiptV1({**receipt, "sha256": canonical_sha256(receipt)}))

    def _enumerate_project_continuations(self, normalized, prefix, *, maximum_sequences,
                                         maximum_seconds, cancel_check=None):
        """Exhaust native declaration choices, not play paths or strategies."""
        if not self.native_cp_sat or type(maximum_sequences) is not int or maximum_sequences < 1:
            raise ValueError("native project enumeration requires a positive sequence limit")
        remaining = _normalization_clock(maximum_seconds, cancel_check)
        normalized.require_integrity()
        if (normalized.rules_hash != self.rules.sha256 or normalized.prefix_hash != _effective_prefix_hash(prefix)
                or normalized.normalization_receipt.payload.get("backendId") != _BACKEND_ID_V3
                or not prefix.project_actions_present or len(prefix.play_actions) >= 4
                or normalized.decision_kind not in {"play", "project"}):
            raise ValueError("project enumeration requires its normalized first-trick prefix")
        remaining()
        owners = {Card.parse(card): seat for seat in Seat
                  for card in normalized.state["initialHands"][seat.name.lower()]}
        model = cp_model.CpModel()
        fragment = attach_project_lifecycle_fragment_v1(model,
            owner_of_card=tuple(model.new_int_var(int(owners[Card(card)]), int(owners[Card(card)]),
                f"project_continuation_owner_{card}") for card in range(32)),
            fixed_card_owners=tuple(owners[Card(card)] for card in range(32)),
            contract=Contract.parse(normalized.state["contract"]), dealer=Seat.parse(prefix.dealer),
            trump=Suit.parse(normalized.state["trumpSuit"]) if normalized.state.get("trumpSuit") else None,
            domain=self.rules.project_domain, name="project_continuations", compact_fixed_states=True,
            declaration_only=self.rules.project_domain.tie_break_order == "second_trick_reveal")
        order = self.rules.project_domain.declaration_seats(Seat.parse(prefix.dealer))
        for index, row in enumerate(prefix.project_actions):
            model.add_bool_and(_native_project_action_literals(fragment,
                ProjectDeclarationActionV1.build(row, actor=order[index])))
        pending = order[len(prefix.project_actions):]
        variables = tuple(model.get_bool_var_from_proto_index(index) for index in sorted({
            project.declared.index for project in fragment.projects if project.owner in pending}))
        scope = {"kind": "native_project_lifecycle_continuations_v1", "rulesHash": self.rules.sha256,
            "sourceStateHash": normalized.state_hash, "prefixHash": prefix.sha256,
            "fragmentHash": fragment.fragment_hash, "pendingSeats": [seat.name.lower() for seat in pending]}
        sequences, queries = [], []
        try:
            while True:
                remaining()
                model_hash = _cp_model_hash(model, authority=scope)
                solver = _deterministic_solver(seed=0, maximum_seconds=remaining())
                status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
                queries.append({"modelHash": model_hash, "status": solver.status_name(status).lower()})
                remaining()
                if status == cp_model.INFEASIBLE:
                    if not sequences:
                        raise ValueError("normalized prefix disagrees with native project lifecycle")
                    core = {"schemaVersion": 1, "status": "complete", "scope": scope,
                        "sequences": sorted(sequences, key=canonical_sha256), "queries": queries}
                    result = {**core, "sha256": canonical_sha256(core)}
                    remaining()
                    return result
                if status not in {cp_model.FEASIBLE, cp_model.OPTIMAL}:
                    raise ExactNormalizationStopped(solver.status_name(status).lower())
                if len(sequences) >= maximum_sequences:
                    raise ExactNormalizationStopped("maximum_sequences")
                actions = []
                for seat in pending:
                    declared = [project for project in fragment.projects
                                if project.owner == seat and solver.boolean_value(project.declared)]
                    if len(declared) > 1:
                        raise ValueError("native project lifecycle has multiple declarations for one actor")
                    actions.append(ProjectDeclarationActionV1(seat,
                        "declare" if declared else "pass", declared[0].project_id if declared else None).to_dict())
                sequences.append(actions)
                queries[-1]["sequence"] = actions
                if variables:
                    model.add_forbidden_assignments(variables, [[solver.value(variable) for variable in variables]])
                else:
                    model.add_bool_or([])
        except ExactNormalizationStopped as exc:
            raise ExactNormalizationStopped(exc.partial_receipt["exhaustionReason"], evidence={
                "scope": scope, "sequences": sequences, "queries": queries}) from exc

    def normalize_play_or_project(
        self,
        prefix: ExactTracePrefixV1,
        *,
        maximum_seconds: float = 15.0,
        cancel_check: Callable[[], bool] | None = None,
    ) -> OrToolsNormalizedExactStateV1:
        """Route public prefix syntax; native normalization validates the stage."""
        if not self.native_cp_sat or not isinstance(prefix, ExactTracePrefixV1):
            raise TypeError("automatic play/project routing requires a native exact prefix")
        decision: dict[str, Any] = {"kind": "play"}
        order = self.rules.project_domain.declaration_seats(Seat.parse(prefix.dealer))
        if (
            prefix.project_actions_present
            and len(prefix.play_actions) < len(order)
            and len(prefix.project_actions) == len(prefix.play_actions)
        ):
            decision = {
                "kind": "project",
                "selector": {"owner": order[len(prefix.project_actions)].name.lower()},
            }
        return self.normalize(
            prefix, decision_point=decision, maximum_seconds=maximum_seconds,
            cancel_check=cancel_check,
        )

    def append_normalized_action(
        self,
        prefix: ExactTracePrefixV1,
        normalized: OrToolsNormalizedExactStateV1,
        action_code: str,
    ) -> ExactTracePrefixV1:
        """Append certified action syntax; callers normalize the resulting prefix.

        This internal transition helper consumes a normalizer result, never a
        client-authored runtime state. It does not evaluate legality itself.
        """
        if not isinstance(prefix, ExactTracePrefixV1) or not isinstance(normalized, OrToolsNormalizedExactStateV1):
            raise TypeError("action append requires an exact prefix and normalization")
        normalized.require_integrity()
        if normalized.rules_hash != self.rules.sha256 or normalized.prefix_hash != _effective_prefix_hash(prefix):
            raise ValueError("action normalization belongs to a different prefix or rules")
        if type(action_code) is not str:
            raise TypeError("selected action code must be a string")
        matches = [row for row in normalized.legal_actions if row["code"] == action_code]
        if len(matches) != 1 or normalized.next_actor is None:
            raise ValueError("selected action is not a unique legal action")
        row = matches[0]
        actor = normalized.next_actor
        kind = normalized.decision_kind
        if row.get("actor") not in {None, actor}:
            raise ValueError("selected action belongs to another actor")
        if kind == "play":
            play = LegalPlayActionV1.from_code(action_code)
            field = "playActions"
            action = {"actor": actor, "card": play.card.id, "ekka": play.ekka}
        elif kind in {"bid", "project", "escalation"}:
            field = {"bid": "auctionActions", "project": "projectActions", "escalation": "escalationActions"}[kind]
            if kind == "escalation" and row["action"] == EscalationDomainIRV1.from_rules(self.rules.data).closure_action:
                raise ValueError("escalation closure requires an explicit stage-transition record")
            action = {"actor": actor, "action": row["action"]}
            for key in {"bid": ("trumpSuit",), "project": ("projectId",), "escalation": ("mode",)}[kind]:
                if row.get(key) is not None:
                    action[key] = row[key]
        else:
            raise ValueError("action append decision kind is unsupported")
        value = prefix.to_dict()
        value.pop("sha256", None)
        value["stateAssertions"] = {}
        if kind != "play":
            value["projectAssertions"] = []
        value[field] = [*value.get(field, []), action]
        return ExactTracePrefixV1.build(value)

    @staticmethod
    def _semantic_vector(
        *,
        spec: ExactTracePrefixV1,
        solution: OrToolsScenarioDealSolutionV1,
        fragment: OrToolsScenarioAuctionFragmentV1,
    ) -> tuple[list[Any], list[int]]:
        if solution.auction is None or solution.purchase_suit is None:
            raise AssertionError("satisfied exact-input solve lacks auction evidence")
        observation = solution.auction
        variables: list[Any] = []
        values: list[int] = []
        # The caller prepends the model's card variables. Their values are the
        # fixed authored deck order, so the vector binds purchase semantics too.
        values.extend(spec.deck_order)
        values.append(int(solution.purchase_suit))
        values.append(int(observation.prefix_terminal))
        for name, literal in sorted(fragment.legal_next_action_literals.items()):
            variables.append(literal)
            values.append(int(name in observation.legal_next_actions))
        for name, literal in sorted(fragment.legal_next_choice_literals.items()):
            variables.append(literal)
            values.append(int(name in observation.legal_next_choices))
        if observation.prefix_terminal:
            if (
                observation.contract is None
                or observation.buyer is None
                or observation.purchase_round is None
                or observation.prefix_purchase_recipient is None
                or observation.caller is None
                or observation.declarer is None
            ):
                raise AssertionError("terminal auction prefix lacks derived values")
            variables.extend(
                [
                    fragment.contract,
                    fragment.trump,
                    fragment.purchase_round,
                    fragment.caller,
                    fragment.buyer,
                    fragment.purchase_recipient,
                    fragment.declarer,
                ]
            )
            values.extend(
                [
                    _CONTRACT_IDS[observation.contract],
                    (
                        int(observation.trump)
                        if observation.trump is not None
                        else _NO_TRUMP
                    ),
                    observation.purchase_round,
                    int(observation.caller),
                    int(observation.buyer),
                    int(observation.prefix_purchase_recipient),
                    int(observation.declarer),
                ]
            )
        return variables, values

    def _normalize_deal_auction(self, spec, seed, remaining, cancel_check):
        remaining()
        key = canonical_sha256({
            "rules": self.rules.sha256, "dealer": spec.dealer,
            "deck": list(spec.deck_order), "auction": [dict(row) for row in spec.auction_actions],
        })
        def certificate_hash(values):
            first, uniqueness, semantic_hash, roles, domain_hash = values
            return canonical_sha256({
                "solution": first.to_dict(), "uniqueness": uniqueness.to_dict(),
                "semanticHash": semantic_hash, "roles": roles.to_dict() if roles else None,
                "domainHash": domain_hash,
            })
        if self.native_cp_sat and self._deal_prefix_certificate is not None:
            stored_key, values, digest = self._deal_prefix_certificate
            if stored_key == key:
                if certificate_hash(values) != digest:
                    raise ValueError("native deal certificate integrity mismatch")
                remaining()
                return values
        model = OrToolsScenarioModelV1(
            rules=self.rules,
            dealer=Seat.parse(spec.dealer),
        )
        for position, card_id in enumerate(spec.deck_order):
            model.model.add(model.card_at_position[position] == card_id)
        fragment = (model.add_native_auction_fragment if self.native_cp_sat else model.add_auction_fragment)(
            required_actions=spec.auction_actions,
            required_actions_complete=False,
            required_purchase_card=spec.deck_order[model.purchase_position],
        )
        first = model.solve(seed=seed, maximum_seconds=remaining(), cancel_check=cancel_check)
        remaining()
        if first.status == "infeasible":
            raise ValueError("exact trace deal or auction prefix is infeasible")
        if first.status != "satisfied" or first.solver_status != "optimal":
            raise ExactNormalizationStopped(
                first.solver_status,
                evidence={"sourceStage": "deal_auction_witness"},
            )
        if first.deck_order != spec.deck_order:
            raise AssertionError("CP-SAT exact normalization changed the authored deck")

        extra_variables, semantic_values = self._semantic_vector(
            spec=spec,
            solution=first,
            fragment=fragment,
        )
        semantic_variables = [
            *model.card_at_position,
            model.purchase_suit,
            fragment.terminal,
            *extra_variables,
        ]
        if first.auction is not None and first.auction.prefix_terminal:
            if first.auction.buyer is None:
                raise AssertionError("terminal auction prefix lacks buyer")
            semantic_variables.append(model.buyer)
            semantic_values.append(int(first.auction.buyer))
        if len(semantic_variables) != len(semantic_values):
            raise AssertionError("exact normalization semantic vector is malformed")
        deal_auction_semantic_hash = canonical_sha256(
            {"schemaVersion": 1, "values": semantic_values}
        )
        model.model.add_forbidden_assignments(
            semantic_variables,
            [semantic_values],
        )
        uniqueness = model.solve(seed=seed, maximum_seconds=remaining(), cancel_check=cancel_check)
        remaining()
        if uniqueness.status == "satisfied":
            raise AssertionError("exact trace normalization is semantically ambiguous")
        if uniqueness.status != "infeasible":
            raise ExactNormalizationStopped(
                uniqueness.solver_status,
                evidence={"sourceStage": "deal_auction_uniqueness"},
            )

        role_projection = None
        if (
            self.native_cp_sat
            and first.auction is not None
            and first.auction.prefix_terminal
        ):
            if (
                first.auction.buyer is None
                or first.auction.prefix_purchase_recipient is None
                or first.auction.contract is None
                or first.auction.caller is None
                or first.auction.declarer is None
            ):
                raise ValueError(
                    "terminal auction lacks a native purchased-role binding"
                )
            role_projection = _native_role_projection(
                rules=self.rules,
                dealer=Seat.parse(spec.dealer),
                contract=first.auction.contract,
                caller=first.auction.caller,
                buyer=first.auction.buyer,
                purchase_recipient=first.auction.prefix_purchase_recipient,
                declarer=first.auction.declarer,
                seed=seed,
                maximum_seconds=remaining(),
                remaining_seconds=remaining,
                cancel_check=cancel_check,
            )
        values = (first, uniqueness, deal_auction_semantic_hash, role_projection, fragment.domain_hash)
        if self.native_cp_sat:
            self._deal_prefix_certificate = (key, values, certificate_hash(values))
        return values

    def normalize(
        self,
        prefix: ExactTracePrefixV1 | Mapping[str, Any],
        *,
        decision_point: Mapping[str, Any] | None = None,
        maximum_seconds: float = 15.0,
        cancel_check: Callable[[], bool] | None = None,
        recorded_escalation_opportunities: Sequence[Mapping[str, Any]] | None = None,
        opportunity_order: Sequence[Seat | str | int] | None = None,
        _play_template: tuple[str, str, OrToolsPlayTailBuildV1, OrToolsProjectLifecycleFragmentV1] | None = None,
    ) -> OrToolsNormalizedExactStateV1:
        remaining = _normalization_clock(maximum_seconds, cancel_check)
        remaining()
        spec = (
            prefix
            if isinstance(prefix, ExactTracePrefixV1)
            else ExactTracePrefixV1.build(prefix)
        )
        _validate_supported_slice(spec, self.rules)
        decision = dict(decision_point or {})
        decision_kind = str(decision.get("kind") or "bid")
        if (recorded_escalation_opportunities is None) != (opportunity_order is None):
            raise ValueError("recorded escalation normalization requires events and opportunity order")
        if recorded_escalation_opportunities is not None and (not self.native_cp_sat or decision_kind != "escalation"):
            raise ValueError("recorded opportunities require native escalation normalization")
        opportunity_replay = None
        if (self.native_cp_sat and len(spec.auction_actions) == 8
                and all(row.get("action") == "pass" for row in spec.auction_actions)):
            return self._normalize_all_pass(spec, decision_kind, remaining, cancel_check)
        if decision_kind not in {"bid", "escalation", "project", "play"}:
            raise ValueError(
                "OR-Tools exact normalization decision kind is unsupported"
            )
        if spec.project_actions_present:
            played_first_trick = min(len(spec.play_actions), 4)
            maximum_project_actions = min(played_first_trick + 1, 4)
            if not played_first_trick and decision_kind in {"bid", "escalation"}:
                maximum_project_actions = 0
            if not (
                played_first_trick
                <= len(spec.project_actions)
                <= maximum_project_actions
            ):
                raise ValueError(
                    "project actions must occur in order before first-trick plays"
                )
            if decision_kind == "project" and len(spec.project_actions) != played_first_trick:
                raise ValueError(
                    "project decision prefix cannot include the current actor action"
                )
            if (
                decision_kind == "play"
                and played_first_trick < 4
                and len(spec.project_actions) != maximum_project_actions
            ):
                raise ValueError(
                    "first-trick play prefix requires the current project action"
                )
        prefix_hash = _effective_prefix_hash(spec)
        seed = int(prefix_hash[:16], 16) & 0x7FFFFFFF
        first, uniqueness, deal_auction_semantic_hash, role_projection, auction_domain_hash = self._normalize_deal_auction(
            spec, seed, remaining, cancel_check,
        )
        state_document, legal_actions, next_actor, phase = _build_state(
            rules=self.rules,
            spec=spec,
            prefix_hash=prefix_hash,
            solution=first,
            native_roles=role_projection,
        )
        state = dict(state_document)
        escalation_result = None
        if spec.escalation_actions and decision_kind == "bid":
            raise ValueError("bid decision trace cannot contain escalation actions")
        if spec.escalation_actions or decision_kind in {
            "escalation",
            "project",
            "play",
        }:
            if first.auction is None or not first.auction.prefix_terminal:
                raise ValueError(
                    "exact escalation prefix requires a terminal purchased auction"
                )
            if (
                first.auction.buyer is None
                or first.auction.declarer is None
                or first.auction.contract is None
            ):
                raise AssertionError("terminal auction lacks escalation authority")
            escalation_domain = EscalationDomainIRV1.from_rules(self.rules.data)
            escalation_stage = OrToolsEscalationStageV1(
                escalation_domain,
                declarer=first.auction.declarer,
                contract=first.auction.contract,
                match_score=spec.match_score,
                native_cp_sat=self.native_cp_sat,
            )
            non_announcements = None
            if recorded_escalation_opportunities is not None:
                opportunity_replay = escalation_stage.replay_opportunities(
                    required_actions=(), events=recorded_escalation_opportunities,
                    expected_final_actions=spec.escalation_actions, opportunity_order=opportunity_order,
                    maximum_seconds=remaining(), declared_maximum_seconds=maximum_seconds,
                    cancel_check=cancel_check)
                if opportunity_replay["status"] == "infeasible":
                    raise ValueError("recorded escalation history is infeasible")
                if opportunity_replay["status"] != "satisfied":
                    raise ExactNormalizationStopped(
                        opportunity_replay.get("stopReason", "escalation_replay_incomplete"),
                        evidence={"escalationOpportunityReplay": opportunity_replay})
                actor = opportunity_replay["nextActor"]
                if actor is None:
                    raise ValueError("recorded escalation history has no live decision")
                selector = dict(decision.get("selector") or {})
                if selector.get("actingSeat") is not None and Seat.parse(selector["actingSeat"]).name.lower() != actor:
                    raise ValueError("escalation acting seat disagrees with recorded opportunity history")
                decision = {**decision, "selector": {**selector, "actingSeat": actor}}
                non_announcements = opportunity_replay["finalState"]["opportunityState"]["nonAnnouncingSeats"]
                state["escalationOpportunityHistoryHash"] = canonical_sha256({
                    "events": opportunity_replay["events"], "order": opportunity_replay["opportunityOrder"]})
            terminal_key = (canonical_sha256({
                "rules": self.rules.sha256, "declarer": int(first.auction.declarer),
                "contract": first.auction.contract.value, "matchScore": list(spec.match_score),
                "actions": [dict(action) for action in spec.escalation_actions],
            }) if self.native_cp_sat and decision_kind in {"project", "play"} else None)
            certificate = self._terminal_escalation_certificate
            if terminal_key is not None and certificate is not None and certificate[0] == terminal_key:
                _, escalation_result, digest = certificate
                if canonical_sha256(escalation_result.to_dict()) != digest:
                    raise ValueError("native escalation certificate integrity mismatch")
            else:
                escalation_result = escalation_stage.solve(
                    required_actions=spec.escalation_actions,
                    **({"opportunity_non_announcements": non_announcements} if non_announcements is not None else {}),
                    required_actions_complete=True,
                    required_terminal=True if terminal_key is not None else None,
                    seed=seed,
                    maximum_seconds=remaining(),
                    cancel_check=cancel_check,
                )
            remaining()
            if escalation_result.status == "infeasible":
                raise ValueError("exact escalation prefix is infeasible")
            if self.native_cp_sat and escalation_result.solver_status not in {"optimal", "infeasible"}:
                raise ExactNormalizationStopped(escalation_result.solver_status,
                    evidence={"sourceStage": "exact_escalation"})
            if (
                escalation_result.status != "satisfied"
                or escalation_result.solver_status != "optimal"
                or escalation_result.observation is None
            ):
                raise RuntimeError(
                    "exact escalation normalization was not completed within budget"
                )
            _apply_escalation(state, escalation_result.observation)
            if terminal_key is not None:
                self._terminal_escalation_certificate = (
                    terminal_key, escalation_result, canonical_sha256(escalation_result.to_dict()),
                )
            if decision_kind == "escalation":
                selector = dict(decision.get("selector") or {})
                acting_seat = selector.get("actingSeat")
                if acting_seat is None:
                    raise ValueError(
                        "escalation decisionPoint requires selector.actingSeat"
                    )
                actor = Seat.parse(str(acting_seat))
                legal_actions = _legal_escalation_rows(
                    escalation_result.observation,
                    actor=actor,
                    domain=escalation_domain,
                    include_actor=self.native_cp_sat,
                )
                next_actor = actor.name.lower()
        if (
            spec.project_actions
            or spec.project_declarations
            or spec.project_assertions
            or spec.play_actions
        ) and decision_kind in {"bid", "escalation"}:
            raise ValueError(
                f"{decision_kind} decision trace cannot contain project or play input"
            )

        project_result = None
        play_result = None
        native_play_normalization = None
        native_project_projection = None
        scoring_result = None
        projects: list[Project] = []
        baloot_teams: frozenset[Team] = frozenset()
        project_declaration_count = 0
        project_assertion_count = 0
        if decision_kind in {"project", "play"} or (
            spec.project_actions
            or spec.project_declarations
            or spec.project_assertions
            or spec.play_actions
        ):
            if first.auction is None or not first.auction.prefix_terminal:
                raise ValueError(
                    "exact project/play prefix requires a terminal purchased auction"
                )
            if first.auction.buyer is None or first.auction.contract is None:
                raise AssertionError("terminal auction lacks project/play authority")
            contract = first.auction.contract
            trump = first.auction.trump
            hands = {
                seat: tuple(
                    Card.parse(card)
                    for card in state["initialHands"][seat.name.lower()]
                )
                for seat in Seat
            }
            first_leader_raw = state.get("firstLeader")
            if first_leader_raw is None:
                raise AssertionError("purchased deal lacks first leader")
            first_leader = Seat.parse(first_leader_raw)
            scoring_domain = ScoringDomainIRV1.from_rules(self.rules.data)
            if self.native_cp_sat:
                native_play_normalization = _native_fixed_play_prefix(
                    spec=spec,
                    hands=hands,
                    contract=contract,
                    trump=trump,
                    first_leader=first_leader,
                    dealer=Seat.parse(spec.dealer),
                    buyer=first.auction.buyer,
                    escalation=str(state["escalation"]),
                    last_escalating_team=(
                        Team[str(state["escalationHistory"][-1]["team"]).upper()]
                        if state["escalationHistory"]
                        else None
                    ),
                    starting_match_score=tuple(state["matchScore"]),
                    locked=bool(state["locked"]),
                    rules=self.rules,
                    seed=seed,
                    maximum_seconds=remaining(),
                    remaining_seconds=remaining,
                    cancel_check=cancel_check,
                    play_template=_play_template,
                    first_trick_templates=self._first_trick_templates,
                    paired_pass_cache=self._paired_pass_cache,
                )
                play_result = native_play_normalization.result
                project_result = native_play_normalization.project_result
                native_project_projection = (
                    native_play_normalization.project_projection
                )
                projects, baloot_teams = _render_native_play_prefix(
                    state,
                    play=play_result,
                    projection=native_project_projection,
                    trump=trump,
                    project_actions=(
                        spec.project_actions
                        if spec.project_actions_present
                        else None
                    ),
                )
                scoring_result = _score_terminal_state(
                    state,
                    play=play_result,
                    projects=projects,
                    baloot_teams=baloot_teams,
                    contract=contract,
                    declarer=first.auction.declarer,
                    scoring=scoring_domain,
                    rules=self.rules,
                    native_cp_sat=True,
                    native_result=native_play_normalization.scoring_result,
                )
            else:
                play_prefix = _derive_play_prefix(
                    spec,
                    first_leader=first_leader,
                    contract=contract,
                    trump=trump,
                    rules=self.rules,
                )
                ekka_leads = [False] * 8
                for index, action in enumerate(play_prefix):
                    if action.ekka:
                        if index % 4 != 0:
                            raise ValueError(
                                "Ekka may only be declared on a trick lead"
                            )
                        ekka_leads[index // 4] = True
                ownership = {
                    card: seat for seat, hand in hands.items() for card in hand
                }
                play_step_count = min(
                    32, max(4, ((len(play_prefix) + 4) // 4) * 4)
                )
                play_result = OrToolsPlayTailStageV1().solve(
                    cards=tuple(Card(card_id) for card_id in range(32)),
                    ownership=ownership,
                    contract=contract,
                    trump=trump,
                    first_leader=first_leader,
                    play_domain=self.rules.play_domain,
                    structural_domain=self.rules.structural_domain,
                    scoring_domain=scoring_domain,
                    prefix=play_prefix,
                    tail=(),
                    play_step_count=play_step_count,
                    ekka_leads=tuple(ekka_leads),
                    locked=bool(state["locked"]),
                    maximum_time_seconds=remaining(),
                    cancel_check=cancel_check,
                )
                if play_result.solver_status != "optimal":
                    raise RuntimeError(
                        "exact play normalization was not completed within budget"
                    )
                project_result = OrToolsProjectLifecycleModelV1().solve(
                    hands=hands,
                    contract=contract,
                    dealer=Seat.parse(spec.dealer),
                    trump=trump,
                    domain=self.rules.project_domain,
                    # Selection is extracted without award flags; replay applies
                    # comparison only once the first trick is actually complete.
                    second_trick_leader=play_result.trick_winners[0],
                    timeout_seconds=remaining(),
                    required_declaration_actions=spec.project_actions,
                    cancel_check=cancel_check,
                )
                if project_result.solver_status != "optimal":
                    raise RuntimeError(
                        "exact project normalization was not completed within budget"
                    )
                projects = _selected_projects(
                    project_result,
                    dealer=Seat.parse(spec.dealer),
                )
                projects, baloot_teams = _replay_play_prefix(
                    state,
                    play=play_result,
                    projects=projects,
                    contract=contract,
                    trump=trump,
                    rules=self.rules,
                )
                scoring_result = _score_terminal_state(
                    state,
                    play=play_result,
                    projects=projects,
                    baloot_teams=baloot_teams,
                    contract=contract,
                    declarer=first.auction.declarer,
                    scoring=scoring_domain,
                    rules=self.rules,
                    native_cp_sat=False,
                )
            actual_declarations = _project_declarations(state["events"])
            expected_declarations = [dict(item) for item in spec.project_declarations]
            if (
                not spec.project_actions_present or expected_declarations
            ) and actual_declarations != expected_declarations:
                raise ValueError(
                    "exact-state project declarations do not match CP-SAT events"
                )
            project_declaration_count = len(actual_declarations)
            if self.native_cp_sat:
                if native_project_projection is None:
                    raise AssertionError("native project projection is missing")
                project_assertion_count = _verify_native_project_assertions(
                    project_result,
                    native_project_projection,
                    spec.project_assertions,
                )
            else:
                project_assertion_count = _verify_project_assertions(
                    projects,
                    spec.project_assertions,
                    domain=self.rules.project_domain,
                    trump=trump,
                    second_trick_leader=(play_result.trick_winners[0] if play_result.trick_winners else None),
                )
            if decision_kind == "project":
                selector = dict(decision.get("selector") or {})
                owner_raw = selector.get("owner")
                if owner_raw is None:
                    raise ValueError("project decisionPoint requires selector.owner")
                owner = Seat.parse(str(owner_raw))
                if self.native_cp_sat:
                    if native_play_normalization is None:
                        raise AssertionError("native project action bindings are missing")
                    native_actions = native_play_normalization.legal_project_actions
                    if not native_actions or native_actions[0].actor != owner:
                        raise ValueError(
                            "project decisionPoint owner must be the current actor"
                        )
                    legal_actions = tuple(
                        {"code": action.code, **action.to_dict()}
                        for action in native_actions
                    )
                    # A forced declaration/pass is still a legal exact state.
                    # Scenario action-count constraints own tactical interest;
                    # normalization must also support whole-round auditing.
                    next_actor = owner.name.lower()
                elif spec.project_actions_present:
                    declaration_state = self.rules.project_domain.declaration_state(
                        _selected_projects(
                            project_result,
                            dealer=Seat.parse(spec.dealer),
                        ),
                        dealer=Seat.parse(spec.dealer),
                    )
                    for action in spec.project_actions:
                        declaration_state, _announced = (
                            self.rules.project_domain.apply_declaration_action(
                                declaration_state,
                                action,
                                trump_suit=trump,
                            )
                        )
                    if declaration_state.current_actor != owner:
                        raise ValueError(
                            "project action prefix does not reach the selected owner"
                        )
                    legal_actions = tuple(
                        {
                            "code": action.code,
                            **action.to_dict(),
                        }
                        for action in self.rules.project_domain.legal_declaration_actions(
                            declaration_state,
                            trump_suit=trump,
                        )
                    )
                    if (
                        self.rules.project_domain.declaration_mode == "optional"
                        and len(legal_actions) < 2
                    ):
                        raise ProjectDecisionNotStrategicError()
                    next_actor = owner.name.lower()
                else:
                    _updated, announced = self.rules.project_domain.announce_for_turn(
                        projects,
                        owner,
                        trump_suit=trump,
                    )
                    if announced is None:
                        raise ValueError(
                            "project decisionPoint has no legal declaration"
                        )
                if not self.native_cp_sat and not spec.project_actions_present:
                    project_index = int(selector.get("projectIndex", 0))
                    owned = sorted(
                        [project for project in projects if project.owner == owner],
                        key=lambda item: (
                            -item.value,
                            -item.category_priority,
                            -item.high_rank,
                        ),
                    )
                    authoritative_index = next(
                        (
                            index
                            for index, project in enumerate(owned)
                            if (
                                project.kind,
                                project.value,
                                project.high_rank,
                            )
                            == (
                                announced.kind,
                                announced.value,
                                announced.high_rank,
                            )
                        ),
                        None,
                    )
                    if (
                        authoritative_index is None
                        or project_index != authoritative_index
                    ):
                        raise ValueError(
                            "project decisionPoint index is not the authoritative declaration"
                        )
                    legal_actions = (
                        {
                            "code": (
                                f"declare:{announced.kind}:{announced.value}:"
                                f"{announced.high_rank}"
                            ),
                            "action": "declare",
                            "owner": owner.name.lower(),
                            "kind": announced.kind,
                            "value": announced.value,
                            "highRank": announced.high_rank,
                        },
                    )
                    next_actor = owner.name.lower()
            else:
                legal_actions = _legal_play_rows(
                    native_play_normalization.legal_play_actions
                    if native_play_normalization is not None
                    else play_result.legal_play_actions
                )
                next_actor = (
                    play_result.next_actor.name.lower()
                    if play_result.next_actor is not None
                    else None
                )
            phase = str(state["phase"])
        for path, expected in spec.state_assertions.items():
            actual = _mapping_path(state, str(path))
            if canonical_sha256(actual) != canonical_sha256(expected):
                raise ValueError(
                    f"exact-state assertion mismatch at {path}: "
                    f"expected={expected!r}, actual={actual!r}"
                )
        state_hash, event_chain_hash, digest = _exact_state_identity(
            prefix_hash=prefix_hash, rules_hash=self.rules.sha256, state=state,
            phase=phase, decision_kind=decision_kind, next_actor=next_actor, legal_actions=legal_actions,
        )
        exact_state_id = f"STATE-{digest[:20].upper()}"
        semantic_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "dealAuctionSemanticSolutionVectorHash": (deal_auction_semantic_hash),
                "escalationEvidenceHash": (
                    escalation_result.evidence_hash
                    if escalation_result is not None
                    else None
                ),
                "projectEvidenceHash": (
                    project_result.evidence_hash if project_result is not None else None
                ),
                "playEvidenceHash": (
                    play_result.evidence_hash if play_result is not None else None
                ),
                "roleEvidenceHash": (
                    role_projection.evidence_hash
                    if role_projection is not None
                    else None
                ),
                "nativePlaySemanticVectorHash": (
                    native_play_normalization.semantic_vector_hash
                    if native_play_normalization is not None
                    else None
                ),
                "nativeProjectPrefixEvidenceHash": (
                    native_project_projection.evidence_hash
                    if native_project_projection is not None
                    else None
                ),
                "scoringEvidenceHash": (
                    scoring_result.evidence_hash if scoring_result is not None else None
                ),
            }
        )
        observer_information = (
            build_canonical_observer_information_v1(
                complete_state=state,
                acting_seat=next_actor,
            )
            if next_actor is not None
            else None
        )
        project_document = (
            project_result.to_dict() if project_result is not None else None
        )
        play_document = play_result.to_dict() if play_result is not None else None
        if not self.native_cp_sat:
            for document in (project_document, play_document):
                if document is not None:
                    document.pop("variableCount", None)
                    document.pop("constraintCount", None)
        receipt_core = {
            "schemaVersion": 1,
            "backendId": (_BACKEND_ID_V3 if self.native_cp_sat else _BACKEND_ID_V2),
            "normalizerId": _NORMALIZER_ID,
            "solverVersion": _ORTOOLS_VERSION,
            "rulesHash": self.rules.sha256,
            "structuralDomainIrHash": self.rules.structural_domain.sha256,
            "auctionDomainIrHash": auction_domain_hash,
            "prefixHash": prefix_hash,
            "decisionKind": decision_kind,
            "modelHash": first.model_hash,
            "solutionHash": first.solution_hash,
            "semanticSolutionVectorHash": semantic_hash,
            "dealAuctionSemanticSolutionVectorHash": (deal_auction_semantic_hash),
            "solverStatus": first.solver_status,
            "uniquenessStatus": "proven_unique",
            "uniquenessModelHash": uniqueness.model_hash,
            "uniquenessSolverStatus": uniqueness.solver_status,
            "uniquenessStages": {
                "dealAuction": "cp_sat_second_solve_infeasible",
                "roles": (
                    "cp_sat_second_solve_infeasible"
                    if role_projection is not None
                    else "not_applicable"
                ),
                "escalation": (
                    "cp_sat_fixed_trace_optimal"
                    if self.native_cp_sat and escalation_result is not None
                    else "domain_ir_deterministic_transition"
                    if escalation_result is not None
                    else "not_applicable"
                ),
                "projects": (
                    "cp_sat_fixed_hands_and_prefix_unique"
                    if native_project_projection is not None
                    else "cp_sat_fixed_hands_optimal"
                    if project_result is not None
                    else "not_applicable"
                ),
                "play": (
                    "cp_sat_fixed_cards_second_solve_infeasible"
                    if native_play_normalization is not None
                    else "cp_sat_fixed_prefix_optimal"
                    if play_result is not None
                    else "not_applicable"
                ),
                "scoring": (
                    "cp_sat_fixed_inputs_optimal"
                    if scoring_result is not None
                    else "not_applicable"
                ),
            },
            "derivedAuction": {
                "prefixTerminal": first.auction.prefix_terminal,
                "buyer": (
                    first.auction.buyer.name.lower()
                    if first.auction.buyer is not None
                    else None
                ),
                "contract": (
                    first.auction.contract.value
                    if first.auction.contract is not None
                    else None
                ),
                "trumpSuit": (
                    first.auction.trump.name.lower()
                    if first.auction.trump is not None
                    else None
                ),
                "purchaseRound": first.auction.purchase_round,
                **(
                    {
                        "caller": (
                            first.auction.caller.name.lower()
                            if first.auction.caller is not None
                            else None
                        ),
                        "purchaseRecipient": (
                            first.auction.prefix_purchase_recipient.name.lower()
                            if first.auction.prefix_purchase_recipient is not None
                            else None
                        ),
                        "declarer": (
                            first.auction.declarer.name.lower()
                            if first.auction.declarer is not None
                            else None
                        ),
                    }
                    if first.auction.prefix_terminal
                    else {}
                ),
                "currentActor": (
                    first.auction.current_actor.name.lower()
                    if first.auction.current_actor is not None
                    else None
                ),
                "currentRound": first.auction.current_round,
                "legalNextActions": list(first.auction.legal_next_actions),
                "legalNextChoices": list(first.auction.legal_next_choices),
                "traceHash": first.auction.trace_hash,
            },
            "derivedEscalation": (
                {
                    "domainIrHash": escalation_result.domain_ir_hash,
                    "modelHash": escalation_result.model_hash,
                    "evidenceHash": escalation_result.evidence_hash,
                    "observation": escalation_result.observation.to_dict(),
                }
                if escalation_result is not None
                and escalation_result.observation is not None
                else None
            ),
            **({"derivedEscalationOpportunities": opportunity_replay} if opportunity_replay is not None else {}),
            "derivedProjects": (
                project_document
            ),
            "derivedPlay": play_document,
            "derivedRoles": (
                role_projection.to_dict() if role_projection is not None else None
            ),
            "derivedProjectPrefix": (
                native_project_projection.to_dict()
                if native_project_projection is not None
                else None
            ),
            "nativePlayNormalization": (
                native_play_normalization.to_dict()
                if native_play_normalization is not None
                else None
            ),
            "derivedScoring": (
                scoring_result.to_dict() if scoring_result is not None else None
            ),
            "observerInformation": observer_information,
            "exactStateId": exact_state_id,
            "stateHash": state_hash,
        }
        if self.native_cp_sat:
            receipt_core["variableCount"] = first.variable_count
            receipt_core["constraintCount"] = first.constraint_count
            receipt_core["stageModelSizes"] = {
                "dealAuction": {
                    "variables": first.variable_count,
                    "constraints": first.constraint_count,
                },
                "escalation": {
                    "variables": (
                        escalation_result.variable_count
                        if escalation_result is not None
                        else 0
                    ),
                    "constraints": (
                        escalation_result.constraint_count
                        if escalation_result is not None
                        else 0
                    ),
                },
                "projects": {
                    "variables": (
                        project_result.variable_count
                        if project_result is not None
                        else 0
                    ),
                    "constraints": (
                        project_result.constraint_count
                        if project_result is not None
                        else 0
                    ),
                },
                "play": {
                    "variables": (
                        play_result.variable_count if play_result is not None else 0
                    ),
                    "constraints": (
                        play_result.constraint_count if play_result is not None else 0
                    ),
                },
                "scoring": {
                    "variables": (
                        scoring_result.variable_count
                        if scoring_result is not None
                        else 0
                    ),
                    "constraints": (
                        scoring_result.constraint_count
                        if scoring_result is not None
                        else 0
                    ),
                },
            }
        receipt = OrToolsExactNormalizationReceiptV1(
            {**receipt_core, "sha256": canonical_sha256(receipt_core)}
        )
        remaining()
        return OrToolsNormalizedExactStateV1(
            exact_state_id,
            prefix_hash,
            self.rules.sha256,
            state_hash,
            phase,
            decision_kind,
            next_actor,
            legal_actions,
            state,
            project_declaration_count,
            project_assertion_count,
            event_chain_hash,
            digest,
            receipt,
            native_play_normalization.play_template if native_play_normalization is not None else None,
        )

    def preview_legal_next_actions(
        self,
        prefix: ExactTracePrefixV1 | Mapping[str, Any],
        *,
        maximum_seconds: float = 15.0,
        cancel_check: Callable[[], bool] | None = None,
        recorded_escalation_opportunities: Sequence[Mapping[str, Any]] | None = None,
        opportunity_order: Sequence[Seat | str | int] | None = None,
    ) -> dict[str, Any]:
        """Derive the next operator choices from one fixed exact prefix."""

        remaining = _normalization_clock(maximum_seconds, cancel_check)
        if (recorded_escalation_opportunities is None) != (opportunity_order is None):
            raise ValueError("recorded events and opportunity order must be supplied together")
        if recorded_escalation_opportunities is not None and not self.native_cp_sat:
            raise ValueError("recorded opportunities require native preview")
        remaining()
        spec = (
            prefix
            if isinstance(prefix, ExactTracePrefixV1)
            else ExactTracePrefixV1.build(prefix)
        )
        auction_only = spec.to_dict()
        auction_only.pop("sha256", None)
        for field in (
            "escalationActions",
            "projectActions",
            "projectDeclarations",
            "projectAssertions",
            "playActions",
        ):
            if field in auction_only:
                auction_only[field] = []
        # Assertions describe the authored decision state, not the auction-only
        # intermediate used to derive its fixed purchase roles.
        auction_only["stateAssertions"] = {}
        auction = self.normalize(
            auction_only,
            decision_point={"kind": "bid", "selector": {}},
            maximum_seconds=remaining(),
            cancel_check=cancel_check,
        )
        auction_receipt = auction.normalization_receipt.to_dict()["derivedAuction"]
        if recorded_escalation_opportunities and (not auction_receipt["prefixTerminal"] or auction.state.get("allPassed")):
            raise ValueError("recorded opportunities require a purchased auction")
        if not bool(auction_receipt["prefixTerminal"]):
            if any(
                (
                    spec.escalation_actions,
                    spec.project_actions,
                    spec.project_declarations,
                    spec.project_assertions,
                    spec.play_actions,
                )
            ):
                raise ValueError(
                    "exact trace cannot contain post-auction actions before purchase"
                )
            legal = [
                {
                    "stage": "auction",
                    "actor": auction.next_actor,
                    **copy.deepcopy(dict(action)),
                }
                for action in auction.legal_actions
            ]
            core = {
                "schemaVersion": 1,
                "prefixHash": auction.prefix_hash,
                "currentStage": "auction",
                "nextActor": auction.next_actor,
                "terminal": False,
                "legalActions": legal,
                "previewStateHash": auction.state_hash,
                "previewReceiptHash": auction.normalization_receipt.sha256,
            }
            remaining()
            return {**core, "sha256": canonical_sha256(core)}

        if auction.state.get("allPassed") is True:
            auction = self.normalize(spec, decision_point={"kind": "bid"},
                maximum_seconds=remaining(), cancel_check=cancel_check)
            core = {"schemaVersion": 1, "prefixHash": auction.prefix_hash,
                "currentStage": "auction", "nextActor": None, "terminal": True,
                "legalActions": [], "previewStateHash": auction.state_hash,
                "previewReceiptHash": auction.normalization_receipt.sha256}
            remaining()
            return {**core, "sha256": canonical_sha256(core)}

        buyer = Seat.parse(str(auction.state["buyer"]))
        declarer = Seat.parse(str(auction.state["declarer"]))
        contract = Contract.parse(str(auction.state["contract"]))
        trump = (
            Suit.parse(str(auction.state["trumpSuit"]))
            if auction.state.get("trumpSuit") is not None
            else None
        )
        escalation_domain = EscalationDomainIRV1.from_rules(self.rules.data)
        escalation_stage = OrToolsEscalationStageV1(
            escalation_domain,
            declarer=declarer,
            contract=contract,
            match_score=spec.match_score,
            native_cp_sat=self.native_cp_sat,
        )
        replay = None
        if recorded_escalation_opportunities is not None:
            replay = escalation_stage.replay_opportunities(required_actions=(),
                events=recorded_escalation_opportunities, expected_final_actions=spec.escalation_actions,
                opportunity_order=opportunity_order, maximum_seconds=remaining(),
                declared_maximum_seconds=maximum_seconds, cancel_check=cancel_check)
            if replay["status"] == "infeasible":
                raise ValueError("recorded escalation history is infeasible")
            if replay["status"] != "satisfied":
                raise ExactNormalizationStopped(replay.get("stopReason", "escalation_replay_incomplete"),
                    evidence={"escalationOpportunityReplay": replay})
            if replay["nextActor"] is not None:
                if spec.project_actions or spec.play_actions or spec.project_declarations:
                    raise ValueError("recorded escalation opportunities must close before later actions")
                normalized = self.normalize(spec, decision_point={"kind": "escalation"},
                    recorded_escalation_opportunities=recorded_escalation_opportunities,
                    opportunity_order=opportunity_order, maximum_seconds=remaining(), cancel_check=cancel_check)
                core = {"schemaVersion": 1, "prefixHash": normalized.prefix_hash,
                    "currentStage": "escalation", "nextActor": normalized.next_actor, "terminal": False,
                    "legalActions": [{**dict(action), "stage": "escalation"} for action in normalized.legal_actions],
                    "previewStateHash": normalized.state_hash,
                    "previewReceiptHash": normalized.normalization_receipt.sha256}
                return {**core, "sha256": canonical_sha256(core)}
        escalation_result = escalation_stage.solve(
            required_actions=spec.escalation_actions,
            **({"opportunity_non_announcements": replay["finalState"]["opportunityState"]["nonAnnouncingSeats"]}
               if replay is not None else {}),
            required_actions_complete=True,
            seed=int(auction.prefix_hash[:16], 16) & 0x7FFFFFFF,
            maximum_seconds=remaining(),
            cancel_check=cancel_check,
        )
        remaining()
        if escalation_result.status == "infeasible":
            raise ValueError("exact escalation prefix is infeasible")
        if self.native_cp_sat and escalation_result.solver_status not in {"optimal", "infeasible"}:
            raise ExactNormalizationStopped(escalation_result.solver_status,
                evidence={"sourceStage": "exact_escalation"})
        if (
            escalation_result.status != "satisfied"
            or escalation_result.observation is None
        ):
            raise RuntimeError("exact escalation preview exceeded its budget")
        escalation = escalation_result.observation
        if (
            escalation.legal_next_choices
            and not any(
                (
                    spec.project_actions,
                    spec.project_declarations,
                    spec.project_assertions,
                    spec.play_actions,
                )
            )
        ):
            legal = [
                {
                    "stage": (
                        "play"
                        if choice.action == escalation_domain.closure_action
                        else "escalation"
                    ),
                    "code": choice.code,
                    "actor": choice.actor.name.lower(),
                    "action": choice.action,
                    "mode": choice.mode,
                }
                for choice in escalation.legal_next_choices
            ]
            if escalation_domain.closure_action in escalation.legal_next_actions:
                legal.insert(
                    0,
                    {
                        "stage": "play",
                        "code": escalation_domain.closure_action,
                        "actor": None,
                        "action": escalation_domain.closure_action,
                        "mode": None,
                    },
                )
            core = {
                "schemaVersion": 1,
                "prefixHash": _effective_prefix_hash(spec),
                "currentStage": "escalation",
                "nextActor": None,
                "terminal": False,
                "legalActions": legal,
                "previewStateHash": canonical_sha256(
                    {
                        "auctionStateHash": auction.state_hash,
                        "escalation": escalation.to_dict(),
                    }
                ),
                "previewReceiptHash": canonical_sha256(
                    {
                        "auction": auction.normalization_receipt.sha256,
                        "escalation": escalation_result.evidence_hash,
                        "rules": self.rules.sha256,
                    }
                ),
            }
            remaining()
            return {**core, "sha256": canonical_sha256(core)}
        hands = {
            seat: [
                Card.parse(card)
                for card in auction.state["initialHands"][seat.name.lower()]
            ]
            for seat in Seat
        }
        first_leader = Seat.parse(str(auction.state["firstLeader"]))
        native_joint_prefix = self.native_cp_sat and any(
            (
                spec.project_actions,
                spec.project_declarations,
                spec.project_assertions,
                spec.play_actions,
            )
        )
        if native_joint_prefix:
            native_play = _native_fixed_play_prefix(
                spec=spec,
                hands=hands,
                contract=contract,
                trump=trump,
                first_leader=first_leader,
                dealer=Seat.parse(spec.dealer),
                buyer=buyer,
                escalation=escalation.level,
                last_escalating_team=(
                    escalation.actions[-1].actor.team
                    if escalation.actions
                    else None
                ),
                starting_match_score=spec.match_score,
                locked=escalation.mode == "locked",
                rules=self.rules,
                seed=int(auction.prefix_hash[:16], 16) & 0x7FFFFFFF,
                maximum_seconds=remaining(),
                remaining_seconds=remaining,
                cancel_check=cancel_check,
            )
            play_result = native_play.result
            project_result = native_play.project_result
            project_projection = native_play.project_projection
            projects = list(project_projection.projects)
            declarations = [
                project.public_declaration() for project in projects if project.declared
            ]
            declarations.sort(key=lambda row: int(row["declarationOrder"]))
            if (
                not spec.project_actions_present or spec.project_declarations
            ) and declarations != [dict(item) for item in spec.project_declarations]:
                raise ValueError(
                    "exact-state project declarations do not match CP-SAT projects"
                )
            _verify_native_project_assertions(
                project_result,
                project_projection,
                spec.project_assertions,
            )
        else:
            play_prefix = (
                _derive_play_prefix(
                    spec,
                    first_leader=first_leader,
                    contract=contract,
                    trump=trump,
                    rules=self.rules,
                )
                if spec.play_actions
                else ()
            )
            ownership = {card: seat for seat, hand in hands.items() for card in hand}
            play_step_count = min(32, max(4, ((len(play_prefix) + 4) // 4) * 4))
            play_result = OrToolsPlayTailStageV1(native_conversion=self.native_cp_sat).solve(
                cards=tuple(Card(card_id) for card_id in range(32)),
                ownership=ownership,
                contract=contract,
                trump=trump,
                first_leader=first_leader,
                play_domain=self.rules.play_domain,
                structural_domain=self.rules.structural_domain,
                scoring_domain=ScoringDomainIRV1.from_rules(self.rules.data),
                prefix=play_prefix,
                play_step_count=play_step_count,
                locked=escalation.mode == "locked",
                maximum_time_seconds=remaining(),
                cancel_check=cancel_check,
            )
            first_trick_actors = {action.actor for action in play_prefix[:4]}
            project_result = OrToolsProjectLifecycleModelV1().solve(
                hands=hands,
                contract=contract,
                dealer=Seat.parse(spec.dealer),
                trump=trump,
                domain=self.rules.project_domain,
                # Future witness awards are masked by `compared` below.
                second_trick_leader=play_result.trick_winners[0],
                timeout_seconds=remaining(),
                required_declaration_actions=spec.project_actions,
                cancel_check=cancel_check,
            )
            if project_result.solver_status != "optimal":
                raise RuntimeError("exact project preview exceeded its budget")
            projects = _selected_projects(
                project_result,
                dealer=Seat.parse(spec.dealer),
            )
            second_trick_actors = {action.actor for action in play_prefix[4:8]}
            compared = len(play_prefix) >= 4
            evidence_by_key = {
                (item.owner, item.kind, item.value, item.high_rank): item
                for item in project_result.projects
                if item.selected
            }
            declarations = []
            reached_projects: list[Project] = []
            for project in projects:
                evidence = evidence_by_key[
                    (project.owner, project.kind, project.value, project.high_rank)
                ]
                declared = evidence.declared and project.owner in first_trick_actors
                reached = Project(
                    project.owner,
                    project.kind,
                    project.cards,
                    project.value,
                    project.high_rank,
                    project.category_priority,
                    declared,
                    int(evidence.declaration_order) if declared else 999,
                    evidence.awarded and compared,
                    evidence.awarded and project.owner in second_trick_actors,
                )
                reached_projects.append(reached)
                if reached.declared:
                    declarations.append(reached.public_declaration())
            projects = reached_projects
            declarations.sort(key=lambda row: int(row["declarationOrder"]))
            if (
                not spec.project_actions_present or spec.project_declarations
            ) and declarations != [dict(item) for item in spec.project_declarations]:
                raise ValueError(
                    "exact-state project declarations do not match CP-SAT projects"
                )
            if spec.project_assertions:
                _verify_project_assertions(
                    projects,
                    spec.project_assertions,
                    domain=self.rules.project_domain,
                    trump=trump,
                    second_trick_leader=(play_result.trick_winners[0] if play_result.trick_winners else None),
                )

        terminal = len(spec.play_actions) == 32
        next_actor = play_result.next_actor
        announced: Project | None = None
        project_actions: list[dict[str, Any]] = []
        if native_joint_prefix and native_play.legal_project_actions:
            next_actor = native_play.legal_project_actions[0].actor
            project_actions = [
                {
                    "stage": "project",
                    "code": action.code,
                    **action.to_dict(),
                }
                for action in native_play.legal_project_actions
            ]
        elif (
            not native_joint_prefix
            and
            spec.project_actions_present
            and next_actor is not None
            and len(spec.play_actions) < 4
        ):
            declaration_state = self.rules.project_domain.declaration_state(
                _selected_projects(
                    project_result,
                    dealer=Seat.parse(spec.dealer),
                ),
                dealer=Seat.parse(spec.dealer),
            )
            for action in spec.project_actions:
                declaration_state, _announced = (
                    self.rules.project_domain.apply_declaration_action(
                        declaration_state,
                        action,
                        trump_suit=trump,
                    )
                )
            if declaration_state.current_actor == next_actor:
                project_actions = [
                    {
                        "stage": "project",
                        "code": action.code,
                        **action.to_dict(),
                    }
                    for action in self.rules.project_domain.legal_declaration_actions(
                        declaration_state,
                        trump_suit=trump,
                    )
                ]
        elif (
            not native_joint_prefix
            and next_actor is not None
            and len(spec.play_actions) < 4
        ):
            if self.rules.project_domain.should_announce(len(play_prefix) // 4):
                announced = next(
                    (
                        project
                        for project in projects
                        if project.owner == next_actor
                        and evidence_by_key[
                            (
                                project.owner,
                                project.kind,
                                project.value,
                                project.high_rank,
                            )
                        ].declared
                    ),
                    None,
                )
        play_actions: list[dict[str, Any]] = []
        if next_actor is not None and not project_actions:
            for action in (
                native_play.legal_play_actions
                if native_joint_prefix
                else play_result.legal_play_actions
            ):
                row = {
                    "stage": "play",
                    "actor": next_actor.name.lower(),
                    **_legal_play_rows((action,))[0],
                }
                if announced is not None:
                    row["projectDeclarations"] = [announced.public_declaration()]
                play_actions.append(row)

        escalation_actions: list[dict[str, Any]] = []
        if not spec.play_actions:
            escalation_actions = [
                {
                    "stage": "escalation",
                    "code": choice.code,
                    "actor": choice.actor.name.lower(),
                    "action": choice.action,
                    "mode": choice.mode,
                }
                for choice in escalation.legal_next_choices
            ]
        current_stage = (
            "terminal"
            if terminal
            else "escalation"
            if escalation_actions
            else "project"
            if project_actions or announced is not None
            else "play"
        )
        legal = (
            []
            if terminal
            else [
                *escalation_actions,
                *(project_actions if project_actions else play_actions),
            ]
        )
        core = {
            "schemaVersion": 1,
            "prefixHash": _effective_prefix_hash(spec),
            "currentStage": current_stage,
            "nextActor": (
                None
                if current_stage == "escalation"
                else (next_actor.name.lower() if next_actor is not None else None)
            ),
            "terminal": terminal,
            "legalActions": legal,
            "previewStateHash": canonical_sha256(
                {
                    "hands": {
                        seat.name.lower(): [
                            card.id for card in play_result.remaining_hands[int(seat)]
                        ]
                        for seat in Seat
                    },
                    "leader": (
                        play_result.current_trick[0].actor.name.lower()
                        if play_result.current_trick
                        else (
                            play_result.next_actor.name.lower()
                            if play_result.next_actor is not None
                            else None
                        )
                    ),
                    "trick": [
                        {
                            "actor": play.actor.name.lower(),
                            "card": play.card.id,
                            "ekka": play.ekka,
                        }
                        for play in play_result.current_trick
                    ],
                    "contract": contract.value,
                    "trumpSuit": trump.name.lower() if trump is not None else None,
                    "escalation": escalation.to_dict(),
                    "projectDeclarations": declarations,
                }
            ),
            "previewReceiptHash": canonical_sha256(
                {
                    "auction": auction.normalization_receipt.sha256,
                    "escalation": escalation_result.evidence_hash,
                    "projects": project_result.evidence_hash,
                    "play": play_result.evidence_hash,
                    "rules": self.rules.sha256,
                }
            ),
        }
        remaining()
        return {**core, "sha256": canonical_sha256(core)}


__all__ = [
    "ExactNormalizationStopped",
    "OrToolsExactNormalizationReceiptV1",
    "OrToolsExactStateNormalizerV1",
    "OrToolsNormalizedExactStateV1",
    "ProjectDecisionNotStrategicError",
]
