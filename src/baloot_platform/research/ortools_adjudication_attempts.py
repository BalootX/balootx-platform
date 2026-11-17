"""Native CP-SAT branches for governed illegal-action attempts.

These branches model an attempted action and its adjudication remedy.  They do
not append the illegal attempt to the ordinary auction or play history.  The
caller must bind the actor and attempted-card/token arguments to the existing
auction or play domain variables; this module never creates a parallel trace.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, Mapping

from ortools.sat.python import cp_model

from baloot_platform.core.enums import Seat

from .ortools_infeasible_core import sound_infeasible_assumption_indexes
from .scenario_contracts import canonical_sha256


_HASH = re.compile(r"^[0-9a-f]{64}$")
_ASSUMPTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")


class AdjudicationAttemptKindV1(StrEnum):
    OUT_OF_TURN = "out_of_turn"
    PURCHASE_BEFORE_TURN = "purchase_before_turn"
    FALSE_EKKA = "false_ekka_attempt"


class PurchaseAttemptTokenV1(IntEnum):
    SUN = 0
    HOKUM = 1
    PASS = 2
    NO = 3


class AdjudicationRemedyV1(IntEnum):
    NONE = 0
    CONTINUE_PLAY = 1
    OPPONENT_REGISTER_WITH_PROJECTS = 2
    OPPONENT_TAKWEESH = 3


_REMEDIES_BY_KIND = {
    AdjudicationAttemptKindV1.OUT_OF_TURN: (
        AdjudicationRemedyV1.OPPONENT_REGISTER_WITH_PROJECTS,
        AdjudicationRemedyV1.CONTINUE_PLAY,
    ),
    AdjudicationAttemptKindV1.PURCHASE_BEFORE_TURN: (
        AdjudicationRemedyV1.OPPONENT_TAKWEESH,
        AdjudicationRemedyV1.CONTINUE_PLAY,
    ),
    AdjudicationAttemptKindV1.FALSE_EKKA: (
        AdjudicationRemedyV1.OPPONENT_REGISTER_WITH_PROJECTS,
        AdjudicationRemedyV1.CONTINUE_PLAY,
    ),
}


def _parse_kind(value: AdjudicationAttemptKindV1 | str) -> AdjudicationAttemptKindV1:
    try:
        return AdjudicationAttemptKindV1(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported adjudication attempt kind: {value!r}") from exc


def _parse_remedy(value: AdjudicationRemedyV1 | str | None) -> AdjudicationRemedyV1 | None:
    if value is None or isinstance(value, AdjudicationRemedyV1):
        return value
    try:
        return AdjudicationRemedyV1[str(value).strip().upper()]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unsupported adjudication remedy: {value!r}") from exc


def _require_int_var(
    model: cp_model.CpModel,
    value: Any,
    label: str,
) -> cp_model.IntVar:
    if not isinstance(value, cp_model.IntVar):
        raise TypeError(f"{label} must be a CP-SAT IntVar")
    if value.model_proto is not model.proto:
        raise ValueError(f"{label} must belong to the supplied CP-SAT model")
    return value


@dataclass(frozen=True, slots=True)
class OrToolsAdjudicationAttemptResultV1:
    status: str
    solver_status: str
    kind: AdjudicationAttemptKindV1
    event_present: bool | None
    expected_actor: Seat | None
    attempted_actor: Seat | None
    attempted_action: Mapping[str, Any] | None
    allowed_remedies: tuple[AdjudicationRemedyV1, ...]
    selected_remedy: AdjudicationRemedyV1 | None
    authority_id: str
    authority_hash: str
    model_hash: str
    evidence_hash: str | None
    assumption_core: tuple[str, ...]
    branches: int
    conflicts: int

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schemaVersion": 1,
            "status": self.status,
            "solverStatus": self.solver_status,
            "kind": self.kind.value,
            "eventPresent": self.event_present,
            "expectedActor": (
                self.expected_actor.name.lower() if self.expected_actor is not None else None
            ),
            "attemptedActor": (
                self.attempted_actor.name.lower()
                if self.attempted_actor is not None
                else None
            ),
            "attemptedAction": (
                dict(self.attempted_action) if self.attempted_action is not None else None
            ),
            "allowedRemedies": [remedy.name.lower() for remedy in self.allowed_remedies],
            "ordinaryActionHistoryMutated": False,
            "authorityId": self.authority_id,
            "authorityHash": self.authority_hash,
            "modelHash": self.model_hash,
            "evidenceHash": self.evidence_hash,
            "assumptionCore": list(self.assumption_core),
            "branches": self.branches,
            "conflicts": self.conflicts,
        }
        if self.selected_remedy is not None:
            result["selectedRemedy"] = self.selected_remedy.name.lower()
        return result


@dataclass(frozen=True, slots=True)
class OrToolsAdjudicationAttemptBranchV1:
    model: cp_model.CpModel
    kind: AdjudicationAttemptKindV1
    expected_actor: cp_model.IntVar
    attempted_actor: cp_model.IntVar
    attempted_action: cp_model.IntVar
    event_present: cp_model.IntVar
    admissible_remedy: cp_model.IntVar
    authority_id: str
    authority_hash: str
    assumption_id: str | None
    assumption: cp_model.IntVar | None
    required_remedy: AdjudicationRemedyV1 | None

    def _model_hash(self) -> str:
        return canonical_sha256(
            {
                "schemaVersion": 1,
                "backendId": "ortools_cp_sat_v3",
                "kind": self.kind.value,
                "authorityId": self.authority_id,
                "authorityHash": self.authority_hash,
                "cpModelProtoHash": hashlib.sha256(
                    str(self.model.proto).encode("utf-8")
                ).hexdigest(),
            }
        )

    def solve(
        self,
        *,
        seed: int = 0,
        maximum_seconds: float = 5.0,
    ) -> OrToolsAdjudicationAttemptResultV1:
        if type(seed) is not int:
            raise TypeError("adjudication attempt seed must be an integer")
        if type(maximum_seconds) not in {int, float} or isinstance(
            maximum_seconds, bool
        ):
            raise TypeError("adjudication attempt maximum_seconds must be numeric")
        if maximum_seconds <= 0:
            raise ValueError("adjudication attempt maximum_seconds must be positive")

        model_hash = self._model_hash()
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = seed & 0x7FFFFFFF
        solver.parameters.max_time_in_seconds = float(maximum_seconds)
        status = solver.solve(self.model)
        solver_status = solver.status_name(status).lower()
        branches = int(solver.num_branches)
        conflicts = int(solver.num_conflicts)

        if status == cp_model.INFEASIBLE:
            own_core = ()
            if self.assumption is not None and (
                self.assumption.index
                in sound_infeasible_assumption_indexes(
                    solver, (self.assumption.index,)
                )
            ):
                own_core = (str(self.assumption_id),)
            return OrToolsAdjudicationAttemptResultV1(
                "infeasible",
                solver_status,
                self.kind,
                None,
                None,
                None,
                None,
                _REMEDIES_BY_KIND[self.kind],
                None,
                self.authority_id,
                self.authority_hash,
                model_hash,
                None,
                own_core,
                branches,
                conflicts,
            )
        if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            return OrToolsAdjudicationAttemptResultV1(
                "incomplete",
                solver_status,
                self.kind,
                None,
                None,
                None,
                None,
                _REMEDIES_BY_KIND[self.kind],
                None,
                self.authority_id,
                self.authority_hash,
                model_hash,
                None,
                (),
                branches,
                conflicts,
            )

        event_present = solver.boolean_value(self.event_present)
        expected_actor = Seat(solver.value(self.expected_actor))
        attempted_actor = Seat(solver.value(self.attempted_actor))
        action_value = solver.value(self.attempted_action)
        attempted_action: Mapping[str, Any]
        if self.kind in {
            AdjudicationAttemptKindV1.OUT_OF_TURN,
            AdjudicationAttemptKindV1.FALSE_EKKA,
        }:
            attempted_action = {"type": "card", "cardId": action_value}
        else:
            attempted_action = {
                "type": "auction_token",
                "token": PurchaseAttemptTokenV1(action_value).name.lower(),
            }
        selected_remedy = (
            AdjudicationRemedyV1(solver.value(self.admissible_remedy))
            if self.required_remedy is not None
            else None
        )
        allowed_remedies = _REMEDIES_BY_KIND[self.kind]
        evidence_core = {
            "schemaVersion": 1,
            "backendId": "ortools_cp_sat_v3",
            "kind": self.kind.value,
            "eventPresent": event_present,
            "expectedActor": expected_actor.name.lower(),
            "attemptedActor": attempted_actor.name.lower(),
            "attemptedAction": dict(attempted_action),
            "allowedRemedies": [remedy.name.lower() for remedy in allowed_remedies],
            "ordinaryActionHistoryMutated": False,
            "authorityId": self.authority_id,
            "authorityHash": self.authority_hash,
            "modelHash": model_hash,
        }
        if selected_remedy is not None:
            evidence_core["selectedRemedy"] = selected_remedy.name.lower()
        return OrToolsAdjudicationAttemptResultV1(
            "witness" if event_present else "no_event",
            solver_status,
            self.kind,
            event_present,
            expected_actor,
            attempted_actor,
            attempted_action,
            allowed_remedies,
            selected_remedy,
            self.authority_id,
            self.authority_hash,
            model_hash,
            canonical_sha256(evidence_core),
            (),
            branches,
            conflicts,
        )


def add_adjudication_attempt_branch_v1(
    model: cp_model.CpModel,
    *,
    kind: AdjudicationAttemptKindV1 | str,
    expected_actor: cp_model.IntVar,
    attempted_actor: cp_model.IntVar,
    attempted_action: cp_model.IntVar,
    authority_id: str,
    authority_hash: str,
    require_event: bool = False,
    required_remedy: AdjudicationRemedyV1 | str | None = None,
    assumption_id: str | None = None,
) -> OrToolsAdjudicationAttemptBranchV1:
    """Attach a branch using the caller's existing actor and action variables."""

    if not isinstance(model, cp_model.CpModel):
        raise TypeError("adjudication attempt model must be CpModel")
    parsed_kind = _parse_kind(kind)
    expected = _require_int_var(model, expected_actor, "expected_actor")
    attempted = _require_int_var(model, attempted_actor, "attempted_actor")
    if type(authority_id) is not str or not authority_id.strip():
        raise ValueError("adjudication authority_id must be a non-empty string")
    if type(authority_hash) is not str or not _HASH.fullmatch(authority_hash):
        raise ValueError("adjudication authority_hash must be a lowercase SHA-256")
    if type(require_event) is not bool:
        raise TypeError("require_event must be a boolean")
    if assumption_id is not None and (
        type(assumption_id) is not str or not _ASSUMPTION_ID.fullmatch(assumption_id)
    ):
        raise ValueError("assumption_id must be a stable 1-96 character identifier")

    model.add_allowed_assignments((expected,), tuple((seat,) for seat in range(4)))
    model.add_allowed_assignments((attempted,), tuple((seat,) for seat in range(4)))

    action = _require_int_var(model, attempted_action, "attempted_action")
    allowed_actions = (
        tuple(range(32))
        if parsed_kind
        in {AdjudicationAttemptKindV1.OUT_OF_TURN, AdjudicationAttemptKindV1.FALSE_EKKA}
        else tuple(int(token) for token in PurchaseAttemptTokenV1)
    )
    model.add_allowed_assignments((action,), tuple((value,) for value in allowed_actions))

    same_actor = model.new_bool_var("adjudication_attempt_actor_is_expected")
    event_present = model.new_bool_var("adjudication_attempt_event_present")
    model.add(expected == attempted).only_enforce_if(same_actor)
    model.add(expected != attempted).only_enforce_if(same_actor.Not())
    if parsed_kind == AdjudicationAttemptKindV1.FALSE_EKKA:
        model.add(event_present == same_actor)
    else:
        model.add(event_present + same_actor == 1)

    admissible_remedy = model.new_int_var(
        int(AdjudicationRemedyV1.NONE),
        int(AdjudicationRemedyV1.OPPONENT_TAKWEESH),
        "adjudication_attempt_admissible_remedy",
    )
    allowed_remedies = _REMEDIES_BY_KIND[parsed_kind]
    model.add_allowed_assignments(
        (event_present, admissible_remedy),
        (
            (0, int(AdjudicationRemedyV1.NONE)),
            *((1, int(value)) for value in allowed_remedies),
        ),
    )

    parsed_remedy = _parse_remedy(required_remedy)
    if parsed_remedy is not None and parsed_remedy not in allowed_remedies:
        raise ValueError(
            f"remedy {parsed_remedy.name.lower()} is not valid for {parsed_kind.value}"
        )

    assumption: cp_model.IntVar | None = None
    if assumption_id is not None:
        assumption = model.new_bool_var(f"assume_adjudication_{assumption_id}")
        model.add(event_present == 1).only_enforce_if(assumption)
        if parsed_remedy is not None:
            model.add(admissible_remedy == int(parsed_remedy)).only_enforce_if(
                assumption
            )
        model.add_assumption(assumption)
    else:
        if require_event or parsed_remedy is not None:
            model.add(event_present == 1)
        if parsed_remedy is not None:
            model.add(admissible_remedy == int(parsed_remedy))

    return OrToolsAdjudicationAttemptBranchV1(
        model,
        parsed_kind,
        expected,
        attempted,
        action,
        event_present,
        admissible_remedy,
        authority_id.strip(),
        authority_hash,
        assumption_id,
        assumption,
        parsed_remedy,
    )


__all__ = [
    "AdjudicationAttemptKindV1",
    "AdjudicationRemedyV1",
    "OrToolsAdjudicationAttemptBranchV1",
    "OrToolsAdjudicationAttemptResultV1",
    "PurchaseAttemptTokenV1",
    "add_adjudication_attempt_branch_v1",
]
