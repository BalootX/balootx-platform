"""Engine-free CP-SAT escalation stage for Research V4."""

from __future__ import annotations

import hashlib
import copy
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.escalation_ir import EscalationDomainIRV1, EscalationStateV1
from baloot_platform.core.enums import Contract, Seat, Team

from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1
from .scenario_contracts import canonical_sha256


_LEVELS = ("normal", "double", "triple", "four", "gahwa")


def attach_native_escalation_transition_v1(
    model: cp_model.CpModel, domain: EscalationDomainIRV1, *, contract: Contract,
    declarer: cp_model.IntVar, match_score: tuple[cp_model.IntVar, cp_model.IntVar],
    active: cp_model.IntVar, actor: cp_model.IntVar, action: cp_model.IntVar,
    requested_mode: cp_model.IntVar, before: tuple[cp_model.IntVar, cp_model.IntVar, cp_model.IntVar],
    after: tuple[cp_model.IntVar, cp_model.IntVar, cp_model.IntVar], name: str,
    apply_transition: bool = True,
) -> Mapping[tuple[str, str | None], cp_model.IntVar]:
    """Reified legality and transitions from frozen rule fields, not an interpreter."""
    if not isinstance(model, cp_model.CpModel) or not isinstance(domain, EscalationDomainIRV1):
        raise TypeError("native escalation requires a CP model and Domain IR")
    if any(not isinstance(value, cp_model.IntVar) for value in
           (declarer, *match_score, active, actor, action, requested_mode, *before, *after)):
        raise TypeError("native escalation state must contain solver variables")
    if len(match_score) != 2 or len(before) != 3 or len(after) != 3:
        raise ValueError("native escalation state shape is invalid")
    contract = Contract.parse(contract)
    # Reuse the common exact Boolean constraint primitives.
    from .ortools_scenario_model import _auction_logic_v1
    equal, both, either = _auction_logic_v1(model, name)
    for variable, low, high in ((declarer, 0, 3), (actor, 0, 4), (active, 0, 1),
                                (action, 0, 4), (requested_mode, 0, 2),
                                *[(row[0], 0, 4) for row in (before, after)],
                                *[(row[1], 0, 1) for row in (before, after)],
                                *[(row[2], 0, 4) for row in (before, after)]):
        model.add(variable >= low)
        model.add(variable <= high)
    declarer_team = model.new_int_var(0, 1, f"{name}_declarer_team")
    actor_team = model.new_int_var(0, 1, f"{name}_actor_team")
    model.add_modulo_equality(declarer_team, declarer, 2)
    model.add_modulo_equality(actor_team, actor, 2)
    same_team = equal(actor_team, declarer_team, "same_team")
    real_actor = equal(actor, 4, "absent_actor").Not()
    split_score = model.new_constant(1)
    if contract.effective == Contract.SUN and domain.sun_double_requires_split_score:
        high_flags, low_flags = [], []
        for index, score in enumerate(match_score):
            model.add(score >= 0)
            high = model.new_bool_var(f"{name}_score_high_{index}")
            low = model.new_bool_var(f"{name}_score_low_{index}")
            model.add(score > domain.sun_score_threshold).only_enforce_if(high)
            model.add(score <= domain.sun_score_threshold).only_enforce_if(high.Not())
            limit = domain.sun_score_threshold - int(not domain.sun_lower_score_inclusive)
            model.add(score <= limit).only_enforce_if(low)
            model.add(score > limit).only_enforce_if(low.Not())
            high_flags.append(high)
            low_flags.append(low)
        split_score = both("split", either("high", *high_flags), either("low", *low_flags))
    legal, selected = {}, []
    chain = domain.sun_chain if contract.effective == Contract.SUN else domain.hokum_chain
    for action_index, action_name in enumerate(domain.supported_actions):
        predecessors = [domain.default_level if index == 0 else chain[index - 1]
                        for index, value in enumerate(chain) if value == action_name
                        and (contract.effective != Contract.SUN or index == 0)]
        predecessor = either(f"{action_name}_predecessor", *[
            equal(before[0], _LEVELS.index(value), f"{action_name}_after_{value}") for value in predecessors]) if predecessors else model.new_constant(0)
        if contract.effective == Contract.SUN or action_name == "double":
            role = same_team.Not()
        elif action_name in {"triple", "gahwa"}:
            role = equal(actor, declarer, f"{action_name}_declarer") if domain.response_actor_policy == "declarer_and_original_doubler" else same_team
        else:
            role = both("four_role", equal(before[2], 4, "no_doubler").Not(),
                        equal(actor, before[2], "original_doubler") if domain.response_actor_policy == "declarer_and_original_doubler" else same_team.Not())
        for mode_index in ((0, 1, 2) if action_name in {"double", "four"} else (0,)):
            mode = (None, "open", "locked")[mode_index]
            allowed = both(f"legal_{action_name}_{mode_index}", real_actor, predecessor, role,
                           split_score, model.new_constant(int(domain.enabled)))
            legal[action_name, mode] = allowed
            if not apply_transition:
                continue
            chosen = both(f"selected_{action_name}_{mode_index}", active,
                          equal(action, action_index, f"action_{action_name}_{mode_index}"),
                          equal(requested_mode, mode_index, f"mode_{action_name}_{mode_index}"))
            selected.append(chosen)
            model.add(chosen <= allowed)
            model.add(after[0] == _LEVELS.index(action_name)).only_enforce_if(chosen)
            locked = contract.effective == Contract.HOKUM and action_name in {"double", "four"} and (mode == "locked" or mode is None and domain.default_mode == "locked")
            model.add(after[1] == int(locked)).only_enforce_if(chosen)
            model.add(after[2] == (actor if contract.effective == Contract.SUN or action_name == "double" else before[2])).only_enforce_if(chosen)
    if apply_transition:
        model.add(sum(selected) == active)
        for variable, value in ((actor, 4), (action, 4), (requested_mode, 0), *zip(after, before)):
            model.add(variable == value).only_enforce_if(active.Not())
    return legal


@dataclass(frozen=True, slots=True)
class OrToolsEscalationActionV1:
    actor: Seat
    action: str
    requested_mode: str | None
    resolved_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.name.lower(),
            "team": self.actor.team.name.lower(),
            "action": self.action,
            "requestedMode": self.requested_mode,
            "resolvedMode": self.resolved_mode,
        }


@dataclass(frozen=True, slots=True)
class OrToolsEscalationChoiceV1:
    actor: Seat
    action: str
    mode: str | None

    @property
    def code(self) -> str:
        suffix = f":{self.mode}" if self.mode is not None else ""
        return f"{self.actor.name.lower()}:{self.action}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "actor": self.actor.name.lower(),
            "team": self.actor.team.name.lower(),
            "action": self.action,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class _EscalationCandidateV1:
    actions: tuple[OrToolsEscalationActionV1, ...]
    level: str
    locked: bool
    double_seat: Seat | None
    terminal: bool
    choices: tuple[OrToolsEscalationChoiceV1, ...]
    closure_action: str | None
    fingerprint: str

    @property
    def eligible_seats(self) -> tuple[Seat, ...]:
        return tuple(sorted({choice.actor for choice in self.choices}, key=int))

    @property
    def eligible_teams(self) -> tuple[Team, ...]:
        return tuple(sorted({seat.team for seat in self.eligible_seats}, key=int))

    @property
    def legal_actions(self) -> tuple[str, ...]:
        closure = (self.closure_action,) if self.closure_action is not None else ()
        return (*closure, *sorted({choice.action for choice in self.choices}))


@dataclass(frozen=True, slots=True)
class OrToolsEscalationObservationV1:
    actions: tuple[OrToolsEscalationActionV1, ...]
    level: str
    mode: str
    double_seat: Seat | None
    terminal: bool
    eligible_seats: tuple[Seat, ...]
    eligible_teams: tuple[Team, ...]
    legal_next_actions: tuple[str, ...]
    legal_next_choices: tuple[OrToolsEscalationChoiceV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "actions": [action.to_dict() for action in self.actions],
            "level": self.level,
            "mode": self.mode,
            "doubleSeat": (
                self.double_seat.name.lower() if self.double_seat is not None else None
            ),
            "terminal": self.terminal,
            "eligibleSeats": [seat.name.lower() for seat in self.eligible_seats],
            "eligibleTeams": [team.name.lower() for team in self.eligible_teams],
            "legalNextActions": list(self.legal_next_actions),
            "legalNextChoices": [
                choice.to_dict() for choice in self.legal_next_choices
            ],
        }


@dataclass(frozen=True, slots=True)
class OrToolsEscalationStageResultV1:
    status: str
    solver_status: str
    observation: OrToolsEscalationObservationV1 | None
    domain_ir_hash: str
    model_hash: str
    evidence_hash: str | None
    branches: int
    conflicts: int
    variable_count: int = 0
    constraint_count: int = 0
    opportunity_state: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "status": self.status,
            "solverStatus": self.solver_status,
            "observation": (
                self.observation.to_dict() if self.observation is not None else None
            ),
            "domainIrHash": self.domain_ir_hash,
            "modelHash": self.model_hash,
            "evidenceHash": self.evidence_hash,
            "branches": self.branches,
            "conflicts": self.conflicts,
            "variableCount": self.variable_count,
            "constraintCount": self.constraint_count,
            **({"opportunityState": copy.deepcopy(dict(self.opportunity_state))}
               if self.opportunity_state is not None else {}),
        }


def _normalize_team(value: Team | str | int | None) -> Team | None:
    if value is None:
        return None
    if isinstance(value, Team):
        return value
    if type(value) is int:
        return Team(value)
    return Team[str(value).strip().upper()]


def _normalize_actions(
    values: Sequence[Mapping[str, Any] | tuple[Seat | str | int, str, str | None]],
    *,
    domain: EscalationDomainIRV1,
) -> tuple[tuple[Seat, str, str | None], ...]:
    normalized: list[tuple[Seat, str, str | None]] = []
    for ordinal, raw in enumerate(values, start=1):
        if isinstance(raw, Mapping):
            item = dict(raw)
            unknown = set(item) - {
                "ordinal",
                "actor",
                "action",
                "mode",
                "requestedMode",
            }
            if unknown:
                raise ValueError(
                    f"escalation action {ordinal} has unknown fields: "
                    + ", ".join(sorted(unknown))
                )
            if item.get("ordinal") is not None and item["ordinal"] != ordinal:
                raise ValueError("escalation action ordinals must be contiguous")
            if item.get("actor") is None or item.get("action") is None:
                raise ValueError("escalation actions require actor and action")
            if item.get("mode") is not None and item.get("requestedMode") is not None:
                raise ValueError("escalation action cannot contain both mode aliases")
            actor, action = item["actor"], item["action"]
            mode = item.get("requestedMode", item.get("mode"))
        else:
            if len(raw) != 3:
                raise ValueError("escalation action tuples require actor, action, mode")
            actor, action, mode = raw
        actor = Seat.parse(actor)
        action = str(action).strip().lower()
        if action not in domain.supported_actions:
            raise ValueError(f"unsupported escalation action {action!r}")
        if mode is not None:
            mode = str(mode).strip().lower()
            if mode not in {"open", "locked"}:
                raise ValueError("escalation action mode must be open or locked")
            if action not in {"double", "four"}:
                raise ValueError("only Double or Four may specify a mode")
        normalized.append((actor, action, mode))
    if len(normalized) > len(domain.hokum_chain):
        raise ValueError("escalation action sequence exceeds the active chain")
    return tuple(normalized)


def _legal_choices(
    domain: EscalationDomainIRV1,
    *,
    state: EscalationStateV1,
    declarer: Seat,
    contract: Contract,
    match_score: tuple[int, int],
) -> tuple[OrToolsEscalationChoiceV1, ...]:
    return tuple(
        OrToolsEscalationChoiceV1(
            decision.actor,
            decision.action,
            (decision.resolved_mode if decision.action in {"double", "four"} else None),
        )
        for decision in domain.legal_decisions(
            state,
            declarer=declarer,
            contract=contract,
            match_score=match_score,
            include_closure=False,
        )
        if decision.actor is not None
    )


class OrToolsEscalationStageV1:
    """Select one exact legal escalation state from the shared Domain IR table."""

    def __init__(
        self,
        domain: EscalationDomainIRV1,
        *,
        declarer: Seat | str | int,
        contract: Contract | str,
        match_score: tuple[int, int],
        native_cp_sat: bool = False,
    ) -> None:
        if not isinstance(domain, EscalationDomainIRV1):
            raise TypeError("escalation stage requires EscalationDomainIRV1")
        if (
            not isinstance(match_score, tuple)
            or len(match_score) != 2
            or any(type(value) is not int or value < 0 for value in match_score)
        ):
            raise ValueError("match score must contain two non-negative integers")
        self.domain = domain
        self.declarer = Seat.parse(declarer)
        self.contract = Contract.parse(contract)
        self.match_score = match_score
        if type(native_cp_sat) is not bool:
            raise TypeError("native_cp_sat must be boolean")
        self.native_cp_sat = native_cp_sat

    def _candidates(
        self, cancel_check: Callable[[], bool] | None = None
    ) -> tuple[_EscalationCandidateV1, ...]:
        """Materialize compatibility views from the Domain IR, never qualification."""

        candidates: list[_EscalationCandidateV1] = []
        pending: list[
            tuple[EscalationStateV1, tuple[OrToolsEscalationActionV1, ...]]
        ] = [(self.domain.initial_state(), ())]
        seen: set[
            tuple[str, bool, Seat | None, tuple[tuple[int, str, str | None], ...]]
        ] = set()
        while pending:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("escalation solve cancelled")
            state, actions = pending.pop()
            key = (
                state.level,
                state.locked,
                state.double_seat,
                tuple(
                    (int(item.actor), item.action, item.requested_mode)
                    for item in actions
                ),
            )
            if key in seen:
                continue
            seen.add(key)
            choices = _legal_choices(
                self.domain,
                state=state,
                declarer=self.declarer,
                contract=self.contract,
                match_score=self.match_score,
            )
            closure_action = (
                self.domain.closure_action if self.domain.can_close(state) else None
            )
            core = {
                "domainIrHash": self.domain.sha256,
                "declarer": self.declarer.name.lower(),
                "contract": self.contract.value,
                "matchScore": list(self.match_score),
                "actions": [action.to_dict() for action in actions],
                "level": state.level,
                "locked": state.locked,
                "doubleSeat": (
                    state.double_seat.name.lower()
                    if state.double_seat is not None
                    else None
                ),
                "terminal": not choices,
                "legalNextChoices": [choice.to_dict() for choice in choices],
            }
            candidates.append(
                _EscalationCandidateV1(
                    actions,
                    state.level,
                    state.locked,
                    state.double_seat,
                    not choices,
                    choices,
                    closure_action,
                    canonical_sha256(core),
                )
            )
            if choices and closure_action is not None:
                closed_core = {
                    **core,
                    "terminal": True,
                    "legalNextChoices": [],
                }
                candidates.append(
                    _EscalationCandidateV1(
                        actions,
                        state.level,
                        state.locked,
                        state.double_seat,
                        True,
                        (),
                        closure_action,
                        canonical_sha256(closed_core),
                    )
                )
            for choice in choices:
                requested_modes = (
                    (None, choice.mode)
                    if self.contract == Contract.HOKUM
                    and choice.action in {"double", "four"}
                    else (choice.mode,)
                )
                for requested_mode in requested_modes:
                    transition = self.domain.transition(
                        state,
                        declarer=self.declarer,
                        contract=self.contract,
                        match_score=self.match_score,
                        actor=choice.actor,
                        action=choice.action,
                        mode=requested_mode,
                    )
                    pending.append(
                        (
                            transition.state,
                            (
                                *actions,
                                OrToolsEscalationActionV1(
                                    choice.actor,
                                    choice.action,
                                    requested_mode,
                                    transition.resolved_mode,
                                ),
                            ),
                        )
                    )
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    len(item.actions),
                    item.terminal,
                    item.fingerprint,
                ),
            )
        )

    def _result(
        self,
        candidate: _EscalationCandidateV1,
        *,
        model_hash: str,
        solver_status: str,
        branches: int,
        conflicts: int,
        opportunity_state: Mapping[str, Any] | None = None,
    ) -> OrToolsEscalationStageResultV1:
        observation = OrToolsEscalationObservationV1(
            candidate.actions,
            candidate.level,
            "locked" if candidate.locked else "open",
            candidate.double_seat,
            candidate.terminal,
            candidate.eligible_seats,
            candidate.eligible_teams,
            candidate.legal_actions,
            candidate.choices,
        )
        evidence = {
            "schemaVersion": 1,
            "domainIrHash": self.domain.sha256,
            "modelHash": model_hash,
            "observation": observation.to_dict(),
            **({"opportunityState": copy.deepcopy(dict(opportunity_state))}
               if opportunity_state is not None else {}),
        }
        return OrToolsEscalationStageResultV1(
            "satisfied",
            solver_status,
            observation,
            self.domain.sha256,
            model_hash,
            canonical_sha256(evidence),
            branches,
            conflicts,
            opportunity_state=copy.deepcopy(dict(opportunity_state)) if opportunity_state is not None else None,
        )

    def solve(
        self,
        *,
        required_actions: Sequence[
            Mapping[str, Any] | tuple[Seat | str | int, str, str | None]
        ] = (),
        required_actions_complete: bool = False,
        required_level: str | None = None,
        required_mode: str | None = None,
        required_eligible_seat: Seat | str | int | None = None,
        required_eligible_team: Team | str | int | None = None,
        required_response: str | None = None,
        required_terminal: bool | None = None,
        allowed_candidate_fingerprints: Sequence[str] | None = None,
        seed: int = 0,
        maximum_seconds: float = 15.0,
        maximum_conflicts: int = 250_000,
        cancel_check: Callable[[], bool] | None = None,
        opportunity_non_announcements: Sequence[Seat | str | int] | None = None,
        required_pending_seat: Seat | str | int | None = None,
        configure_model: Callable[[cp_model.CpModel, Mapping[str, Any]], None] | None = None,
    ) -> OrToolsEscalationStageResultV1:
        """Solve announcements; optional private opportunities are not public actions.

        Opportunity mode applies only after a fixed announcement prefix. It does
        not choose an actor order or certify coverage of all such orders.
        """
        if cancel_check is not None and cancel_check():
            raise InterruptedError("escalation solve cancelled")
        if configure_model is not None and (not callable(configure_model) or not self.native_cp_sat):
            raise ValueError("escalation model configuration requires a native stage and callback")
        if type(required_actions_complete) is not bool:
            raise TypeError("required_actions_complete must be boolean")
        if required_terminal is not None and type(required_terminal) is not bool:
            raise TypeError("required_terminal must be boolean")
        non_announcing = None
        if required_pending_seat is not None and opportunity_non_announcements is None:
            raise ValueError("pending seat requires opportunity mode")
        if required_pending_seat is not None and type(required_pending_seat) not in {Seat, str, int}:
            raise TypeError("pending seat must be a seat")
        pending_seat = Seat.parse(required_pending_seat) if required_pending_seat is not None else None
        if opportunity_non_announcements is not None:
            if (not isinstance(opportunity_non_announcements, Sequence)
                    or isinstance(opportunity_non_announcements, (str, bytes))
                    or any(type(item) not in {Seat, str, int} for item in opportunity_non_announcements)):
                raise TypeError("opportunity non-announcements require a sequence of seats")
            non_announcing = tuple(sorted((Seat.parse(item) for item in opportunity_non_announcements), key=int))
            if len(non_announcing) != len(set(non_announcing)):
                raise ValueError("opportunity non-announcements must not repeat a seat")
            if not required_actions_complete or allowed_candidate_fingerprints is not None:
                raise ValueError("opportunity mode requires an exact announcement prefix without fingerprint filtering")
        allowed_fingerprints = (
            None
            if allowed_candidate_fingerprints is None
            else frozenset(allowed_candidate_fingerprints)
        )
        if allowed_fingerprints is not None and any(
            type(item) is not str or len(item) != 64 for item in allowed_fingerprints
        ):
            raise ValueError("allowed escalation fingerprints must be SHA-256 digests")
        if self.native_cp_sat and allowed_fingerprints is not None:
            raise ValueError("native escalation requires action constraints, not legacy candidate fingerprints")
        if type(seed) is not int:
            raise TypeError("escalation solver seed must be an integer")
        if (
            type(maximum_seconds) not in {int, float}
            or isinstance(maximum_seconds, bool)
            or maximum_seconds <= 0
        ):
            raise ValueError("escalation solver maximum_seconds must be positive")
        if type(maximum_conflicts) is not int or maximum_conflicts < 1:
            raise ValueError("escalation solver maximum_conflicts must be positive")

        actions = _normalize_actions(required_actions, domain=self.domain)
        level = str(required_level).strip().lower() if required_level else None
        if level not in {None, *_LEVELS}:
            raise ValueError("unsupported escalation level")
        mode = str(required_mode).strip().lower() if required_mode else None
        if mode not in {None, "open", "locked"}:
            raise ValueError("unsupported escalation mode")
        seat = (
            Seat.parse(required_eligible_seat)
            if required_eligible_seat is not None
            else None
        )
        team = _normalize_team(required_eligible_team)
        response = str(required_response).strip().lower() if required_response else None
        if response not in {
            None,
            self.domain.closure_action,
            *self.domain.supported_actions,
        }:
            raise ValueError("unsupported required escalation response")

        model = cp_model.CpModel()
        levels = {value: index for index, value in enumerate(_LEVELS)}
        action_names = tuple(self.domain.supported_actions)
        action_indexes = {value: index for index, value in enumerate(action_names)}
        mode_names = {0: None, 1: "open", 2: "locked"}
        mode_indexes = {value: key for key, value in mode_names.items()}
        sentinel = 4
        maximum_steps = max(
            1,
            len(self.domain.hokum_chain),
            len(self.domain.sun_chain),
        )
        state_space = tuple(
            (level_index, locked_value, double_value)
            for level_index in range(len(_LEVELS))
            for locked_value in (0, 1)
            for double_value in range(5)
        )
        transition_rows: list[tuple[int, ...]] = []
        for level_index, locked_value, double_value in (() if self.native_cp_sat else state_space):
            transition_rows.append(
                (
                    0,
                    level_index,
                    locked_value,
                    double_value,
                    sentinel,
                    sentinel,
                    0,
                    level_index,
                    locked_value,
                    double_value,
                )
            )
            state_value = EscalationStateV1(
                _LEVELS[level_index],
                bool(locked_value),
                None if double_value == sentinel else Seat(double_value),
            )
            for actor_value in Seat:
                for action_name in action_names:
                    requested_modes = (
                        (None, "open", "locked")
                        if action_name in {"double", "four"}
                        else (None,)
                    )
                    for requested_mode in requested_modes:
                        try:
                            transition = self.domain.transition(
                                state_value,
                                declarer=self.declarer,
                                contract=self.contract,
                                match_score=self.match_score,
                                actor=actor_value,
                                action=action_name,
                                mode=requested_mode,
                            )
                        except ValueError:
                            continue
                        transition_rows.append(
                            (
                                1,
                                level_index,
                                locked_value,
                                double_value,
                                int(actor_value),
                                action_indexes[action_name],
                                mode_indexes[requested_mode],
                                levels[transition.state.level],
                                int(transition.state.locked),
                                (
                                    sentinel
                                    if transition.state.double_seat is None
                                    else int(transition.state.double_seat)
                                ),
                            )
                        )

        active = tuple(
            model.new_bool_var(f"escalation_active_{step}")
            for step in range(maximum_steps)
        )
        actors = tuple(
            model.new_int_var(0, sentinel, f"escalation_actor_{step}")
            for step in range(maximum_steps)
        )
        action_vars = tuple(
            model.new_int_var(0, sentinel, f"escalation_action_{step}")
            for step in range(maximum_steps)
        )
        requested_modes = tuple(
            model.new_int_var(0, 2, f"escalation_mode_{step}")
            for step in range(maximum_steps)
        )
        state_levels = tuple(
            model.new_int_var(0, len(_LEVELS) - 1, f"escalation_level_{step}")
            for step in range(maximum_steps + 1)
        )
        state_locked = tuple(
            model.new_bool_var(f"escalation_locked_{step}")
            for step in range(maximum_steps + 1)
        )
        double_seats = tuple(
            model.new_int_var(0, sentinel, f"escalation_double_seat_{step}")
            for step in range(maximum_steps + 1)
        )
        model.add(state_levels[0] == levels[self.domain.default_level])
        model.add(state_locked[0] == 0)
        model.add(double_seats[0] == sentinel)
        for step in range(maximum_steps):
            if step:
                model.add(active[step - 1] >= active[step])
            if self.native_cp_sat:
                attach_native_escalation_transition_v1(
                    model, self.domain, contract=self.contract,
                    declarer=model.new_constant(int(self.declarer)),
                    match_score=tuple(model.new_constant(value) for value in self.match_score),
                    active=active[step], actor=actors[step], action=action_vars[step],
                    requested_mode=requested_modes[step],
                    before=(state_levels[step], state_locked[step], double_seats[step]),
                    after=(state_levels[step + 1], state_locked[step + 1], double_seats[step + 1]),
                    name=f"native_escalation_{step}",
                )
                continue
            model.add_allowed_assignments(
                [
                    active[step],
                    state_levels[step],
                    state_locked[step],
                    double_seats[step],
                    actors[step],
                    action_vars[step],
                    requested_modes[step],
                    state_levels[step + 1],
                    state_locked[step + 1],
                    double_seats[step + 1],
                ],
                transition_rows,
            )
        length = model.new_int_var(0, maximum_steps, "escalation_length")
        model.add(length == sum(active))

        choices_by_state: dict[
            tuple[int, int, int], tuple[OrToolsEscalationChoiceV1, ...]
        ] = {}
        closure_by_state: dict[tuple[int, int, int], bool] = {}
        all_choice_codes: set[str] = set()
        for state_key in (() if self.native_cp_sat else state_space):
            level_index, locked_value, double_value = state_key
            state_value = EscalationStateV1(
                _LEVELS[level_index],
                bool(locked_value),
                None if double_value == sentinel else Seat(double_value),
            )
            choices_value = _legal_choices(
                self.domain,
                state=state_value,
                declarer=self.declarer,
                contract=self.contract,
                match_score=self.match_score,
            )
            choices_by_state[state_key] = choices_value
            closure_by_state[state_key] = self.domain.can_close(state_value)
            all_choice_codes.update(item.code for item in choices_value)

        choice_literals: dict[str, cp_model.IntVar] = {}
        native_choices: dict[str, OrToolsEscalationChoiceV1] = {}
        if self.native_cp_sat:
            final_state = (state_levels[-1], state_locked[-1], double_seats[-1])
            for candidate_actor in Seat:
                legal = attach_native_escalation_transition_v1(
                    model, self.domain, contract=self.contract,
                    declarer=model.new_constant(int(self.declarer)),
                    match_score=tuple(model.new_constant(value) for value in self.match_score),
                    active=model.new_constant(0), actor=model.new_constant(int(candidate_actor)),
                    action=model.new_constant(4), requested_mode=model.new_constant(0),
                    before=final_state, after=final_state, name=f"native_legal_{int(candidate_actor)}",
                    apply_transition=False,
                )
                for action_name in action_names:
                    modes = ("open", "locked") if self.contract.effective == Contract.HOKUM and action_name in {"double", "four"} else (None,)
                    for requested_mode in modes:
                        choice = OrToolsEscalationChoiceV1(candidate_actor, action_name,
                            requested_mode if requested_mode is not None else "open" if action_name in {"double", "four"} else None)
                        native_choices[choice.code] = choice
                        choice_literals[choice.code] = legal[action_name, requested_mode]
            all_choice_codes.update(native_choices)
        for code in (() if self.native_cp_sat else sorted(all_choice_codes)):
            literal = model.new_bool_var("escalation_choice_" + code.replace(":", "_"))
            model.add_allowed_assignments(
                [state_levels[-1], state_locked[-1], double_seats[-1], literal],
                [
                    (*state_key, int(any(item.code == code for item in choices_value)))
                    for state_key, choices_value in choices_by_state.items()
                ],
            )
            choice_literals[code] = literal

        def any_literal(
            values: Sequence[cp_model.IntVar], name: str
        ) -> cp_model.IntVar:
            result = model.new_bool_var(name)
            if values:
                model.add_max_equality(result, list(values))
            else:
                model.add(result == 0)
            return result

        has_choices = any_literal(
            tuple(choice_literals.values()), "escalation_has_choices"
        )
        closure_available = model.new_bool_var("escalation_closure_available")
        if self.native_cp_sat:
            model.add(closure_available == int(self.domain.closure_policy == "advance_to_projects_without_announcement"))
        else:
            model.add_allowed_assignments(
            [
                state_levels[-1],
                state_locked[-1],
                double_seats[-1],
                closure_available,
            ],
                [(*state_key, int(value)) for state_key, value in closure_by_state.items()],
            )
        closed = model.new_bool_var("escalation_closed")
        model.add(closed <= closure_available)
        model.add(closed <= has_choices)
        terminal = model.new_bool_var("escalation_terminal")
        model.add(terminal >= closed)
        model.add(terminal >= 1 - has_choices)
        model.add(terminal <= closed + 1 - has_choices)

        seat_eligible = {
            seat_value: any_literal(
                tuple(
                    literal
                    for code, literal in choice_literals.items()
                    if code.startswith(seat_value.name.lower() + ":")
                ),
                f"escalation_seat_{int(seat_value)}_eligible",
            )
            for seat_value in Seat
        }
        remaining_seats: dict[Seat, cp_model.IntVar] = {}
        if non_announcing is not None:
            for actor in Seat:
                pending = model.new_bool_var(f"escalation_opportunity_{int(actor)}_remaining")
                if actor in non_announcing:
                    model.add(seat_eligible[actor] == 1)
                    model.add(pending == 0)
                else:
                    model.add(pending == seat_eligible[actor])
                remaining_seats[actor] = pending
            has_remaining = any_literal(tuple(remaining_seats.values()), "escalation_opportunities_remaining")
            model.add(closed <= 1 - has_remaining)
            model.add(closed >= has_choices - has_remaining)
            if pending_seat is not None:
                model.add(remaining_seats[pending_seat] == 1)
        team_eligible = {
            team_value: any_literal(
                tuple(
                    seat_eligible[seat_value]
                    for seat_value in Seat
                    if seat_value.team == team_value
                ),
                f"escalation_team_{int(team_value)}_eligible",
            )
            for team_value in Team
        }
        action_available = {
            action_name: any_literal(
                tuple(
                    literal
                    for code, literal in choice_literals.items()
                    if code.split(":", 2)[1] == action_name
                ),
                f"escalation_action_{action_name}_available",
            )
            for action_name in action_names
        }

        for step, (actor_value, action_name, requested_mode) in enumerate(actions):
            model.add(active[step] == 1)
            model.add(actors[step] == int(actor_value))
            model.add(action_vars[step] == action_indexes[action_name])
            model.add(requested_modes[step] == mode_indexes[requested_mode])
        if required_actions_complete:
            model.add(length == len(actions))
        if level is not None:
            model.add(state_levels[-1] == levels[level])
        if mode is not None:
            model.add(state_locked[-1] == int(mode == "locked"))
        if seat is not None:
            model.add(seat_eligible[seat] == 1)
        if team is not None:
            model.add(team_eligible[team] == 1)
        if response is not None:
            model.add(
                (
                    closure_available
                    if response == self.domain.closure_action
                    else action_available[response]
                )
                == 1
            )
        if required_terminal is not None:
            model.add(terminal == int(required_terminal))

        if allowed_fingerprints is not None:
            candidates = tuple(
                item
                for item in self._candidates(cancel_check)
                if item.fingerprint in allowed_fingerprints
            )
            fingerprint_rows = []
            for candidate in candidates:
                row = [len(candidate.actions), int(candidate.terminal)]
                for step in range(maximum_steps):
                    if step < len(candidate.actions):
                        item = candidate.actions[step]
                        row.extend(
                            (
                                int(item.actor),
                                action_indexes[item.action],
                                mode_indexes[item.requested_mode],
                            )
                        )
                    else:
                        row.extend((sentinel, sentinel, 0))
                fingerprint_rows.append(tuple(row))
            fingerprint_variables = [length, terminal]
            for step in range(maximum_steps):
                fingerprint_variables.extend(
                    (actors[step], action_vars[step], requested_modes[step])
                )
            model.add_allowed_assignments(
                fingerprint_variables,
                fingerprint_rows,
            )

        if configure_model is not None:
            def visible(value, name):
                result = model.new_bool_var(name)
                model.add_min_equality(result, [value, terminal.Not()])
                return result
            predicate_seats = {seat: visible(value, f"predicate_seat_{int(seat)}") for seat, value in seat_eligible.items()}
            predicate_teams = {team: visible(value, f"predicate_team_{int(team)}") for team, value in team_eligible.items()}
            grouped_choices: dict[tuple[str, str | None], list[cp_model.IntVar]] = {}
            for code, choice in native_choices.items():
                grouped_choices.setdefault((choice.action, choice.mode), []).append(choice_literals[code])
            predicate_legal = {key: visible(any_literal(values, f"predicate_action_{key}"), f"predicate_visible_{key}")
                               for key, values in grouped_choices.items()}
            predicate_legal[self.domain.closure_action, None] = closure_available
            configure_model(model, dict(levels=levels, actions=action_indexes, modes=mode_indexes,
                length=length, level=state_levels[-1], locked=state_locked[-1], terminal=terminal,
                stateLocked=state_locked, doubleSeat=double_seats[-1],
                actionSlots=tuple(zip(actors, action_vars, requested_modes)),
                seatEligible=predicate_seats, teamEligible=predicate_teams, legal=predicate_legal))

        model.add_decision_strategy(
            [*active, terminal, *actors, *action_vars, *requested_modes],
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MIN_VALUE,
        )
        request = {
            "schemaVersion": 1,
            "domainIrHash": self.domain.sha256,
            "declarer": self.declarer.name.lower(),
            "contract": self.contract.value,
            "matchScore": list(self.match_score),
            "requiredActions": [
                [actor.name.lower(), action, action_mode]
                for actor, action, action_mode in actions
            ],
            "requiredActionsComplete": required_actions_complete,
            "requiredLevel": level,
            "requiredMode": mode,
            "requiredEligibleSeat": seat.name.lower() if seat is not None else None,
            "requiredEligibleTeam": team.name.lower() if team is not None else None,
            "requiredResponse": response,
            "requiredTerminal": required_terminal,
            "allowedCandidateFingerprints": (
                sorted(allowed_fingerprints)
                if allowed_fingerprints is not None
                else None
            ),
            "lowering": "native_escalation_transition_table_v1",
            **({"transitionAuthority": "native_constraints_v1"} if self.native_cp_sat else
               {"transitionTableHash": canonical_sha256({"rows": transition_rows})}),
            "legalChoiceCodes": sorted(all_choice_codes),
            "cpModelProtoHash": hashlib.sha256(
                str(model.proto).encode("utf-8")
            ).hexdigest(),
            **({"opportunityNonAnnouncements": [actor.name.lower() for actor in non_announcing]}
               if non_announcing is not None else {}),
            **({"requiredPendingSeat": pending_seat.name.lower()} if pending_seat is not None else {}),
        }
        model_hash = canonical_sha256(request)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = seed & 0x7FFFFFFF
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        solver.parameters.max_time_in_seconds = float(maximum_seconds)
        solver.parameters.max_number_of_conflicts = maximum_conflicts
        status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
        solver_status = solver.status_name(status).lower()
        try:
            branches = int(solver.num_branches)
            conflicts = int(solver.num_conflicts)
        except RuntimeError:
            if status != cp_model.UNKNOWN:
                raise
            branches, conflicts = 0, 0
        variable_count = len(model.proto.variables)
        constraint_count = len(model.proto.constraints)
        if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            return OrToolsEscalationStageResultV1(
                "infeasible" if status == cp_model.INFEASIBLE else "incomplete",
                solver_status,
                None,
                self.domain.sha256,
                model_hash,
                None,
                branches,
                conflicts,
                variable_count,
                constraint_count,
            )
        if status != cp_model.OPTIMAL:
            return OrToolsEscalationStageResultV1(
                "incomplete",
                solver_status,
                None,
                self.domain.sha256,
                model_hash,
                None,
                branches,
                conflicts,
                variable_count,
                constraint_count,
            )

        solved_length = solver.value(length)
        solved_actions = tuple(
            OrToolsEscalationActionV1(
                Seat(solver.value(actors[step])),
                action_names[solver.value(action_vars[step])],
                mode_names[solver.value(requested_modes[step])],
                "locked" if solver.value(state_locked[step + 1]) else "open",
            )
            for step in range(solved_length)
        )
        solved_state = EscalationStateV1(
            _LEVELS[solver.value(state_levels[-1])],
            bool(solver.value(state_locked[-1])),
            (
                None
                if solver.value(double_seats[-1]) == sentinel
                else Seat(solver.value(double_seats[-1]))
            ),
        )
        solved_terminal = bool(solver.value(terminal))
        solved_choices = (
            ()
            if solved_terminal
            else tuple(native_choices[code] for code in sorted(native_choices) if solver.boolean_value(choice_literals[code]))
            if self.native_cp_sat else choices_by_state[
                (
                    solver.value(state_levels[-1]),
                    solver.value(state_locked[-1]),
                    solver.value(double_seats[-1]),
                )
            ]
        )
        candidate_core = {
            "domainIrHash": self.domain.sha256,
            "declarer": self.declarer.name.lower(),
            "contract": self.contract.value,
            "matchScore": list(self.match_score),
            "actions": [item.to_dict() for item in solved_actions],
            "level": solved_state.level,
            "locked": solved_state.locked,
            "doubleSeat": (
                solved_state.double_seat.name.lower()
                if solved_state.double_seat is not None
                else None
            ),
            "terminal": solved_terminal,
            "legalNextChoices": [item.to_dict() for item in solved_choices],
        }
        candidate = _EscalationCandidateV1(
            solved_actions,
            solved_state.level,
            solved_state.locked,
            solved_state.double_seat,
            solved_terminal,
            solved_choices,
            (
                self.domain.closure_action
                if solver.boolean_value(closure_available)
                else None
            ),
            canonical_sha256(candidate_core),
        )
        opportunity_state = None
        if non_announcing is not None:
            opportunity_state = {
                "schemaVersion": 1, "kind": "individual_opportunities_before_closure_v1",
                "qualificationAuthority": "cp_sat", "modelHash": model_hash,
                "nonAnnouncingSeats": [actor.name.lower() for actor in non_announcing],
                "remainingEligibleSeats": [actor.name.lower() for actor in Seat if solver.boolean_value(remaining_seats[actor])],
                "remainingChoices": [choice.to_dict() for choice in solved_choices
                                     if solver.boolean_value(remaining_seats[choice.actor])],
                "windowClosed": solved_terminal,
                "variableBindings": {
                    "remainingEligibleSeats": {actor.name.lower(): remaining_seats[actor].name for actor in Seat},
                    "windowClosed": terminal.name,
                },
            }
            opportunity_state["sha256"] = canonical_sha256(opportunity_state)
        return replace(
            self._result(
                candidate,
                model_hash=model_hash,
                solver_status=solver_status,
                branches=branches,
                conflicts=conflicts,
                opportunity_state=opportunity_state,
            ),
            variable_count=variable_count,
            constraint_count=constraint_count,
        )


    def replay_opportunities(
        self, *, required_actions: Sequence[Any], events: Sequence[Mapping[str, Any]],
        maximum_seconds: float, cancel_check: Callable[[], bool] | None = None,
        declared_maximum_seconds: float | None = None,
        expected_final_actions: Sequence[Any] | None = None,
        opportunity_order: Sequence[Seat | str | int] | None = None,
    ) -> dict[str, Any]:
        """Derive an immutable private/public event history through native transitions."""
        if (not isinstance(events, Sequence) or isinstance(events, (str, bytes))
                or len(events) > len(_LEVELS) * len(Seat)):
            raise ValueError("opportunity events must fit the finite escalation action universe")
        normalized_events = []
        order = None
        if opportunity_order is not None:
            if (not isinstance(opportunity_order, Sequence) or isinstance(opportunity_order, (str, bytes))
                    or any(type(seat) not in {Seat, str, int} for seat in opportunity_order)):
                raise ValueError("opportunity order must list every seat exactly once")
            order = tuple(Seat.parse(seat).name.lower() for seat in opportunity_order)
            if len(order) != len(Seat) or len(set(order)) != len(Seat):
                raise ValueError("opportunity order must list every seat exactly once")
        for event in events:
            if (not isinstance(event, Mapping) or set(event) != {"actor", "actionCode"}
                    or type(event["actor"]) is not str or type(event["actionCode"]) is not str):
                raise ValueError("opportunity event requires only actor and actionCode strings")
            normalized_events.append({"actor": Seat.parse(event["actor"]).name.lower(),
                                      "actionCode": event["actionCode"]})
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("opportunity replay time limit must be positive and finite")
        declared = maximum_seconds if declared_maximum_seconds is None else declared_maximum_seconds
        if type(declared) not in {int, float} or not math.isfinite(declared) or declared < maximum_seconds:
            raise ValueError("declared wall limit must cover the runtime allowance")
        deadline = time.monotonic() + maximum_seconds
        core = {"schemaVersion": 1, "kind": "native_escalation_opportunity_replay_v1",
                "domainIrHash": self.domain.sha256, "limits": {"maximumSeconds": declared},
                "events": normalized_events, "status": "incomplete", "completedEvents": 0,
                "initialState": None, "finalState": None, "transitions": []}
        if order is not None:
            core["opportunityOrder"] = list(order)
        expected = None if expected_final_actions is None else _normalize_actions(expected_final_actions, domain=self.domain)
        if expected is not None:
            core["expectedFinalActions"] = [
                {"actor": actor.name.lower(), "action": action, "requestedMode": mode}
                for actor, action, mode in expected
            ]
        def stopped():
            reason = ("cancelled" if cancel_check is not None and cancel_check() else
                      "maximum_seconds" if time.monotonic() >= deadline else None)
            if reason is not None:
                core["status"], core["stopReason"] = "incomplete", reason
            return reason is not None
        if not stopped():
            try:
                initial = self.solve(required_actions=required_actions, required_actions_complete=True,
                    opportunity_non_announcements=(), maximum_seconds=max(deadline - time.monotonic(), 1e-9),
                    cancel_check=cancel_check).to_dict()
                core.update(initialState=initial, finalState=initial, status=initial["status"])
            except InterruptedError:
                core["stopReason"] = "cancelled"
        for event in normalized_events:
            if core["status"] != "satisfied" or stopped():
                break
            current = core["finalState"]
            if order is not None and event["actor"] != next((seat for seat in order
                    if seat in current["opportunityState"]["remainingEligibleSeats"]), None):
                raise ValueError("recorded opportunity actor disagrees with declared order")
            actions = tuple((row["actor"], row["action"], row["requestedMode"])
                            for row in current["observation"]["actions"])
            transition = self.transition_opportunity(required_actions=actions,
                non_announcements=current["opportunityState"]["nonAnnouncingSeats"],
                actor=event["actor"], action_code=event["actionCode"],
                maximum_seconds=max(deadline - time.monotonic(), 1e-9),
                declared_maximum_seconds=declared, cancel_check=cancel_check)
            core["transitions"].append(transition)
            core["status"] = transition["status"]
            if transition["status"] == "satisfied":
                core["finalState"] = transition["after"]
                core["completedEvents"] += 1
            else:
                if "stopReason" in transition:
                    core["stopReason"] = transition["stopReason"]
                break
        if core["status"] == "satisfied":
            stopped()
        if core["status"] == "satisfied" and expected is not None:
            observed = _normalize_actions([
                (row["actor"], row["action"], row["requestedMode"])
                for row in core["finalState"]["observation"]["actions"]
            ], domain=self.domain)
            if observed != expected:
                raise ValueError("recorded opportunities disagree with public escalation actions")
        if order is not None and core["status"] == "satisfied":
            final = core["finalState"]
            core["nextActor"] = None if final["observation"]["terminal"] else next(
                (seat for seat in order if seat in final["opportunityState"]["remainingEligibleSeats"]), None)
        return {**core, "sha256": canonical_sha256(core)}


    def transition_opportunity(
        self, *, required_actions: Sequence[Any], non_announcements: Sequence[Seat | str | int],
        actor: Seat | str | int, action_code: str, maximum_seconds: float,
        cancel_check: Callable[[], bool] | None = None,
        declared_maximum_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Record a native legal opportunity transition, not strategic quality.

        A non-announcement never enters public announcement history. Both
        normalization solves share the caller's wall allowance.
        """
        if type(actor) not in {Seat, str, int}:
            raise TypeError("opportunity actor must be a seat")
        seat = Seat.parse(actor)
        if type(action_code) is not str:
            raise TypeError("opportunity action code must be a string")
        parts = action_code.split(":")
        if len(parts) not in {1, 2} or parts[0] not in {*_LEVELS[1:], self.domain.closure_action}:
            raise ValueError("unsupported opportunity action")
        mode = parts[1] if len(parts) == 2 else None
        if mode is not None and (mode not in {"open", "locked"} or parts[0] not in {"double", "four"}):
            raise ValueError("unsupported opportunity action mode")
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("opportunity transition time limit must be positive and finite")
        declared_seconds = maximum_seconds if declared_maximum_seconds is None else declared_maximum_seconds
        if (type(declared_seconds) not in {int, float} or not math.isfinite(declared_seconds)
                or declared_seconds < maximum_seconds):
            raise ValueError("declared wall limit must be finite and cover the runtime allowance")
        deadline = time.monotonic() + maximum_seconds
        private = parts[0] == self.domain.closure_action
        core = {"schemaVersion": 1, "kind": "native_escalation_opportunity_transition_v1",
                "domainIrHash": self.domain.sha256,
                "limits": {"maximumSeconds": declared_seconds},
                "event": {"actor": seat.name.lower(), "actionCode": action_code,
                          "visibility": "private" if private else "public"},
                "status": "incomplete", "before": None, "after": None}
        def solve(**kwargs):
            try:
                return self.solve(**kwargs)
            except InterruptedError:
                core["stopReason"] = "cancelled"
                return None
        before = solve(required_actions=required_actions, required_actions_complete=True,
            opportunity_non_announcements=non_announcements, required_pending_seat=seat,
            maximum_seconds=maximum_seconds, cancel_check=cancel_check)
        if before is None:
            return {**core, "sha256": canonical_sha256(core)}
        core["before"] = before.to_dict()
        if before.status != "satisfied":
            core["status"] = before.status
        elif time.monotonic() < deadline:
            actions = tuple((item.actor, item.action, item.requested_mode) for item in before.observation.actions)
            declines = before.opportunity_state["nonAnnouncingSeats"]
            after = solve(
                required_actions=actions if private else (*actions, (seat, parts[0], mode)),
                required_actions_complete=True,
                opportunity_non_announcements=(*declines, seat) if private else (),
                maximum_seconds=max(deadline - time.monotonic(), 1e-9), cancel_check=cancel_check)
            if after is not None:
                core["after"] = after.to_dict()
                core["status"] = after.status
        else:
            core["stopReason"] = "maximum_seconds"
        return {**core, "sha256": canonical_sha256(core)}


__all__ = [
    "OrToolsEscalationActionV1",
    "OrToolsEscalationChoiceV1",
    "OrToolsEscalationObservationV1",
    "OrToolsEscalationStageResultV1",
    "OrToolsEscalationStageV1",
]
