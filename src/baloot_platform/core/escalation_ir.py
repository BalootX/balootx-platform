from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .enums import Contract, Seat


_SUPPORTED_ACTIONS = ("double", "triple", "four", "gahwa")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    import hashlib
    import json

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class EscalationStateV1:
    level: str
    locked: bool = False
    double_seat: Seat | None = None


@dataclass(frozen=True, slots=True)
class EscalationTransitionV1:
    state: EscalationStateV1
    requested_mode: str | None
    resolved_mode: str


@dataclass(frozen=True, slots=True)
class EscalationDecisionV1:
    """One governed escalation decision, including the actorless close choice."""

    actor: Seat | None
    action: str
    requested_mode: str | None
    resolved_mode: str | None
    state: EscalationStateV1

    @property
    def is_closure(self) -> bool:
        return self.actor is None

    @property
    def code(self) -> str:
        if self.is_closure:
            return self.action
        assert self.actor is not None
        suffix = (
            f":{self.resolved_mode}"
            if self.action in {"double", "four"} and self.resolved_mode is not None
            else ""
        )
        return f"{self.actor.name.lower()}:{self.action}{suffix}"


@dataclass(frozen=True, slots=True)
class EscalationDomainIRV1:
    """Validated escalation semantics shared by runtime, reference, and CP-SAT."""

    enabled: bool
    hokum_chain: tuple[str, ...]
    sun_chain: tuple[str, ...]
    default_level: str
    default_mode: str
    opening_actor_policy: str
    closure_policy: str
    response_actor_policy: str
    sun_score_threshold: int
    sun_double_requires_split_score: bool
    sun_lower_score_inclusive: bool
    sha256: str

    @classmethod
    def from_rules(cls, rules: Mapping[str, Any]) -> "EscalationDomainIRV1":
        try:
            value = dict(rules["escalation"])
            if any(type(item) is not bool for item in (
                value["enabled"], value["sun_double_requires_split_score"],
                value.get("sun_lower_score_inclusive", False),
            )):
                raise TypeError("escalation switches must be booleans")
            if not isinstance(value["hokum_chain"], (list, tuple)) or not isinstance(
                value["sun_chain"], (list, tuple)
            ):
                raise TypeError("escalation chains must be arrays")
            if type(value["sun_score_threshold"]) is not int:
                raise TypeError("Sun score threshold must be an integer")
            core = {
                "schemaVersion": 1,
                "enabled": bool(value["enabled"]),
                "hokumChain": [str(item) for item in value["hokum_chain"]],
                "sunChain": [str(item) for item in value["sun_chain"]],
                "defaultLevel": str(value["default_level"]),
                "defaultMode": str(value["default_mode"]),
                "openingActorPolicy": str(value["opening_actor_policy"]),
                "closurePolicy": str(value["closure_policy"]),
                "responseActorPolicy": str(value["response_actor_policy"]),
                "sunScoreThreshold": int(value["sun_score_threshold"]),
                "sunDoubleRequiresSplitScore": bool(
                    value["sun_double_requires_split_score"]
                ),
                "sunLowerScoreInclusive": bool(
                    value.get("sun_lower_score_inclusive", False)
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("incomplete or invalid escalation domain configuration") from exc

        hokum_chain = tuple(core["hokumChain"])
        sun_chain = tuple(core["sunChain"])
        if hokum_chain != _SUPPORTED_ACTIONS[: len(hokum_chain)]:
            raise ValueError("Hokum escalation chain must be an ordered supported prefix")
        if sun_chain != _SUPPORTED_ACTIONS[: len(sun_chain)] or len(sun_chain) > 1:
            raise ValueError("Sun escalation chain must be empty or contain Double")
        if core["defaultLevel"] != "normal":
            raise ValueError("only Normal is supported as the default escalation level")
        if core["defaultMode"] not in {"open", "locked"}:
            raise ValueError("default escalation mode must be open or locked")
        if core["openingActorPolicy"] != "opponent_team_any_seat":
            raise ValueError("unsupported escalation opening actor policy")
        if core["closurePolicy"] != "advance_to_projects_without_announcement":
            raise ValueError("unsupported escalation closure policy")
        if core["responseActorPolicy"] not in {
            "declarer_and_original_doubler",
            "team",
        }:
            raise ValueError("unsupported escalation response actor policy")
        if core["sunScoreThreshold"] < 0:
            raise ValueError("Sun escalation score threshold must be non-negative")

        return cls(
            core["enabled"],
            hokum_chain,
            sun_chain,
            core["defaultLevel"],
            core["defaultMode"],
            core["openingActorPolicy"],
            core["closurePolicy"],
            core["responseActorPolicy"],
            core["sunScoreThreshold"],
            core["sunDoubleRequiresSplitScore"],
            core["sunLowerScoreInclusive"],
            _canonical_sha256(core),
        )

    @property
    def supported_actions(self) -> tuple[str, ...]:
        return _SUPPORTED_ACTIONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "enabled": self.enabled,
            "hokumChain": list(self.hokum_chain),
            "sunChain": list(self.sun_chain),
            "defaultLevel": self.default_level,
            "defaultMode": self.default_mode,
            "openingActorPolicy": self.opening_actor_policy,
            "closurePolicy": self.closure_policy,
            "responseActorPolicy": self.response_actor_policy,
            "sunScoreThreshold": self.sun_score_threshold,
            "sunDoubleRequiresSplitScore": self.sun_double_requires_split_score,
            "sunLowerScoreInclusive": self.sun_lower_score_inclusive,
            "sha256": self.sha256,
        }

    def initial_state(self) -> EscalationStateV1:
        return EscalationStateV1(self.default_level)

    @property
    def closure_action(self) -> str:
        return "decline"

    def opening_actors(self, declarer: Seat | str | int) -> tuple[Seat, ...]:
        """Return every seat allowed to initiate escalation, with no invented turn order."""

        parsed = Seat.parse(declarer)
        if self.opening_actor_policy != "opponent_team_any_seat":
            raise ValueError("unsupported escalation opening actor policy")
        return tuple(seat for seat in Seat if seat.team != parsed.team)

    def can_close(self, state: EscalationStateV1) -> bool:
        """Return whether the escalation window may advance without an announcement."""

        if not isinstance(state, EscalationStateV1):
            raise TypeError("escalation closure requires EscalationStateV1")
        return self.closure_policy == "advance_to_projects_without_announcement"

    def legal_decisions(
        self,
        state: EscalationStateV1,
        *,
        declarer: Seat | str | int,
        contract: Contract | str,
        match_score: tuple[int, int],
        actors: Sequence[Seat | str | int] | None = None,
        include_closure: bool = True,
    ) -> tuple[EscalationDecisionV1, ...]:
        """Enumerate stable legal decisions from the same transition authority."""

        if not isinstance(state, EscalationStateV1):
            raise TypeError("escalation decisions require EscalationStateV1")
        parsed_declarer = Seat.parse(declarer)
        parsed_contract = Contract.parse(contract)
        candidate_actors = (
            tuple(Seat.parse(actor) for actor in actors)
            if actors is not None
            else (
                self.opening_actors(parsed_declarer)
                if state.level == self.default_level
                else tuple(Seat)
            )
        )
        retained: dict[str, EscalationDecisionV1] = {}
        if include_closure and self.can_close(state):
            close = EscalationDecisionV1(
                None,
                self.closure_action,
                None,
                None,
                state,
            )
            retained[close.code] = close
        for actor in sorted(set(candidate_actors), key=int):
            for action in self.supported_actions:
                modes: tuple[str | None, ...] = (
                    ("open", "locked")
                    if parsed_contract.effective == Contract.HOKUM
                    and action in {"double", "four"}
                    else (None,)
                )
                for mode in modes:
                    try:
                        transition = self.transition(
                            state,
                            declarer=parsed_declarer,
                            contract=parsed_contract,
                            match_score=match_score,
                            actor=actor,
                            action=action,
                            mode=mode,
                        )
                    except ValueError:
                        continue
                    decision = EscalationDecisionV1(
                        actor,
                        action,
                        mode,
                        transition.resolved_mode,
                        transition.state,
                    )
                    retained[decision.code] = decision
        return tuple(retained[key] for key in sorted(retained))

    def transition(
        self,
        state: EscalationStateV1,
        *,
        declarer: Seat,
        contract: Contract,
        match_score: tuple[int, int],
        actor: Seat,
        action: str,
        mode: str | None = None,
    ) -> EscalationTransitionV1:
        if not isinstance(state, EscalationStateV1):
            raise TypeError("escalation transition requires EscalationStateV1")
        if not self.enabled:
            raise ValueError("escalation is disabled")
        declarer = Seat.parse(declarer)
        contract = Contract.parse(contract)
        actor = Seat.parse(actor)
        score = tuple(int(item) for item in match_score)
        if len(score) != 2 or any(item < 0 for item in score):
            raise ValueError("match score must contain two non-negative values")
        action = str(action).strip().lower()
        if action not in self.supported_actions:
            raise ValueError(f"unsupported escalation action {action}")

        if contract in {Contract.SUN, Contract.ASHKAL}:
            if (
                not self.sun_chain
                or action != self.sun_chain[0]
                or state.level != self.default_level
                or actor.team == declarer.team
            ):
                raise ValueError("illegal Sun/Ashkal escalation")
            high, low = max(score), min(score)
            low_meets = (
                low <= self.sun_score_threshold
                if self.sun_lower_score_inclusive
                else low < self.sun_score_threshold
            )
            if self.sun_double_requires_split_score and not (
                high > self.sun_score_threshold and low_meets
            ):
                raise ValueError("Sun split-score threshold is not met")
            next_state = EscalationStateV1(self.sun_chain[0], False, actor)
        else:
            current_index = (
                -1
                if state.level == self.default_level
                else (
                    self.hokum_chain.index(state.level)
                    if state.level in self.hokum_chain
                    else len(self.hokum_chain)
                )
            )
            expected = (
                self.hokum_chain[current_index + 1]
                if current_index + 1 < len(self.hokum_chain)
                else None
            )
            if action != expected:
                raise ValueError(
                    f"illegal escalation {action} after {state.level}; expected {expected}"
                )
            double_seat = state.double_seat
            if action == "double":
                if actor.team == declarer.team:
                    raise ValueError("only a declarer's opponent may double")
                double_seat = actor
            elif action in {"triple", "gahwa"}:
                if self.response_actor_policy == "declarer_and_original_doubler":
                    if actor != declarer:
                        raise ValueError(f"only the declarer may announce {action}")
                elif actor.team != declarer.team:
                    raise ValueError(f"only the declarer's team may announce {action}")
            elif action == "four":
                if double_seat is None:
                    raise ValueError("Four requires a prior Double")
                if self.response_actor_policy == "declarer_and_original_doubler":
                    if actor != double_seat:
                        raise ValueError("the player who doubled must announce Four")
                elif actor.team == declarer.team:
                    raise ValueError("only the opponents may announce Four")

            if action in {"double", "four"}:
                selected_mode = mode or self.default_mode
                if selected_mode not in {"open", "locked"}:
                    raise ValueError("mode must be open or locked")
                locked = selected_mode == "locked"
            else:
                locked = False
            next_state = EscalationStateV1(action, locked, double_seat)

        return EscalationTransitionV1(
            next_state,
            mode,
            "locked" if next_state.locked else "open",
        )


__all__ = [
    "EscalationDecisionV1",
    "EscalationDomainIRV1",
    "EscalationStateV1",
    "EscalationTransitionV1",
]
