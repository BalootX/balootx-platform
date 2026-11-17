from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Any

from baloot_platform.core.escalation_ir import (
    EscalationDomainIRV1,
    EscalationStateV1,
)
from baloot_platform.core.enums import Contract, Seat

from .scenario_contracts import canonical_sha256


@dataclass(frozen=True, slots=True)
class ReferenceEscalationActionV1:
    actor: Seat
    action: str
    requested_mode: str | None
    resolved_mode: str


ReferenceEscalationStateV1 = EscalationStateV1


@dataclass(frozen=True, slots=True)
class ReferenceEscalationPlanV1:
    actions: tuple[ReferenceEscalationActionV1, ...]
    state: ReferenceEscalationStateV1
    fingerprint: str


class ReferenceEscalationInterpreterV1:
    """Small independent interpreter for the governed escalation transition IR."""

    def __init__(self, domain: EscalationDomainIRV1) -> None:
        if not isinstance(domain, EscalationDomainIRV1):
            raise TypeError(
                "ReferenceEscalationInterpreterV1 requires a compiled EscalationDomainIRV1"
            )
        self.domain = domain

    @classmethod
    def from_rules(
        cls, rules: Mapping[str, Any]
    ) -> "ReferenceEscalationInterpreterV1":
        return cls(EscalationDomainIRV1.from_rules(rules))

    def apply(
        self,
        state: ReferenceEscalationStateV1,
        *,
        declarer: Seat,
        contract: Contract,
        match_score: tuple[int, int],
        actor: Seat,
        action: str,
        mode: str | None = None,
    ) -> tuple[ReferenceEscalationStateV1, ReferenceEscalationActionV1]:
        transition = self.domain.transition(
            state,
            declarer=declarer,
            contract=contract,
            match_score=match_score,
            actor=actor,
            action=action,
            mode=mode,
        )
        return transition.state, ReferenceEscalationActionV1(
            Seat.parse(actor),
            str(action).strip().lower(),
            mode,
            transition.resolved_mode,
        )

    def apply_sequence(
        self,
        *,
        declarer: Seat,
        contract: Contract,
        match_score: tuple[int, int],
        actions: Iterable[tuple[Seat, str, str | None]],
    ) -> ReferenceEscalationPlanV1:
        state = self.domain.initial_state()
        applied: list[ReferenceEscalationActionV1] = []
        for actor, action, mode in actions:
            state, record = self.apply(
                state,
                declarer=declarer,
                contract=contract,
                match_score=match_score,
                actor=actor,
                action=action,
                mode=mode,
            )
            applied.append(record)
        core = {
            "domainIrHash": self.domain.sha256,
            "declarer": Seat.parse(declarer).name.lower(),
            "contract": Contract.parse(contract).value,
            "matchScore": list(match_score),
            "actions": [
                [item.actor.name.lower(), item.action, item.requested_mode]
                for item in applied
            ],
            "level": state.level,
            "locked": state.locked,
            "doubleSeat": (
                state.double_seat.name.lower()
                if state.double_seat is not None
                else None
            ),
        }
        return ReferenceEscalationPlanV1(
            tuple(applied), state, canonical_sha256(core)
        )

    def legal_plans(
        self,
        *,
        declarer: Seat,
        contract: Contract,
        match_score: tuple[int, int],
    ) -> tuple[ReferenceEscalationPlanV1, ...]:
        retained: dict[str, ReferenceEscalationPlanV1] = {}

        def visit(actions: tuple[tuple[Seat, str, str | None], ...]) -> None:
            try:
                plan = self.apply_sequence(
                    declarer=declarer,
                    contract=contract,
                    match_score=match_score,
                    actions=actions,
                )
            except ValueError:
                return
            retained[plan.fingerprint] = plan
            if len(actions) >= 4:
                return
            for actor in Seat:
                for action in self.domain.supported_actions:
                    modes: Iterable[str | None] = (
                        (None, "open", "locked")
                        if Contract.parse(contract) == Contract.HOKUM
                        and action in {"double", "four"}
                        else (None,)
                    )
                    for mode in modes:
                        visit((*actions, (actor, action, mode)))

        visit(())
        return tuple(retained[key] for key in sorted(retained))
