from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.enums import Contract, GamePhase, Seat
from baloot_platform.core.models import RoundState
from baloot_platform.core.rules import RuleBook
from baloot_platform.plugins.base import PluginDescriptor

from .contracts import canonical_sha256
from .reference_escalation import ReferenceEscalationInterpreterV1


_LEVELS = ("normal", "double", "triple", "four", "gahwa")


@dataclass(frozen=True, slots=True)
class EscalationActionV1:
    ordinal: int
    actor: Seat
    action: str
    requested_mode: str | None
    resolved_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "actor": self.actor.name.lower(),
            "action": self.action,
            "requestedMode": self.requested_mode,
            "resolvedMode": self.resolved_mode,
        }


@dataclass(frozen=True, slots=True)
class EscalationPlanV1:
    actions: tuple[EscalationActionV1, ...]
    level: str
    locked: bool
    double_seat: Seat | None
    model_hash: str
    solver_status: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "actions": [item.to_dict() for item in self.actions],
            "level": self.level,
            "mode": "locked" if self.locked else "open",
            "doubleSeat": (
                self.double_seat.name.lower()
                if self.double_seat is not None
                else None
            ),
            "modelHash": self.model_hash,
            "solverStatus": self.solver_status,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _LegalEscalation:
    actions: tuple[EscalationActionV1, ...]
    level: str
    locked: bool
    double_seat: Seat | None
    fingerprint: str


class OrToolsCpSatEscalationCompilerV1:
    """Compile the independent escalation Domain IR into an exact CP-SAT table."""

    descriptor = PluginDescriptor(
        id="ortools_cp_sat_escalation_compiler_v1",
        kind="domain_lowering",
        version="1.1.0",
        description="Independent Domain-IR escalation table compiled to CP-SAT",
        capabilities=(
            "complete_escalation",
            "sun_threshold_and_split_score",
            "hokum_chain_and_actor_policy",
            "open_locked_mode",
            "runtime_transition_table_parity",
        ),
        deterministic=True,
        resource_class="cpu",
        timeout_seconds=30,
        maximum_memory_mb=256,
        contract_id="cp_sat_lowering",
        contract_version="1",
        safety_guards=(
            "compiled_domain_authority",
            "authoritative_legality",
            "hard_gate_preservation",
        ),
    )

    @staticmethod
    def _root_state(
        *,
        rules: RuleBook,
        dealer: Seat,
        declarer: Seat,
        contract: Contract,
        match_score: tuple[int, int],
        domain: EscalationDomainIRV1,
    ) -> RoundState:
        state = RoundState(rules.sha256, 0, 1, dealer)
        state.phase = GamePhase.FINAL_DEALT
        state.match_score = [int(match_score[0]), int(match_score[1])]
        state.contract = Contract.parse(contract)
        state.buyer = declarer
        state.declarer = declarer
        state.escalation = domain.default_level
        return state

    @classmethod
    def _legal_escalations(
        cls,
        *,
        dealer: Seat,
        declarer: Seat,
        contract: Contract,
        match_score: tuple[int, int],
        rules: RuleBook,
    ) -> tuple[_LegalEscalation, ...]:
        domain = EscalationDomainIRV1.from_rules(rules.data)
        reference = ReferenceEscalationInterpreterV1(domain)
        plans = [
            _LegalEscalation(
                tuple(
                    EscalationActionV1(
                        ordinal,
                        action.actor,
                        action.action,
                        action.requested_mode,
                        action.resolved_mode,
                    )
                    for ordinal, action in enumerate(plan.actions, start=1)
                ),
                plan.state.level,
                plan.state.locked,
                plan.state.double_seat,
                plan.fingerprint,
            )
            for plan in reference.legal_plans(
                declarer=declarer,
                contract=contract,
                match_score=match_score,
            )
        ]
        plans.sort(key=lambda item: item.fingerprint)
        if len({item.fingerprint for item in plans}) != len(plans):
            raise AssertionError("escalation compiler produced duplicate traces")
        return tuple(plans)

    @classmethod
    def _replay(
        cls,
        plan: _LegalEscalation,
        *,
        dealer: Seat,
        declarer: Seat,
        contract: Contract,
        match_score: tuple[int, int],
        rules: RuleBook,
    ) -> None:
        domain = EscalationDomainIRV1.from_rules(rules.data)
        engine = BalootEngine(rules)
        state = cls._root_state(
            rules=rules,
            dealer=dealer,
            declarer=declarer,
            contract=contract,
            match_score=match_score,
            domain=domain,
        )
        for action in plan.actions:
            engine.apply_escalation(
                state,
                seat=action.actor,
                action=action.action,
                mode=action.requested_mode,
            )
        actual = (state.escalation, state.locked, state.double_seat)
        expected = (plan.level, plan.locked, plan.double_seat)
        if actual != expected:
            raise AssertionError("compiled escalation differs from runtime replay")

    @classmethod
    def reference_runtime_parity(
        cls,
        *,
        dealer: Seat,
        declarer: Seat,
        contract: Contract,
        match_score: tuple[int, int],
        rules: RuleBook,
    ) -> dict[str, Any]:
        domain = EscalationDomainIRV1.from_rules(rules.data)
        plans = cls._legal_escalations(
            dealer=dealer,
            declarer=declarer,
            contract=contract,
            match_score=match_score,
            rules=rules,
        )
        for plan in plans:
            cls._replay(
                plan,
                dealer=dealer,
                declarer=declarer,
                contract=contract,
                match_score=match_score,
                rules=rules,
            )
        core = {
            "schemaVersion": 1,
            "rulesHash": rules.sha256,
            "escalationDomainIrHash": domain.sha256,
            "dealer": Seat.parse(dealer).name.lower(),
            "declarer": Seat.parse(declarer).name.lower(),
            "contract": Contract.parse(contract).value,
            "matchScore": [int(item) for item in match_score],
            "traceCount": len(plans),
            "traceSetHash": canonical_sha256(
                [item.fingerprint for item in plans]
            ),
            "targetsObserved": ["reference", "runtime"],
        }
        return {**core, "proofHash": canonical_sha256(core)}

    def compile(
        self,
        *,
        dealer: Seat,
        declarer: Seat,
        contract: Contract,
        match_score: tuple[int, int],
        rules: RuleBook,
        required_level: str | None = None,
        required_mode: str | None = None,
        required_actions: Iterable[tuple[Seat | str | int, str, str | None]] | None = None,
        required_actions_complete: bool = True,
        maximum_conflicts: int = 250_000,
        timeout_seconds: float = 15.0,
    ) -> EscalationPlanV1 | None:
        from ortools.sat.python import cp_model

        dealer = Seat.parse(dealer)
        declarer = Seat.parse(declarer)
        contract = Contract.parse(contract)
        score = tuple(int(item) for item in match_score)
        if len(score) != 2 or any(item < 0 for item in score):
            raise ValueError("match score must contain two non-negative values")
        level = str(required_level).strip().lower() if required_level else None
        if level not in {None, *_LEVELS}:
            raise ValueError("unsupported escalation level")
        mode = str(required_mode).strip().lower() if required_mode else None
        if mode not in {None, "open", "locked"}:
            raise ValueError("unsupported escalation mode")
        if mode is not None and level not in {"double", "four"}:
            raise ValueError("an escalation mode requires Double or Four")
        action_sequence = (
            tuple(
                (
                    Seat.parse(actor),
                    str(action).strip().lower(),
                    None if requested_mode is None else str(requested_mode).strip().lower(),
                )
                for actor, action, requested_mode in required_actions
            )
            if required_actions is not None
            else None
        )

        domain = EscalationDomainIRV1.from_rules(rules.data)

        plans = self._legal_escalations(
            dealer=dealer,
            declarer=declarer,
            contract=contract,
            match_score=(score[0], score[1]),
            rules=rules,
        )
        model = cp_model.CpModel()
        selected = [
            model.new_bool_var(f"escalation_{index}")
            for index in range(len(plans))
        ]
        model.add_exactly_one(selected)
        eligible: list[int] = []
        for index, plan in enumerate(plans):
            plan_mode = "locked" if plan.locked else "open"
            matches = (
                (level is None or plan.level == level)
                and (mode is None or plan_mode == mode)
                and (
                    action_sequence is None
                    or (
                        tuple(
                            (item.actor, item.action, item.requested_mode)
                            for item in plan.actions
                        )
                        == action_sequence
                        if required_actions_complete
                        else tuple(
                            (item.actor, item.action, item.requested_mode)
                            for item in plan.actions[: len(action_sequence)]
                        )
                        == action_sequence
                    )
                )
            )
            if matches:
                eligible.append(index)
            else:
                model.add(selected[index] == 0)

        # Prefer the shortest legal history, then a content-derived stable tie-break.
        # This makes an unrestricted recipe select the normal, non-escalated state.
        fingerprint_costs = [int(item.fingerprint[:10], 16) for item in plans]
        model.minimize(
            sum(
                (len(plan.actions) * (1 << 42) + fingerprint_costs[index])
                * selected[index]
                for index, plan in enumerate(plans)
            )
        )
        model_core = {
            "schemaVersion": 1,
            "rulesHash": rules.sha256,
            "escalationDomainIrHash": domain.sha256,
            "dealer": dealer.name.lower(),
            "declarer": declarer.name.lower(),
            "contract": contract.value,
            "matchScore": list(score),
            "requiredLevel": level,
            "requiredMode": mode,
            "requiredActions": (
                [
                    [actor.name.lower(), action, requested_mode]
                    for actor, action, requested_mode in action_sequence
                ]
                if action_sequence is not None
                else None
            ),
            "requiredActionsComplete": bool(required_actions_complete),
            "legalPlanFingerprints": [item.fingerprint for item in plans],
            "eligiblePlanIndexes": eligible,
        }
        model_hash = canonical_sha256(model_core)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_number_of_conflicts = int(maximum_conflicts)
        solver.parameters.max_time_in_seconds = float(timeout_seconds)
        status = solver.solve(model)
        if status == cp_model.INFEASIBLE:
            return None
        if status != cp_model.OPTIMAL:
            raise RuntimeError(
                "CP-SAT escalation compiler did not prove an optimal result: "
                + solver.status_name(status)
            )
        chosen = [
            index
            for index, variable in enumerate(selected)
            if solver.boolean_value(variable)
        ]
        if len(chosen) != 1:
            raise AssertionError("escalation compiler did not select one plan")
        plan = plans[chosen[0]]
        self._replay(
            plan,
            dealer=dealer,
            declarer=declarer,
            contract=contract,
            match_score=(score[0], score[1]),
            rules=rules,
        )
        core = {
            "schemaVersion": 1,
            "actions": [item.to_dict() for item in plan.actions],
            "level": plan.level,
            "locked": plan.locked,
            "doubleSeat": (
                plan.double_seat.name.lower()
                if plan.double_seat is not None
                else None
            ),
            "modelHash": model_hash,
            "solverStatus": "optimal",
        }
        return EscalationPlanV1(
            plan.actions,
            plan.level,
            plan.locked,
            plan.double_seat,
            model_hash,
            "optimal",
            canonical_sha256(core),
        )
