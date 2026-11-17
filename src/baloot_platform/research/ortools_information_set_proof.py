"""Engine-free CP-SAT proof for finite actor information sets.

Callers compile legal worlds and continuation trees from the shared domain IR.
This module owns only pure-strategy qualification: one target action is bound
to each observation across every indistinguishable world, and deterministic
CP-SAT master/adversary solves certify the exact maximin policy.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from threading import Condition, Lock, Thread
from typing import Any, Callable, Iterator, Mapping, Sequence

from baloot_platform.core.exact_state_view import (
    canonical_observer_information_hash_v1,
)

from .scenario_contracts import canonical_sha256


_ACTORS = frozenset({"target", "opponent"})
_CLAIM = "winning_root_action_count"
_CANCELLATION_POLL_SECONDS = 0.005
_CANCELLATION_CONDITION = Condition()
_CANCELLATION_ENTRIES: dict[object, dict[str, Any]] = {}
_CANCELLATION_THREAD: Thread | None = None


def _canonical_policy_objectives_v1(
    keys: Sequence[tuple[str, str]],
    domains: Mapping[tuple[str, str], tuple[str, ...]],
    choices: Mapping[tuple[tuple[str, str], str], Any],
) -> Iterator[Any]:
    ordinal, width = 0, 1
    for key in keys:
        size = len(domains[key])
        if size == 1:
            continue
        # Mixed-radix blocks preserve lexicographic order within safe int64 bounds.
        if width * size > 2**31 - 1:
            yield ordinal
            ordinal, width = 0, 1
        ordinal = ordinal * size + sum(
            index * choices[(key, action)] for index, action in enumerate(domains[key]))
        width *= size
    if width > 1:
        yield ordinal


def native_bid_branch_specs_v1(rules, dealer):
    """Finite rule-owned partition; native legality rejects empty cells."""
    from baloot_platform.core.enums import Contract, Seat, Suit
    domain = rules.play_domain
    if domain.first_leader == "dealer_right":
        leaders = (rules.structural_domain.dealer_right(dealer),)
    elif domain.first_leader == "configured":
        leaders = (domain.configured_first_leader,)
    elif domain.first_leader == "buyer":
        leaders = tuple(Seat)
    else:
        raise ValueError("first-leader rule lacks native round binding")
    return (None, *((contract, trump, leader, locked)
        for contract in Contract
        for trump in (tuple(Suit) if contract.effective == Contract.HOKUM else (None,))
        for leader in leaders for locked in (False, True)))


def build_native_threshold_refutation_model_v1(upper_bound: int, threshold: int):
    """Compile a threshold comparison from certified native bound premises."""
    from ortools.sat.python import cp_model
    if type(upper_bound) is not int or type(threshold) is not int:
        raise ValueError("native threshold refutation requires integer bounds")
    model = cp_model.CpModel()
    upper = model.new_int_var(upper_bound, upper_bound, "certified_world_upper")
    required = model.new_int_var(threshold, threshold, "required_threshold")
    refuted = model.new_bool_var("threshold_refuted")
    model.add(upper < required).only_enforce_if(refuted)
    model.add(upper >= required).only_enforce_if(refuted.Not())
    return model, refuted


def build_native_bid_dominance_model_v1(value: int, alternatives: Sequence[Mapping[str, Any]]):
    """Compile certified native bounds; callers must verify their premises."""
    from ortools.sat.python import cp_model
    if type(value) is not int or any(type(row.get("upperBound")) is not int for row in alternatives):
        raise ValueError("native bid dominance requires integer certified bounds")
    model = cp_model.CpModel()
    selected = model.new_int_var(value, value, "selected_native_value")
    comparisons, bindings, uppers = [], [], []
    for index, row in enumerate(alternatives):
        bound = row["upperBound"]
        upper = model.new_int_var(bound, bound, f"alternative_upper_{index}")
        dominates = model.new_bool_var(f"dominates_{index}")
        model.add(selected >= upper).only_enforce_if(dominates)
        model.add(selected < upper).only_enforce_if(dominates.Not())
        comparisons.append(dominates)
        uppers.append(upper)
        bindings.append({"action": row["action"], "upperVariable": upper.name,
            "dominanceVariable": dominates.name, "sourceProofHash": canonical_sha256(row)})
    claim = model.new_bool_var("selected_action_optimal")
    if comparisons:
        model.add_min_equality(claim, comparisons)
    else:
        model.add(claim == 1)  # The only legal action is optimal, not necessarily winning.
    values = [value, *(row["upperBound"] for row in alternatives)]
    lower, upper = min(values), max(values)
    best = model.new_int_var(lower, upper, "certified_best_value")
    model.add_max_equality(best, [selected, *uppers])
    regret = model.new_int_var(0, upper - lower, "certified_action_regret")
    model.add(regret == best - selected)
    return model, selected, claim, best, regret, bindings


def _watch_cp_sat_cancellations_v1() -> None:
    while True:
        with _CANCELLATION_CONDITION:
            while not _CANCELLATION_ENTRIES:
                _CANCELLATION_CONDITION.wait()
            entries = tuple(_CANCELLATION_ENTRIES.values())
        for entry in entries:
            with entry["lock"]:
                if entry["done"]:
                    continue
                try:
                    cancelled = entry["cancel_check"]()
                except BaseException as exc:  # pragma: no cover - defensive thread handoff
                    entry["error"] = exc
                    cancelled = True
                if cancelled:
                    entry["solver"].stop_search()
        time.sleep(_CANCELLATION_POLL_SECONDS)


def _register_cp_sat_cancellation_v1(solver: Any, cancel_check: Callable[[], bool]) -> tuple[object, dict[str, Any]]:
    global _CANCELLATION_THREAD
    token = object()
    entry = {"solver": solver, "cancel_check": cancel_check, "lock": Lock(), "done": False, "error": None}
    with _CANCELLATION_CONDITION:
        _CANCELLATION_ENTRIES[token] = entry
        if _CANCELLATION_THREAD is None or not _CANCELLATION_THREAD.is_alive():
            _CANCELLATION_THREAD = Thread(
                target=_watch_cp_sat_cancellations_v1,
                name="actor-info-cancel-watch",
                daemon=True,
            )
            _CANCELLATION_THREAD.start()
        _CANCELLATION_CONDITION.notify()
    return token, entry


def _solve_cp_sat_with_cancel_v1(
    solver: Any,
    model: Any,
    cancel_check: Callable[[], bool] | None,
    solution_callback: Any | None = None,
) -> int:
    """Stop a blocking CP-SAT solve promptly when the governed run is cancelled."""

    def solve() -> int:
        return (
            solver.solve(model, solution_callback)
            if solution_callback is not None
            else solver.solve(model)
        )

    if cancel_check is None:
        return solve()
    if cancel_check():
        from ortools.sat.python import cp_model

        return cp_model.UNKNOWN
    token, entry = _register_cp_sat_cancellation_v1(solver, cancel_check)
    try:
        status = solve()
    finally:
        with entry["lock"]:
            entry["done"] = True
            callback_error = entry["error"]
        with _CANCELLATION_CONDITION:
            _CANCELLATION_ENTRIES.pop(token, None)
    if callback_error is not None:
        raise callback_error
    return status


class _FiniteProofBudgetExceeded(RuntimeError):
    def __init__(self, reason: str, *, compiled_nodes: int = 0) -> None:
        super().__init__(reason)
        self.compiled_nodes = compiled_nodes


class _InformationSetModelError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class FiniteInformationNodeV1:
    actor: str | None
    observation_id: str | None
    children: tuple[tuple[str, "FiniteInformationNodeV1"], ...]
    utility: int | None
    forced_prefix: tuple[tuple[str, str, str], ...] = ()

    def with_forced_prefix(self, steps: Sequence[tuple[str, str, str]]) -> "FiniteInformationNodeV1":
        prefix = tuple(steps) + self.forced_prefix
        if any(not isinstance(step, tuple) or len(step) != 3
               or step[0] not in _ACTORS
               or any(type(value) is not str or not value.strip() for value in step)
               for step in prefix):
            raise ValueError("forced information prefix requires actor, observation and action")
        return replace(self, forced_prefix=prefix)

    @classmethod
    def terminal(cls, utility: int) -> "FiniteInformationNodeV1":
        if type(utility) is not int:
            raise TypeError("terminal utility must be an integer")
        return cls(None, None, (), utility)

    @classmethod
    def decision(
        cls,
        actor: str,
        *,
        complete_state: Mapping[str, Any],
        acting_seat: str,
        children: Mapping[str, "FiniteInformationNodeV1"],
    ) -> "FiniteInformationNodeV1":
        relation = str(actor).strip()
        if relation not in _ACTORS:
            raise ValueError("information-set actor must be target or opponent")
        observation = canonical_observer_information_hash_v1(
            complete_state=complete_state,
            acting_seat=acting_seat,
        )
        normalized = tuple(sorted((str(action), child) for action, child in children.items()))
        if not normalized or any(not action for action, _child in normalized):
            raise ValueError("information-set decision requires named actions")
        if len({action for action, _child in normalized}) != len(normalized):
            raise ValueError("information-set decision actions must be unique")
        if any(not isinstance(child, cls) for _action, child in normalized):
            raise TypeError("information-set children must be finite nodes")
        return cls(relation, observation, normalized, None)

    @classmethod
    def decision_observed(
        cls,
        actor: str,
        *,
        observation_id: str,
        children: Mapping[str, "FiniteInformationNodeV1"],
    ) -> "FiniteInformationNodeV1":
        """Build a decision from a compiler-certified non-play observation."""

        relation = str(actor).strip()
        observation = str(observation_id).strip()
        if relation not in _ACTORS:
            raise ValueError("information-set actor must be target or opponent")
        if not observation:
            raise ValueError("information-set observation identity is required")
        normalized = tuple(
            sorted((str(action), child) for action, child in children.items())
        )
        if not normalized or any(not action for action, _child in normalized):
            raise ValueError("information-set decision requires named actions")
        if len({action for action, _child in normalized}) != len(normalized):
            raise ValueError("information-set decision actions must be unique")
        if any(not isinstance(child, cls) for _action, child in normalized):
            raise TypeError("information-set children must be finite nodes")
        return cls(relation, observation, normalized, None)

    def to_dict(self) -> dict[str, Any]:
        if self.utility is not None:
            document = {"utility": self.utility}
        else:
            document = {
            "actor": self.actor,
            "observationId": self.observation_id,
            "children": [
                {"action": action, "node": child.to_dict()}
                for action, child in self.children
            ],
            }
        if self.forced_prefix:
            document["forcedPrefix"] = [
                {"actor": actor, "observationId": observation, "action": action}
                for actor, observation, action in self.forced_prefix]
        return document


@dataclass(frozen=True, slots=True)
class FiniteInformationWorldV1:
    world_id: str
    roots: tuple[tuple[str, FiniteInformationNodeV1], ...]

    @classmethod
    def build(
        cls,
        world_id: str,
        roots: Mapping[str, FiniteInformationNodeV1],
    ) -> "FiniteInformationWorldV1":
        identity = str(world_id).strip()
        normalized = tuple(sorted((str(action), node) for action, node in roots.items()))
        if not identity:
            raise ValueError("information-set world id is required")
        if not normalized or any(not action for action, _node in normalized):
            raise ValueError("information-set world requires visible root actions")
        if any(not isinstance(node, FiniteInformationNodeV1) for _action, node in normalized):
            raise TypeError("information-set roots must be finite nodes")
        return cls(identity, normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worldId": self.world_id,
            "roots": [
                {"action": action, "node": node.to_dict()}
                for action, node in self.roots
            ],
        }


@dataclass(frozen=True, slots=True)
class FiniteInformationSetProblemV1:
    problem_id: str
    worlds: tuple[FiniteInformationWorldV1, ...]
    success_threshold: int

    @classmethod
    def build(
        cls,
        *,
        problem_id: str,
        worlds: Sequence[FiniteInformationWorldV1],
        success_threshold: int,
    ) -> "FiniteInformationSetProblemV1":
        identity = str(problem_id).strip()
        normalized = tuple(sorted(worlds, key=lambda world: world.world_id))
        if not identity:
            raise ValueError("information-set problem id is required")
        if not normalized:
            raise ValueError("information-set proof requires at least one world")
        if type(success_threshold) is not int:
            raise TypeError("information-set success threshold must be an integer")
        if len({world.world_id for world in normalized}) != len(normalized):
            raise ValueError("information-set world ids must be unique")
        root_domain = tuple(action for action, _node in normalized[0].roots)
        if any(tuple(action for action, _node in world.roots) != root_domain for world in normalized):
            raise ValueError("all hidden worlds must expose the same visible root actions")
        return cls(identity, normalized, success_threshold)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "problemId": self.problem_id,
            "successThreshold": self.success_threshold,
            "worlds": [world.to_dict() for world in self.worlds],
        }


@dataclass(frozen=True, slots=True)
class InformationSetProofBudgetV1:
    maximum_nodes: int = 100_000
    maximum_worlds: int = 4_096
    maximum_policies: int = 100_000
    maximum_iterations: int = 10_000
    maximum_wall_seconds: float = 10.0
    policy_search: str | None = None

    def __post_init__(self) -> None:
        if self.policy_search is not None and self.policy_search != "time_budgeted_symbolic_v1":
            raise ValueError("information-set policy search is unsupported")
        integers = (
            self.maximum_nodes,
            self.maximum_worlds,
            self.maximum_policies,
            self.maximum_iterations,
        )
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError("information-set proof budget bounds must be positive integers")
        if (type(self.maximum_wall_seconds) not in {int, float}
                or not math.isfinite(self.maximum_wall_seconds) or self.maximum_wall_seconds <= 0):
            raise ValueError("information-set proof wall budget must be positive and finite")

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximumNodes": self.maximum_nodes,
            "maximumWorlds": self.maximum_worlds,
            "maximumPolicies": self.maximum_policies,
            "maximumIterations": self.maximum_iterations,
            "maximumWallSeconds": float(self.maximum_wall_seconds),
            **({"policySearch": self.policy_search} if self.policy_search is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class InformationRootProofV1:
    root_action: str
    completion_status: str
    robust_value: int | None
    satisfies: bool | None
    target_policy: tuple[tuple[str, str], ...]
    target_policy_hash: str | None
    worst_world_id: str | None
    worst_opponent_policy_hash: str | None
    principal_variation: tuple[str, ...]
    model_hash: str | None
    solver_status: str
    target_policy_count: int
    opponent_policy_count: int
    iterations: int
    cut_hashes: tuple[str, ...]
    master_model_hash: str | None
    adversary_model_hash: str | None
    exhaustion_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rootAction": self.root_action,
            "completionStatus": self.completion_status,
            "robustValue": self.robust_value,
            "satisfies": self.satisfies,
            "targetPolicy": [
                {"observationId": observation, "action": action}
                for observation, action in self.target_policy
            ],
            "targetPolicyHash": self.target_policy_hash,
            "worstWorldId": self.worst_world_id,
            "worstOpponentPolicyHash": self.worst_opponent_policy_hash,
            "principalVariation": list(self.principal_variation),
            "modelHash": self.model_hash,
            "solverStatus": self.solver_status,
            "targetPolicyCount": self.target_policy_count,
            "opponentPolicyCount": self.opponent_policy_count,
            "iterations": self.iterations,
            "cutHashes": list(self.cut_hashes),
            "masterModelHash": self.master_model_hash,
            "adversaryModelHash": self.adversary_model_hash,
            "exhaustionReason": self.exhaustion_reason,
        }


@dataclass(frozen=True, slots=True)
class NativeRootValueCertificateV1:
    """A solved scoped value, without invented finite-graph policy metadata."""
    root_action: str
    completion_status: str
    robust_value: int | None
    satisfies: bool | None
    model_hash: str
    solver_status: str
    exhaustion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ActorInformationSetProofReceiptV1:
    problem_hash: str
    budget_hash: str
    completion_status: str
    roots: tuple[InformationRootProofV1, ...]
    winning_root_action_count: int | None
    winning_root_action_lower_bound: int
    winning_root_action_upper_bound: int
    compiled_nodes: int
    elapsed_wall_seconds: float
    qualification_authority: str = "cp_sat"
    compiler_oracle: str = "counterexample_guided_master_adversary_v1"
    model_status: str = "valid"
    quarantine_reason: str | None = None
    qualification_model_hash: str | None = None
    qualification_model_proto: str | None = None
    qualification_variable_bindings: tuple[tuple[str, str, str], ...] = ()
    qualification_variable_count: int = 0
    qualification_constraint_count: int = 0
    qualification_claim_satisfied: bool | None = None
    qualification_claim_binding: tuple[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        semantic_core = {
            "schemaVersion": 1,
            "problemHash": self.problem_hash,
            "budgetHash": self.budget_hash,
            "completionStatus": self.completion_status,
            "roots": [root.to_dict() for root in self.roots],
            "winningRootActionCount": self.winning_root_action_count,
            "winningRootActionLowerBound": self.winning_root_action_lower_bound,
            "winningRootActionUpperBound": self.winning_root_action_upper_bound,
            "compiledNodes": self.compiled_nodes,
            "qualificationAuthority": self.qualification_authority,
            "compilerOracle": self.compiler_oracle,
            "modelStatus": self.model_status,
            "quarantineReason": self.quarantine_reason,
            "qualificationModelHash": self.qualification_model_hash,
            "qualificationVariableBindings": [
                {
                    "rootAction": action,
                    "valueVariable": value_variable,
                    "winningVariable": winning_variable,
                }
                for action, value_variable, winning_variable
                in self.qualification_variable_bindings
            ],
            "qualificationVariableCount": self.qualification_variable_count,
            "qualificationConstraintCount": self.qualification_constraint_count,
            "qualificationClaimSatisfied": self.qualification_claim_satisfied,
            "qualificationClaimBinding": (
                {
                    "variable": self.qualification_claim_binding[0],
                    "literalIndex": self.qualification_claim_binding[1],
                }
                if self.qualification_claim_binding is not None
                else None
            ),
        }
        core = {
            **semantic_core,
            "semanticSha256": canonical_sha256(semantic_core),
            "elapsedWallSeconds": self.elapsed_wall_seconds,
        }
        return {**core, "sha256": canonical_sha256(core)}


@dataclass(slots=True)
class _Catalog:
    domains: dict[tuple[str, str], tuple[str, ...]]
    node_count: int
    utilities: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _CounterexampleCut:
    utility: int
    target_actions: tuple[tuple[str, str], ...]

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                "utility": self.utility,
                "targetActions": [list(item) for item in self.target_actions],
            }
        )


@dataclass(frozen=True, slots=True)
class _MasterResult:
    value: int
    policy: tuple[tuple[str, str], ...]
    model_hash: str
    status: str


@dataclass(frozen=True, slots=True)
class _AdversaryResult:
    value: int
    world_id: str
    opponent_policy_hash: str
    path: tuple[str, ...]
    target_actions: tuple[tuple[str, str], ...]
    model_hash: str
    status: str


@dataclass(frozen=True, slots=True)
class _QualificationResult:
    status: str
    roots: tuple[InformationRootProofV1 | NativeRootValueCertificateV1, ...]
    winning_count: int | None
    model_hash: str
    model_proto: str
    bindings: tuple[tuple[str, str, str], ...]
    variable_count: int
    constraint_count: int
    claim_satisfied: bool
    claim_variable: str
    claim_literal_index: int
    model_core: Mapping[str, Any] | None = None


class OrToolsActorInformationSetProofV1:
    """Exact pure-strategy maximin over a finite, caller-supplied world set."""

    @staticmethod
    def _opponent_stop_problem(problem, *, maximum_nodes, deadline, cancel_check=None):
        """Dualize policies, not target maximin values or the native Kabout target."""
        memo = {}
        roles = {"target": "opponent", "opponent": "target"}

        def visit(node):
            if OrToolsActorInformationSetProofV1._stopped(deadline, cancel_check):
                raise _FiniteProofBudgetExceeded("cancelled_or_wall_time")
            if id(node) in memo:
                return memo[id(node)]
            if len(memo) >= maximum_nodes:
                raise _FiniteProofBudgetExceeded("maximum_nodes")
            if node.utility is not None and (type(node.utility) is not int or node.utility not in {-1, 1}):
                raise ValueError("opponent stopping requires signed native Kabout utilities")
            result = replace(node,
                actor=roles[node.actor] if node.actor is not None else None,
                utility=-node.utility if node.utility is not None else None,
                children=tuple((action, visit(child)) for action, child in node.children),
                forced_prefix=tuple((roles[actor], observation, action)
                    for actor, observation, action in node.forced_prefix))
            memo[id(node)] = result
            return result

        return FiniteInformationSetProblemV1.build(problem_id=f"{problem.problem_id}:opponent-stop",
            success_threshold=1, worlds=tuple(FiniteInformationWorldV1.build(world.world_id,
                {action: visit(node) for action, node in world.roots}) for world in problem.worlds))

    def prove(
        self,
        problem: FiniteInformationSetProblemV1,
        *,
        budget: InformationSetProofBudgetV1 | None = None,
        claim: str = _CLAIM,
        selected_action: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
        runtime_maximum_seconds: float | None = None,
    ) -> ActorInformationSetProofReceiptV1:
        if claim == "winning_continuation_line_count":
            raise ValueError("actor-information-set continuation counts are unsupported")
        if claim not in {
            "outcome_reachable",
            "outcome_forced",
            "selected_action_is_unique_winner",
            "target_team_has_no_salvage",
            "winning_root_action_count",
        }:
            raise ValueError("actor-information-set proof claim is unsupported")
        limits = budget or InformationSetProofBudgetV1()
        runtime_seconds = limits.maximum_wall_seconds if runtime_maximum_seconds is None else runtime_maximum_seconds
        if (type(runtime_seconds) not in {int, float} or not math.isfinite(runtime_seconds)
                or not 0 < runtime_seconds <= limits.maximum_wall_seconds):
            raise ValueError("runtime wall allowance must be positive and within the declared budget")
        started = time.monotonic()
        deadline = started + runtime_seconds
        root_actions = tuple(action for action, _node in problem.worlds[0].roots)
        if len(problem.worlds) > limits.maximum_worlds:
            roots = tuple(
                self._incomplete(action, "maximum_worlds") for action in root_actions
            )
            return ActorInformationSetProofReceiptV1(
                problem.sha256,
                limits.sha256,
                "proof_incomplete",
                roots,
                None,
                0,
                len(root_actions),
                0,
                time.monotonic() - started,
            )
        try:
            catalog = self._catalog(problem, limits.maximum_nodes, deadline=deadline, cancel_check=cancel_check)
        except _InformationSetModelError as exc:
            roots = tuple(
                self._model_error(action, exc.code) for action in root_actions
            )
            return ActorInformationSetProofReceiptV1(
                problem.sha256,
                limits.sha256,
                "model_error",
                roots,
                None,
                0,
                len(root_actions),
                0,
                time.monotonic() - started,
                model_status="quarantined",
                quarantine_reason=exc.code,
            )
        except _FiniteProofBudgetExceeded as exc:
            roots = tuple(
                self._incomplete(action, str(exc)) for action in root_actions
            )
            return ActorInformationSetProofReceiptV1(
                problem.sha256,
                limits.sha256,
                "proof_incomplete",
                roots,
                None,
                0,
                len(root_actions),
                exc.compiled_nodes,
                time.monotonic() - started,
            )
        roots: list[InformationRootProofV1] = []
        for root_action in root_actions:
            if self._stopped(deadline, cancel_check):
                roots.append(self._incomplete(root_action, "cancelled_or_wall_time"))
                continue
            roots.append(
                self._prove_root(
                    problem,
                    root_action,
                    limits,
                    deadline,
                    cancel_check,
                )
            )
        completed = tuple(root for root in roots if root.completion_status == "exact")
        qualification: _QualificationResult | None = None
        if completed:
            qualification = self._certify_root_thresholds(
                completed,
                success_threshold=problem.success_threshold,
                problem_hash=problem.sha256,
                claim=claim,
                selected_action=selected_action,
                deadline=deadline,
                cancel_check=cancel_check,
            )
            certified = {
                root.root_action: root for root in qualification.roots
            }
            roots = [certified.get(root.root_action, root) for root in roots]
        certified_completed = [
            root
            for root in roots
            if root.completion_status == "exact" and root.satisfies is not None
        ]
        lower = (
            int(qualification.winning_count)
            if qualification is not None
            and qualification.winning_count is not None
            else 0
        )
        incomplete = len(roots) - len(certified_completed)
        exact = incomplete == 0
        return ActorInformationSetProofReceiptV1(
            problem.sha256,
            limits.sha256,
            "exact_complete" if exact else "proof_incomplete",
            tuple(roots),
            lower if exact else None,
            lower,
            lower + incomplete,
            catalog.node_count,
            time.monotonic() - started,
            qualification_model_hash=(qualification.model_hash if qualification else None),
            qualification_model_proto=(qualification.model_proto if qualification else None),
            qualification_variable_bindings=(qualification.bindings if qualification else ()),
            qualification_variable_count=(qualification.variable_count if qualification else 0),
            qualification_constraint_count=(qualification.constraint_count if qualification else 0),
            qualification_claim_satisfied=(
                qualification.claim_satisfied if qualification else None
            ),
            qualification_claim_binding=(
                (qualification.claim_variable, qualification.claim_literal_index)
                if qualification
                else None
            ),
        )

    @staticmethod
    def _stopped(deadline: float, cancel_check: Callable[[], bool] | None) -> bool:
        return time.monotonic() >= deadline or (
            cancel_check is not None and bool(cancel_check())
        )

    def _certify_root_thresholds(
        self,
        roots: Sequence[InformationRootProofV1 | NativeRootValueCertificateV1],
        *,
        success_threshold: int,
        problem_hash: str,
        claim: str,
        selected_action: str | None,
        deadline: float,
        cancel_check: Callable[[], bool] | None,
        model: Any | None = None,
        variable_prefix: str = "",
    ) -> _QualificationResult:
        """Bind root wins and their count to CP-SAT, never a Python comparison."""

        from ortools.sat.python import cp_model

        if not roots or any(root.completion_status != "exact" for root in roots):
            raise ValueError("root qualification requires completed exact root proofs")
        values = tuple(
            int(root.robust_value)
            for root in roots
            if root.robust_value is not None
        )
        if len(values) != len(roots):
            raise ValueError("exact information-set roots lack robust values")
        lower = min(*values, success_threshold - 1)
        upper = max(*values, success_threshold)
        model = cp_model.CpModel() if model is None else model
        bindings: list[tuple[str, str, str]] = []
        winning_variables: list[Any] = []
        value_variables: list[Any] = []
        for index, (root, value) in enumerate(zip(roots, values, strict=True)):
            value_name = f"{variable_prefix}root_value_{index}"
            winning_name = f"{variable_prefix}root_wins_{index}"
            value_variable = model.new_int_var(lower, upper, value_name)
            winning_variable = model.new_bool_var(winning_name)
            model.add(value_variable == value)
            model.add(value_variable >= success_threshold).only_enforce_if(
                winning_variable
            )
            model.add(value_variable < success_threshold).only_enforce_if(
                winning_variable.Not()
            )
            value_variables.append(value_variable)
            winning_variables.append(winning_variable)
            bindings.append((root.root_action, value_name, winning_name))
        count = model.new_int_var(0, len(roots), f"{variable_prefix}winning_root_action_count")
        model.add(count == sum(winning_variables))
        claim_literal = model.new_bool_var(f"{variable_prefix}proof_claim_satisfied")
        if claim == "outcome_reachable":
            model.add(count >= 1).only_enforce_if(claim_literal)
            model.add(count == 0).only_enforce_if(claim_literal.Not())
        elif claim == "outcome_forced":
            model.add(count == len(roots)).only_enforce_if(claim_literal)
            model.add(count != len(roots)).only_enforce_if(claim_literal.Not())
        elif claim == "selected_action_is_unique_winner":
            unique = model.new_bool_var(f"{variable_prefix}proof_unique_winner")
            model.add(count == 1).only_enforce_if(unique)
            model.add(count != 1).only_enforce_if(unique.Not())
            if selected_action is None:
                model.add(claim_literal == unique)
            else:
                try:
                    selected_index = next(
                        index
                        for index, root in enumerate(roots)
                        if root.root_action == selected_action
                    )
                except StopIteration as exc:
                    raise ValueError(
                        "selected actor-information root action is unavailable"
                    ) from exc
                model.add_min_equality(
                    claim_literal, [unique, winning_variables[selected_index]]
                )
        elif claim == "target_team_has_no_salvage":
            model.add(count == 0).only_enforce_if(claim_literal)
            model.add(count != 0).only_enforce_if(claim_literal.Not())
        elif claim == "winning_root_action_count":
            model.add(claim_literal == 1)
        else:  # pragma: no cover - guarded by prove()
            raise AssertionError("unsupported actor-information claim")
        model_core = {
            "schemaVersion": 1,
            "kind": "actor_information_set_root_qualification_v1",
            "problemHash": problem_hash,
            "successThreshold": success_threshold,
            "roots": [
                {
                    "action": root.root_action,
                    "robustValue": value,
                    "sourceModelHash": root.model_hash,
                    "valueVariable": bindings[index][1],
                    "winningVariable": bindings[index][2],
                }
                for index, (root, value) in enumerate(zip(roots, values, strict=True))
            ],
            "countVariable": f"{variable_prefix}winning_root_action_count",
            "claim": claim,
            "selectedAction": selected_action,
            "claimVariable": claim_literal.name,
            "claimLiteralIndex": int(claim_literal.index),
        }
        model_proto = str(model.proto)
        model_hash = canonical_sha256({"core": model_core, "modelProto": model_proto})
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        statuses: list[int] = []
        if not self._solve_optimal(
            solver,
            model,
            deadline,
            statuses,
            cancel_check,
        ):
            status = solver.status_name(statuses[-1]).lower()
            incomplete = tuple(
                replace(
                    root,
                    completion_status="incomplete",
                    satisfies=None,
                    solver_status=status,
                    exhaustion_reason="qualification_not_optimal",
                )
                for root in roots
            )
            return _QualificationResult(
                status,
                incomplete,
                None,
                model_hash,
                model_proto,
                tuple(bindings),
                len(model.proto.variables),
                len(model.proto.constraints),
                False,
                claim_literal.name,
                int(claim_literal.index),
                model_core,
            )
        certified = tuple(
            replace(
                root,
                satisfies=bool(solver.value(winning_variables[index])),
                model_hash=canonical_sha256(
                    {
                        "cegisModelHash": root.model_hash,
                        "qualificationModelHash": model_hash,
                        "rootAction": root.root_action,
                    }
                ),
            )
            for index, root in enumerate(roots)
        )
        return _QualificationResult(
            "optimal",
            certified,
            int(solver.value(count)),
            model_hash,
            model_proto,
            tuple(bindings),
            len(model.proto.variables),
            len(model.proto.constraints),
            bool(solver.boolean_value(claim_literal)),
            claim_literal.name,
            int(claim_literal.index),
            model_core,
        )

    @staticmethod
    def _model_error(root_action: str, reason: str) -> InformationRootProofV1:
        return InformationRootProofV1(
            root_action,
            "model_error",
            None,
            None,
            (),
            None,
            None,
            None,
            (),
            None,
            "not_started",
            0,
            0,
            0,
            (),
            None,
            None,
            reason,
        )

    @staticmethod
    def _incomplete(
        root_action: str,
        reason: str,
        *,
        target_policy_count: int = 0,
        opponent_policy_count: int = 0,
        iterations: int = 0,
        cut_hashes: Sequence[str] = (),
        solver_status: str = "not_started",
        master_model_hash: str | None = None,
        adversary_model_hash: str | None = None,
    ) -> InformationRootProofV1:
        return InformationRootProofV1(
            root_action,
            "incomplete",
            None,
            None,
            (),
            None,
            None,
            None,
            (),
            master_model_hash,
            solver_status,
            target_policy_count,
            opponent_policy_count,
            iterations,
            tuple(cut_hashes),
            master_model_hash,
            adversary_model_hash,
            reason,
        )

    @staticmethod
    def _catalog(
        problem: FiniteInformationSetProblemV1,
        maximum_nodes: int,
        root_action: str | None = None,
        *, deadline: float = math.inf, cancel_check: Callable[[], bool] | None = None,
    ) -> _Catalog:
        domains: dict[tuple[str, str], tuple[str, ...]] = {}
        observed_actor: dict[str, str] = {}
        utilities: list[int] = []
        visited: set[int] = set()
        active: set[int] = set()

        def register(actor, observation, actions):
            previous_actor = observed_actor.setdefault(observation, actor)
            if previous_actor != actor:
                raise ValueError("one observation id cannot belong to both teams")
            key = (actor, observation)
            if domains.setdefault(key, actions) != actions:
                raise _InformationSetModelError("observation_action_domain_mismatch")

        def visit(node: FiniteInformationNodeV1) -> None:
            if OrToolsActorInformationSetProofV1._stopped(deadline, cancel_check):
                raise _FiniteProofBudgetExceeded("cancelled_or_wall_time", compiled_nodes=len(visited))
            identity = id(node)
            if identity in active:
                raise ValueError("information-set continuation graph contains a cycle")
            if identity in visited:
                return
            if len(visited) >= maximum_nodes:
                raise _FiniteProofBudgetExceeded("maximum_nodes", compiled_nodes=len(visited))
            visited.add(identity)
            active.add(identity)
            for step in node.forced_prefix:
                if OrToolsActorInformationSetProofV1._stopped(deadline, cancel_check):
                    raise _FiniteProofBudgetExceeded("cancelled_or_wall_time", compiled_nodes=len(visited))
                if (not isinstance(step, tuple) or len(step) != 3 or step[0] not in _ACTORS
                        or any(type(value) is not str or not value.strip() for value in step)):
                    raise ValueError("forced information prefix requires actor, observation and action")
                actor, observation, action = step
                register(actor, observation, (action,))
            if node.utility is not None:
                if node.actor is not None or node.observation_id is not None or node.children:
                    raise ValueError("terminal information-set node shape is invalid")
                utilities.append(node.utility)
            else:
                if node.actor not in _ACTORS or not node.observation_id or not node.children:
                    raise ValueError("decision information-set node shape is invalid")
                actions = tuple(action for action, _child in node.children)
                register(node.actor, node.observation_id, actions)
                for _action, child in node.children:
                    visit(child)
            active.remove(identity)

        for world in problem.worlds:
            for action, node in world.roots:
                if root_action is None or action == root_action:
                    visit(node)
        if not utilities:
            raise ValueError("information-set problem has no terminal utilities")
        return _Catalog(domains, len(visited), tuple(sorted(set(utilities))))

    def _prove_root(
        self,
        problem: FiniteInformationSetProblemV1,
        root_action: str,
        budget: InformationSetProofBudgetV1,
        deadline: float,
        cancel_check: Callable[[], bool] | None,
    ) -> InformationRootProofV1:
        try:
            catalog = self._catalog(problem, budget.maximum_nodes, root_action,
                                    deadline=deadline, cancel_check=cancel_check)
        except _FiniteProofBudgetExceeded as exc:
            return self._incomplete(root_action, str(exc))
        target_keys = tuple(sorted(key for key in catalog.domains if key[0] == "target"))
        opponent_keys = tuple(sorted(key for key in catalog.domains if key[0] == "opponent"))
        target_domains = tuple(catalog.domains[key] for key in target_keys)
        opponent_domains = tuple(catalog.domains[key] for key in opponent_keys)
        target_count = _product_size(target_domains)
        opponent_count = _product_size(opponent_domains)
        if budget.policy_search is None and target_count > budget.maximum_policies:
            return self._incomplete(
                root_action,
                "maximum_policies_target",
                target_policy_count=target_count,
                opponent_policy_count=opponent_count,
            )
        if budget.policy_search is None and opponent_count > budget.maximum_policies:
            return self._incomplete(
                root_action,
                "maximum_policies_opponent",
                target_policy_count=target_count,
                opponent_policy_count=opponent_count,
            )
        cuts: list[_CounterexampleCut] = []
        cut_hashes: set[str] = set()
        last_master: _MasterResult | None = None
        last_adversary: _AdversaryResult | None = None
        for iteration in range(1, budget.maximum_iterations + 1):
            if self._stopped(deadline, cancel_check):
                return self._incomplete(
                    root_action,
                    "cancelled_or_wall_time",
                    target_policy_count=target_count,
                    opponent_policy_count=opponent_count,
                    iterations=iteration - 1,
                    cut_hashes=tuple(cut.sha256 for cut in cuts),
                    master_model_hash=(last_master.model_hash if last_master else None),
                    adversary_model_hash=(
                        last_adversary.model_hash if last_adversary else None
                    ),
                )
            last_master = self._solve_master(
                problem.sha256,
                root_action,
                catalog,
                cuts,
                deadline,
                cancel_check,
            )
            if last_master.status != "optimal":
                return self._incomplete(
                    root_action,
                    "master_not_optimal",
                    target_policy_count=target_count,
                    opponent_policy_count=opponent_count,
                    iterations=iteration,
                    cut_hashes=tuple(cut.sha256 for cut in cuts),
                    solver_status=last_master.status,
                    master_model_hash=last_master.model_hash,
                )
            try:
                last_adversary = self._solve_adversary(
                    problem,
                    root_action,
                    catalog,
                    last_master.policy,
                    budget.maximum_nodes,
                    deadline,
                    cancel_check,
                )
            except _FiniteProofBudgetExceeded as exc:
                return self._incomplete(
                    root_action,
                    str(exc),
                    target_policy_count=target_count,
                    opponent_policy_count=opponent_count,
                    iterations=iteration,
                    cut_hashes=tuple(cut.sha256 for cut in cuts),
                    master_model_hash=last_master.model_hash,
                )
            if last_adversary.status != "optimal":
                return self._incomplete(
                    root_action,
                    "adversary_not_optimal",
                    target_policy_count=target_count,
                    opponent_policy_count=opponent_count,
                    iterations=iteration,
                    cut_hashes=tuple(cut.sha256 for cut in cuts),
                    solver_status=last_adversary.status,
                    master_model_hash=last_master.model_hash,
                    adversary_model_hash=last_adversary.model_hash,
                )
            if last_adversary.value > last_master.value:
                raise ValueError("counterexample-guided proof bounds are inconsistent")
            if last_adversary.value == last_master.value:
                combined_hash = canonical_sha256(
                    {
                        "masterModelHash": last_master.model_hash,
                        "adversaryModelHash": last_adversary.model_hash,
                        "cutHashes": [cut.sha256 for cut in cuts],
                    }
                )
                return InformationRootProofV1(
                    root_action,
                    "exact",
                    last_master.value,
                    None,
                    last_master.policy,
                    canonical_sha256(last_master.policy),
                    last_adversary.world_id,
                    last_adversary.opponent_policy_hash,
                    last_adversary.path,
                    combined_hash,
                    "optimal_converged",
                    target_count,
                    opponent_count,
                    iteration,
                    tuple(cut.sha256 for cut in cuts),
                    last_master.model_hash,
                    last_adversary.model_hash,
                    None,
                )
            cut = _CounterexampleCut(
                last_adversary.value,
                tuple(sorted(set(last_adversary.target_actions))),
            )
            if cut.sha256 in cut_hashes:
                raise ValueError("counterexample-guided proof produced a duplicate cut")
            cuts.append(cut)
            cut_hashes.add(cut.sha256)

        return self._incomplete(
            root_action,
            "maximum_iterations",
            target_policy_count=target_count,
            opponent_policy_count=opponent_count,
            iterations=budget.maximum_iterations,
            cut_hashes=tuple(cut.sha256 for cut in cuts),
            solver_status="optimal_not_converged",
            master_model_hash=last_master.model_hash if last_master else None,
            adversary_model_hash=(
                last_adversary.model_hash if last_adversary else None
            ),
        )

    def _solve_master(
        self,
        problem_hash: str,
        root_action: str,
        catalog: _Catalog,
        cuts: Sequence[_CounterexampleCut],
        deadline: float,
        cancel_check: Callable[[], bool] | None,
        *, fixed_actions: Mapping[str, str] | None = None,
        canonicalize_policy: bool = True,
        maximum_conflicts: int | None = None,
        preferred_actions: Mapping[str, str] | None = None,
        allowed_actions: Mapping[str, tuple[str, ...]] | None = None,
    ) -> _MasterResult:
        from ortools.sat.python import cp_model

        if type(canonicalize_policy) is not bool:
            raise TypeError("master policy canonicalization must be boolean")
        if maximum_conflicts is not None and (type(maximum_conflicts) is not int or maximum_conflicts < 1):
            raise ValueError("master conflict limit must be a positive integer")
        if preferred_actions is not None and (not isinstance(preferred_actions, Mapping)
                or any(type(key) is not str or type(action) is not str
                       or action not in catalog.domains.get(("target", key), ())
                       for key, action in preferred_actions.items())):
            raise ValueError("preferred policy action is absent from the target observation domain")
        preferred = tuple(sorted((preferred_actions or {}).items()))
        if allowed_actions is not None and (not isinstance(allowed_actions, Mapping)
                or any(type(key) is not str or type(actions) is not tuple or not actions
                       or any(type(action) is not str or action not in catalog.domains.get(("target", key), ())
                              for action in actions)
                       or len(set(actions)) != len(actions)
                       for key, actions in allowed_actions.items())):
            raise ValueError("allowed policy actions must be distinct legal target choices")
        allowed = tuple((key, tuple(sorted(actions))) for key, actions in sorted((allowed_actions or {}).items()))

        model = cp_model.CpModel()
        target_keys = tuple(sorted(key for key in catalog.domains if key[0] == "target"))
        selected = {
            (key, action): model.new_bool_var(f"target_{index}_{action_index}")
            for index, key in enumerate(target_keys)
            for action_index, action in enumerate(catalog.domains[key])
        }
        for key in target_keys:
            model.add_exactly_one(selected[(key, action)] for action in catalog.domains[key])
        for observation, actions in allowed:
            model.add(sum(selected[(("target", observation), action)] for action in actions) == 1)
        fixed = tuple(sorted((fixed_actions or {}).items()))
        for observation, action in fixed:
            binding = selected.get((("target", observation), action))
            if binding is None:
                raise ValueError("fixed policy action is absent from the target observation domain")
            model.add(binding == 1)
        lower, upper = catalog.utilities[0], catalog.utilities[-1]
        value = model.new_int_var(lower, upper, "robust_value")
        big_m = upper - lower
        ordered_cuts = tuple(sorted(cuts, key=lambda cut: (cut.utility, cut.target_actions)))
        bound_guards = set()
        for cut in ordered_cuts:
            # Ascending utility keeps the tightest bound for identical guards.
            # Retain every original cut in model_core and evidence below.
            if cut.target_actions in bound_guards:
                continue
            bound_guards.add(cut.target_actions)
            matches = [
                selected[(('target', observation), action)]
                for observation, action in cut.target_actions
            ]
            model.add(value <= cut.utility + big_m * sum(1 - match for match in matches))
        model_core = {
            "schemaVersion": 1,
            "kind": "actor_information_set_master_v1",
            "problemHash": problem_hash,
            "rootAction": root_action,
            "targetDomains": [
                {"observationId": key[1], "actions": list(catalog.domains[key])}
                for key in target_keys
            ],
            "cuts": [
                {
                    "utility": cut.utility,
                    "targetActions": [list(item) for item in cut.target_actions],
                    "sha256": cut.sha256,
                }
                for cut in ordered_cuts
            ],
            "phases": (["maximizeRobustValue"]
                       + (["minimizePolicyChanges"] if preferred else [])
                       + (["minimizeCanonicalActions"] if canonicalize_policy else [])),
            **({"fixedActions": [list(item) for item in fixed]} if fixed else {}),
            **({"preferredActions": [list(item) for item in preferred]} if preferred else {}),
            **({"allowedActions": [[key, list(actions)] for key, actions in allowed]} if allowed else {}),
        }
        model_hash = canonical_sha256(model_core)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        if maximum_conflicts is not None:
            solver.parameters.max_number_of_conflicts = maximum_conflicts
        statuses: list[int] = []
        model.maximize(value)
        if not self._solve_optimal(
            solver, model, deadline, statuses, cancel_check
        ):
            return _MasterResult(0, (), model_hash, solver.status_name(statuses[-1]).lower())
        optimum = int(solver.value(value))
        if preferred:
            # Preserve the proved search value first. A stable strategy avoids
            # needlessly rebuilding choices unrelated to the new counterexample.
            model.add(value == optimum)
            changes = sum(1 - selected[(("target", key), action)] for key, action in preferred)
            model.minimize(changes)
            if not self._solve_optimal(solver, model, deadline, statuses, cancel_check):
                return _MasterResult(optimum, (), model_hash, solver.status_name(statuses[-1]).lower())
            model.add(changes == int(solver.value(changes)))
        if not canonicalize_policy:
            policy = tuple(
                (
                    key[1],
                    next(
                        action
                        for action in catalog.domains[key]
                        if solver.value(selected[(key, action)])
                    ),
                )
                for key in target_keys
            )
            return _MasterResult(optimum, policy, model_hash, "optimal")
        model.add(value == optimum)
        for ordinal in _canonical_policy_objectives_v1(target_keys, catalog.domains, selected):
            model.minimize(ordinal)
            if not self._solve_optimal(
                solver, model, deadline, statuses, cancel_check
            ):
                return _MasterResult(
                    optimum, (), model_hash, solver.status_name(statuses[-1]).lower()
                )
            model.add(ordinal == int(solver.value(ordinal)))
        policy = tuple(
            (
                key[1],
                next(
                    action
                    for action in catalog.domains[key]
                    if solver.value(selected[(key, action)])
                ),
            )
            for key in target_keys
        )
        return _MasterResult(optimum, policy, model_hash, "optimal")

    def _solve_adversary(
        self,
        problem: FiniteInformationSetProblemV1,
        root_action: str,
        catalog: _Catalog,
        target_policy: Sequence[tuple[str, str]],
        maximum_nodes: int,
        deadline: float,
        cancel_check: Callable[[], bool] | None,
    ) -> _AdversaryResult:
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        target = {("target", observation): action for observation, action in target_policy}
        opponent_keys = tuple(sorted(key for key in catalog.domains if key[0] == "opponent"))
        opponent = {
            (key, action): model.new_bool_var(f"opponent_{index}_{action_index}")
            for index, key in enumerate(opponent_keys)
            for action_index, action in enumerate(catalog.domains[key])
        }
        for key in opponent_keys:
            model.add_exactly_one(opponent[(key, action)] for action in catalog.domains[key])
        worlds = tuple(sorted(problem.worlds, key=lambda world: world.world_id))
        world_active = [model.new_bool_var(f"world_{index}") for index in range(len(worlds))]
        model.add_exactly_one(world_active)
        terminals: list[tuple[Any, int]] = []
        compiled_nodes = 0
        activations: dict[int, tuple[Any, list[Any]]] = {}

        def lower(node: FiniteInformationNodeV1, incoming: Any) -> None:
            nonlocal compiled_nodes
            if (cancel_check is not None and cancel_check()) or time.monotonic() >= deadline:
                raise _FiniteProofBudgetExceeded("cancelled_or_wall_time", compiled_nodes=compiled_nodes)
            identity = id(node)
            if identity in activations:
                activations[identity][1].append(incoming)
                return
            compiled_nodes += 1
            if compiled_nodes > maximum_nodes:
                raise _FiniteProofBudgetExceeded("maximum_nodes")
            active = model.new_bool_var(f"node_active_{compiled_nodes}")
            activations[identity] = (active, [incoming])
            if node.utility is not None:
                terminals.append((active, node.utility))
                return
            assert node.actor is not None and node.observation_id is not None
            key = (node.actor, node.observation_id)
            if node.actor == "target":
                lower(dict(node.children)[target[key]], active)
                return
            for action, child in node.children:
                edge = model.new_bool_var(f"edge_{compiled_nodes}_{len(terminals)}_{action}")
                chosen = opponent[(key, action)]
                model.add(edge <= active)
                model.add(edge <= chosen)
                model.add(edge >= active + chosen - 1)
                lower(child, edge)

        for index, world in enumerate(worlds):
            lower(dict(world.roots)[root_action], world_active[index])
        for active, incoming in activations.values():
            model.add_max_equality(active, incoming)
        model.add(sum(active for active, _utility in terminals) == 1)
        lower_utility, upper_utility = catalog.utilities[0], catalog.utilities[-1]
        utility = model.new_int_var(lower_utility, upper_utility, "adversary_utility")
        model.add(utility == sum(active * value for active, value in terminals))
        model_core = {
            "schemaVersion": 1,
            "kind": "actor_information_set_adversary_v1",
            "graphLowering": "shared_node_flow_v1",
            "problemHash": problem.sha256,
            "rootAction": root_action,
            "targetPolicy": [list(item) for item in target_policy],
            "opponentDomains": [
                {"observationId": key[1], "actions": list(catalog.domains[key])}
                for key in opponent_keys
            ],
            "compiledNodes": compiled_nodes,
            "phases": [
                "minimizeUtility",
                "minimizeWorldId",
                "minimizeCanonicalOpponentActions",
            ],
        }
        model_hash = canonical_sha256(model_core)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        statuses: list[int] = []
        model.minimize(utility)
        if not self._solve_optimal(
            solver, model, deadline, statuses, cancel_check
        ):
            return _AdversaryResult(
                0, "", "", (), (), model_hash, solver.status_name(statuses[-1]).lower()
            )
        optimum = int(solver.value(utility))
        model.add(utility == optimum)
        world_ordinal = sum(index * active for index, active in enumerate(world_active))
        model.minimize(world_ordinal)
        if not self._solve_optimal(
            solver, model, deadline, statuses, cancel_check
        ):
            return _AdversaryResult(
                optimum, "", "", (), (), model_hash, solver.status_name(statuses[-1]).lower()
            )
        model.add(world_ordinal == int(solver.value(world_ordinal)))
        for ordinal in _canonical_policy_objectives_v1(opponent_keys, catalog.domains, opponent):
            model.minimize(ordinal)
            if not self._solve_optimal(
                solver, model, deadline, statuses, cancel_check
            ):
                return _AdversaryResult(
                    optimum, "", "", (), (), model_hash, solver.status_name(statuses[-1]).lower()
                )
            model.add(ordinal == int(solver.value(ordinal)))
        world = worlds[next(index for index, active in enumerate(world_active) if solver.value(active))]
        opponent_policy = tuple(
            (
                key[1],
                next(
                    action
                    for action in catalog.domains[key]
                    if solver.value(opponent[(key, action)])
                ),
            )
            for key in opponent_keys
        )
        played_value, suffix, target_actions = self._play(
            dict(world.roots)[root_action],
            target,
            {("opponent", observation): action for observation, action in opponent_policy},
        )
        if played_value != optimum:
            raise ValueError("adversary witness does not match its CP-SAT utility")
        return _AdversaryResult(
            optimum,
            world.world_id,
            canonical_sha256(opponent_policy),
            (root_action, *suffix),
            target_actions,
            model_hash,
            "optimal",
        )

    @staticmethod
    def _solve_optimal(
        solver: Any,
        model: Any,
        deadline: float,
        statuses: list[int],
        cancel_check: Callable[[], bool] | None = None,
    ) -> bool:
        from ortools.sat.python import cp_model

        remaining = deadline - time.monotonic()
        if remaining <= 0 or (cancel_check is not None and cancel_check()):
            statuses.append(cp_model.UNKNOWN)
            return False
        solver.parameters.max_time_in_seconds = max(0.001, remaining)
        status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
        statuses.append(status)
        return status == cp_model.OPTIMAL

    @staticmethod
    def _play(
        node: FiniteInformationNodeV1,
        target_policy: Mapping[tuple[str, str], str],
        opponent_policy: Mapping[tuple[str, str], str],
    ) -> tuple[int, tuple[str, ...], tuple[tuple[str, str], ...]]:
        path: list[str] = []
        target_actions: list[tuple[str, str]] = []
        current = node
        while True:
            for actor, observation, action in current.forced_prefix:
                policy = target_policy if actor == "target" else opponent_policy
                if policy.get((actor, observation)) != action:
                    raise ValueError("forced information action disagrees with policy")
                path.append(action)
                if actor == "target":
                    target_actions.append((observation, action))
            if current.utility is not None:
                return current.utility, tuple(path), tuple(target_actions)
            assert current.actor is not None and current.observation_id is not None
            key = (current.actor, current.observation_id)
            policy = target_policy if current.actor == "target" else opponent_policy
            action = policy[key]
            if current.actor == "target":
                target_actions.append((current.observation_id, action))
            current = dict(current.children)[action]
            path.append(action)


def _product_size(domains: Sequence[Sequence[str]]) -> int:
    result = 1
    for domain in domains:
        result *= len(domain)
    return result


__all__ = [
    "ActorInformationSetProofReceiptV1",
    "FiniteInformationNodeV1",
    "FiniteInformationSetProblemV1",
    "FiniteInformationWorldV1",
    "InformationRootProofV1",
    "InformationSetProofBudgetV1",
    "OrToolsActorInformationSetProofV1",
    "_solve_cp_sat_with_cancel_v1",
]
