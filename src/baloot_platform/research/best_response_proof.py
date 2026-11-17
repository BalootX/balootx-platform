"""Bounded late-game partner/opponent best-response policy proofs.

The game-tree subproblem is zero-sum relative to the root actor's team:
partners maximize the governed margin and opponents minimize it.  Every CP-SAT
value is checked against an independent recursive backward-induction reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from baloot_platform.analysis.models import InformationState
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat, Team
from baloot_platform.core.models import RoundState
from baloot_platform.plugins.base import PluginDescriptor

from .contracts import canonical_sha256
from .policy_proof import (
    PROOF_SCHEMA_VERSION,
    OrToolsReducedPolicyProofV1,
    ReducedActorSafePolicyProblemV1,
    ReducedPolicyProfileV1,
    ReducedPolicyProofV1,
    ReducedWorldOutcomeV1,
    exhaustive_reference_policy_proof,
)


BEST_RESPONSE_ENGINE_ID = "bounded_best_response_policy_proof"
BEST_RESPONSE_ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class ReducedGameNodeV1:
    """A finite continuation node; internal actors are team-relative."""

    actor_relation: str | None = None
    information_state_hash: str | None = None
    children: tuple[tuple[int, "ReducedGameNodeV1"], ...] = ()
    terminal_margin: int | None = None

    @classmethod
    def terminal(cls, margin: int) -> "ReducedGameNodeV1":
        return cls(terminal_margin=int(margin))

    @classmethod
    def decision(
        cls,
        actor_relation: str,
        children: dict[Card | int | str, "ReducedGameNodeV1"],
        *,
        information_state_hash: str,
    ) -> "ReducedGameNodeV1":
        relation = str(actor_relation).strip().lower()
        if relation not in {"partner", "opponent"}:
            raise ValueError("continuation actor must be partner or opponent")
        normalized = tuple(
            sorted((Card.parse(action).id, child) for action, child in children.items())
        )
        if not normalized or len({action for action, _ in normalized}) != len(normalized):
            raise ValueError("continuation actions must be non-empty and unique")
        information_hash = str(information_state_hash).strip()
        if not information_hash:
            raise ValueError("continuation information-state hash is required")
        return cls(relation, information_hash, normalized, None)

    def validate(self) -> int:
        if self.terminal_margin is not None:
            if self.actor_relation is not None or self.information_state_hash is not None or self.children:
                raise ValueError("terminal node cannot have an actor or children")
            return 1
        if (
            self.actor_relation not in {"partner", "opponent"}
            or not self.information_state_hash
            or not self.children
        ):
            raise ValueError("decision node requires an actor relation and children")
        return 1 + sum(child.validate() for _, child in self.children)

    def terminal_bounds(self) -> tuple[int, int]:
        if self.terminal_margin is not None:
            return self.terminal_margin, self.terminal_margin
        bounds = [child.terminal_bounds() for _, child in self.children]
        return min(item[0] for item in bounds), max(item[1] for item in bounds)

    def to_dict(self) -> dict[str, Any]:
        if self.terminal_margin is not None:
            return {"terminalMargin": self.terminal_margin}
        return {
            "actorRelation": self.actor_relation,
            "informationStateHash": self.information_state_hash,
            "children": [
                {"action": Card(action).code, "node": child.to_dict()}
                for action, child in self.children
            ],
        }


@dataclass(frozen=True, slots=True)
class ReducedRootActionV1:
    action: Card
    continuation: ReducedGameNodeV1
    immediate_margin: int
    immediate_margin_certified: bool
    strength_percentile_bps: int

    @classmethod
    def build(
        cls,
        action: Card | int | str,
        continuation: ReducedGameNodeV1,
        *,
        immediate_margin: int = 0,
        immediate_margin_certified: bool = False,
        strength_percentile_bps: int = 10_000,
    ) -> "ReducedRootActionV1":
        percentile = int(strength_percentile_bps)
        if not 0 <= percentile <= 10_000:
            raise ValueError("card strength percentile must be between 0 and 10000")
        continuation.validate()
        return cls(
            Card.parse(action),
            continuation,
            int(immediate_margin),
            bool(immediate_margin_certified),
            percentile,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.code,
            "immediateMargin": self.immediate_margin,
            "immediateMarginCertified": self.immediate_margin_certified,
            "strengthPercentileBps": self.strength_percentile_bps,
            "continuation": self.continuation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ReducedGameWorldV1:
    world_fingerprint: str
    root_actions: tuple[ReducedRootActionV1, ...]

    @classmethod
    def build(
        cls,
        world_fingerprint: str,
        root_actions: Sequence[ReducedRootActionV1],
    ) -> "ReducedGameWorldV1":
        actions = tuple(sorted(root_actions, key=lambda item: item.action.id))
        if not world_fingerprint or len(actions) < 2:
            raise ValueError("world requires a fingerprint and at least two root actions")
        if len({item.action.id for item in actions}) != len(actions):
            raise ValueError("root actions must be unique")
        return cls(str(world_fingerprint), actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worldFingerprint": self.world_fingerprint,
            "rootActions": [item.to_dict() for item in self.root_actions],
        }


@dataclass(frozen=True, slots=True)
class ReducedBestResponseProfileV1:
    policy: ReducedPolicyProfileV1 = ReducedPolicyProfileV1()
    maximum_nodes: int = 2_000
    maximum_regret: int = 0
    small_card_max_percentile_bps: int = 3_500
    delayed_gain_minimum: int = 10
    delayed_gain_maximum_depth: int = 4
    strength_profile_hash: str = ""
    solver_backend: str = "deterministic_reference"

    def __post_init__(self) -> None:
        if self.maximum_nodes < 1 or self.maximum_regret < 0:
            raise ValueError("best-response bounds must be positive")
        if not 0 <= self.small_card_max_percentile_bps <= 10_000:
            raise ValueError("small-card percentile is outside its valid range")
        if self.delayed_gain_minimum < 1 or self.delayed_gain_maximum_depth < 1:
            raise ValueError("delayed-gain thresholds must be positive")
        if self.solver_backend not in {"deterministic_reference", "batched_cp_sat"}:
            raise ValueError("unsupported best-response solver backend")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": PROOF_SCHEMA_VERSION,
            "policy": self.policy.to_dict(),
            "maximumNodes": self.maximum_nodes,
            "maximumRegret": self.maximum_regret,
            "smallCardMaxPercentileBps": self.small_card_max_percentile_bps,
            "delayedGainMinimum": self.delayed_gain_minimum,
            "delayedGainMaximumDepth": self.delayed_gain_maximum_depth,
            "strengthProfileHash": self.strength_profile_hash,
            "solverBackend": self.solver_backend,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ReducedBestResponseProblemV1:
    information_state_hash: str
    actor: str
    coverage_method: str
    worlds: tuple[ReducedGameWorldV1, ...]
    profile: ReducedBestResponseProfileV1

    @classmethod
    def build(
        cls,
        *,
        information_state_hash: str,
        actor: str,
        coverage_method: str,
        worlds: Sequence[ReducedGameWorldV1],
        profile: ReducedBestResponseProfileV1 | None = None,
    ) -> "ReducedBestResponseProblemV1":
        value = cls(
            str(information_state_hash),
            str(actor).lower(),
            str(coverage_method),
            tuple(sorted(worlds, key=lambda item: item.world_fingerprint)),
            profile or ReducedBestResponseProfileV1(),
        )
        value.validate()
        return value

    def validate(self) -> None:
        if not self.information_state_hash or not self.actor or not self.worlds:
            raise ValueError("best-response problem identity is incomplete")
        fingerprints = [world.world_fingerprint for world in self.worlds]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("best-response world fingerprints must be unique")
        expected = tuple(item.action.id for item in self.worlds[0].root_actions)
        total_nodes = 0
        information_sets: dict[str, str] = {}
        metadata = tuple(
            (
                item.action.id,
                item.immediate_margin,
                item.immediate_margin_certified,
                item.strength_percentile_bps,
            )
            for item in self.worlds[0].root_actions
        )
        for world in self.worlds:
            if tuple(item.action.id for item in world.root_actions) != expected:
                raise ValueError("all worlds must expose the same actor root actions")
            candidate_metadata = tuple(
                (
                    item.action.id,
                    item.immediate_margin,
                    item.immediate_margin_certified,
                    item.strength_percentile_bps,
                )
                for item in world.root_actions
            )
            if candidate_metadata != metadata:
                raise ValueError("public root-action metadata must match across hidden worlds")
            total_nodes += sum(item.continuation.validate() for item in world.root_actions)
            for item in world.root_actions:
                self._validate_information_sets(item.continuation, information_sets)
        if total_nodes > self.profile.maximum_nodes:
            raise ValueError("best-response tree exceeds the governed node budget")

    @classmethod
    def _validate_information_sets(
        cls,
        node: ReducedGameNodeV1,
        seen: dict[str, str],
    ) -> None:
        if node.terminal_margin is not None:
            return
        assert node.information_state_hash is not None
        # This bounded slice supports repeated information sets only when their
        # complete action/value subtree is identical. Otherwise a solver could
        # choose differently using hidden-world identity (anticipativity).
        fingerprint = canonical_sha256(node.to_dict())
        previous = seen.setdefault(node.information_state_hash, fingerprint)
        if previous != fingerprint:
            raise ValueError(
                "non-identical nodes share an actor information state; "
                "non-anticipativity is not proven"
            )
        for _, child in node.children:
            cls._validate_information_sets(child, seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": PROOF_SCHEMA_VERSION,
            "informationStateHash": self.information_state_hash,
            "actor": self.actor,
            "coverageMethod": self.coverage_method,
            "worlds": [world.to_dict() for world in self.worlds],
            "profile": self.profile.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


class EngineReducedBestResponseProblemBuilderV1:
    """Enumerate a bounded authoritative-engine tree for each hidden world."""

    def build(
        self,
        *,
        states: Sequence[RoundState],
        engine: BalootEngine,
        actor: Seat,
        coverage_method: str,
        profile: ReducedBestResponseProfileV1 | None = None,
        maximum_plies: int = 12,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ReducedBestResponseProblemV1:
        if not states or maximum_plies < 1:
            raise ValueError("engine best-response builder requires states and a ply bound")
        effective_profile = profile or ReducedBestResponseProfileV1()
        node_budget = [effective_profile.maximum_nodes]
        visible_hash: str | None = None
        expected_actions: tuple[Card, ...] | None = None
        worlds: list[ReducedGameWorldV1] = []
        for state in states:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("best-response tree construction cancelled")
            if state.current_player != actor or state.rules_hash != engine.rules.sha256:
                raise ValueError("root state actor or rules do not match the requested proof")
            actor_hash = canonical_sha256(InformationState.from_complete(state, actor).to_dict())
            if visible_hash is None:
                visible_hash = actor_hash
            elif actor_hash != visible_hash:
                raise ValueError("root hidden states are not actor-indistinguishable")
            legal = engine.legal_actions(state)
            if expected_actions is None:
                expected_actions = legal
            elif legal != expected_actions:
                raise ValueError("root legal actions differ across hidden worlds")
            if len(legal) < 2:
                raise ValueError("best-response proof requires a non-forced root decision")
            ordered = sorted(
                legal,
                key=lambda card: (
                    engine.rules.strength(card, state.contract, state.trump_suit),
                    card.id,
                ),
            )
            percentiles = {
                card.id: (index * 10_000 // max(1, len(ordered) - 1))
                for index, card in enumerate(ordered)
            }
            roots: list[ReducedRootActionV1] = []
            working = state.clone()
            for action in legal:
                token = engine.apply(
                    working,
                    action,
                    verify_identity=False,
                    emit_events=False,
                )
                try:
                    continuation = self._expand(
                        working,
                        engine=engine,
                        root_team=actor.team,
                        remaining_plies=maximum_plies - 1,
                        cancel_check=cancel_check,
                        node_budget=node_budget,
                    )
                finally:
                    engine.undo(working, token)
                roots.append(
                    ReducedRootActionV1.build(
                        action,
                        continuation,
                        # Mid-trick point deltas are not authoritative awarded
                        # margins, so this builder does not certify them.
                        immediate_margin=0,
                        immediate_margin_certified=False,
                        strength_percentile_bps=percentiles[action.id],
                    )
                )
            worlds.append(ReducedGameWorldV1.build(engine.state_hash(state), roots))
        assert visible_hash is not None
        return ReducedBestResponseProblemV1.build(
            information_state_hash=visible_hash,
            actor=actor.name.lower(),
            coverage_method=coverage_method,
            worlds=worlds,
            profile=effective_profile,
        )

    def _expand(
        self,
        state: RoundState,
        *,
        engine: BalootEngine,
        root_team: Team,
        remaining_plies: int,
        cancel_check: Callable[[], bool] | None,
        node_budget: list[int],
    ) -> ReducedGameNodeV1:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("best-response tree construction cancelled")
        node_budget[0] -= 1
        if node_budget[0] < 0:
            raise ValueError("best-response tree exceeds the governed node budget")
        if state.current_player is None:
            if state.round_score is None:
                raise AssertionError("terminal engine branch has no round score")
            points = state.round_score.awarded_points
            return ReducedGameNodeV1.terminal(
                points[int(root_team)] - points[int(root_team.other)]
            )
        if remaining_plies <= 0:
            raise ValueError("late-game continuation exceeds the governed ply bound")
        seat = state.current_player
        relation = "partner" if seat.team == root_team else "opponent"
        information_hash = canonical_sha256(InformationState.from_complete(state, seat).to_dict())
        children: dict[Card, ReducedGameNodeV1] = {}
        for action in engine.legal_actions(state):
            token = engine.apply(
                state,
                action,
                verify_identity=False,
                emit_events=False,
            )
            try:
                children[action] = self._expand(
                    state,
                    engine=engine,
                    root_team=root_team,
                    remaining_plies=remaining_plies - 1,
                    cancel_check=cancel_check,
                    node_budget=node_budget,
                )
            finally:
                engine.undo(state, token)
        return ReducedGameNodeV1.decision(
            relation,
            children,
            information_state_hash=information_hash,
        )


@dataclass(frozen=True, slots=True)
class BestResponseEvidenceV1:
    world_fingerprint: str
    root_action: Card
    cp_sat_margin: int
    reference_margin: int
    principal_actions: tuple[Card, ...]
    payoff_depth: int
    node_count: int
    solver_status: str
    parity: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "worldFingerprint": self.world_fingerprint,
            "rootAction": self.root_action.code,
            "cpSatMargin": self.cp_sat_margin,
            "referenceMargin": self.reference_margin,
            "principalActions": [action.code for action in self.principal_actions],
            "payoffDepth": self.payoff_depth,
            "nodeCount": self.node_count,
            "solverStatus": self.solver_status,
            "parity": self.parity,
        }


@dataclass(frozen=True, slots=True)
class RegretEvidenceV1:
    actual_action: Card
    recommended_action: Card
    maximum_paired_regret: int
    total_paired_regret: int
    maximum_allowed: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "actualAction": self.actual_action.code,
            "recommendedAction": self.recommended_action.code,
            "maximumPairedRegret": self.maximum_paired_regret,
            "totalPairedRegret": self.total_paired_regret,
            "maximumAllowed": self.maximum_allowed,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class SmallCardBigGainWitnessV1:
    selected_action: Card
    strength_percentile_bps: int
    immediate_margin: int
    minimum_delayed_gain: int
    maximum_payoff_depth: int
    counterfactual_margin_gap: int
    strength_profile_hash: str
    scope: str = "exhaustive_reduced_counterfactual_tree"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": PROOF_SCHEMA_VERSION,
            "selectedAction": self.selected_action.code,
            "strengthPercentileBps": self.strength_percentile_bps,
            "immediateMargin": self.immediate_margin,
            "minimumDelayedGain": self.minimum_delayed_gain,
            "maximumPayoffDepth": self.maximum_payoff_depth,
            "counterfactualMarginGap": self.counterfactual_margin_gap,
            "strengthProfileHash": self.strength_profile_hash,
            "scope": self.scope,
        }


@dataclass(frozen=True, slots=True)
class ReducedBestResponseProofV1:
    problem_hash: str
    profile_hash: str
    policy_proof: ReducedPolicyProofV1
    continuations: tuple[BestResponseEvidenceV1, ...]
    regret: RegretEvidenceV1 | None
    qualification_status: str
    small_card_big_gain: SmallCardBigGainWitnessV1 | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": PROOF_SCHEMA_VERSION,
            "proofEngine": {
                "id": BEST_RESPONSE_ENGINE_ID,
                "version": BEST_RESPONSE_ENGINE_VERSION,
            },
            "problemHash": self.problem_hash,
            "profileHash": self.profile_hash,
            "policyProof": self.policy_proof.to_dict(),
            "continuations": [item.to_dict() for item in self.continuations],
            "regret": self.regret.to_dict() if self.regret else None,
            "qualificationStatus": self.qualification_status,
            "smallCardBigGain": (
                self.small_card_big_gain.to_dict() if self.small_card_big_gain else None
            ),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def exhaustive_backward_induction(
    node: ReducedGameNodeV1,
) -> tuple[int, tuple[Card, ...]]:
    """Independent recursive reference with stable action tie-breaking."""
    if node.terminal_margin is not None:
        return node.terminal_margin, ()
    candidates = []
    for action_id, child in node.children:
        margin, suffix = exhaustive_backward_induction(child)
        candidates.append((margin, action_id, (Card(action_id), *suffix)))
    if node.actor_relation == "partner":
        margin, _action_id, line = max(candidates, key=lambda item: (item[0], -item[1]))
    else:
        margin, _action_id, line = min(candidates, key=lambda item: (item[0], item[1]))
    return margin, line


class OrToolsBoundedBestResponseProofV1:
    descriptor = PluginDescriptor(
        id="ortools_bounded_best_response_v1",
        kind="best_response",
        version=BEST_RESPONSE_ENGINE_VERSION,
        description="Bounded actor-safe best-response proof with independent reference parity",
        capabilities=(
            "actor_safe_worlds",
            "authoritative_legal_actions",
            "cp_sat_reference_parity",
            "bounded_regret",
        ),
        deterministic=True,
        resource_class="analysis-cpu",
        timeout_seconds=60,
        maximum_memory_mb=512,
        contract_id="best_response",
        contract_version="1",
        safety_guards=(
            "authoritative_legality",
            "observer_safe",
            "hard_gate_preservation",
            "frozen_provenance",
        ),
        cancellation_mode="cooperative",
    )

    @staticmethod
    def _solve_reference_batch(
        nodes: Sequence[ReducedGameNodeV1],
    ) -> tuple[tuple[int, ...], str, tuple[int, ...]]:
        """Iterative evaluator independent from the recursive audit reference."""
        values: list[int] = []
        counts: list[int] = []
        for root in nodes:
            computed: dict[int, int] = {}
            stack: list[tuple[ReducedGameNodeV1, bool]] = [(root, False)]
            count = 0
            while stack:
                node, visited = stack.pop()
                identity = id(node)
                if identity in computed:
                    continue
                if node.terminal_margin is not None:
                    computed[identity] = node.terminal_margin
                    count += 1
                    continue
                if not visited:
                    stack.append((node, True))
                    stack.extend((child, False) for _, child in reversed(node.children))
                    continue
                child_values = [computed[id(child)] for _, child in node.children]
                computed[identity] = (
                    max(child_values)
                    if node.actor_relation == "partner"
                    else min(child_values)
                )
                count += 1
            values.append(computed[id(root)])
            counts.append(count)
        return tuple(values), "exact_reference", tuple(counts)

    def _solve_batch(
        self,
        nodes: Sequence[ReducedGameNodeV1],
        *,
        maximum_seconds: float = 5.0,
    ) -> tuple[tuple[int, ...], str, tuple[int, ...]]:
        """Lower all independent continuations into one reusable CP-SAT model."""
        from ortools.sat.python import cp_model

        if not nodes:
            raise ValueError("best-response batch cannot be empty")
        model = cp_model.CpModel()
        serial = 0

        def lower(current: ReducedGameNodeV1):
            nonlocal serial
            serial += 1
            lower_bound, upper_bound = current.terminal_bounds()
            value = model.new_int_var(lower_bound, upper_bound, f"value_{serial}")
            if current.terminal_margin is not None:
                model.add(value == current.terminal_margin)
                return value
            children = [lower(child) for _, child in current.children]
            if current.actor_relation == "partner":
                model.add_max_equality(value, children)
            else:
                model.add_min_equality(value, children)
            return value

        roots = []
        counts = []
        for node in nodes:
            before = serial
            roots.append(lower(node))
            counts.append(serial - before)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_time_in_seconds = max(0.001, float(maximum_seconds))
        status = solver.solve(model)
        if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            raise RuntimeError(
                "best-response batch did not complete: " + solver.status_name(status)
            )
        return (
            tuple(int(solver.value(root)) for root in roots),
            solver.status_name(status).lower(),
            tuple(counts),
        )

    def _solve_tree(
        self,
        node: ReducedGameNodeV1,
        *,
        maximum_seconds: float = 5.0,
    ) -> tuple[int, str, int]:
        values, status, counts = self._solve_batch(
            (node,),
            maximum_seconds=maximum_seconds,
        )
        return values[0], status, counts[0]

    def prove(
        self,
        problem: ReducedBestResponseProblemV1,
        *,
        actual_action: Card | int | str | None = None,
        cancel_check: Callable[[], bool] | None = None,
        maximum_seconds_per_subproblem: float = 5.0,
    ) -> ReducedBestResponseProofV1:
        if maximum_seconds_per_subproblem <= 0:
            raise ValueError("subproblem time bound must be positive")
        problem.validate()
        continuations: list[BestResponseEvidenceV1] = []
        outcome_worlds: list[ReducedWorldOutcomeV1] = []
        values_by_world: dict[str, dict[int, int]] = {}
        flattened = tuple(
            (world, root)
            for world in problem.worlds
            for root in world.root_actions
        )
        if cancel_check is not None and cancel_check():
            raise InterruptedError("best-response proof cancelled")
        continuation_nodes = tuple(root.continuation for _world, root in flattened)
        if problem.profile.solver_backend == "batched_cp_sat":
            batch_values, batch_status, batch_counts = self._solve_batch(
                continuation_nodes,
                maximum_seconds=maximum_seconds_per_subproblem,
            )
        else:
            batch_values, batch_status, batch_counts = self._solve_reference_batch(
                continuation_nodes
            )
        batch_results = {
            (world.world_fingerprint, root.action.id): (margin, node_count)
            for (world, root), margin, node_count in zip(
                flattened,
                batch_values,
                batch_counts,
                strict=True,
            )
        }
        for world in problem.worlds:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("best-response proof cancelled")
            margins: dict[Card, int] = {}
            values_by_world[world.world_fingerprint] = {}
            for root in world.root_actions:
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("best-response proof cancelled")
                cp_margin, node_count = batch_results[
                    (world.world_fingerprint, root.action.id)
                ]
                reference_margin, principal = exhaustive_backward_induction(root.continuation)
                parity = cp_margin == reference_margin
                margins[root.action] = cp_margin
                values_by_world[world.world_fingerprint][root.action.id] = cp_margin
                continuations.append(
                    BestResponseEvidenceV1(
                        world.world_fingerprint,
                        root.action,
                        cp_margin,
                        reference_margin,
                        principal,
                        len(principal),
                        node_count,
                        batch_status,
                        parity,
                    )
                )
            outcome_worlds.append(ReducedWorldOutcomeV1.build(world.world_fingerprint, margins))

        reduced = ReducedActorSafePolicyProblemV1.build(
            information_state_hash=problem.information_state_hash,
            actor=problem.actor,
            coverage_method=problem.coverage_method,
            worlds=outcome_worlds,
            profile=problem.profile.policy,
        )
        if problem.profile.solver_backend == "batched_cp_sat":
            policy_proof = OrToolsReducedPolicyProofV1().prove(
                reduced,
                actual_action=actual_action,
            )
        else:
            policy_proof = exhaustive_reference_policy_proof(
                reduced,
                actual_action=actual_action,
            )
        continuation_parity = all(item.parity for item in continuations)
        continuation_optimal = all(
            item.solver_status in {"optimal", "exact_reference"}
            for item in continuations
        )
        if not continuation_parity or not continuation_optimal:
            # Do not allow the root model to promote truth if any subproblem is incomplete.
            policy_proof = ReducedPolicyProofV1(
                policy_proof.problem_hash,
                policy_proof.profile_hash,
                policy_proof.selected_action,
                policy_proof.action_evidence,
                policy_proof.model_feasibility_status,
                "provisional",
                policy_proof.reference_action,
                policy_proof.actor_choice_bound_once,
                policy_proof.all_actions_evaluated,
                None,
                policy_proof.action_cut,
                policy_proof.model_hash,
            )

        regret = None
        qualification = "not_evaluated"
        if actual_action is not None:
            actual = Card.parse(actual_action)
            recommended = policy_proof.selected_action
            if actual.id not in values_by_world[problem.worlds[0].world_fingerprint]:
                raise ValueError("actual action is outside the root action set")
            paired = [
                max(0, values[recommended.id] - values[actual.id])
                for values in values_by_world.values()
            ]
            maximum = max(paired)
            passed = maximum <= problem.profile.maximum_regret
            regret = RegretEvidenceV1(
                actual,
                recommended,
                maximum,
                sum(paired),
                problem.profile.maximum_regret,
                passed,
            )
            if policy_proof.strategic_truth != "exact_reduced_actor_safe":
                qualification = "rejected_incomplete_truth"
            else:
                qualification = "qualified" if passed else "rejected_maximum_regret"

        small_witness = self._small_card_witness(problem, policy_proof, continuations)
        return ReducedBestResponseProofV1(
            problem.sha256,
            problem.profile.sha256,
            policy_proof,
            tuple(continuations),
            regret,
            qualification,
            small_witness,
        )

    @staticmethod
    def _small_card_witness(
        problem: ReducedBestResponseProblemV1,
        proof: ReducedPolicyProofV1,
        continuations: Sequence[BestResponseEvidenceV1],
    ) -> SmallCardBigGainWitnessV1 | None:
        profile = problem.profile
        if (
            proof.strategic_truth != "exact_reduced_actor_safe"
            or not profile.strength_profile_hash
        ):
            return None
        selected = proof.selected_action
        selected_roots = [
            next(item for item in world.root_actions if item.action == selected)
            for world in problem.worlds
        ]
        percentile = selected_roots[0].strength_percentile_bps
        immediate = selected_roots[0].immediate_margin
        selected_lines = [item for item in continuations if item.root_action == selected]
        minimum_gain = min(item.cp_sat_margin - immediate for item in selected_lines)
        maximum_depth = max(item.payoff_depth for item in selected_lines)
        selected_worst = min(item.cp_sat_margin for item in selected_lines)
        alternative_best = max(
            item.cp_sat_margin for item in continuations if item.root_action != selected
        )
        gap = selected_worst - alternative_best
        if (
            percentile <= profile.small_card_max_percentile_bps
            and all(item.immediate_margin_certified for item in selected_roots)
            and immediate <= 0
            and 1 <= maximum_depth <= profile.delayed_gain_maximum_depth
            and minimum_gain >= profile.delayed_gain_minimum
            and gap >= profile.delayed_gain_minimum
        ):
            return SmallCardBigGainWitnessV1(
                selected,
                percentile,
                immediate,
                minimum_gain,
                maximum_depth,
                gap,
                profile.strength_profile_hash,
            )
        return None
