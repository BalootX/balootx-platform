"""Actor-safe reduced late-game policy proofs.

This module is intentionally independent from Research services and storage.  It
proves a *single shared action* over an already enumerated observer information
set.  CP-SAT's model status is recorded separately from strategic truth: a
legal/optimal solve is never, by itself, an optimal-play claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from baloot_platform.analysis.models import InformationState
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat
from baloot_platform.core.models import RoundState

from .contracts import ReducedPolicyProfileV1, canonical_sha256


PROOF_SCHEMA_VERSION = 1
PROOF_ENGINE_ID = "reduced_actor_safe_policy_proof"
PROOF_ENGINE_VERSION = "1.0.0"
EXHAUSTIVE_COVERAGE_METHODS = frozenset(
    {
        "exhaustive_constrained_hidden_deals",
        "exhaustive_project_conditioned_hidden_deals",
        "exhaustive_reference_fixture",
    }
)


@dataclass(frozen=True, slots=True)
class ReducedWorldOutcomeV1:
    """One hidden world, identified only by a content fingerprint."""

    world_fingerprint: str
    action_margins: tuple[tuple[int, int], ...]

    @classmethod
    def build(
        cls,
        world_fingerprint: str,
        action_margins: Mapping[Card | int | str, int],
    ) -> "ReducedWorldOutcomeV1":
        fingerprint = str(world_fingerprint).strip()
        if not fingerprint:
            raise ValueError("world fingerprint is required")
        normalized = tuple(
            sorted(
                (Card.parse(action).id, int(margin))
                for action, margin in action_margins.items()
            )
        )
        if not normalized or len({action for action, _ in normalized}) != len(normalized):
            raise ValueError("world actions must be non-empty and unique")
        return cls(fingerprint, normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worldFingerprint": self.world_fingerprint,
            "actionMargins": [
                {"action": Card(action).code, "margin": margin}
                for action, margin in self.action_margins
            ],
        }


@dataclass(frozen=True, slots=True)
class ReducedActorSafePolicyProblemV1:
    """Complete reduced information set with one actor-visible action domain."""

    information_state_hash: str
    actor: str
    coverage_method: str
    worlds: tuple[ReducedWorldOutcomeV1, ...]
    profile: ReducedPolicyProfileV1

    @classmethod
    def build(
        cls,
        *,
        information_state_hash: str,
        actor: str,
        coverage_method: str,
        worlds: Sequence[ReducedWorldOutcomeV1],
        profile: ReducedPolicyProfileV1 | None = None,
    ) -> "ReducedActorSafePolicyProblemV1":
        value = cls(
            str(information_state_hash).strip(),
            str(actor).strip().lower(),
            str(coverage_method).strip(),
            tuple(sorted(worlds, key=lambda item: item.world_fingerprint)),
            profile or ReducedPolicyProfileV1(),
        )
        value.validate()
        return value

    @property
    def is_exhaustive(self) -> bool:
        return self.coverage_method.split("+", 1)[0] in EXHAUSTIVE_COVERAGE_METHODS

    @property
    def legal_action_ids(self) -> tuple[int, ...]:
        return tuple(action for action, _ in self.worlds[0].action_margins)

    def validate(self) -> None:
        if not self.information_state_hash or not self.actor or not self.coverage_method:
            raise ValueError("information hash, actor, and coverage method are required")
        if not self.worlds:
            raise ValueError("at least one hidden world is required")
        fingerprints = [world.world_fingerprint for world in self.worlds]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("hidden-world fingerprints must be unique")
        expected = tuple(action for action, _ in self.worlds[0].action_margins)
        if len(expected) < 2:
            raise ValueError("a policy proof requires at least two legal actions")
        for world in self.worlds:
            actions = tuple(action for action, _ in world.action_margins)
            if actions != expected:
                raise ValueError("all indistinguishable worlds must expose the same actor action set")

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


class ForcedLateGameProblemBuilderV1:
    """Lower exhaustive late-game states into a reduced actor-safe problem.

    This first vertical slice deliberately accepts only continuations that are
    forced after the root choice. It therefore proves a real decision without
    pretending that unresolved future policy has been optimized.
    """

    def build(
        self,
        *,
        states: Sequence[RoundState],
        engine: BalootEngine,
        actor: Seat,
        coverage_method: str,
        profile: ReducedPolicyProfileV1 | None = None,
    ) -> ReducedActorSafePolicyProblemV1:
        if not states:
            raise ValueError("late-game proof requires at least one hidden state")
        visible_hash: str | None = None
        expected_actions: tuple[Card, ...] | None = None
        worlds: list[ReducedWorldOutcomeV1] = []
        for state in states:
            if state.current_player != actor:
                raise ValueError("every hidden state must have the same acting seat")
            if state.rules_hash != engine.rules.sha256:
                raise ValueError("hidden state and engine rules do not match")
            information = InformationState.from_complete(state, actor)
            candidate_hash = canonical_sha256(information.to_dict())
            if visible_hash is None:
                visible_hash = candidate_hash
            elif candidate_hash != visible_hash:
                raise ValueError("hidden states are not actor-indistinguishable")
            legal = engine.legal_actions(state)
            if expected_actions is None:
                expected_actions = legal
            elif legal != expected_actions:
                raise ValueError("actor-visible legal actions differ between hidden states")
            if len(legal) < 2:
                raise ValueError("late-game proof requires a non-forced root decision")

            margins: dict[Card, int] = {}
            for action in legal:
                branch = state.clone()
                engine.play_card(branch, action, emit_events=False)
                while branch.current_player is not None:
                    continuation = engine.legal_actions(branch)
                    if len(continuation) != 1:
                        raise ValueError(
                            "reduced slice supports only forced post-root continuations"
                        )
                    engine.play_card(branch, continuation[0], emit_events=False)
                if branch.round_score is None:
                    raise AssertionError("terminal reduced branch has no round score")
                points = branch.round_score.awarded_points
                margins[action] = points[int(actor.team)] - points[int(actor.team.other)]
            worlds.append(
                ReducedWorldOutcomeV1.build(engine.state_hash(state), margins)
            )

        assert visible_hash is not None
        return ReducedActorSafePolicyProblemV1.build(
            information_state_hash=visible_hash,
            actor=actor.name.lower(),
            coverage_method=coverage_method,
            worlds=worlds,
            profile=profile,
        )


@dataclass(frozen=True, slots=True)
class ReducedActionEvidenceV1:
    action: Card
    worst_margin: int
    best_margin: int
    total_margin: int
    preserving_worlds: int
    world_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.code,
            "worstMargin": self.worst_margin,
            "bestMargin": self.best_margin,
            "totalMargin": self.total_margin,
            "preservingWorlds": self.preserving_worlds,
            "worldCount": self.world_count,
        }


@dataclass(frozen=True, slots=True)
class OnlyWinningMoveWitnessV1:
    selected_action: Card
    rejected_alternatives: tuple[Card, ...]
    outcome_boundary: int
    selected_worst_margin: int
    best_alternative_margin: int
    scenario_count: int
    scope: str = "exhaustive_reduced_actor_information_set"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": PROOF_SCHEMA_VERSION,
            "selectedAction": self.selected_action.code,
            "rejectedAlternatives": [card.code for card in self.rejected_alternatives],
            "outcomeBoundary": self.outcome_boundary,
            "selectedWorstMargin": self.selected_worst_margin,
            "bestAlternativeMargin": self.best_alternative_margin,
            "scenarioCount": self.scenario_count,
            "scope": self.scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ActorVisibleActionCutV1:
    information_state_hash: str
    forbidden_action: Card
    proven_action: Card
    reason: str = "reduced_actor_safe_counterexample"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": PROOF_SCHEMA_VERSION,
            "informationStateHash": self.information_state_hash,
            "forbiddenAction": self.forbidden_action.code,
            "provenAction": self.proven_action.code,
            "reason": self.reason,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ReducedPolicyProofV1:
    problem_hash: str
    profile_hash: str
    selected_action: Card
    action_evidence: tuple[ReducedActionEvidenceV1, ...]
    model_feasibility_status: str
    strategic_truth: str
    reference_action: Card
    actor_choice_bound_once: bool
    all_actions_evaluated: bool
    only_winning_move: OnlyWinningMoveWitnessV1 | None
    action_cut: ActorVisibleActionCutV1 | None
    model_hash: str

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schemaVersion": PROOF_SCHEMA_VERSION,
            "proofEngine": {"id": PROOF_ENGINE_ID, "version": PROOF_ENGINE_VERSION},
            "problemHash": self.problem_hash,
            "profileHash": self.profile_hash,
            "selectedAction": self.selected_action.code,
            "actionEvidence": [item.to_dict() for item in self.action_evidence],
            "modelFeasibilityStatus": self.model_feasibility_status,
            "strategicTruth": self.strategic_truth,
            "referenceAction": self.reference_action.code,
            "actorChoiceBoundOnce": self.actor_choice_bound_once,
            "allActionsEvaluated": self.all_actions_evaluated,
            "onlyWinningMove": self.only_winning_move.to_dict() if self.only_winning_move else None,
            "actionCut": self.action_cut.to_dict() if self.action_cut else None,
            "modelHash": self.model_hash,
        }
        return payload

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _evidence(problem: ReducedActorSafePolicyProblemV1) -> tuple[ReducedActionEvidenceV1, ...]:
    result: list[ReducedActionEvidenceV1] = []
    for action_id in problem.legal_action_ids:
        margins = [dict(world.action_margins)[action_id] for world in problem.worlds]
        result.append(
            ReducedActionEvidenceV1(
                Card(action_id),
                min(margins),
                max(margins),
                sum(margins),
                sum(margin >= problem.profile.outcome_boundary for margin in margins),
                len(margins),
            )
        )
    return tuple(result)


def exhaustive_reference_action(problem: ReducedActorSafePolicyProblemV1) -> Card:
    """Independent exhaustive reference for the governed lexicographic policy."""
    evidence = _evidence(problem)
    return max(
        evidence,
        key=lambda item: (item.worst_margin, item.total_margin, -item.action.id),
    ).action


class OrToolsReducedPolicyProofV1:
    """CP-SAT accelerator for an exhaustive reduced actor information set."""

    def prove(
        self,
        problem: ReducedActorSafePolicyProblemV1,
        *,
        actual_action: Card | int | str | None = None,
    ) -> ReducedPolicyProofV1:
        from ortools.sat.python import cp_model

        problem.validate()
        evidence = _evidence(problem)
        model = cp_model.CpModel()
        chosen = {
            item.action.id: model.new_bool_var(f"choose_{item.action.id}")
            for item in evidence
        }
        model.add_exactly_one(chosen.values())
        worst = sum(item.worst_margin * chosen[item.action.id] for item in evidence)
        total = sum(item.total_margin * chosen[item.action.id] for item in evidence)
        stable_id = sum(item.action.id * chosen[item.action.id] for item in evidence)
        model_core = {
            "schemaVersion": PROOF_SCHEMA_VERSION,
            "problemHash": problem.sha256,
            "profileHash": problem.profile.sha256,
            "actions": [item.to_dict() for item in evidence],
            "actorChoiceVariables": sorted(chosen),
            "phases": ["maxWorstMargin", "maxTotalMargin", "minCardId"],
        }
        model_hash = canonical_sha256(model_core)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        phase_statuses: list[int] = []
        for objective, maximize in ((worst, True), (total, True), (stable_id, False)):
            (model.maximize if maximize else model.minimize)(objective)
            status = solver.solve(model)
            phase_statuses.append(status)
            if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
                raise RuntimeError("reduced policy model did not complete: " + solver.status_name(status))
            optimum = int(round(solver.objective_value))
            model.add(objective == optimum)

        selected = Card(next(action for action, variable in chosen.items() if solver.value(variable)))
        reference = exhaustive_reference_action(problem)
        parity = selected == reference
        all_optimal = all(status == cp_model.OPTIMAL for status in phase_statuses)
        feasibility_status = "optimal" if all_optimal else "feasible"
        # Strategic truth additionally requires exhaustive actor-safe coverage
        # and independent reference parity. CP-SAT status alone is insufficient.
        strategic_truth = (
            "exact_reduced_actor_safe"
            if problem.is_exhaustive and all_optimal and parity
            else "provisional"
        )

        winner = next(item for item in evidence if item.action == selected)
        alternatives = tuple(item for item in evidence if item.action != selected)
        witness = None
        if (
            strategic_truth == "exact_reduced_actor_safe"
            and winner.preserving_worlds == winner.world_count
            and all(item.preserving_worlds == 0 for item in alternatives)
        ):
            witness = OnlyWinningMoveWitnessV1(
                selected,
                tuple(item.action for item in alternatives),
                problem.profile.outcome_boundary,
                winner.worst_margin,
                max(item.best_margin for item in alternatives),
                winner.world_count,
            )

        cut = None
        if actual_action is not None:
            actual = Card.parse(actual_action)
            if actual.id not in chosen:
                raise ValueError("actual action is outside the actor-visible legal action set")
            if actual != selected:
                cut = ActorVisibleActionCutV1(problem.information_state_hash, actual, selected)

        return ReducedPolicyProofV1(
            problem.sha256,
            problem.profile.sha256,
            selected,
            evidence,
            feasibility_status,
            strategic_truth,
            reference,
            True,
            True,
            witness,
            cut,
            model_hash,
        )


def exhaustive_reference_policy_proof(
    problem: ReducedActorSafePolicyProblemV1,
    *,
    actual_action: Card | int | str | None = None,
) -> ReducedPolicyProofV1:
    """Low-startup exact enumeration for already bounded outcome matrices."""
    problem.validate()
    evidence = _evidence(problem)
    selected = exhaustive_reference_action(problem)
    strategic_truth = (
        "exact_reduced_actor_safe" if problem.is_exhaustive else "provisional"
    )
    winner = next(item for item in evidence if item.action == selected)
    alternatives = tuple(item for item in evidence if item.action != selected)
    witness = None
    if (
        strategic_truth == "exact_reduced_actor_safe"
        and winner.preserving_worlds == winner.world_count
        and all(item.preserving_worlds == 0 for item in alternatives)
    ):
        witness = OnlyWinningMoveWitnessV1(
            selected,
            tuple(item.action for item in alternatives),
            problem.profile.outcome_boundary,
            winner.worst_margin,
            max(item.best_margin for item in alternatives),
            winner.world_count,
        )
    cut = None
    if actual_action is not None:
        actual = Card.parse(actual_action)
        if actual.id not in problem.legal_action_ids:
            raise ValueError("actual action is outside the actor-visible legal action set")
        if actual != selected:
            cut = ActorVisibleActionCutV1(problem.information_state_hash, actual, selected)
    model_hash = canonical_sha256(
        {
            "schemaVersion": PROOF_SCHEMA_VERSION,
            "engine": "exhaustive_reference_policy",
            "problemHash": problem.sha256,
            "profileHash": problem.profile.sha256,
            "actions": [item.to_dict() for item in evidence],
        }
    )
    return ReducedPolicyProofV1(
        problem.sha256,
        problem.profile.sha256,
        selected,
        evidence,
        "not_applicable_reference",
        strategic_truth,
        selected,
        True,
        True,
        witness,
        cut,
        model_hash,
    )
