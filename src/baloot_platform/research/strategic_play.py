"""Actor-safe CP-SAT play selection for bounded late-game Research decisions.

This plugin is intentionally narrower than a full-round optimal policy.  It
uses only the acting player's information state, binds one root action across
all compatible hidden worlds, evaluates one governed observer-safe continuation
policy, and keeps CP-SAT/reference parity evidence separate from the later
full-line legality proof.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from baloot_platform.analysis.hidden_deals import generate_hidden_deals
from baloot_platform.analysis.models import InformationState
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.models import RoundState
from baloot_platform.plugins.base import PluginDescriptor

from .policy_proof import (
    OrToolsReducedPolicyProofV1,
    ReducedActorSafePolicyProblemV1,
    ReducedPolicyProfileV1,
    ReducedWorldOutcomeV1,
    exhaustive_reference_policy_proof,
)
from .contracts import ActorSafeCpSatPlayProfileV1, canonical_sha256


class ActorSafeCpSatPlaySelectorV1:
    descriptor = PluginDescriptor(
        id="actor_safe_cp_sat_late_game_v1",
        kind="research_play_selector",
        version="1.0.0",
        description=(
            "Observer-safe CP-SAT root-action selection over bounded late-game "
            "policy-conditioned continuations"
        ),
        capabilities=(
            "actor_safe_information_set",
            "non_anticipative_root_action",
            "batched_cp_sat_policy_conditioned_selection",
            "independent_reference_parity",
            "bounded_late_game_only",
        ),
        resource_class="analysis-cpu",
        deterministic=True,
        timeout_seconds=120,
        maximum_memory_mb=2048,
    )

    def __init__(self, profile: ActorSafeCpSatPlayProfileV1):
        self.profile = profile

    def provenance(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "plugin": self.descriptor.to_dict(),
            "profile": self.profile.to_dict(),
        }

    @property
    def policy_profile(self) -> ReducedPolicyProfileV1:
        return ReducedPolicyProfileV1(
            profile_id=self.profile.policy_profile_id,
            revision=self.profile.policy_revision,
            outcome_boundary=self.profile.outcome_boundary,
            utility=self.profile.utility,
            tie_break=self.profile.tie_break,
        )

    def choose(
        self,
        *,
        engine: BalootEngine,
        state: RoundState,
        legal_cards: tuple[Card, ...],
        seed: int,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[Card | None, dict[str, Any]] | None:
        if len(state.played_cards) < self.profile.minimum_played_cards:
            return None
        if len(legal_cards) < 2:
            return None
        if state.current_player is None:
            raise ValueError("strategic selector requires an active player")
        actor = state.current_player
        information = InformationState.from_complete(state, actor)
        started = time.monotonic()
        remaining_step_budget = self.profile.maximum_continuation_steps

        def check_budget() -> None:
            nonlocal remaining_step_budget
            if cancel_check is not None and cancel_check():
                raise InterruptedError("strategic play selection cancelled")
            if (
                time.monotonic() - started
                >= self.profile.maximum_selection_seconds
            ):
                raise RuntimeError("strategic play selection exceeded its time bound")
            remaining_step_budget -= 1
            if remaining_step_budget < 0:
                raise ValueError(
                    "strategic play selection exceeded its continuation-step bound"
                )

        def bounded_cancel_check() -> bool:
            return bool(
                (cancel_check is not None and cancel_check())
                or time.monotonic() - started
                >= self.profile.maximum_selection_seconds
            )

        try:
            cohort = generate_hidden_deals(
                information,
                engine,
                requested_samples=self.profile.requested_samples,
                exact_allocation_limit=self.profile.exact_allocation_limit,
                maximum_attempts=self.profile.maximum_sampling_attempts,
                seed=int(seed),
                cancel_check=bounded_cancel_check,
            )
            if len(cohort.states) > self.profile.maximum_worlds:
                raise ValueError("strategic hidden-world cohort exceeds its governed bound")
            worlds: list[ReducedWorldOutcomeV1] = []
            continuation_hashes: set[str] = set()
            continuation_choices: dict[str, tuple[tuple[int, ...], int]] = {}
            root_information_hash = canonical_sha256(information.to_dict())
            for hidden_state in cohort.states:
                check_budget()
                hidden_information = InformationState.from_complete(hidden_state, actor)
                if canonical_sha256(hidden_information.to_dict()) != root_information_hash:
                    raise AssertionError(
                        "strategic hidden world changed the actor information state"
                    )
                hidden_legal = engine.legal_actions(hidden_state)
                if tuple(card.id for card in hidden_legal) != tuple(
                    card.id for card in legal_cards
                ):
                    raise AssertionError(
                        "strategic hidden worlds disagree on the legal root actions"
                    )
                margins: dict[Card, int] = {}
                for root_action in legal_cards:
                    check_budget()
                    branch = hidden_state.clone()
                    engine.play_card(
                        branch,
                        root_action,
                        emit_events=False,
                    )
                    continuation_trace: list[dict[str, Any]] = []
                    continuation_plies = 1
                    while branch.current_player is not None:
                        check_budget()
                        continuation_plies += 1
                        if continuation_plies > self.profile.maximum_plies:
                            raise ValueError(
                                "strategic continuation exceeded its ply bound"
                            )
                        branch_legal = engine.legal_actions(branch)
                        branch_actor = branch.current_player
                        branch_information = InformationState.from_complete(
                            branch, branch_actor
                        )
                        information_hash = canonical_sha256(
                            branch_information.to_dict()
                        )
                        policy_seed = int(
                            canonical_sha256({
                                "profileHash": self.profile.sha256,
                                "informationStateHash": information_hash,
                                "publicPlayOrdinal": len(branch.played_cards),
                            })[:16],
                            16,
                        )
                        continuation_card = (
                            branch_legal[0]
                            if len(branch_legal) == 1
                            else HeuristicBot(
                                engine, BotProfile(policy_seed)
                            ).choose(branch)
                        )
                        if continuation_card not in branch_legal:
                            raise AssertionError(
                                "observer-safe continuation selected an illegal card"
                            )
                        choice = (
                            tuple(card.id for card in branch_legal),
                            continuation_card.id,
                        )
                        prior_choice = continuation_choices.setdefault(
                            information_hash, choice
                        )
                        if prior_choice != choice:
                            raise AssertionError(
                                "observer-safe continuation violated non-anticipativity"
                            )
                        continuation_trace.append({
                            "informationStateHash": information_hash,
                            "action": continuation_card.code,
                        })
                        engine.play_card(
                            branch,
                            continuation_card,
                            emit_events=False,
                        )
                    if branch.round_score is None:
                        raise AssertionError(
                            "strategic continuation did not reach an awarded score"
                        )
                    points = branch.round_score.awarded_points
                    margins[root_action] = (
                        points[int(actor.team)] - points[int(actor.team.other)]
                    )
                    continuation_hashes.add(canonical_sha256(continuation_trace))
                worlds.append(
                    ReducedWorldOutcomeV1.build(
                        engine.state_hash(hidden_state), margins
                    )
                )
            problem = ReducedActorSafePolicyProblemV1.build(
                information_state_hash=root_information_hash,
                actor=actor.name.lower(),
                coverage_method=cohort.method + "+policy_conditioned",
                worlds=worlds,
                profile=self.policy_profile,
            )
            policy = OrToolsReducedPolicyProofV1().prove(problem)
            reference = exhaustive_reference_policy_proof(problem)
            if (
                reference.selected_action != policy.selected_action
                or tuple(item.to_dict() for item in reference.action_evidence)
                != tuple(item.to_dict() for item in policy.action_evidence)
            ):
                raise AssertionError(
                    "CP-SAT strategic root selection differs from its reference"
                )
        except InterruptedError:
            if cancel_check is not None and cancel_check():
                raise
            return None, {
                "schemaVersion": 1,
                "method": self.descriptor.id,
                "selectionReason": "actor_safe_cp_sat_unavailable",
                "profileHash": self.profile.sha256,
                "qualificationBlocked": True,
                "failure": "selection_timeout",
                "failureMessage": "strategic play selection exceeded its time bound",
                "hiddenStateIdentifiersExposed": False,
            }
        except (RuntimeError, ValueError) as exc:
            return None, {
                "schemaVersion": 1,
                "method": self.descriptor.id,
                "selectionReason": "actor_safe_cp_sat_unavailable",
                "profileHash": self.profile.sha256,
                "qualificationBlocked": True,
                "failure": type(exc).__name__,
                "failureMessage": str(exc),
                "hiddenStateIdentifiersExposed": False,
            }

        exhaustive = cohort.coverage_complete
        exact = exhaustive and policy.all_actions_evaluated
        blocked = self.profile.require_exact_for_qualification and not exact
        alternatives = [
            {
                "action": item.action.code,
                "worstMargin": item.worst_margin,
                "bestMargin": item.best_margin,
                "totalMargin": item.total_margin,
                "preservingWorlds": item.preserving_worlds,
                "worldCount": item.world_count,
            }
            for item in policy.action_evidence
        ]
        return policy.selected_action, {
            "schemaVersion": 1,
            "method": self.descriptor.id,
            "selectionReason": "actor_safe_cp_sat_maximin_then_total",
            "profileHash": self.profile.sha256,
            "proofHash": canonical_sha256({
                "problemHash": problem.sha256,
                "policyProofHash": policy.sha256,
                "referenceProofHash": reference.sha256,
                "continuationPolicy": self.profile.continuation_policy,
                "continuationHashes": sorted(continuation_hashes),
            }),
            "modelHash": policy.model_hash,
            "informationStateHash": problem.information_state_hash,
            "coverageMethod": cohort.method,
            "hiddenWorldCount": len(cohort.states),
            "coverageComplete": cohort.coverage_complete,
            "allAlternativesEvaluated": policy.all_actions_evaluated,
            "actorChoiceBoundOnce": policy.actor_choice_bound_once,
            "strategicTruth": (
                "exact_policy_conditioned_actor_safe"
                if exact else "provisional_policy_conditioned_actor_safe"
            ),
            "modelFeasibilityStatus": policy.model_feasibility_status,
            "recommendedAction": policy.selected_action.code,
            "alternatives": alternatives,
            "continuationPolicy": self.profile.continuation_policy,
            "continuationPolicyHash": canonical_sha256({
                "id": self.profile.continuation_policy,
                "rulesHash": engine.rules.sha256,
            }),
            "continuationParity": True,
            "qualificationBlocked": blocked,
            "proofScope": "bounded_late_game_actor_safe_policy_conditioned",
            "hiddenStateIdentifiersExposed": False,
        }
