"""Unwired adapter from a trusted trace decision to DecisionAudit V2 evidence."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from baloot_platform.analysis.models import InformationState
from baloot_platform.challenge.replay import CompleteTraceReplayerV1
from baloot_platform.challenge.trace import CompleteGameTraceV1
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.models import RoundState
from baloot_platform.core.rules import RuleBook

from .best_response_proof import (
    BEST_RESPONSE_ENGINE_ID,
    BEST_RESPONSE_ENGINE_VERSION,
    EngineReducedBestResponseProblemBuilderV1,
    OrToolsBoundedBestResponseProofV1,
    ReducedBestResponseProfileV1,
)
from .contracts import canonical_sha256
from .decision_audit import InformationSetEvidenceV1, SequentialResponseEvidenceV1


_SERVICE_COHORT_METHODS = frozenset(
    {
        "exhaustive_constrained_hidden_deals",
        "exhaustive_project_conditioned_hidden_deals",
        "constructive_uniform_hidden_deals",
        "constructive_uniform_project_conditioned_hidden_deals",
    }
)


@dataclass(frozen=True, slots=True)
class DecisionProofResourcesV1:
    maximum_worlds: int = 128
    maximum_nodes: int = 20_000
    maximum_plies: int = 12
    maximum_wall_seconds: float = 30.0
    maximum_subproblem_seconds: float = 2.0

    def __post_init__(self) -> None:
        if min(self.maximum_worlds, self.maximum_nodes, self.maximum_plies) < 1:
            raise ValueError("decision-proof count bounds must be positive")
        if self.maximum_wall_seconds <= 0 or self.maximum_subproblem_seconds <= 0:
            raise ValueError("decision-proof time bounds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "maximumWorlds": self.maximum_worlds,
            "maximumNodes": self.maximum_nodes,
            "maximumPlies": self.maximum_plies,
            "maximumWallSeconds": self.maximum_wall_seconds,
            "maximumSubproblemSeconds": self.maximum_subproblem_seconds,
        }


@dataclass(frozen=True, slots=True)
class DecisionAuditV2SolverEvidenceV1:
    decision_id: str
    ordinal: int
    actor: str
    actual_action: dict[str, Any]
    recommended_action: dict[str, Any]
    alternatives: tuple[dict[str, Any], ...]
    information_set_evidence: InformationSetEvidenceV1
    sequential_response_evidence: SequentialResponseEvidenceV1
    truth_label: str
    qualification_status: str
    tactical_witnesses: tuple[dict[str, Any], ...]
    proof_hash: str
    provenance: dict[str, Any]
    resource_evidence: dict[str, Any]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 2,
            "evidenceSchema": "decision_audit_v2_solver_evidence_v1",
            "decisionId": self.decision_id,
            "ordinal": self.ordinal,
            "actor": self.actor,
            "actualAction": dict(self.actual_action),
            "recommendedAction": dict(self.recommended_action),
            "alternatives": [dict(item) for item in self.alternatives],
            "informationSetEvidence": self.information_set_evidence.to_dict(),
            "sequentialResponseEvidence": self.sequential_response_evidence.to_dict(),
            "truthLabel": self.truth_label,
            "qualificationStatus": self.qualification_status,
            "tacticalWitnesses": [dict(item) for item in self.tactical_witnesses],
            "proofHash": self.proof_hash,
            "provenance": dict(self.provenance),
            "resourceEvidence": dict(self.resource_evidence),
            "hiddenStateIdentifiersExposed": False,
            "sha256": self.sha256,
        }


class TraceDecisionPolicyProofAdapterV1:
    """Prepare solver evidence without writing storage or calling services."""

    def __init__(self, rules: RuleBook):
        self.rules = rules

    def adapt(
        self,
        *,
        trace: CompleteGameTraceV1,
        decision_id: str,
        hidden_worlds: Sequence[RoundState],
        coverage_method: str,
        profile: ReducedBestResponseProfileV1 | None = None,
        resources: DecisionProofResourcesV1 | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> DecisionAuditV2SolverEvidenceV1:
        limits = resources or DecisionProofResourcesV1()
        started = time.monotonic()

        def cancelled() -> bool:
            return bool(
                (cancel_check is not None and cancel_check())
                or time.monotonic() - started >= limits.maximum_wall_seconds
            )

        if cancelled():
            raise InterruptedError("decision proof cancelled before replay")
        if trace.rules_hash != self.rules.sha256:
            raise ValueError("trace and proof adapter rules do not match")
        cohort_method = str(coverage_method).split("+", 1)[0]
        if cohort_method not in _SERVICE_COHORT_METHODS:
            raise ValueError("decision proof requires a governed actor-safe cohort method")
        if not hidden_worlds or len(hidden_worlds) > limits.maximum_worlds:
            raise ValueError("actor-safe hidden-world cohort exceeds its governed bound")
        replay = CompleteTraceReplayerV1(self.rules).replay(trace)
        try:
            replayed = next(
                item for item in replay.decisions if item.decision.decision_id == decision_id
            )
        except StopIteration as exc:
            raise KeyError(decision_id) from exc
        record = replayed.decision
        if record.decision_kind != "play":
            raise ValueError("the reduced policy adapter currently supports play decisions")
        if record.forced:
            raise ValueError("a best-response policy proof requires a non-forced decision")

        effective_profile = profile or ReducedBestResponseProfileV1()
        if effective_profile.maximum_nodes > limits.maximum_nodes:
            effective_profile = replace(effective_profile, maximum_nodes=limits.maximum_nodes)
        engine = BalootEngine(self.rules)
        problem = EngineReducedBestResponseProblemBuilderV1().build(
            states=hidden_worlds,
            engine=engine,
            actor=record.acting_seat,
            coverage_method=coverage_method,
            profile=effective_profile,
            maximum_plies=limits.maximum_plies,
            cancel_check=cancelled,
        )
        replay_information = InformationState.from_complete(
            replayed.state,
            record.acting_seat,
        )
        if canonical_sha256(replay_information.to_dict()) != problem.information_state_hash:
            raise ValueError("hidden-world cohort does not match the trace decision projection")
        actual_card = Card.parse(dict(record.chosen_action))
        proof = OrToolsBoundedBestResponseProofV1().prove(
            problem,
            actual_action=actual_card,
            cancel_check=cancelled,
            maximum_seconds_per_subproblem=min(
                limits.maximum_subproblem_seconds,
                limits.maximum_wall_seconds,
            ),
        )
        if cancelled():
            raise InterruptedError("decision proof exceeded its wall-time bound")

        exact = proof.policy_proof.strategic_truth == "exact_reduced_actor_safe"
        alternatives = tuple(
            {
                "action": item.action.to_dict(),
                "expectedRoundMargin": item.total_margin / item.world_count,
                "worstRoundMargin": item.worst_margin,
                "bestRoundMargin": item.best_margin,
                "preservingWorlds": item.preserving_worlds,
                "worldCount": item.world_count,
                "searchCompletionStatus": "exact" if exact else "bounded_complete",
                "exactShare": 1.0 if exact else 0.0,
                "solverStatus": proof.policy_proof.model_feasibility_status,
            }
            for item in proof.policy_proof.action_evidence
        )
        legal_actions = tuple(card.to_dict() for card in engine.legal_actions(replayed.state))
        information_evidence = InformationSetEvidenceV1.build(
            observer=record.acting_seat,
            information_state=replay_information.to_dict(),
            legal_actions=legal_actions,
            alternatives=alternatives,
            forced=False,
            analysis_hash=proof.sha256,
            method=coverage_method,
            hidden_deal_support=len(hidden_worlds),
            effective_sample_size=float(len(hidden_worlds)),
        )
        sequential_evidence = SequentialResponseEvidenceV1.build(
            actual_action=actual_card.to_dict(),
            recommended_action=proof.policy_proof.selected_action.to_dict(),
            legal_actions=legal_actions,
            alternatives=alternatives,
            information_set=information_evidence,
            expected_regret=float(
                proof.regret.maximum_paired_regret if proof.regret is not None else 0
            ),
            forced=False,
        )
        witnesses = tuple(
            item
            for item in (
                (
                    {
                        "kind": "only_winning_move",
                        "evidence": proof.policy_proof.only_winning_move.to_dict(),
                    }
                    if proof.policy_proof.only_winning_move is not None
                    else None
                ),
                (
                    {
                        "kind": "small_card_big_gain",
                        "evidence": proof.small_card_big_gain.to_dict(),
                    }
                    if proof.small_card_big_gain is not None
                    else None
                ),
            )
            if item is not None
        )
        provenance = {
            "traceHash": trace.sha256,
            "gameHash": trace.game_hash,
            "rulesHash": trace.rules_hash,
            "bundleHash": trace.bundle_hash,
            "engineVersion": trace.engine_version,
            "solverEngine": BEST_RESPONSE_ENGINE_ID,
            "solverVersion": BEST_RESPONSE_ENGINE_VERSION,
            "solverBackend": effective_profile.solver_backend,
            "profileHash": effective_profile.sha256,
            "coverageMethod": coverage_method,
        }
        resource_evidence = {
            **limits.to_dict(),
            "hiddenWorldsUsed": len(hidden_worlds),
            "continuationNodes": sum(item.node_count for item in proof.continuations),
            "cancelled": False,
        }
        core = {
            "schemaVersion": 2,
            "evidenceSchema": "decision_audit_v2_solver_evidence_v1",
            "decisionId": record.decision_id,
            "ordinal": record.ordinal,
            "actor": record.acting_seat.name.lower(),
            "actualAction": actual_card.to_dict(),
            "recommendedAction": proof.policy_proof.selected_action.to_dict(),
            "alternatives": list(alternatives),
            "informationSetEvidence": information_evidence.to_dict(),
            "sequentialResponseEvidence": sequential_evidence.to_dict(),
            "truthLabel": proof.policy_proof.strategic_truth,
            "qualificationStatus": proof.qualification_status,
            "tacticalWitnesses": list(witnesses),
            "proofHash": proof.sha256,
            "provenance": provenance,
            "resourceEvidence": resource_evidence,
            "hiddenStateIdentifiersExposed": False,
        }
        return DecisionAuditV2SolverEvidenceV1(
            record.decision_id,
            record.ordinal,
            record.acting_seat.name.lower(),
            actual_card.to_dict(),
            proof.policy_proof.selected_action.to_dict(),
            alternatives,
            information_evidence,
            sequential_evidence,
            proof.policy_proof.strategic_truth,
            proof.qualification_status,
            witnesses,
            proof.sha256,
            provenance,
            resource_evidence,
            canonical_sha256(core),
        )
