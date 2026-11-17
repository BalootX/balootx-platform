from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import threading
import time
import uuid
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from baloot_platform.agents.profiles import canonical_plain_profile
from baloot_platform.analysis.decision_scores import (
    BID_DECISION_OBJECTIVE_V1,
    BID_DECISION_TIE_BREAK_V1,
    PLAY_DECISION_TIE_BREAK_V1,
    PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
    canonical_json_bytes_v1,
    validate_primary_decision_score_evidence_v1,
)
from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.search_errors import SearchTimeBudgetExceeded
from baloot_platform.challenge.analysis_pipeline import TraceAnalysisResultV1
from baloot_platform.challenge.decision_cohort import (
    PRIMARY_DECISION_COHORT_SALT_V1,
    derive_actor_observer_cohort_seed_v1,
)
from baloot_platform.challenge.constraint_deal_search import (
    DealConstraintSpecV1,
    OrToolsCpSatDealGeneratorV1,
)
from baloot_platform.challenge.evidence_v3 import TruthMode, canonical_sha256
from baloot_platform.challenge.finalist_adjudication import (
    frozen_adaptive_finalist_profile_hash_v1,
)
from baloot_platform.challenge.oracle import (
    ActualDealOracleVerifierV1,
    TacticalVerificationResultV1,
)
from baloot_platform.challenge.tactical_evidence import (
    TacticalDetectorProfileV1,
    TacticalEvidenceV1,
    TacticalKind,
)
from baloot_platform.challenge.replay import TraceDecisionProjectionStoreV1
from baloot_platform.challenge.quality_diversity import QualityDiversityProfileV1
from baloot_platform.challenge.simulation import (
    NaturalLaffaSimulatorV1,
    bidding_machine_projection_v1,
)
from baloot_platform.challenge.trace import CompleteGameTraceV1
from baloot_platform.core.cards import FULL_DECK, Card
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, GamePhase, Rank, Seat, Suit
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.project_ir import ProjectDomainIRV1
from baloot_platform.core.scoring import resolve_counting_team
from baloot_platform.core.exact_state_view import build_observer_safe_exact_state_v1
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.core.rule_source_catalog import load_bound_rule_source_catalog
from baloot_platform.config import platform_root
from baloot_platform.plugins.base import PluginDescriptor
from baloot_platform.version import __version__
from baloot_platform.adjudication.research import AdjudicationResearchAuthorityV1

from .contracts import (
    BalootDomainSpecV2,
    ResearchEstimateV1,
    ResearchRecipeV1,
    ResearchRecipeV2,
    ResearchRecipeV3,
    SemanticDisagreementV2,
    build_research_recipe,
    build_research_configuration_selector_v1,
    effective_disagreement_category,
    research_recipe_configuration_v1,
    truth_rank,
)

from .construction_pre_screen import (
    ConstructionPreScreenProfileV1,
    ConstructionPreScreenV1,
    TacticalViabilityPreScreenProfileV1,
    TacticalViabilityPreScreenV1,
)
from .domain_compiler import BalootDomainSpecV3
from .execution_context import (
    ResearchExecutionContextFactoryV1,
    ResearchExecutionContextV1,
)
from .decision_audit import (
    DECISION_AUDIT_V2_MEDIA_TYPE,
    InformationSetEvidenceV1,
    SequentialResponseEvidenceV1,
    decision_consistency_qualification_failures,
    load_decision_audit_v2,
    validate_governed_decision_audit_v2,
    validate_selected_decision_audit_v2,
)
from .decision_selection import (
    apply_decision_selection_qualification_v1,
    evaluate_decision_selection_v1,
)
from .auction_compiler import AuctionPlanV1, OrToolsCpSatAuctionCompilerV1
from .audit_trace import COMPLETE_TRACE_MEDIA_TYPE, ResearchAuditTraceRepositoryV1
from .escalation_compiler import EscalationPlanV1, OrToolsCpSatEscalationCompilerV1
from .reference_escalation import (
    ReferenceEscalationInterpreterV1,
    ReferenceEscalationStateV1,
)
from .project_compiler import (
    CompiledProjectV1,
    OrToolsCpSatProjectCompilerV1,
    ProjectPlanV1,
)
from .play_compiler import OrToolsCpSatPlayCompilerV1, PlayPlanV1
from .reference_project import ReferenceProjectInterpreterV1
from .reference_scoring import ReferenceScoringInterpreterV1
from .performance_stats import (
    CALLING_THREAD_CPU_BOUNDARY,
    PROCESS_PEAK_RSS_BOUNDARY,
    build_research_environment_declaration,
    build_research_performance_by_lane,
    build_research_performance_rollup,
    build_sequential_search_receipt_page,
    build_sequential_search_receipt_rollup,
    process_lifetime_peak_rss_bytes,
    qualified_scenario_identity_hash_v1,
    validate_v4_partial_performance_receipt,
    validate_v4_performance_receipt,
)
from .sharding import ResearchShardSpecV1, partition_research_shards
from .result_diversity import select_research_results
from .result_library import validate_scenario_result_authority_v1
from .sequential_candidate_search import (
    SequentialCandidateEvaluationV1,
    SequentialCandidateOutcomeKindV1,
    run_sequential_candidate_search_v1,
)
from .recipe_api import (
    RecipeListCursorV1,
    recipe_mutation_intent_hash,
    replay_recipe_mutation_receipt,
    validate_idempotency_key,
)
from .recipe_conversion import preview_v3_to_v4_conversion
from .strategic_play import (
    ActorSafeCpSatPlayProfileV1,
    ActorSafeCpSatPlaySelectorV1,
)
from .strategic_auction import (
    ActorSafeCpSatAuctionProfileV1,
    ActorSafeCpSatAuctionSelectorV1,
)
from .exact_state import ExactStateNormalizerV1
from .recipe_v4 import ResearchRecipeV4
from .scenario_contracts import (
    ExactTracePrefixV1,
    LINE_UNIVERSES,
    OPERATOR_PROOF_BOUNDARY_EVALUATORS,
    PROOF_CLAIMS,
    PROOF_HORIZON_UNITS,
    PROOF_MODES,
    PROOF_STRATEGY_CLASSES,
    PROOF_UTILITIES,
    PROOF_VISIBILITIES,
    SCENARIO_DECISION_KINDS,
    SCENARIO_SECTIONS,
    ScenarioDiagnosticV1,
    canonical_sha256 as scenario_sha256,
)
from .scenario_backend import (
    ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3,
    OrToolsScenarioResearchBackendV2,
    materialize_default_policy_profiles_v1,
)
from .ortools_research_executor import operator_candidate_progress_summary_v1
from .scenario_execution_capabilities import (
    build_scenario_execution_capabilities_v1,
    project_scenario_configuration_execution_v1,
)
from .ortools_exact_state import ExactNormalizationStopped, OrToolsExactStateNormalizerV1
from .scenario_evidence import ScenarioRuntimeVerifierV1
from .scenario_registry import (
    SCENARIO_RULE_APPLICABILITY_CODES,
    ScenarioPredicateRegistryV1,
)
from .scenario_search import ScenarioSearchExecutorV1
from .proof_service import (
    ScenarioProofBudgetRegistryV1,
    UnifiedScenarioProofServiceV1,
)
from .scenario_result import (
    SCENARIO_ARTIFACT_MEDIA_TYPES,
    build_constraint_evidence_v1,
    build_scenario_result_v4,
    canonical_json_bytes as scenario_json_bytes,
    scenario_decision_audit_v1,
    validate_scenario_result_v4,
)
from .result_authority import (
    ORTOOLS_AUTHORITY_ARTIFACT_MEDIA_TYPES,
    ORTOOLS_RESULT_IDENTITY_KIND,
    ORTOOLS_RESULT_IDENTITY_V3_KIND,
    ORTOOLS_RESEARCH_BACKENDS,
    ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES,
    ResearchAuthorityV1,
    _build_ortools_research_result_v3,
    build_ortools_research_result_v3,
    native_featured_play_proof_supported_v1,
    validate_ortools_result_artifact_documents_v3,
    validate_ortools_result_sections_v1,
    validate_native_interest_artifact_v1,
)
from .runtime_compatibility import (
    check_runtime_compatibility_v1,
    quarantine_runtime_compatibility_failure_v1,
    require_runtime_compatibility_passed_v1,
)
from .ortools_proof import required_line_count_limbs_v1
from .scenario_predicate_manifest_v2 import (
    shared_scenario_predicate_catalog_lowering_manifest_v2,
)
from .tail_first_stage_planner import TailFirstStagePlannerV1

ResearchRecipe = (
    ResearchRecipeV1 | ResearchRecipeV2 | ResearchRecipeV3 | ResearchRecipeV4
)

_TRACE_ANALYSIS_MEDIA_TYPE = "application/vnd.balootx.trace-analysis-v1+json"
_PRIMARY_DECISION_SCORE_ANALYSIS_MEDIA_TYPE = (
    "application/vnd.balootx.primary-decision-score-analysis-v1+json"
)
_DECISION_SCORE_PARITY_MEDIA_TYPE = (
    "application/vnd.balootx.decision-score-parity-v1+json"
)
_RESULT_ARTIFACT_REFERENCE_KEYS = frozenset(
    {"uri", "sha256", "bytes", "mediaType", "name", "ownerKind", "ownerHash"}
)
_CAS_ARTIFACT_REFERENCE_KEYS = frozenset(
    {"uri", "sha256", "bytes", "mediaType", "name"}
)

_ORTOOLS_CAMPAIGN_RECEIPT_FIELDS = frozenset(
    {
        "schemaVersion",
        "status",
        "requestedResultCount",
        "maximumCandidates",
        "symmetryPolicyId",
        "distancePolicyId",
        "minimumDistance",
        "completedCandidates",
        "executionHashes",
        "executionSemanticHashes",
        "acceptedExecutionHashes",
        "acceptedSemanticHashes",
        "noGoodVectors",
        "variantReceipts",
        "wallTimeSeconds",
        "cpuSeconds",
        "semanticHash",
        "sha256",
    }
)
_ORTOOLS_COORDINATED_CAMPAIGN_RECEIPT_FIELDS = _ORTOOLS_CAMPAIGN_RECEIPT_FIELDS | {
    "coordinatorReceipt"
}
_ORTOOLS_COORDINATOR_RECEIPT_FIELDS = frozenset(
    {
        "schemaVersion",
        "status",
        "shards",
        "budget",
        "workerDeclaration",
        "duplicatePolicyId",
        "symmetryPolicyId",
        "distancePolicyId",
        "minimumDistance",
        "targetSelectionPolicyId",
        "attempts",
        "variants",
        "acceptedVariants",
        "semanticSha256",
        "wallTimeNs",
        "cpuTimeNs",
        "receiptSha256",
    }
)


def _project_ortools_campaign_statistics_run(
    run: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Project only hash-bound campaign timing into the legacy stats rollup."""

    identity = run.get("scenarioExecutorIdentity")
    backend = (
        dict(identity.get("backendIdentity") or {})
        if isinstance(identity, Mapping)
        else {}
    )
    value = dict(receipt)
    receipt_fields = set(value)
    if (
        backend.get("backendId") not in ORTOOLS_RESEARCH_BACKENDS
        or receipt_fields
        not in {
            _ORTOOLS_CAMPAIGN_RECEIPT_FIELDS,
            _ORTOOLS_COORDINATED_CAMPAIGN_RECEIPT_FIELDS,
        }
        or value.get("schemaVersion") != 1
        or value.get("status")
        not in {
            "completed",
            "structurally_infeasible",
            "not_found_within_budget",
            "semantic_postcondition_miss",
            "proof_failed",
            "incomplete",
        }
    ):
        raise ValueError("OR-Tools Research campaign performance receipt is invalid")
    no_goods = value["noGoodVectors"]
    if not isinstance(no_goods, list):
        raise ValueError("OR-Tools Research campaign performance receipt is invalid")
    coordinator_semantic_hash = None
    if receipt_fields == _ORTOOLS_COORDINATED_CAMPAIGN_RECEIPT_FIELDS:
        coordinator = value["coordinatorReceipt"]
        if coordinator is not None:
            if not isinstance(coordinator, Mapping):
                raise ValueError(
                    "OR-Tools Research campaign performance receipt is invalid"
                )
            coordinator = dict(coordinator)
            coordinator_core = {
                key: item
                for key, item in coordinator.items()
                if key != "receiptSha256"
            }
            accepted_variants = coordinator.get("acceptedVariants")
            budget = coordinator.get("budget")
            shards = coordinator.get("shards")
            if (
                set(coordinator) not in (
                    _ORTOOLS_COORDINATOR_RECEIPT_FIELDS,
                    _ORTOOLS_COORDINATOR_RECEIPT_FIELDS | {"cpuTimeCoverage"},
                )
                or "cpuTimeCoverage" in coordinator and (
                    type(coordinator["cpuTimeCoverage"]) is not str
                    or coordinator["cpuTimeCoverage"] not in {"terminal_measurements", "sampled_lower_bound"})
                or coordinator.get("schemaVersion") != 1
                or coordinator.get("status")
                not in {
                    "completed",
                    "not_found_within_budget",
                    "budget_exhausted",
                    "cancelled",
                    "incomplete",
                }
                or not isinstance(accepted_variants, list)
                or not isinstance(budget, Mapping)
                or not isinstance(shards, list)
                or coordinator.get("receiptSha256") != scenario_sha256(coordinator_core)
            ):
                raise ValueError(
                    "OR-Tools Research campaign performance receipt is invalid"
                )
            accepted_variant_hashes = []
            accepted_result_hashes = []
            for variant in accepted_variants:
                if not isinstance(variant, Mapping) or not isinstance(
                    variant.get("semanticHash"), str
                ):
                    raise ValueError(
                        "OR-Tools Research campaign performance receipt is invalid"
                    )
                accepted_variant_hashes.append(str(variant["semanticHash"]))
                semantic_document = variant.get("semanticDocument")
                semantic_result_hash = (
                    semantic_document.get("semanticResultHash")
                    if isinstance(semantic_document, Mapping)
                    else variant.get("semanticHash")
                )
                if not isinstance(semantic_result_hash, str):
                    raise ValueError(
                        "OR-Tools Research campaign performance receipt is invalid"
                    )
                accepted_result_hashes.append(semantic_result_hash)
            coordinator_semantic_hash = coordinator.get("semanticSha256")
            if coordinator["status"] in {"completed", "not_found_within_budget"}:
                # Historical proposal streams sealed only their operational receipt.
                legacy_stream = (coordinator_semantic_hash is None
                    and coordinator.get("cpuTimeCoverage") == "sampled_lower_bound"
                    and bool(shards) and all(
                        isinstance(item, Mapping) and item.get("runCode") == "proposal_stream"
                        and item.get("seedCount") == item.get("maximumAttempts") == 1
                        and item.get("seedStart") == item.get("shardIndex")
                        and scenario_sha256(item) == scenario_sha256(ResearchShardSpecV1.from_dict(item).to_dict())
                        for item in shards))
                coordinator_semantic_core = {
                    "schemaVersion": 1,
                    "status": coordinator["status"],
                    "recipeHash": (
                        shards[0].get("recipeHash")
                        if shards and isinstance(shards[0], Mapping)
                        else None
                    ),
                    "shardPartitions": [
                        {
                            "shardIndex": item.get("shardIndex"),
                            "seedStart": item.get("seedStart"),
                            "seedCount": item.get("seedCount"),
                            "maximumAttempts": item.get("maximumAttempts"),
                        }
                        if isinstance(item, Mapping)
                        else None
                        for item in shards
                    ],
                    "requestedResultCount": budget.get("resultCount"),
                    "duplicatePolicyId": coordinator["duplicatePolicyId"],
                    "symmetryPolicyId": coordinator["symmetryPolicyId"],
                    "distancePolicyId": coordinator["distancePolicyId"],
                    "minimumDistance": coordinator["minimumDistance"],
                    "targetSelectionPolicyId": coordinator["targetSelectionPolicyId"],
                    "acceptedSemanticHashes": accepted_variant_hashes,
                }
                if not legacy_stream and coordinator_semantic_hash != scenario_sha256(
                    coordinator_semantic_core
                ):
                    raise ValueError(
                        "OR-Tools Research campaign performance receipt is invalid"
                    )
            elif coordinator_semantic_hash is not None:
                raise ValueError(
                    "OR-Tools Research campaign performance receipt is invalid"
                )
            if accepted_result_hashes != value["acceptedSemanticHashes"]:
                raise ValueError(
                    "OR-Tools Research campaign performance receipt is invalid"
                )
    semantic_core = {
        key: value[key]
        for key in (
            "schemaVersion",
            "status",
            "requestedResultCount",
            "maximumCandidates",
            "symmetryPolicyId",
            "distancePolicyId",
            "minimumDistance",
            "executionSemanticHashes",
            "acceptedSemanticHashes",
            "variantReceipts",
        )
    }
    semantic_core["noGoodHashes"] = [scenario_sha256(item) for item in no_goods]
    if receipt_fields == _ORTOOLS_COORDINATED_CAMPAIGN_RECEIPT_FIELDS:
        semantic_core["coordinatorSemanticHash"] = coordinator_semantic_hash
    wall_seconds = value["wallTimeSeconds"]
    cpu_seconds = value["cpuSeconds"]
    attempted = run.get("attemptedCandidates")
    completed = value["completedCandidates"]
    observed_attempts: dict[tuple[str, int], int] = {}
    legacy_timeout_attempts: set[tuple[str, int]] = set()
    coordinator = value.get("coordinatorReceipt")
    if isinstance(coordinator, Mapping):
        if not isinstance(coordinator["attempts"], list):
            raise ValueError("OR-Tools Research campaign performance receipt is invalid")
        for attempt in coordinator["attempts"]:
            if not isinstance(attempt, Mapping):
                raise ValueError("OR-Tools Research campaign performance receipt is invalid")
            partial = attempt.get("partialEvidence")
            if not isinstance(partial, Mapping):
                continue
            legacy_failed = ("completedCandidates" not in partial
                and attempt.get("status") == "failed"
                and partial.get("currentStage") == "worker_failed"
                and isinstance(attempt.get("errorCode"), str)
                and partial.get("errorCode") == attempt["errorCode"])
            if "completedCandidates" not in partial and not legacy_failed:
                continue
            count = partial.get("completedCandidates", 0)
            shard_id, ordinal = attempt.get("shardId"), attempt.get("attempt")
            if (type(count) is not int or count < 0
                    or type(shard_id) is not str or not shard_id
                    or type(ordinal) is not int or ordinal < 1
                    or partial.get("shardId") != shard_id
                    or partial.get("attempt") != ordinal):
                raise ValueError("OR-Tools Research campaign performance receipt is invalid")
            key = (shard_id, ordinal)
            observed_attempts[key] = max(observed_attempts.get(key, 0), count)
            if legacy_failed or (count == 0 and attempt.get("status") == "completed"
                    and partial.get("currentStage") == "shard_budget_exhausted"
                    and partial.get("shardBudgetExhausted") is True
                    and partial.get("campaignStatus") == "incomplete"):
                legacy_timeout_attempts.add(key)
    # A cancelled worker can retain candidate progress without returning an execution.
    expected_attempted = max(completed, sum(observed_attempts.values())) if type(completed) is int else None
    # Historical workers counted empty timed-out/failed shards as one attempt.
    # Recognize only that hash-owned shape, then project actual candidate counts.
    legacy_attempted = (max(completed, sum(max(count, int(key in legacy_timeout_attempts))
        for key, count in observed_attempts.items())) if type(completed) is int else None)
    result_count = run.get("resultCount")
    accepted = value["acceptedSemanticHashes"]
    if (
        value["semanticHash"] != scenario_sha256(semantic_core)
        or value["sha256"]
        != scenario_sha256(
            {
                **semantic_core,
                "semanticHash": value["semanticHash"],
                "wallTimeSeconds": wall_seconds,
                "cpuSeconds": cpu_seconds,
                **(
                    {"coordinatorReceipt": value["coordinatorReceipt"]}
                    if receipt_fields == _ORTOOLS_COORDINATED_CAMPAIGN_RECEIPT_FIELDS
                    else {}
                ),
            }
        )
        or type(completed) is not int
        or completed < 0
        or type(attempted) is not int
        or (attempted != expected_attempted
            and not (value["status"] == "incomplete"
                and attempted in (completed, legacy_attempted)))
        or type(result_count) is not int
        or not isinstance(accepted, list)
        or result_count != len(accepted)
        or isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(float(wall_seconds))
        or wall_seconds < 0
    ):
        raise ValueError("OR-Tools Research campaign performance receipt is invalid")
    projected = dict(run)
    projected["attemptedCandidates"] = expected_attempted
    yield_counts = projected.get("scenarioYieldCounts")
    if isinstance(yield_counts, Mapping):
        for target, source in (
            ("structuralCandidates", "structurallyFeasible"),
            ("runtimeParityPassed", "legalReplay"),
            ("semanticReplayPassed", "legalReplay"),
            ("proofCompleted", "proofCompleted"),
        ):
            count = yield_counts.get(source)
            if type(count) is not int or count < 0:
                raise ValueError("OR-Tools Research campaign yield counts are invalid")
            projected[target] = count
    requested = projected.get("requestedResultCount")
    if type(requested) is int and requested >= 0:
        projected["requested"] = requested
    projected["attemptedSeeds"] = 0
    projected["elapsedSeconds"] = float(wall_seconds)
    projected.pop("cpuSeconds", None)  # Campaign CPU lacks a measurement boundary.
    return projected


class ResearchLaunchAdmissionError(ValueError):
    """A deterministic Estimate V2 refusal raised before run creation."""

    def __init__(self, diagnostic: Mapping[str, Any]):
        self.diagnostic = copy.deepcopy(dict(diagnostic))
        super().__init__(
            json.dumps(
                self.diagnostic,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )


_ORTOOLS_CANDIDATE_RECEIPT_HISTORY_LIMIT = 200
_ORTOOLS_CANDIDATE_RECEIPT_CLASSIFICATIONS = (
    "qualified",
    "rejected",
    "duplicate",
    "incomplete",
)


def _operator_incomplete_compilation_v1(value: Any) -> dict[str, Any] | None:
    """Expose diagnostics, never private traces, models or decision audits."""
    if not isinstance(value, Mapping):
        return None
    projected = {
        key: value[key]
        for key in (
            "stage", "sourceStage", "status", "exhaustionReason",
            "completionStatus", "sha256", "candidateHash", "targetActor",
            "generatedActorAtTarget", "qualityStatus", "qualityFailureStage",
            "qualityFailureCauseStage", "qualityFailureReason",
        )
        if isinstance(value.get(key), str)
    }
    if value.get("reason") == "hard_project_award_miss":
        projected["reason"] = "hard_project_award_miss"
    for key in ("targetPlayIndex", "firstConstructedPlayDivergenceIndex"):
        if type(value.get(key)) is int:
            projected[key] = value[key]
    return projected


def _operator_stage_counters_v1(value: Any) -> dict[str, int] | None:
    """Expose counts, never solver assignments or hidden-card evidence."""

    if not isinstance(value, Mapping):
        return None
    counters = {
        key: value[key]
        for key in ("tailCandidates", "proofBoundaryAttempts", "prefixLiftAttempts")
        if type(value.get(key)) is int and value[key] >= 0
    }
    return counters or None


def _operator_quality_solver_progress_v1(value: Any) -> dict[str, Any] | None:
    """Project only safe work counters, never candidate paths or hidden cards."""

    if not isinstance(value, Mapping) or value.get("kind") != "quality_solver_progress_v1":
        return None
    if value.get("phase") not in {"preparation", "master", "counterexample", "binding"}:
        return None
    if any(type(value.get(key)) is not int or value[key] < 0
           for key in ("masterCount", "queryCount", "bindingCount")):
        return None
    result = {key: value[key] for key in ("phase", "masterCount", "queryCount", "bindingCount")}
    for key, allowed in (("decisionKind", SCENARIO_DECISION_KINDS),
                         ("proofPhase", {"best", "selected", "fallback"})):
        if value.get(key) in allowed:
            result[key] = value[key]
    return result


def _operator_family_progress_v1(event: Mapping[str, Any]) -> dict[str, Any]:
    """Expose construction work/rejections, never an auction or card assignment."""
    status = event.get('constructionStatus')
    if (event.get('type') != 'progress' or event.get('currentStage') != 'family_proposal'
            or type(status) is not str or status not in {'proposal', 'seed_miss', 'infeasible', 'incomplete'}):
        return {}
    result = {'constructionStatus': status}
    reason = event.get('constructionReason')
    if type(reason) is str and reason in {
        'opening_mismatch', 'negative_project_mismatch', 'history_suit_count_mismatch',
        'decision_actor_mismatch', 'observer_suit_count_mismatch', 'history_expression_infeasible',
        'fixed_deal_projects_infeasible', 'fixed_deal_projects_incomplete',
        'auction_quality_incomplete', 'cancelled_or_budget',
    }:
        result['constructionReason'] = reason
    core = event.get('constructionInfeasibleCore')
    if (status in {'seed_miss', 'infeasible'} and isinstance(core, list) and len(core) <= 2
            and all(type(item) is str and item in {'decision_actor', 'scenario_expression'} for item in core)
            and len(set(core)) == len(core)):
        result['constructionInfeasibleCore'] = list(core)
    seconds = event.get('constructionElapsedSeconds')
    if type(seconds) in {int, float}:
        try:
            if math.isfinite(seconds) and seconds >= 0:
                result['constructionElapsedSeconds'] = seconds
        except OverflowError:
            pass  # Malformed persisted counters never enter the public projection.
    return result


def _operator_research_run_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded Foundry view while full evidence stays hash-owned."""

    source = dict(value)
    receipt = source.get("performanceReceipt")
    if isinstance(receipt, Mapping) and "coordinatorReceipt" in receipt:
        source.pop("performanceReceipt")
    events = source.pop("events", ())
    source.pop("auctionOpenings", None)
    projected = copy.deepcopy(source)
    if source.get("status") == "completed":
        openings = []
        seen = set()
        for event in events:
            if (not isinstance(event, Mapping) or event.get("type") != "progress"
                    or event.get("currentStage") != "family_proposal"):
                continue
            seed, dealer = event.get("screenedSeed"), event.get("screenedDealer")
            qualification = event.get("auctionQualification")
            if (type(seed) is not int or seed < 0 or type(dealer) is not str
                    or dealer not in {seat.name.lower() for seat in Seat}
                    or not isinstance(qualification, Mapping)
                    or qualification.get("status") != "complete"
                    or qualification.get("contract") != "sun"
                    or qualification.get("allPassed") is not False
                    or (seed, dealer) in seen):
                continue
            seen.add((seed, dealer))
            openings.append({"seed": seed, "dealer": dealer})
        if openings:
            projected["auctionOpenings"] = openings
    progress = projected.get("campaignProgress")
    if projected.get("status") == "running" and isinstance(progress, dict):
        for event in reversed(events):
            if not isinstance(event, Mapping) or event.get("type") != "progress":
                continue
            accounting = event.get("accounting")
            if not isinstance(accounting, Mapping) or accounting.get("elapsedSeconds") != progress.get("elapsedSeconds"):
                break
            try:
                sampled_at = datetime.fromisoformat(event["at"])
                age = max(0.0, (datetime.now(timezone.utc) - sampled_at).total_seconds())
                elapsed = float(progress["elapsedSeconds"]) + age
                limit = float(progress["maximumCampaignSeconds"])
            except (KeyError, TypeError, ValueError):
                break
            # Display-only clock; persisted receipts and solver deadlines stay unchanged.
            progress["elapsedSeconds"] = elapsed
            progress["remainingCampaignSeconds"] = max(0.0, limit - elapsed)
            break
    for key in ("campaignProgress", "campaignAccounting"):
        accounting = projected.get(key)
        if isinstance(accounting, Mapping):
            compact = copy.deepcopy(dict(accounting))
            compact["latestIncompleteCompilation"] = (
                _operator_incomplete_compilation_v1(
                    accounting.get("latestIncompleteCompilation")
                )
            )
            projected[key] = compact
    projected["events"] = [
        {
            key: copy.deepcopy(event[key])
            for key in (
                "type",
                "sequence",
                "currentStage",
                "candidateStatus",
                "proofBoundaryStatus",
                "proofBoundaryReason",
                "scenarioResearchStatus",
                "resultCount",
                "quarantineCount",
                "compilerDisagreement",
                "at",
            )
            if key in event and (
                key not in {"proofBoundaryStatus", "proofBoundaryReason"}
                or (
                    type(event[key]) is str
                    and event[key] in {
                        "feasible", "infeasible", "incomplete",
                        "tail_boundary_infeasible", "tail_boundary_budget",
                    }
                )
            )
        } | _operator_family_progress_v1(event)
        for event in events
        if isinstance(event, Mapping)
    ]
    return projected


def _operator_candidate_progress_receipt_v1(
    evidence: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Project one observer-safe candidate status into a stable live receipt."""

    status = evidence.get("candidateStatus")
    if not isinstance(status, str) or not status:
        return None
    if status == "qualified":
        classification = "qualified"
    elif status in {"duplicate", "symmetry", "decision_state_distance"}:
        classification = "duplicate"
    elif status in {
        "incomplete",
        "proof_incomplete",
        "search_incomplete",
        "compiler_disagreement",
    }:
        classification = "incomplete"
    else:
        classification = "rejected"
    stage = str(evidence.get("currentStage") or "candidate")
    candidate_hash = evidence.get("candidateHash")
    core: dict[str, Any] = {
        "schemaVersion": 1,
        "classification": classification,
        "status": status,
        "stage": stage,
        "candidateHash": (
            None if stage == "variant_selection_terminal" else candidate_hash
        ),
    }
    if stage == "variant_selection_terminal" and candidate_hash is not None:
        core["variantSemanticHash"] = copy.deepcopy(candidate_hash)
    for key in (
        "shardIndex",
        "shardId",
        "attempt",
        "executionOrdinal",
        "retryAttempt",
        "completedCandidates",
        "qualifiedCandidates",
        "proofCompletionStatus",
        "nearestAcceptedDistance",
    ):
        if evidence.get(key) is not None:
            core[key] = copy.deepcopy(evidence[key])
    for key in ("candidateSummary",):
        value = evidence.get(key)
        if isinstance(value, Mapping):
            core[key] = copy.deepcopy(dict(value))
    incomplete = _operator_incomplete_compilation_v1(evidence.get("incompleteCompilation"))
    if incomplete is not None:
        core["incompleteCompilation"] = incomplete
    if isinstance(evidence.get("candidateAttemptReason"), str):
        core["candidateAttemptReason"] = evidence["candidateAttemptReason"]
    return {**core, "receiptId": scenario_sha256(core)}


def _operator_candidate_progress_snapshot_v1(
    receipts: Iterable[Mapping[str, Any]],
    *,
    history_limit: int = _ORTOOLS_CANDIDATE_RECEIPT_HISTORY_LIMIT,
) -> dict[str, Any]:
    """Return separate aggregate counts and a deterministic bounded history."""

    if type(history_limit) is not int or history_limit < 1:
        raise ValueError("candidate receipt history limit must be a positive integer")
    values = [copy.deepcopy(dict(item)) for item in receipts]
    terminal_candidate_keys = {
        (item.get("shardId"), item.get("attempt"), item.get("executionOrdinal"),
         item.get("candidateHash"), item.get("status"))
        for item in values
        if item.get("stage") == "candidate_terminal"
        and type(item.get("executionOrdinal")) is int
        and isinstance(item.get("candidateHash"), str)
    }
    # Worker and coordinator can both report the same deadline. Keep the
    # candidate terminal receipt, not a second candidate for worker shutdown.
    values = [item for item in values if not (
        item.get("stage") == "shard_budget_exhausted"
        and (item.get("shardId"), item.get("attempt"), item.get("executionOrdinal"),
             item.get("candidateHash"), item.get("status")) in terminal_candidate_keys
    )]
    completed_execution_keys = {
        (
            item.get("shardId"),
            item.get("attempt"),
            item.get("status"),
        )
        for item in values
        if item.get("stage") == "candidate_terminal"
        and type(item.get("executionOrdinal")) is int
    }
    values = [
        item
        for item in values
        if not (
            item.get("stage") == "candidate_terminal"
            and item.get("executionOrdinal") is None
            and (
                item.get("shardId"),
                item.get("attempt"),
                item.get("status"),
            )
            in completed_execution_keys
        )
    ]
    values.sort(
        key=lambda item: (
            int(item.get("shardIndex"))
            if type(item.get("shardIndex")) is int
            else 2**31,
            str(item.get("shardId") or ""),
            int(item.get("attempt")) if type(item.get("attempt")) is int else 0,
            int(item.get("executionOrdinal"))
            if type(item.get("executionOrdinal")) is int
            else 0,
            str(item.get("stage") or ""),
            str(item.get("status") or ""),
            str(item.get("candidateHash") or ""),
            str(item.get("variantSemanticHash") or ""),
            str(item.get("receiptId") or ""),
        )
    )
    status_counts = {
        status: sum(item.get("status") == status for item in values)
        for status in sorted(
            {
                str(item["status"])
                for item in values
                if isinstance(item.get("status"), str) and item["status"]
            }
        )
    }
    return {
        "candidateReceiptCount": len(values),
        "candidateReceiptCounts": {
            classification: sum(
                item.get("classification") == classification for item in values
            )
            for classification in _ORTOOLS_CANDIDATE_RECEIPT_CLASSIFICATIONS
        },
        "candidateStatusCounts": status_counts,
        "candidateReceiptHistoryLimit": history_limit,
        "candidateReceiptHistoryTruncated": len(values) > history_limit,
        "candidateReceipts": values[-history_limit:],
    }


def _canonical_research_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _v4_output_receipt(
    recipe: ResearchRecipeV4,
    *,
    retained_count: int,
    qualified_count: int,
    selected_count: int,
) -> dict[str, Any]:
    render_policy = copy.deepcopy(dict(recipe.render_policy))
    if "videoCount" not in render_policy:
        render_policy["videoCount"] = (
            0 if render_policy.get("mode") == "disabled" else 1
        )
    core = {
        "schemaVersion": 1,
        "recipeHash": recipe.sha256,
        "output": copy.deepcopy(dict(recipe.output)),
        "renderPolicy": render_policy,
        "requestedResultCount": int(recipe.output["resultCount"]),
        "requestedVideoCount": int(render_policy["videoCount"]),
        "retainedResultCount": int(retained_count),
        "qualifiedResultCount": int(qualified_count),
        "selectedResultCount": int(selected_count),
        "resultTargetSatisfied": selected_count >= int(recipe.output["resultCount"]),
    }
    return {**core, "sha256": scenario_sha256(core)}


def _v4_partial_performance_receipt(core: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "schemaVersion": 1,
        "measurementScope": "research_v4_cancelled_or_incomplete_run_v1",
        **copy.deepcopy(dict(core)),
    }
    value["sha256"] = scenario_sha256(value)
    return validate_v4_partial_performance_receipt(
        value,
        run_code=str(value["runCode"]),
    )


def _select_v4_results(
    recipe: ResearchRecipeV4,
    results: Iterable[Mapping[str, Any]],
):
    profile_payload = dict(recipe.diversity) or {
        "schemaVersion": 1,
        "cellDimensions": [
            "contract",
            "project",
            "decision_kind",
            "proof_mode",
        ],
        "topKPerCell": 3,
        "rejectedAuditModulus": 100,
    }
    profile_payload.pop("sha256", None)
    return select_research_results(
        results,
        profile=QualityDiversityProfileV1.build(profile_payload),
        requested=int(recipe.output["resultCount"]),
    )


def _ortools_line_count_boundaries_v1(
    campaign: Any,
    recipe: ResearchRecipeV4,
    parameters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project immutable, observer-safe line-count evidence for the run drawer."""

    rows: list[dict[str, Any]] = []
    for execution in campaign.executions:
        proof = dict(execution.proof or {})
        continuation = proof.get("continuationLines")
        if not isinstance(continuation, Mapping):
            continue
        authority = dict(execution.research_authority or {})
        core = {
            "schemaVersion": 1,
            "executionSemanticHash": execution.semantic_hash,
            "executionStatus": execution.status,
            "proofOutcomeHash": authority.get("proofOutcomeHash"),
            "requestedMode": recipe.proof.mode,
            "visibility": recipe.proof.visibility,
            "completionStatus": proof.get("completionStatus"),
            "claimSatisfied": proof.get("claimSatisfied") is True,
            "boundaryEvaluator": (
                recipe.proof.boundary_evaluator or "terminal_utility"
            ),
            "continuationLines": {
                key: copy.deepcopy(continuation.get(key))
                for key in (
                    "complete",
                    "lineUniverse",
                    "exactCount",
                    "lowerBound",
                    "upperBound",
                    "limbBase",
                    "limbCount",
                    "limbLimit",
                    "solverStatus",
                    "modelHash",
                    "exhaustionReason",
                )
            },
            "budget": {
                "profileId": recipe.proof.budget_profile,
                "proofBudgetHash": parameters["proofBudgetHash"],
                "maximumLineCountNodes": parameters["maximumLineCountNodes"],
                "maximumLineCountLimbs": parameters["maximumLineCountLimbs"],
            },
        }
        rows.append({**core, "sha256": scenario_sha256(core)})
    return rows


def _power_expression(base: int, exponent: int) -> str:
    if exponent == 0 or base == 1:
        return "1"
    return f"{base}^{exponent}"


def _power_exceeds_limit(
    base: int,
    exponent: int,
    *,
    multiplier: int,
    additive: int,
    limit: int,
) -> bool:
    """Compare additive + multiplier * base**exponent without huge integers."""

    if additive > limit:
        return True
    remaining = limit - additive
    if multiplier > remaining:
        return True
    threshold = remaining // max(1, multiplier)
    if base > 1 and exponent >= max(1, threshold.bit_length()):
        return True
    value = 1
    for _ in range(exponent):
        if base > 1 and value > threshold // base:
            return True
        value *= base
    return value > threshold


def _geometric_state_upper_bound(branching: int, depth: int) -> int:
    """States in one depth-limited tree, including its root and leaves."""

    total = 0
    level = 1
    for _ in range(depth + 1):
        total += level
        level *= max(1, branching)
    return total


def _play_branch_sequence_upper_bound_v2(
    hand_sizes: Iterable[int],
    *,
    action_multiplier: int = 1,
) -> tuple[int, ...]:
    """Largest per-ply card-action factors for the remaining finite hands."""

    factors = [
        size * action_multiplier
        for hand_size in hand_sizes
        for size in range(1, int(hand_size) + 1)
    ]
    return tuple(sorted(factors, reverse=True))


def _tree_upper_bound_from_branches_v2(branches: Iterable[int]) -> tuple[int, int]:
    nodes = 1
    frontier = 1
    for branch in branches:
        frontier *= max(1, int(branch))
        nodes += frontier
    return nodes, nodes - 1


def _ortools_model_size_upper_bound_v2(
    *,
    compiled_stages: Iterable[str],
    hard_predicate_count: int,
    requested_depth: int,
    graph_nodes: int,
    root_actions: int,
    count_limbs: int | None,
    maximum_prefix_lift_attempts: int,
) -> tuple[int, int]:
    """Conservative maximum of the separately solved construction/proof models."""

    stages = frozenset(compiled_stages)
    construction_variables = 4_096 + hard_predicate_count * 512
    construction_constraints = 8_192 + hard_predicate_count * 1_024
    if "projects" in stages:
        # The finite project lifecycle fragment is the dominant fixed stage.
        construction_variables += 20_000
        construction_constraints += 100_000
    if {"playPrefix", "state", "outcome", "proof", "tactics"} & stages:
        construction_variables += requested_depth * 1_024
        construction_constraints += requested_depth * 2_048
    construction_constraints += maximum_prefix_lift_attempts

    proof_variables = graph_nodes + root_actions + 1
    proof_constraints = graph_nodes + (2 * root_actions) + 1
    if count_limbs is not None:
        proof_variables = max(proof_variables, count_limbs * graph_nodes * 2)
        proof_constraints = max(proof_constraints, count_limbs * graph_nodes * 2)
    return (
        max(construction_variables, proof_variables),
        max(construction_constraints, proof_constraints),
    )


def _sha256_digest_v1(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 hash")
    return value


def _bind_result_artifact_owner_v1(
    reference: Mapping[str, Any],
    *,
    payload: bytes,
    media_type: str,
    owner_kind: str,
    owner_hash: str,
) -> dict[str, Any]:
    """Bind verified semantic identity without changing immutable CAS bytes."""

    raw = copy.deepcopy(dict(reference))
    if set(raw) != _CAS_ARTIFACT_REFERENCE_KEYS:
        raise ValueError("Research retained artifact reference is not closed")
    if raw.get("mediaType") != media_type:
        raise ValueError("Research retained artifact media type mismatch")
    payload_hash = hashlib.sha256(payload).hexdigest()
    if raw.get("sha256") != payload_hash or raw.get("bytes") != len(payload):
        raise ValueError("Research retained artifact CAS identity mismatch")
    semantic_hash = _sha256_digest_v1(
        owner_hash, field="Research retained artifact semantic owner hash"
    )
    return {
        **raw,
        "ownerKind": owner_kind,
        "ownerHash": semantic_hash,
    }


def _read_result_artifact_v1(artifact_store: Any, uri: str) -> bytes:
    try:
        return artifact_store.get(uri)
    except (FileNotFoundError, KeyError) as exc:
        raise RuntimeError("Research result artifact is unavailable") from exc


def _load_owned_result_json_artifact_v1(
    artifact_store: Any,
    reference: Mapping[str, Any],
    *,
    media_type: str,
    owner_kind: str,
) -> tuple[dict[str, Any], str]:
    """Load one current result artifact through exact CAS and semantic ownership."""

    raw = copy.deepcopy(dict(reference))
    if set(raw) != _RESULT_ARTIFACT_REFERENCE_KEYS:
        raise ValueError("Research result artifact reference is not closed or owned")
    if raw.get("mediaType") != media_type:
        raise ValueError("Research result artifact media type mismatch")
    artifact_hash = _sha256_digest_v1(
        raw.get("sha256"), field="Research result artifact hash"
    )
    expected_bytes = raw.get("bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 1
    ):
        raise ValueError("Research result artifact byte count is invalid")
    if raw.get("ownerKind") != owner_kind:
        raise ValueError("Research result artifact semantic owner kind mismatch")
    owner_hash = _sha256_digest_v1(
        raw.get("ownerHash"), field="Research result artifact semantic owner hash"
    )
    payload = _read_result_artifact_v1(
        artifact_store, str(raw.get("uri") or "")
    )
    if (
        len(payload) != expected_bytes
        or hashlib.sha256(payload).hexdigest() != artifact_hash
    ):
        raise ValueError("Research result artifact integrity check failed")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Research result artifact is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("Research result artifact must contain one JSON object")
    document = copy.deepcopy(dict(decoded))
    if _canonical_research_json_bytes(document) != payload:
        raise ValueError("Research result artifact is not canonical JSON")
    return document, owner_hash


def _load_result_json_artifact_v1(
    artifact_store: Any,
    reference: Mapping[str, Any],
    *,
    media_type: str,
) -> dict[str, Any]:
    """Load one closed CAS JSON reference without inventing semantic ownership."""

    raw = copy.deepcopy(dict(reference))
    if set(raw) != _CAS_ARTIFACT_REFERENCE_KEYS:
        raise ValueError("Research result CAS artifact reference is not closed")
    if raw.get("mediaType") != media_type:
        raise ValueError("Research result CAS artifact media type mismatch")
    artifact_hash = _sha256_digest_v1(
        raw.get("sha256"), field="Research result CAS artifact hash"
    )
    expected_bytes = raw.get("bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 1
    ):
        raise ValueError("Research result CAS artifact byte count is invalid")
    payload = _read_result_artifact_v1(
        artifact_store, str(raw.get("uri") or "")
    )
    if (
        len(payload) != expected_bytes
        or hashlib.sha256(payload).hexdigest() != artifact_hash
    ):
        raise ValueError("Research result CAS artifact integrity check failed")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Research result CAS artifact is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("Research result CAS artifact must contain one JSON object")
    document = copy.deepcopy(dict(decoded))
    if _canonical_research_json_bytes(document) != payload:
        raise ValueError("Research result CAS artifact is not canonical JSON")
    return document


def _decision_truth_label_v1(
    *,
    sequential_response_status: str,
    confidence: float,
    debate_score: float,
) -> str:
    """Classify proof strength before optional content-interest signals."""

    if sequential_response_status == "exact_sequential_response":
        return "exact_optimal"
    if (
        sequential_response_status == "robust_policy_conditioned_response"
        and float(confidence) >= 0.75
    ):
        return "robust_best_known"
    if float(debate_score) >= 70.0:
        return "debatable"
    return "provisional"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _shard_failure_evidence(exc: Exception) -> tuple[dict[str, Any], bool]:
    """Classify deterministic solver exhaustion without wasting retries."""
    message = str(exc)
    if isinstance(exc, SearchTimeBudgetExceeded):
        return (
            {
                "type": type(exc).__name__,
                "category": "analysis_incomplete",
                "message": message,
            },
            False,
        )
    if "did not prove an optimal deterministic result" in message:
        return (
            {
                "type": type(exc).__name__,
                "category": "solver_incomplete",
                "message": message,
            },
            False,
        )
    return (
        {
            "type": type(exc).__name__,
            "category": "system_failure",
            "message": message,
        },
        True,
    )


def _strategic_selection_summary(
    trusted_final_state: Mapping[str, Any],
) -> dict[str, Any]:
    traces = []
    for item in trusted_final_state.get("decisionPolicyTraces") or ():
        trace = dict(item.get("trace") or {})
        if trace.get("method") == "actor_safe_cp_sat_late_game_v1":
            traces.append(trace)
            continue
        nested = dict(trace.get("strategicSelection") or {})
        if nested.get("method") == "actor_safe_cp_sat_late_game_v1":
            traces.append(nested)
    truths = sorted(
        {
            str(item.get("strategicTruth"))
            for item in traces
            if item.get("strategicTruth")
        }
    )
    generation = dict(trusted_final_state.get("generation") or {})
    selector = dict(generation.get("strategicPlaySelector") or {})
    profile = dict(selector.get("profile") or {})
    return {
        "schemaVersion": 1,
        "profileHash": profile.get("sha256"),
        "decisionCount": len(traces),
        "exactDecisionCount": sum(
            item.get("strategicTruth") == "exact_policy_conditioned_actor_safe"
            for item in traces
        ),
        "provisionalDecisionCount": sum(
            item.get("strategicTruth") == "provisional_policy_conditioned_actor_safe"
            for item in traces
        ),
        "fallbackDecisionCount": sum(
            bool(item.get("qualificationBlocked")) for item in traces
        ),
        "truthLevels": truths,
        "proofScope": "bounded_late_game_actor_safe_policy_conditioned"
        if traces
        else None,
        "globalOptimalityClaimed": False,
    }


def _tactical_requirement_evaluations(
    requirements: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    tactical_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate configured tactics against observer-safe, perspective-bound evidence."""
    evaluations: list[dict[str, Any]] = []
    for requirement in requirements:
        kind = str(requirement["kind"])
        mode = str(requirement["mode"])
        minimum_score = float(requirement["minimumScore"])
        perspectives = sorted({str(item) for item in requirement["perspectives"]})
        perspective_set = set(perspectives)
        applicable = [
            row
            for row in tactical_rows
            if str(row["kind"]) == kind
            and (not perspective_set or str(row["actingSeat"]) in perspective_set)
        ]
        matching = [
            row
            for row in applicable
            if float(row.get("magnitude", 0.0)) >= minimum_score
        ]
        matching_evidence = sorted(
            (
                {
                    "decisionId": str(row["decisionId"]),
                    "actingSeat": str(row["actingSeat"]),
                    "magnitude": float(row.get("magnitude", 0.0)),
                    "proofHash": (
                        str(row.get("causalProofHash") or row.get("proofHash"))
                        if row.get("causalProofHash") or row.get("proofHash")
                        else None
                    ),
                }
                for row in matching
            ),
            key=lambda row: (
                row["decisionId"],
                row["actingSeat"],
                row["magnitude"],
                row["proofHash"] or "",
            ),
        )
        evaluations.append(
            {
                "kind": kind,
                "mode": mode,
                "minimumScore": minimum_score,
                "weight": float(requirement["weight"]),
                "perspectives": perspectives,
                "score": max(
                    (float(row.get("magnitude", 0.0)) for row in applicable),
                    default=0.0,
                ),
                "applicableEvidenceCount": len(applicable),
                "matchingEvidenceCount": len(matching),
                "matchingEvidence": matching_evidence,
                "eligible": bool(matching),
            }
        )
    return evaluations


def _post_trace_featured_truth_status_ids_v1(
    analysis: Any,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return competence-only and unresolved IDs with fail-closed precedence."""

    adjudication = dict(getattr(analysis, "adjudication_evidence", None) or {})
    unresolved = {
        *(str(item) for item in getattr(analysis, "unresolved_decision_ids", ()) or ()),
        *(str(item) for item in adjudication.get("unresolvedDecisionIds") or ()),
    }
    competent = {str(item) for item in adjudication.get("competentDecisionIds") or ()}
    competent -= unresolved
    return frozenset(competent), frozenset(unresolved)


def _preference_selection_score(
    base_score: float,
    tactic_evaluations: list[Mapping[str, Any]],
) -> float:
    preferences = [row for row in tactic_evaluations if row["mode"] == "preference"]
    if not preferences:
        return float(base_score)
    return (
        float(base_score)
        + sum(
            (
                float(row["score"]) * float(row["weight"])
                if bool(row["eligible"])
                else 0.0
            )
            for row in preferences
        )
    ) / (1.0 + sum(float(row["weight"]) for row in preferences))


def _tactical_verification_required_by_policy(recipe: ResearchRecipe) -> bool:
    policy = str(recipe.resources["deepVerificationPolicy"])
    if policy == "off":
        return False
    if policy == "all_candidates":
        return True
    return bool(recipe.tactics)


def _tactical_verification_provenance(
    recipe: ResearchRecipe,
    tactical_verification_count: int,
) -> dict[str, Any]:
    return {
        "deepVerificationPolicy": recipe.resources["deepVerificationPolicy"],
        "tacticalVerificationRequired": (
            _tactical_verification_required_by_policy(recipe)
        ),
        "tacticalVerificationCount": int(tactical_verification_count),
    }


@dataclass(frozen=True, slots=True)
class _OrganicResearchCandidateV1:
    ordinal: int
    candidate_hash: str
    model_hash: str
    backend_version: str
    solver_status: str = "not_applicable"
    structural_objective_score: int = 0

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "pluginId": "natural_laffa_generator_v1",
            "sourceLane": "organic_runtime",
            "candidateHash": self.candidate_hash,
            "modelHash": self.model_hash,
            "backendVersion": self.backend_version,
            "solverStatus": self.solver_status,
            "solutionOrdinal": self.ordinal,
            "structuralObjectiveScore": 0,
        }


class _ShardLeaseHeartbeat:
    """Renew a shard lease while native solver or analysis work is running."""

    def __init__(
        self,
        store: Any,
        *,
        shard_id: str,
        worker_id: str,
        fencing_token: int,
        lease_seconds: float = 30.0,
    ):
        self.store = store
        self.shard_id = str(shard_id)
        self.worker_id = str(worker_id)
        self.fencing_token = int(fencing_token)
        self.lease_seconds = float(lease_seconds)
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"research-heartbeat-{self.shard_id}",
            daemon=True,
        )

    @property
    def lost(self) -> bool:
        return self._lost.is_set()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        interval = max(0.1, self.lease_seconds / 3.0)
        while not self._stop.wait(interval):
            try:
                self.store.heartbeat_shard(
                    self.shard_id,
                    worker_id=self.worker_id,
                    fencing_token=self.fencing_token,
                    lease_seconds=self.lease_seconds,
                )
            except Exception:
                self._lost.set()
                return


class ResearchFoundryServiceV1:
    descriptor = PluginDescriptor(
        id="ortools_research_foundry_v1",
        kind="research_foundry",
        version="1.2.0",
        description=(
            "Governed OR-Tools construction with actor-safe play selection, "
            "complete CP-SAT replay proof, and full decision auditing"
        ),
        capabilities=(
            "research_recipes",
            "deterministic_cp_sat",
            "all_decision_audit",
            "organic_runtime_baseline",
            "matched_generation_lanes",
            "complete_cp_sat_play_construction",
            "actor_safe_agent_auction_verification",
            "actor_safe_cp_sat_auction_decision_selection",
            "actor_safe_cp_sat_late_game_selection",
            "deterministic_recoverable_shards",
            "semantic_disagreement_quarantine",
            "content_addressed_evidence",
        ),
        resource_class="analysis-cpu",
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=4096,
        contract_id="research_executor",
        contract_version="1",
        safety_guards=(
            "authoritative_legality",
            "observer_safe",
            "hard_gate_preservation",
            "frozen_provenance",
        ),
        cancellation_mode="lifecycle",
        side_effects=("storage_write", "artifact_write"),
    )

    def __init__(
        self,
        *,
        domain: BalootDomainSpecV2,
        compiled_domain: BalootDomainSpecV3 | None = None,
        store: Any,
        artifact_store: Any,
        executor: Executor,
        deal_generator: OrToolsCpSatDealGeneratorV1,
        auction_compiler: OrToolsCpSatAuctionCompilerV1 | None = None,
        escalation_compiler: OrToolsCpSatEscalationCompilerV1 | None = None,
        simulator: NaturalLaffaSimulatorV1,
        analyzers: Mapping[str, Any],
        score_profile: Any,
        decision_score_profile: Any | None = None,
        project_compiler: OrToolsCpSatProjectCompilerV1 | None = None,
        play_compiler: OrToolsCpSatPlayCompilerV1 | None = None,
        tactical_verifier: ActualDealOracleVerifierV1 | None = None,
        maximum_tactical_decisions: int = 8,
        comparison_profile: Mapping[str, Any] | None = None,
        maximum_local_shard_workers: int = 4,
        shard_lease_seconds: float = 30.0,
        maximum_shard_attempts: int = 3,
        execution_context_factory: ResearchExecutionContextFactoryV1 | None = None,
    ):
        if simulator.rules.sha256 != domain.rules_hash:
            raise ValueError(
                "research simulator does not consume the published domain rules"
            )
        self.domain = domain
        product_copy = dict(domain.localization)
        foundry_copy = dict(product_copy.get("foundry_copy") or {})
        copy_groups = dict(foundry_copy.get("groups") or {})
        self._workflow_copy = {
            str(key): str(value)
            for key, value in dict(copy_groups.get("workflow") or {}).items()
        }
        self.compiled_domain = compiled_domain
        if compiled_domain is not None:
            if compiled_domain.rules_hash != domain.rules_hash:
                raise ValueError(
                    "V2 and V3 research domains must consume identical rules"
                )
            if compiled_domain.bundle_hash != domain.bundle_hash:
                raise ValueError(
                    "V2 and V3 research domains must consume the same bundle"
                )
        self.store = store
        self.artifact_store = artifact_store
        self.audit_traces = ResearchAuditTraceRepositoryV1(
            store=store,
            artifact_store=artifact_store,
        )
        self.executor = executor
        self.deal_generator = deal_generator
        self.auction_compiler = auction_compiler
        self.escalation_compiler = escalation_compiler
        self.simulator = simulator
        self.analyzers = dict(analyzers)
        self.score_profile = score_profile
        self.decision_score_profile = decision_score_profile
        self.project_compiler = project_compiler
        self.play_compiler = play_compiler
        self.score_profile_hash = str(score_profile.sha256)
        if len(self.score_profile_hash) != 64:
            raise ValueError("research score profile requires a SHA-256 identity")
        self.decision_score_profile_hash = (
            str(decision_score_profile.sha256)
            if decision_score_profile is not None
            else None
        )
        if (
            self.decision_score_profile_hash is not None
            and len(self.decision_score_profile_hash) != 64
        ):
            raise ValueError(
                "research decision score profile requires a SHA-256 identity"
            )
        self.tactical_verifier = tactical_verifier
        self.maximum_tactical_decisions = int(maximum_tactical_decisions)
        if not 0 <= self.maximum_tactical_decisions <= 64:
            raise ValueError("research tactical decision bound must be in [0, 64]")
        comparison = dict(comparison_profile or {})
        required_comparison = {
            "schemaVersion",
            "id",
            "revision",
            "minimumQualifiedPerLane",
            "minimumDistinctSelectedFinalistsPerLane",
            "minimumAttemptedSeedsPerLane",
            "minimumYieldMultiplier",
            "minimumCostReduction",
            "requiredPromotionCpuBoundary",
            "requireZeroSemanticDefects",
        }
        if set(comparison) != required_comparison:
            raise ValueError(
                "research comparison profile is incomplete or contains unknown keys"
            )
        if int(comparison["schemaVersion"]) != 2:
            raise ValueError("unsupported research comparison profile schema")
        self.comparison_profile = copy.deepcopy(comparison)
        self.comparison_profile_hash = canonical_sha256(comparison)
        self.maximum_local_shard_workers = int(maximum_local_shard_workers)
        if not 1 <= self.maximum_local_shard_workers <= 32:
            raise ValueError("research local shard worker bound must be in [1, 32]")
        self.shard_lease_seconds = float(shard_lease_seconds)
        if not 3.0 <= self.shard_lease_seconds <= 3_600.0:
            raise ValueError("research shard lease must be in [3, 3600] seconds")
        self.maximum_shard_attempts = int(maximum_shard_attempts)
        if not 1 <= self.maximum_shard_attempts <= 10:
            raise ValueError("research shard attempts must be in [1, 10]")
        self.execution_context_factory = execution_context_factory
        # The V2 predicate catalog is immutable and process-wide. Build it at
        # service startup so the first operator validation is not a catalog
        # compilation benchmark.
        shared_scenario_predicate_catalog_lowering_manifest_v2()
        self._service_worker_id = uuid.uuid4().hex
        self._cancelled: set[str] = set()
        self._scheduled_runs: set[str] = set()
        self._lock = threading.RLock()
        self._scenario_backends_v2: dict[str, OrToolsScenarioResearchBackendV2] = {}
        self._scenario_validation_cache_v4: dict[str, tuple[dict[str, Any], int]] = {}
        self._scenario_validation_cache_v4_bytes = 0
        # Coalesce concurrent result/decision/render reads without blocking the
        # campaign/cancellation lock. Only successful semantic checks are kept.
        self._result_verification_lock = threading.RLock()
        self._result_verification_cache: dict[str, None] = {}
        self._scenario_recipe_context_cache_v4: dict[
            str, tuple[ResearchRecipeV4, ResearchExecutionContextV1, int]
        ] = {}
        self._scenario_recipe_context_cache_v4_bytes = 0

        # A process restart invalidates leases owned by the former process. Only
        # runs already using the shard control plane are resumed automatically.
        recovered = self.store.recover_orphaned_shards()
        recoverable = set(recovered)
        for status in ("queued", "running"):
            for item in self.store.list_runs(status=status, limit=10_000):
                if self.store.list_shards(item["runCode"]):
                    recoverable.add(str(item["runCode"]))
        for run_code in sorted(recoverable):
            current = self.store.get_run(run_code)
            if current is None or current["status"] == "cancelled":
                continue
            self.store.update_run(
                run_code,
                status="queued",
                progress=float(current.get("progress", 0.0)),
            )
            self._schedule_run(run_code)

    def _localized(
        self,
        key: str,
        execution_context: ResearchExecutionContextV1 | None = None,
        /,
        **values: Any,
    ) -> str:
        workflow_copy = (
            execution_context.workflow_copy
            if execution_context is not None
            else self._workflow_copy
        )
        template = workflow_copy.get(str(key))
        if template is None:
            raise ValueError(f"research audit localization is missing: {key}")
        try:
            return template.format(**values)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"research audit localization is invalid: {key}") from exc

    def _recipe_configuration(
        self, context: ResearchExecutionContextV1 | None = None
    ) -> dict[str, Any]:
        catalog = research_recipe_configuration_v1()
        simulator = context.simulator if context is not None else self.simulator
        project_domain = ProjectDomainIRV1.from_rules(simulator.rules.data)
        project_options: dict[tuple[str, str, int, str], dict[str, Any]] = {}
        if project_domain.enabled:
            for contract_name in simulator.rules.data["contracts"]["allowed"]:
                contract = Contract.parse(contract_name)
                for project in project_domain.candidates(
                    Seat.NORTH, FULL_DECK, contract
                ):
                    if project.kind not in {
                        "siri",
                        "fifty",
                        "hundred_sequence",
                        "hundred_four_kind",
                        "four_hundred",
                    }:
                        continue
                    high_rank = Rank(int(project.high_rank)).name.lower()
                    key = (
                        contract_name,
                        str(project.kind),
                        int(project.value),
                        high_rank,
                    )
                    project_options[key] = {
                        "contract": contract_name,
                        "kind": str(project.kind),
                        "value": int(project.value),
                        "highRank": high_rank,
                        "highRankId": int(project.high_rank),
                    }
        core = {key: value for key, value in catalog.items() if key != "sha256"}
        core["projectOptions"] = [
            project_options[key] for key in sorted(project_options)
        ]
        core["projectDomainHash"] = project_domain.sha256
        return {**core, "sha256": canonical_sha256(core)}

    @staticmethod
    def _scenario_rule_options(rules) -> dict[str, Any]:
        """Project the selected rules bundle into scenario-authoring choices."""

        data = rules.data
        contracts = [str(item) for item in data["contracts"]["allowed"]]
        escalation = data["escalation"]
        escalation_domain = EscalationDomainIRV1.from_rules(data)
        escalation_enabled = bool(escalation["enabled"])
        escalation_by_contract: list[dict[str, Any]] = []
        for contract_name in contracts:
            contract = Contract.parse(contract_name)
            chain = (
                escalation["hokum_chain"]
                if contract.effective == Contract.HOKUM
                else escalation["sun_chain"]
            )
            enabled_chain = [str(item) for item in chain] if escalation_enabled else []
            escalation_by_contract.append(
                {
                    "contract": contract.value,
                    "levels": ["normal", *enabled_chain],
                    "actions": (
                        [escalation_domain.closure_action, *enabled_chain]
                        if enabled_chain
                        else []
                    ),
                }
            )

        projects = data["projects"]
        baloot = projects["baloot"]
        baloot_contracts: list[str] = []
        if bool(projects["enabled"]) and bool(baloot["enabled"]):
            baloot_contracts = [
                contract_name
                for contract_name in contracts
                if not bool(baloot["only_hokum"])
                or Contract.parse(contract_name).effective == Contract.HOKUM
            ]

        ashkal = data["bidding"]["ashkal"]
        ashkal_enabled = bool(ashkal["enabled"]) and "ashkal" in contracts
        trump_contracts = [
            contract_name
            for contract_name in contracts
            if Contract.parse(contract_name).effective == Contract.HOKUM
        ]
        play_action_options = [LegalPlayActionV1(card).to_dict() for card in FULL_DECK]
        if bool(data["play"]["ekka_enabled"]) and trump_contracts:
            play_action_options.extend(
                LegalPlayActionV1(card, True).to_dict() for card in FULL_DECK
            )
        return {
            "contracts": contracts,
            "auctionActionsByRound": [
                {
                    "round": round_number,
                    "actions": [
                        str(item) for item in data["bidding"][round_key]["allowed"]
                    ],
                }
                for round_number, round_key in ((1, "first_round"), (2, "second_round"))
            ],
            "escalationLevelsByContract": escalation_by_contract,
            "ashkalBuyerRelations": [str(ashkal["taker"])] if ashkal_enabled else [],
            "trumpContracts": trump_contracts,
            "balootContracts": baloot_contracts,
            "playActionOptions": play_action_options,
            "projectsEnabled": bool(projects["enabled"]),
            "kaboutEnabled": bool(data["scoring"]["kabout"]["enabled"]),
        }

    @staticmethod
    def _scenario_recipe_configuration(
        *,
        project_options: Sequence[Mapping[str, Any]] = (),
        project_domain_hash: str = "",
        tactic_options: Sequence[str] = (),
        rule_options: Mapping[str, Any] | None = None,
        policy_catalog: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        registry = ScenarioPredicateRegistryV1().catalog()
        budgets = ScenarioProofBudgetRegistryV1.load_default().to_dict()
        policy_profiles: dict[str, dict[str, Any]] = {}
        if policy_catalog is not None:
            policy_profiles = materialize_default_policy_profiles_v1(policy_catalog)
        for seed, seat in enumerate(("north", "west", "south", "east"), start=1):
            if policy_catalog is None:
                policy_core = {
                    "agentId": "heuristic_bot_v1",
                    "version": "1.0.0",
                    "seed": seed,
                }
                policy_profiles[seat] = {
                    **policy_core,
                    "profileHash": scenario_sha256(policy_core),
                }
        core = {
            "schemaVersion": 1,
            "recipeSchemaVersion": 4,
            "supportedLegacyReadVersions": [1, 2, 3],
            "origins": ["search", "exact_trace_prefix"],
            "outputPolicy": {
                "kinds": ["laffa", "laffa_collection"],
                "resultCount": {"minimum": 1, "maximum": 1_000, "default": 1},
                "renderModes": ["on_demand", "disabled"],
                "videoCount": {
                    "minimum": 0,
                    "maximum": 100,
                    "default": 1,
                    "maximumRelation": "output.resultCount",
                },
            },
            "decisionCapabilities": [
                {
                    "kind": kind,
                    "authoringSupported": True,
                    "exactTraceNormalizationSupported": kind
                    in {"bid", "escalation", "project", "play"},
                    "searchGenerationSupported": kind
                    in {"bid", "escalation", "project", "play"},
                    "proofModes": (
                        ["exact", "bounded", "policy_conditioned"]
                        if kind == "play"
                        else ["policy_conditioned"]
                        if kind in {"bid", "escalation", "project"}
                        else []
                    ),
                    "renderPreparationSupported": kind
                    in {"bid", "escalation", "project", "play"},
                    "unavailableReason": (
                        None
                        if kind in {"bid", "escalation", "project", "play"}
                        else "decision_point_proof_unavailable"
                    ),
                }
                for kind in ("bid", "escalation", "project", "play")
            ],
            "sections": list(SCENARIO_SECTIONS),
            "logicalOperators": ["all", "any", "not", "count"],
            "hardness": ["hard", "prefer", "report"],
            "proofModes": list(PROOF_MODES),
            "proofStrategyClasses": list(PROOF_STRATEGY_CLASSES),
            "proofVisibilities": list(PROOF_VISIBILITIES),
            "proofClaims": list(PROOF_CLAIMS),
            "proofUtilities": list(PROOF_UTILITIES),
            "proofBoundaryEvaluators": list(OPERATOR_PROOF_BOUNDARY_EVALUATORS),
            "proofLineUniverses": list(LINE_UNIVERSES),
            "proofHorizonUnits": list(PROOF_HORIZON_UNITS),
            "predicateRegistry": registry,
            "proofBudgets": budgets,
            "defaultPolicyProfiles": policy_profiles,
            "cardOptions": [card.to_dict() for card in FULL_DECK],
            "tacticOptions": sorted({str(item) for item in tactic_options}),
            "projectOptions": [copy.deepcopy(dict(item)) for item in project_options],
            "projectDomainHash": project_domain_hash,
        }
        if rule_options is not None:
            core["ruleOptions"] = copy.deepcopy(dict(rule_options))
        return {**core, "sha256": scenario_sha256(core)}

    @staticmethod
    def _scenario_rules_applicability_issues(
        recipe: ResearchRecipeV4,
        rules,
        *,
        supported_tactics: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        rule_options = ResearchFoundryServiceV1._scenario_rule_options(rules)
        allowed_contracts = {
            Contract.parse(value) for value in rules.data["contracts"]["allowed"]
        }
        issues: list[dict[str, Any]] = []

        def issue(
            *,
            code: str,
            predicate_id: str,
            details: Mapping[str, Any],
        ) -> dict[str, Any]:
            if code not in SCENARIO_RULE_APPLICABILITY_CODES:
                raise ValueError(f"unregistered scenario rules issue code: {code}")
            core = {
                "code": code,
                "predicateId": predicate_id,
                "details": copy.deepcopy(dict(details)),
                "rulesHash": rules.sha256,
            }
            return {
                "issueId": f"ISS-V4-RULES-{len(issues) + 1:04d}",
                "class": "rules_inapplicable",
                "severity": "error",
                "code": code,
                "predicateIds": [predicate_id],
                "fieldPaths": [f"scenario.predicates.{predicate_id}"],
                "messageArKey": f"research_scenario_issue_{code}",
                "messageEnKey": f"research_scenario_issue_{code}_en",
                "details": copy.deepcopy(dict(details)),
                "sha256": scenario_sha256(core),
            }

        selected_contracts: set[Contract] = set()
        for predicate in recipe.scenario.predicates():
            if predicate.kind != "auction.contract":
                continue
            contract = Contract.parse(predicate.arguments["contract"])
            if predicate.hardness == "hard":
                selected_contracts.add(contract)
            if contract not in allowed_contracts:
                issues.append(
                    issue(
                        code="contract_not_enabled_by_rules",
                        predicate_id=predicate.predicate_id,
                        details={
                            "contract": contract.value,
                            "allowedContracts": sorted(
                                item.value for item in allowed_contracts
                            ),
                        },
                    )
                )
        applicable_contracts = selected_contracts or allowed_contracts

        applicable_contract_values = {
            item.value for item in applicable_contracts if item in allowed_contracts
        }
        hard_rounds = {
            int(predicate.arguments["round"])
            for predicate in recipe.scenario.predicates()
            if predicate.kind == "auction.round" and predicate.hardness == "hard"
        }
        auction_actions_by_round = {
            int(item["round"]): {str(action) for action in item["actions"]}
            for item in rule_options["auctionActionsByRound"]
        }
        applicable_auction_actions = set().union(
            *(
                auction_actions_by_round[round_number]
                for round_number in sorted(hard_rounds or set(auction_actions_by_round))
            )
        )
        escalation_by_contract = {
            str(item["contract"]): item
            for item in rule_options["escalationLevelsByContract"]
        }
        applicable_escalation = [
            escalation_by_contract[contract]
            for contract in sorted(applicable_contract_values)
            if contract in escalation_by_contract
        ]
        applicable_escalation_levels = {
            str(value) for item in applicable_escalation for value in item["levels"]
        }
        applicable_escalation_actions = {
            str(value) for item in applicable_escalation for value in item["actions"]
        }
        ekka_applicable = bool(
            applicable_contract_values & set(rule_options["trumpContracts"])
        ) and any(bool(action["ekka"]) for action in rule_options["playActionOptions"])

        for predicate in recipe.scenario.predicates():
            arguments = dict(predicate.arguments)
            if predicate.kind == "auction.ashkal_buyer_relation":
                relation = str(arguments["relation"])
                if (
                    "ashkal" not in applicable_contract_values
                    or relation not in rule_options["ashkalBuyerRelations"]
                ):
                    issues.append(
                        issue(
                            code="ashkal_not_enabled_by_rules",
                            predicate_id=predicate.predicate_id,
                            details={
                                "relation": relation,
                                "allowedRelations": list(
                                    rule_options["ashkalBuyerRelations"]
                                ),
                                "contracts": sorted(applicable_contract_values),
                            },
                        )
                    )
            elif predicate.kind == "auction.trump" and not (
                applicable_contract_values & set(rule_options["trumpContracts"])
            ):
                issues.append(
                    issue(
                        code="trump_not_applicable_to_contract",
                        predicate_id=predicate.predicate_id,
                        details={
                            "contracts": sorted(applicable_contract_values),
                            "trumpContracts": list(rule_options["trumpContracts"]),
                        },
                    )
                )
            elif predicate.kind in {
                "auction.action_sequence",
                "auction.legal_actions",
            }:
                requested_actions = {
                    str(item["action"]) for item in arguments["actions"]
                }
                unavailable = sorted(requested_actions - applicable_auction_actions)
                if unavailable:
                    issues.append(
                        issue(
                            code="auction_action_not_enabled_by_rules",
                            predicate_id=predicate.predicate_id,
                            details={
                                "actions": unavailable,
                                "allowedActions": sorted(applicable_auction_actions),
                                "rounds": sorted(
                                    hard_rounds or set(auction_actions_by_round)
                                ),
                            },
                        )
                    )
            elif predicate.kind == "escalation.level":
                level = str(arguments["level"])
                if level not in applicable_escalation_levels:
                    issues.append(
                        issue(
                            code="escalation_level_not_enabled_by_rules",
                            predicate_id=predicate.predicate_id,
                            details={
                                "level": level,
                                "allowedLevels": sorted(applicable_escalation_levels),
                                "contracts": sorted(applicable_contract_values),
                            },
                        )
                    )
            elif predicate.kind in {
                "escalation.action_sequence",
                "escalation.legal_actions",
            }:
                requested_actions = {
                    str(item["action"]) for item in arguments["actions"]
                }
                unavailable = sorted(requested_actions - applicable_escalation_actions)
                if unavailable:
                    issues.append(
                        issue(
                            code="escalation_action_not_enabled_by_rules",
                            predicate_id=predicate.predicate_id,
                            details={
                                "actions": unavailable,
                                "allowedActions": sorted(applicable_escalation_actions),
                                "contracts": sorted(applicable_contract_values),
                            },
                        )
                    )
            elif (
                predicate.kind == "escalation.locked"
                and bool(arguments["locked"])
                and not applicable_escalation_actions
            ) or (
                predicate.kind == "escalation.eligible_actor"
                and bool(arguments["eligible"])
                and not applicable_escalation_actions
            ):
                issues.append(
                    issue(
                        code="escalation_not_enabled_by_rules",
                        predicate_id=predicate.predicate_id,
                        details={"contracts": sorted(applicable_contract_values)},
                    )
                )

            if predicate.kind in {
                "play.action_prefix",
                "state.legal_play_actions_include",
            }:
                requests_ekka = any(
                    bool(action["ekka"]) for action in arguments["actions"]
                )
            else:
                requests_ekka = predicate.kind == "play.current_trick_card" and bool(
                    arguments.get("ekka", False)
                )
            if requests_ekka and not ekka_applicable:
                issues.append(
                    issue(
                        code="ekka_not_enabled_by_rules",
                        predicate_id=predicate.predicate_id,
                        details={"kind": predicate.kind},
                    )
                )

        projects_enabled = bool(rule_options["projectsEnabled"])
        positive_project_kinds = {
            "project.exists",
            "project.count",
            "project.combined_value",
            "project.baloot",
            "project.overlap_count",
            "state.current_project_points",
            "outcome.project_points",
        }
        for predicate in recipe.scenario.predicates():
            if projects_enabled or predicate.kind not in positive_project_kinds:
                continue
            arguments = dict(predicate.arguments)
            positive = predicate.kind == "project.exists"
            if predicate.kind == "project.baloot":
                positive = bool(arguments["achieved"])
            elif "minimum" in arguments:
                positive = float(arguments["minimum"]) > 0
            if positive:
                issues.append(
                    issue(
                        code="projects_not_enabled_by_rules",
                        predicate_id=predicate.predicate_id,
                        details={"kind": predicate.kind},
                    )
                )

        for predicate in recipe.scenario.predicates():
            if predicate.kind != "project.exists":
                continue
            if not projects_enabled:
                continue
            arguments = dict(predicate.arguments)
            supported = any(
                project.kind == arguments["kind"]
                and project.value == int(arguments["value"])
                for contract in applicable_contracts
                for project in rules.project_domain.candidates(
                    Seat.NORTH,
                    FULL_DECK,
                    contract,
                )
            )
            if not supported:
                issues.append(
                    issue(
                        code="project_kind_value_not_enabled_by_rules",
                        predicate_id=predicate.predicate_id,
                        details={
                            "kind": arguments["kind"],
                            "value": int(arguments["value"]),
                            "contracts": sorted(
                                item.value for item in applicable_contracts
                            ),
                        },
                    )
                )

        for predicate in recipe.scenario.predicates():
            arguments = dict(predicate.arguments)
            if (
                projects_enabled
                and predicate.kind == "project.baloot"
                and bool(arguments["achieved"])
                and not (
                    applicable_contract_values & set(rule_options["balootContracts"])
                )
            ):
                issues.append(
                    issue(
                        code="baloot_not_enabled_for_contract",
                        predicate_id=predicate.predicate_id,
                        details={
                            "contracts": sorted(applicable_contract_values),
                            "balootContracts": list(rule_options["balootContracts"]),
                        },
                    )
                )
            elif (
                predicate.kind == "outcome.kabout"
                and bool(arguments["achieved"])
                and not bool(rule_options["kaboutEnabled"])
            ):
                issues.append(
                    issue(
                        code="kabout_not_enabled_by_rules",
                        predicate_id=predicate.predicate_id,
                        details={},
                    )
                )
            elif (
                predicate.kind == "outcome.gahwa"
                and bool(arguments["achieved"])
                and "gahwa" not in applicable_escalation_levels
            ):
                issues.append(
                    issue(
                        code="gahwa_not_enabled_for_contract",
                        predicate_id=predicate.predicate_id,
                        details={
                            "contracts": sorted(applicable_contract_values),
                            "allowedLevels": sorted(applicable_escalation_levels),
                        },
                    )
                )
        enabled_tactics = {str(item) for item in supported_tactics}
        for predicate in recipe.scenario.predicates():
            if predicate.kind not in {"tactic.applicable", "tactic.verified"}:
                continue
            requested = str(predicate.arguments["kind"])
            if requested not in enabled_tactics:
                issues.append(
                    issue(
                        code="tactic_detector_not_enabled_by_bundle",
                        predicate_id=predicate.predicate_id,
                        details={
                            "kind": requested,
                            "supportedTactics": sorted(enabled_tactics),
                        },
                    )
                )
        return issues

    @staticmethod
    def _ortools_adjudication_authority_input_v1(
        context: ResearchExecutionContextV1,
        challenge: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        research = dict(challenge.get("scenarioResearchV4") or {})
        if research.get("backend") != ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3:
            return None
        components = getattr(context.bundle, "components", None)
        if not isinstance(components, Mapping) or "adjudication" not in components:
            raise ValueError(
                "OR-Tools Research V3 requires an explicit frozen adjudication component"
            )
        document = context.bundle.component("adjudication")
        if not isinstance(document, Mapping):
            raise TypeError("frozen adjudication component must be an object")
        return copy.deepcopy(dict(document))

    @staticmethod
    def _ortools_scenario_backend_v2(
        context: ResearchExecutionContextV1 | None,
        *,
        repository_root: str | Path | None = None,
    ) -> OrToolsScenarioResearchBackendV2 | None:
        """Resolve V2 only from one explicitly frozen execution context."""

        if context is None:
            return None
        challenge = copy.deepcopy(dict(context.bundle.component("challenge")))
        if "scenarioResearchV4" not in challenge:
            return None
        adjudication_authority = ResearchFoundryServiceV1._ortools_adjudication_authority_input_v1(
            context, challenge
        )
        catalog_snapshot = None
        if dict(challenge["scenarioResearchV4"]).get("backend") == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3:
            catalog = load_bound_rule_source_catalog(
                context.bundle.component("certification"), context.bundle.rules,
                Path(repository_root) if repository_root is not None else platform_root(),
            )
            if catalog is None:
                raise ValueError("native Research requires a frozen rule source catalog binding")
            catalog_snapshot = {"data": catalog.data, "sha256": catalog.sha256}
        return OrToolsScenarioResearchBackendV2(
            challenge_profile=challenge,
            bundle_identity={
                "bundleRevisionId": context.bundle_revision_id,
                "semanticBundleHash": context.bundle.sha256,
                "effectiveRuntimeConfigHash": context.effective_runtime_config_hash,
            },
            rules=context.bundle.rules,
            rule_source_catalog=catalog_snapshot,
            adjudication_authority=adjudication_authority,
        )

    def _cached_ortools_scenario_backend_v2(
        self,
        context: ResearchExecutionContextV1 | None,
    ) -> OrToolsScenarioResearchBackendV2 | None:
        if context is None:
            return None
        challenge = copy.deepcopy(dict(context.bundle.component("challenge")))
        if "scenarioResearchV4" not in challenge:
            return None
        cache_key = canonical_sha256(
            {
                "bundleRevisionId": context.bundle_revision_id,
                "semanticBundleHash": context.bundle.sha256,
                "effectiveRuntimeConfigHash": context.effective_runtime_config_hash,
                "rulesHash": context.bundle.rules.sha256,
                "challenge": challenge,
                "adjudication": self._ortools_adjudication_authority_input_v1(
                    context, challenge
                ),
            }
        )
        with self._lock:
            cached = self._scenario_backends_v2.get(cache_key)
            if cached is not None:
                return cached
        backend = (
            self._ortools_scenario_backend_v2(
                context, repository_root=self.execution_context_factory.root
            )
            if self.execution_context_factory is not None
            else self._ortools_scenario_backend_v2(context)
        )
        if backend is None:
            return None
        with self._lock:
            return self._scenario_backends_v2.setdefault(cache_key, backend)

    @staticmethod
    def _adjudication_research_authority_v1(
        context: ResearchExecutionContextV1,
    ) -> AdjudicationResearchAuthorityV1 | None:
        """Resolve adjudication authoring only from the selected frozen bundle."""

        challenge = copy.deepcopy(dict(context.bundle.component("challenge")))
        document = ResearchFoundryServiceV1._ortools_adjudication_authority_input_v1(
            context, challenge
        )
        if document is None:
            return None
        return AdjudicationResearchAuthorityV1.from_mapping(document)

    @staticmethod
    def _validate_exact_adjudication_events_v1(
        prefix: ExactTracePrefixV1,
        authority: AdjudicationResearchAuthorityV1 | None,
        *,
        current_stage: str,
        next_actor: str | None,
    ) -> None:
        """Bind attempted-action sidecars to one legal trace boundary.

        ``actionSequence`` is a zero-based boundary within the event's stage:
        zero is before the first legal action and ``N`` is after N legal
        actions.  The rejected attempt remains outside the ordinary history.
        """

        if not prefix.adjudication_events:
            return
        if authority is None:
            raise ValueError("selected bundle has no adjudication authority")
        stage_actions: dict[str, tuple[Mapping[str, Any], ...]] = {
            "auction": prefix.auction_actions,
            "play": prefix.play_actions,
            "projects": (
                prefix.project_actions
                if prefix.project_actions_present
                else prefix.project_declarations
            ),
        }
        for event in prefix.adjudication_events:
            capability = authority.validate_authored_event(event)
            actions = stage_actions[capability.stage]
            if event.action_sequence > len(actions):
                raise ValueError(
                    f"{event.kind.value} actionSequence exceeds its legal stage prefix"
                )
            if capability.event_scope != "adjudicated_attempt":
                continue

            action = dict(event.attempted_action)
            if event.kind.value == "out_of_turn":
                if set(action) == {"cardId"}:
                    card_id = action["cardId"]
                    if type(card_id) is not int or not 0 <= card_id < 32:
                        raise ValueError(
                            "out_of_turn attemptedAction.cardId must be an integer 0..31"
                        )
                elif set(action) == {"card"}:
                    if type(action["card"]) is not str:
                        raise TypeError(
                            "out_of_turn attemptedAction.card must be a card code"
                        )
                    Card.parse(action["card"])
                else:
                    raise ValueError(
                        "out_of_turn attemptedAction must contain exactly cardId or card"
                    )
            elif event.kind.value == "invalid_bid":
                if set(action) != {"token"} or action["token"] not in {
                    "sun",
                    "hokum",
                    "pass",
                    "no",
                }:
                    raise ValueError(
                        "invalid_bid attemptedAction must contain one governed purchase token"
                    )
            elif event.kind.value == "false_ekka_attempt":
                if (
                    set(action) != {"cardId", "ekka"}
                    or type(action["cardId"]) is not int
                    or not 0 <= action["cardId"] < 32
                    or action["ekka"] is not True
                ):
                    raise ValueError(
                        "false_ekka_attempt attemptedAction must contain cardId 0..31 and ekka=true"
                    )
            else:  # pragma: no cover - authority scope is closed above
                raise AssertionError("unknown attempted-action adjudication event")

            expected_actor: str | None = None
            if event.action_sequence < len(actions):
                authored_actor = dict(actions[event.action_sequence]).get("actor")
                expected_actor = (
                    str(authored_actor).strip().lower()
                    if authored_actor is not None
                    else None
                )
            elif current_stage == capability.stage:
                expected_actor = next_actor
            if expected_actor is None:
                raise ValueError(
                    f"{event.kind.value} does not identify a legal attempted-action boundary"
                )
            if event.expected_actor_seat != expected_actor:
                raise ValueError(
                    f"{event.kind.value} expectedActorSeat disagrees with the legal trace"
                )

    @staticmethod
    def _ortools_localization_projection_v2(
        context: ResearchExecutionContextV1,
    ) -> dict[str, Any]:
        """Freeze Scenario Studio copy from the selected bundle only."""

        product_artifact = context.bundle.components["product"]
        revision_prefix = "configuration:"
        if not product_artifact.path.startswith(revision_prefix):
            raise ValueError("selected product profile revision is unavailable")
        foundry_copy = dict(
            context.bundle.component("product").get("foundry_copy") or {}
        )
        groups = foundry_copy.get("groups")
        if (
            foundry_copy.get("locale") != "ar-SA"
            or foundry_copy.get("direction") != "rtl"
            or not isinstance(groups, Mapping)
            or not groups
        ):
            raise ValueError("selected product Foundry localization is invalid")
        workflow = groups.get("workflow")
        if (
            not isinstance(workflow, Mapping)
            or not workflow
            or any(
                type(key) is not str or type(value) is not str
                for key, value in workflow.items()
            )
        ):
            raise ValueError("selected product Foundry workflow copy is invalid")
        core = {
            "schemaVersion": 1,
            "bundleRevisionId": context.bundle_revision_id,
            "productRevisionId": product_artifact.path.removeprefix(revision_prefix),
            "productProfileHash": product_artifact.sha256,
            "locale": "ar-SA",
            "direction": "rtl",
            "groups": {
                "workflow": {key: value for key, value in sorted(workflow.items())}
            },
        }
        return {**core, "sha256": scenario_sha256(core)}

    @staticmethod
    def _ortools_preflight_issue_v2(
        issue: Mapping[str, Any],
        *,
        recipe_hash: str,
        workflow_copy: Mapping[str, str],
    ) -> dict[str, Any]:
        code = str(issue["code"])
        diagnostic_class = (
            "schema_invalid"
            if code in {"predicate_arguments_invalid", "resources_invalid",
                        "exact_input_project_actions_required", "exact_input_project_declarations_legacy"}
            else "structurally_infeasible"
            if code == "exact_structural_infeasibility"
            else "unsupported_predicate"
            if code.startswith("predicate_")
            else "rules_inapplicable"
        )
        path = str(issue["path"])
        details = {
            key: copy.deepcopy(value)
            for key, value in issue.items()
            if key not in {"code", "path"}
        }
        structural = details.get("structuralInfeasibility")
        predicate_ids = (
            list(structural.get("namedCorePredicateIds") or [])
            if isinstance(structural, Mapping)
            else list(issue.get("predicateIds") or [])
        )
        issue_hash = scenario_sha256(
            {"code": code, "path": path, "recipeHash": recipe_hash, **details}
        )
        specific_key = f"research_scenario_issue_{code}"
        message_key = (
            specific_key if specific_key in workflow_copy else "research_validation_failed"
        )
        if message_key not in workflow_copy:
            raise ValueError("selected product Foundry validation copy is incomplete")
        return ScenarioDiagnosticV1.build(
            {
                "issueId": f"ISS-V4-ORTOOLS-{issue_hash[:12].upper()}",
                "class": diagnostic_class,
                "severity": "error",
                "code": code,
                "predicateIds": predicate_ids,
                "fieldPaths": [path],
                "messageArKey": message_key,
                "messageEnKey": (
                    f"{specific_key}_en"
                    if f"{specific_key}_en" in workflow_copy
                    else message_key
                ),
                "details": details,
            }
        ).to_dict()

    def capabilities(
        self,
        *,
        include_payloads: bool = False,
        bundle_revision_id: str | None = None,
    ) -> dict[str, Any]:
        context: ResearchExecutionContextV1 | None = None
        if bundle_revision_id is not None:
            if self.execution_context_factory is None:
                raise ValueError("bundle-specific Research execution is unavailable")
            _effective, context = (
                self.execution_context_factory.resolve_bundle_revision(
                    bundle_revision_id, require_published=False
                )
            )
        domain = context.legacy_domain_v2 if context is not None else self.domain
        compiled_domain = (
            context.compiled_domain if context is not None else self.compiled_domain
        )
        simulator = context.simulator if context is not None else self.simulator
        analyzers = context.analyzers if context is not None else self.analyzers
        score_profile = (
            context.score_profile if context is not None else self.score_profile
        )
        decision_score_profile = (
            context.decision_score_profile
            if context is not None
            else self.decision_score_profile
        )
        tactical_verifier = (
            context.tactical_verifier if context is not None else self.tactical_verifier
        )
        maximum_tactical_decisions = (
            context.maximum_tactical_decisions
            if context is not None
            else self.maximum_tactical_decisions
        )
        comparison_profile = (
            context.comparison_profile
            if context is not None
            else self.comparison_profile
        )
        comparison_profile_hash = (
            context.comparison_profile_hash
            if context is not None
            else self.comparison_profile_hash
        )
        recipe_configuration = self._recipe_configuration(context)
        ortools_backend = self._cached_ortools_scenario_backend_v2(context)
        value = domain.to_dict(include_payloads=include_payloads)
        value["compilerV3"] = (
            compiled_domain.to_dict(include_payloads=include_payloads)
            if compiled_domain is not None
            else None
        )
        value["research"] = {
            "recipeSchemaVersion": 4,
            "supportedRecipeSchemaVersions": [1, 2, 3, 4],
            "recipeConfiguration": recipe_configuration,
            "scenarioConfiguration": self._scenario_recipe_configuration(
                project_options=recipe_configuration["projectOptions"],
                project_domain_hash=str(recipe_configuration["projectDomainHash"]),
                tactic_options=domain.tactical_patterns,
                rule_options=self._scenario_rule_options(simulator.rules),
                policy_catalog=(
                    ortools_backend.authority.policy_catalog
                    if ortools_backend is not None
                    else None
                ),
            ),
            "decisionAuditSchemaVersion": 2,
            "solver": self.deal_generator.descriptor.to_dict(),
            "auctionCompiler": (
                self.auction_compiler.descriptor.to_dict()
                if self.auction_compiler is not None
                else None
            ),
            "escalationCompiler": (
                self.escalation_compiler.descriptor.to_dict()
                if self.escalation_compiler is not None
                else None
            ),
            "playCompiler": (
                self.play_compiler.descriptor.to_dict()
                if self.play_compiler is not None
                else None
            ),
            "projectCompiler": (
                self.project_compiler.descriptor.to_dict()
                if self.project_compiler is not None
                else None
            ),
            "solverAvailable": self.deal_generator.available(),
            "generationLanes": ["ortools", "organic_runtime"],
            "auctionConstructionModes": [
                "actor_safe_agent_verified",
                "cp_sat_actor_safe_decision_v1",
                "cp_sat_feasibility_line",
            ],
            "playConstructionModes": [
                "actor_safe_agent_verified",
                "cp_sat_actor_safe_late_game_v1",
                "cp_sat_feasibility_line",
            ],
            "recommendedPlayConstructionMode": "actor_safe_agent_verified",
            "sharding": {
                "schemaVersion": 1,
                "deterministicSeedPartitions": True,
                "durableLeases": True,
                "fencingTokens": True,
                "maximumLocalWorkers": self.maximum_local_shard_workers,
                "leaseSeconds": self.shard_lease_seconds,
                "heartbeatSeconds": self.shard_lease_seconds / 3.0,
                "maximumAttempts": self.maximum_shard_attempts,
                "reductionOrder": "shard_index",
            },
            "analysisModes": sorted(analyzers),
            "deepVerificationPolicies": ["required_gates", "all_candidates", "off"],
            "defaultAgentRelease": simulator.agent_release,
            "analysisProfiles": {
                key: str(value.profile.sha256)
                for key, value in sorted(analyzers.items())
            },
            "defaultScoreProfileHash": str(score_profile.sha256),
            "defaultScoreProfile": score_profile.to_dict(),
            "defaultDecisionScoreProfileHash": (
                str(decision_score_profile.sha256)
                if decision_score_profile is not None
                else None
            ),
            "defaultDecisionScoreProfile": (
                decision_score_profile.to_dict()
                if decision_score_profile is not None
                else None
            ),
            "compilerQualification": {
                "legacyResearchAvailable": True,
                "requiresExecutableBindingCoverage": True,
                "requiresBehavioralOrInvariantCoverage": True,
                "executableBindingComplete": (
                    compiled_domain.compiled.coverage.executable_binding_complete
                    if compiled_domain is not None
                    else False
                ),
                "behavioralOrInvariantComplete": (
                    compiled_domain.compiled.coverage.behavioral_or_invariant_complete
                    if compiled_domain is not None
                    else False
                ),
                "publicationEligible": (
                    compiled_domain.compiled.publishable
                    if compiled_domain is not None
                    else False
                ),
            },
            "comparisonProfile": {
                **copy.deepcopy(dict(comparison_profile)),
                "sha256": comparison_profile_hash,
            },
            "outputModes": {
                "laffa": "available",
                "laffa_collection": "available",
                "contextualized_laffa": "use_context_search",
                "sakka": "use_verified_sakka_composer",
            },
            "truthLevels": [
                "exact_optimal",
                "robust_best_known",
                "debatable",
                "provisional",
                "rejected",
            ],
            "tacticalVerification": {
                "available": tactical_verifier is not None,
                "maximumDecisionsPerLaffa": maximum_tactical_decisions,
                "supportedPatterns": list(domain.tactical_patterns),
                "defaultPolicy": "required_gates",
            },
        }
        if ortools_backend is not None:
            workflow_copy = (
                context.bundle.component("product")
                .get("foundry_copy", {})
                .get("groups", {})
                .get("workflow", {})
            )
            execution_capabilities = build_scenario_execution_capabilities_v1(
                ortools_backend,
                expose_shard_worker_cap=all(
                    key in workflow_copy for key in (
                        "research_shard_worker_count",
                        "research_shard_worker_count_en",
                        "research_scenario_help_shard_worker_count",
                        "research_scenario_help_shard_worker_count_en",
                    )
                ),
                adjudication_authority=self._adjudication_research_authority_v1(
                    context
                ),
            )
            value["research"]["scenarioConfiguration"] = (
                project_scenario_configuration_execution_v1(
                    value["research"]["scenarioConfiguration"],
                    execution_capabilities,
                )
            )
            supported_origins = list(
                dict.fromkeys(
                    item["origin"] for item in execution_capabilities["combinations"]
                )
            )
            supported_proof_visibility = [
                {"mode": mode, "visibility": visibility}
                for mode, visibility in dict.fromkeys(
                    (item["mode"], item["visibility"])
                    for item in execution_capabilities["combinations"]
                )
            ]
            authoring_ready = (
                bool(execution_capabilities["combinations"])
                and context.bundle.bundle_profile["profile"]["status"] == "published"
                and (
                    ortools_backend.authority.backend != ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
                    or "playingQuality" in execution_capabilities["authoringCatalog"]
                )
            )
            capability_core = {
                "schemaVersion": 1,
                "backendIdentity": ortools_backend.identity.to_dict(),
                "localization": self._ortools_localization_projection_v2(context),
                "executionReady": authoring_ready,
                "launchAdmissible": authoring_ready,
                "supportedOrigins": supported_origins,
                "supportedProofVisibility": supported_proof_visibility,
                "executionCapabilities": execution_capabilities,
                "hardPredicateCatalog": ortools_backend.catalog(
                    origin="search",
                    proof_mode="exact",
                    visibility="complete_state",
                    hardness="hard",
                ),
            }
            value["research"]["scenarioResearchV4"] = {
                **capability_core,
                "sha256": scenario_sha256(capability_core),
            }
        if context is not None:
            value["research"]["configurationSelection"] = {
                "schemaVersion": 1,
                "bundleRevisionId": context.bundle_revision_id,
                "bundleId": context.bundle.bundle_profile["profile"]["id"],
                "bundleVersion": context.bundle.bundle_profile["profile"]["version"],
                "semanticBundleHash": context.bundle.sha256,
                "effectiveRuntimeConfigHash": context.effective_runtime_config_hash,
                "contextHash": context.context_hash,
                "domainHash": context.compiled_domain.sha256,
                "domainIrHash": context.compiled_domain.compiled.domain_ir.sha256,
                "legacyDomainV2Hash": context.legacy_domain_v2.sha256,
                "ready": True,
            }
        return value

    def domain_compilation(
        self, compilation_hash: str, *, include_payloads: bool = False
    ) -> dict[str, Any] | None:
        compiled = None
        if (
            self.compiled_domain is not None
            and str(compilation_hash) == self.compiled_domain.compiled.sha256
        ):
            compiled = self.compiled_domain.compiled
        elif self.execution_context_factory is not None:
            selected = self.execution_context_factory.compiled_domain_for_hash(
                compilation_hash
            )
            if selected is not None:
                compiled = selected
        if compiled is None:
            return None
        return compiled.to_dict(include_payloads=include_payloads)

    def domain_compilation_coverage(
        self, compilation_hash: str
    ) -> dict[str, Any] | None:
        compiled = self.domain_compilation(compilation_hash)
        if compiled is None:
            return None
        return copy.deepcopy(dict(compiled["coverage"]))

    def _build_recipe_and_context(
        self, payload: Mapping[str, Any]
    ) -> tuple[ResearchRecipe, ResearchExecutionContextV1 | None]:
        # Recipe builders copy every nested contract section before validation.
        # Only this top-level mapping is rewritten while resolving its bundle.
        raw = dict(payload)
        version = int(raw.get("schemaVersion", 1))
        cache_key: str | None = None
        cache_size = 0
        if version == 4:
            payload_bytes = canonical_json_bytes_v1(raw)
            cache_key = hashlib.sha256(payload_bytes).hexdigest()
            cache_size = len(payload_bytes)
            with self._lock:
                cached = self._scenario_recipe_context_cache_v4.get(cache_key)
                if cached is not None:
                    return cached[0], cached[1]
        if version in {3, 4}:
            configuration = dict(raw.get("configuration") or {})
            selector_keys = {"schemaVersion", "bundleRevisionId"}
            if set(configuration) == selector_keys:
                configuration = build_research_configuration_selector_v1(raw["configuration"])
                if self.execution_context_factory is None:
                    raise ValueError(
                        "bundle-specific Research execution is unavailable"
                    )
                resources = dict(raw.get("resources") or {})
                analysis_mode = str(resources.get("analysisMode") or "cascaded")
                authority, context = self.execution_context_factory.freeze_selector(
                    bundle_revision_id=configuration["bundleRevisionId"],
                    analysis_mode=analysis_mode,
                )
                identities = authority.identities
                if version == 3:
                    raw["profiles"] = {
                        "domainHash": identities["domain"]["sha256"],
                        "bundleHash": authority.semantic_bundle_hash,
                        "rulesHash": identities["rules"]["sha256"],
                        "conventionHash": identities["convention"]["sha256"],
                        "agentRelease": identities["agent"]["id"],
                        "analysisProfileHash": identities["analyzer"]["sha256"],
                        "scoreProfileHash": identities["score"]["sha256"],
                    }
                raw["configuration"] = authority.to_dict()
                recipe = (
                    ResearchRecipeV3.build(raw)
                    if version == 3
                    else ResearchRecipeV4.build(raw)
                )
            else:
                recipe = (
                    ResearchRecipeV3.build(raw)
                    if version == 3
                    else ResearchRecipeV4.build(raw)
                )
                if self.execution_context_factory is None:
                    raise ValueError(
                        "bundle-specific Research execution is unavailable"
                    )
                context = self.execution_context_factory.resolve_recipe(recipe)
            if version == 4:
                assert cache_key is not None
                challenge = dict(context.bundle.component("challenge"))
                research = challenge.get("scenarioResearchV4")
                if isinstance(research, Mapping):
                    limits = dict(research["cacheLimits"])
                    maximum_entries = int(limits["maximumEntries"])
                    maximum_bytes = int(limits["maximumBytes"])
                    if cache_size <= maximum_bytes:
                        with self._lock:
                            while self._scenario_recipe_context_cache_v4 and (
                                len(self._scenario_recipe_context_cache_v4)
                                >= maximum_entries
                                or self._scenario_recipe_context_cache_v4_bytes
                                + cache_size
                                > maximum_bytes
                            ):
                                oldest = next(
                                    iter(self._scenario_recipe_context_cache_v4)
                                )
                                _recipe, _context, removed_size = (
                                    self._scenario_recipe_context_cache_v4.pop(oldest)
                                )
                                self._scenario_recipe_context_cache_v4_bytes -= (
                                    removed_size
                                )
                            if cache_key not in self._scenario_recipe_context_cache_v4:
                                self._scenario_recipe_context_cache_v4[cache_key] = (
                                    recipe,
                                    context,
                                    cache_size,
                                )
                                self._scenario_recipe_context_cache_v4_bytes += (
                                    cache_size
                                )
            return recipe, context
        recipe = build_research_recipe(raw)
        if self.execution_context_factory is None:
            return recipe, None
        return recipe, self.execution_context_factory.resolve_recipe(recipe)

    def preview_exact_trace_legal_actions(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Preview only legal operator actions under one published OR-Tools bundle."""

        raw = copy.deepcopy(dict(payload))
        required = {"schemaVersion", "configuration", "exactTracePrefix"}
        if not required <= set(raw) or set(raw) - required - {"recordedEscalationOpportunities"}:
            raise ValueError("exact trace preview fields are invalid")
        if raw.get("schemaVersion") != 1 or type(raw.get("schemaVersion")) is not int:
            raise ValueError("exact trace preview schemaVersion must be integer 1")
        configuration = build_research_configuration_selector_v1(raw["configuration"])
        bundle_revision_id = configuration["bundleRevisionId"]
        if self.execution_context_factory is None:
            raise ValueError("bundle-specific Research execution is unavailable")
        _authority, context = self.execution_context_factory.freeze_selector(
            bundle_revision_id=bundle_revision_id,
            analysis_mode="screening",
        )
        if context.bundle_revision_id != bundle_revision_id:
            raise ValueError("exact trace preview configuration authority mismatch")
        backend = self._cached_ortools_scenario_backend_v2(context)
        if backend is None:
            raise ValueError("selected bundle does not support OR-Tools Research V4")
        history_options = {}
        if "recordedEscalationOpportunities" in raw:
            execution = backend.authority.quality_execution
            history = raw["recordedEscalationOpportunities"]
            if (backend.identity.backend_id != "ortools_cp_sat_v3" or execution is None
                    or execution.escalation_opportunity_order is None or not isinstance(history, list)
                    or len(history) > execution.maximum_escalation_decisions):
                raise ValueError("recorded preview history requires governed native opportunity order and bounds")
            history_options = {"recorded_escalation_opportunities": history,
                "opportunity_order": execution.escalation_opportunity_order}
        validation_profiles = dict(
            backend.authority.search_budgets.get("profiles") or {}
        )
        validation_budget = dict(
            validation_profiles.get("interactive_validation_v1") or {}
        )
        maximum_seconds = float(validation_budget.get("maximumWallSeconds") or 1.0)
        authority = {
            "bundleRevisionId": context.bundle_revision_id,
            "semanticBundleHash": context.bundle.sha256,
            "effectiveRuntimeConfigHash": context.effective_runtime_config_hash,
            "backendIdentityHash": backend.identity.sha256,
        }
        try:
            prefix = ExactTracePrefixV1.build(raw["exactTracePrefix"])
            preview = OrToolsExactStateNormalizerV1(
                context.bundle.rules,
                native_cp_sat=(backend.identity.backend_id == "ortools_cp_sat_v3"),
            ).preview_legal_next_actions(
                prefix,
                maximum_seconds=maximum_seconds,
                **history_options,
            )
            self._validate_exact_adjudication_events_v1(
                prefix,
                self._adjudication_research_authority_v1(context),
                current_stage=str(preview["currentStage"]),
                next_actor=(
                    str(preview["nextActor"])
                    if preview["nextActor"] is not None
                    else None
                ),
            )
        except (AssertionError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            core = {
                "schemaVersion": 1,
                "configurationAuthority": authority,
                "valid": False,
                "currentStage": None,
                "nextActor": None,
                "terminal": False,
                "legalActions": [],
                "diagnostics": [
                    {
                        "code": (
                            "exact_normalization_incomplete"
                            if isinstance(exc, ExactNormalizationStopped)
                            else "exact_trace_prefix_invalid"
                        ),
                        "fieldPath": "exactTracePrefix",
                    }
                ],
                "prefixHash": None,
                "previewStateHash": None,
                "previewReceiptHash": None,
            }
            return {**core, "sha256": scenario_sha256(core)}
        core = {
            "schemaVersion": 1,
            "configurationAuthority": authority,
            "valid": True,
            "currentStage": preview["currentStage"],
            "nextActor": preview["nextActor"],
            "terminal": preview["terminal"],
            "legalActions": preview["legalActions"],
            "diagnostics": [],
            "prefixHash": preview["prefixHash"],
            "previewStateHash": preview["previewStateHash"],
            "previewReceiptHash": preview["previewReceiptHash"],
        }
        return {**core, "sha256": scenario_sha256(core)}

    def _new_recipe_authoring_issue(
        self, context: ResearchExecutionContextV1 | None,
    ) -> dict[str, Any] | None:
        reason = None
        if context is None or context.bundle.rules.project_domain.declaration_mode != "mandatory":
            reason = "new Research recipes require mandatory project declarations; optional profiles are historical-only"
        else:
            backend = self._cached_ortools_scenario_backend_v2(context)
            if backend is None or backend.identity.backend_id != "ortools_cp_sat_v3":
                reason = "new Research V4 authoring requires a published bundle selecting ortools_cp_sat_v3"
        if reason is None:
            return None
        return ScenarioDiagnosticV1.build({
            "issueId": "ISS-V4-NEW-AUTHORING",
            "class": "rules_inapplicable",
            "severity": "error",
            "code": "configuration_context_unavailable",
            "predicateIds": [],
            "fieldPaths": ["configuration.bundleRevisionId"],
            "messageArKey": "research_configuration_unavailable_help",
            "messageEnKey": "research_configuration_unavailable_help",
            "details": {"reason": reason},
        }).to_dict()

    def validate_recipe(
        self, payload: Mapping[str, Any], *, new_authoring: bool = False,
    ) -> dict[str, Any]:
        recipe, context = self._build_recipe_and_context(payload)
        if isinstance(recipe, ResearchRecipeV4):
            if new_authoring and (issue := self._new_recipe_authoring_issue(context)):
                return {"valid": False, "recipe": recipe.to_dict(), "issues": [issue]}
            return self._validate_recipe_v4_cached(recipe, context)
        domain = context.legacy_domain_v2 if context is not None else self.domain
        simulator = context.simulator if context is not None else self.simulator
        analyzers = context.analyzers if context is not None else self.analyzers
        score_profile_hash = (
            context.score_profile_hash
            if context is not None
            else self.score_profile_hash
        )
        tactical_verifier = (
            context.tactical_verifier if context is not None else self.tactical_verifier
        )
        maximum_tactical_decisions = (
            context.maximum_tactical_decisions
            if context is not None
            else self.maximum_tactical_decisions
        )

        issues: list[dict[str, Any]] = []
        incomplete_primitives = [
            item["id"]
            for item in domain.rule_primitives
            if not all(
                bool(item.get(key))
                for key in (
                    "runtimeCoverage",
                    "solverCoverage",
                    "documentationCoverage",
                    "testCoverage",
                )
            )
        ]
        if incomplete_primitives:
            issues.append(
                {
                    "severity": "error",
                    "code": "incomplete_rule_primitive_coverage",
                    "field": "profiles.rulesHash",
                    "messageAr": self._localized(
                        "research_issue_incomplete_rule_coverage", context
                    ),
                    "messageEn": "At least one active rule primitive lacks complete runtime, compiler, documentation, or test coverage.",
                    "primitiveIds": incomplete_primitives,
                }
            )
        profiles = dict(recipe.profiles)
        expected = {
            "domainHash": (
                context.compiled_domain.sha256
                if context is not None and isinstance(recipe, ResearchRecipeV3)
                else domain.sha256
            ),
            "bundleHash": domain.bundle_hash,
            "rulesHash": domain.rules_hash,
            "conventionHash": domain.convention_hash,
        }
        for key, actual in expected.items():
            if profiles[key] != actual:
                issues.append(
                    {
                        "severity": "error",
                        "code": f"stale_{key}",
                        "field": f"profiles.{key}",
                        "messageAr": self._localized(
                            "research_issue_stale_domain", context
                        ),
                        "messageEn": f"{key} does not match the active published domain.",
                    }
                )
        expected_analysis = (
            str(context.analysis_identity_hashes[recipe.resources["analysisMode"]])
            if context is not None and isinstance(recipe, ResearchRecipeV3)
            else str(analyzers[recipe.resources["analysisMode"]].profile.sha256)
        )
        for key, actual in (
            ("agentRelease", simulator.agent_release),
            ("analysisProfileHash", expected_analysis),
            ("scoreProfileHash", score_profile_hash),
        ):
            if profiles[key] != actual:
                issues.append(
                    {
                        "severity": "error",
                        "code": f"stale_{key}",
                        "field": f"profiles.{key}",
                        "messageAr": self._localized(
                            "research_issue_stale_runtime", context
                        ),
                        "messageEn": f"{key} does not match the selected governed research runtime.",
                    }
                )
        hard_projects = [
            item
            for item in recipe.situation["projects"]
            if item.get("mode", "hard") == "hard"
        ]
        if len(hard_projects) > 1:
            issues.append(
                {
                    "severity": "error",
                    "code": "multiple_hard_projects_not_compiled",
                    "field": "situation.projects",
                    "messageAr": self._localized(
                        "research_issue_multiple_hard_projects", context
                    ),
                    "messageEn": "The current compiler supports one hard project constraint per recipe.",
                }
            )
        active_project_options = {
            (
                str(item["contract"]),
                str(item["kind"]),
                int(item["value"]),
                str(item["highRank"]),
            )
            for item in self._recipe_configuration(context)["projectOptions"]
        }
        for project in hard_projects:
            requested_project = (
                str(project["contract"]),
                str(project["kind"]),
                int(project["value"]),
                str(project["highRank"]),
            )
            if requested_project not in active_project_options:
                issues.append(
                    {
                        "severity": "error",
                        "code": "project_constraint_not_in_active_rules",
                        "field": "situation.projects",
                        "messageAr": self._localized(
                            "research_issue_project_not_in_active_rules",
                            context,
                        ),
                        "messageEn": (
                            "The requested project combination is not produced by "
                            "the active published project rules."
                        ),
                    }
                )
        if recipe.output["kind"] == "sakka":
            issues.append(
                {
                    "severity": "error",
                    "code": "sakka_requires_verified_composer",
                    "field": "output.kind",
                    "messageAr": self._localized(
                        "research_issue_sakka_composer", context
                    ),
                    "messageEn": "Research qualified Laffat first, then use the governed Sakka composer with score-continuity verification.",
                }
            )
        if recipe.output["kind"] == "contextualized_laffa":
            issues.append(
                {
                    "severity": "error",
                    "code": "contextualized_laffa_requires_context_search",
                    "field": "output.kind",
                    "messageAr": self._localized(
                        "research_issue_context_search", context
                    ),
                    "messageEn": "A contextualized Laffa must descend from a qualified Laffa through governed context search.",
                }
            )
        if recipe.situation["escalations"] and self.escalation_compiler is None:
            issues.append(
                {
                    "severity": "error",
                    "code": "escalation_compiler_unavailable",
                    "field": "situation.escalations",
                    "messageAr": self._localized(
                        "research_issue_escalation_unavailable", context
                    ),
                    "messageEn": "The governed escalation compiler is unavailable in the active runtime.",
                }
            )
        hard_tactics = [
            item["kind"] for item in recipe.tactics if item["mode"] == "hard"
        ]
        if hard_tactics:
            issues.append(
                {
                    "severity": "warning",
                    "code": "tactics_verified_after_semantic_replay",
                    "field": "tactics",
                    "messageAr": self._localized(
                        "research_issue_tactics_replay_warning", context
                    ),
                    "messageEn": "OR-Tools guides construction; tactical truth is proven after replay and analysis.",
                }
            )
        if self._tactical_verification_required(recipe) and (
            tactical_verifier is None or maximum_tactical_decisions == 0
        ):
            issues.append(
                {
                    "severity": "error",
                    "code": "tactical_verifier_unavailable",
                    "field": "tactics",
                    "messageAr": self._localized(
                        "research_issue_tactical_verifier_unavailable",
                        context,
                    ),
                    "messageEn": (
                        "The active runtime cannot produce the requested tactical "
                        "qualification or ranking evidence."
                    ),
                }
            )
        if (
            recipe.resources["generationLane"] == "ortools"
            and not self.deal_generator.available()
        ):
            issues.append(
                {
                    "severity": "error",
                    "code": "ortools_unavailable",
                    "field": "resources",
                    "messageAr": self._localized(
                        "research_issue_ortools_unavailable", context
                    ),
                    "messageEn": "OR-Tools is unavailable in the active runtime.",
                }
            )
        valid = not any(item["severity"] == "error" for item in issues)
        return {"valid": valid, "recipe": recipe.to_dict(), "issues": issues}

    def _validate_recipe_v4_cached(
        self,
        recipe: ResearchRecipeV4,
        context: ResearchExecutionContextV1 | None,
    ) -> dict[str, Any]:
        backend = self._cached_ortools_scenario_backend_v2(context)
        if backend is None:
            return self._validate_recipe_v4(recipe, context)
        cache_key = canonical_sha256(
            {
                "recipeHash": recipe.sha256,
                "backendIdentityHash": backend.identity.sha256,
            }
        )
        with self._lock:
            cached = self._scenario_validation_cache_v4.get(cache_key)
            if cached is not None:
                return copy.deepcopy(cached[0])
        result = self._validate_recipe_v4_ortools(recipe, context, backend)
        # Budget exhaustion is an execution observation, not a cached verdict.
        if any(issue.get("class") == "normalization_incomplete" for issue in result.get("issues", ())):
            return result
        size = len(canonical_json_bytes_v1(result))
        maximum_entries = int(backend.authority.cache_limits["maximumEntries"])
        maximum_bytes = int(backend.authority.cache_limits["maximumBytes"])
        if size > maximum_bytes:
            return result
        with self._lock:
            cached = self._scenario_validation_cache_v4.get(cache_key)
            if cached is not None:
                return copy.deepcopy(cached[0])
            while self._scenario_validation_cache_v4 and (
                len(self._scenario_validation_cache_v4) >= maximum_entries
                or self._scenario_validation_cache_v4_bytes + size > maximum_bytes
            ):
                oldest = next(iter(self._scenario_validation_cache_v4))
                _value, removed_size = self._scenario_validation_cache_v4.pop(oldest)
                self._scenario_validation_cache_v4_bytes -= removed_size
            self._scenario_validation_cache_v4[cache_key] = (
                copy.deepcopy(result),
                size,
            )
            self._scenario_validation_cache_v4_bytes += size
        return result

    def _validate_recipe_v4(
        self,
        recipe: ResearchRecipeV4,
        context: ResearchExecutionContextV1 | None,
    ) -> dict[str, Any]:
        """Validate V4 without projecting it into the legacy V3 language."""

        ortools_backend = self._cached_ortools_scenario_backend_v2(context)
        if ortools_backend is not None:
            return self._validate_recipe_v4_ortools(recipe, context, ortools_backend)

        issues = [item.to_dict() for item in recipe.capability_plan.diagnostics]
        normalized_exact_state: dict[str, Any] | None = None
        if context is None:
            issues.append(
                {
                    "issueId": "ISS-V4-CONTEXT",
                    "class": "rules_inapplicable",
                    "severity": "error",
                    "code": "configuration_context_unavailable",
                    "predicateIds": [],
                    "fieldPaths": ["configuration.bundleRevisionId"],
                    "messageArKey": "research_scenario_configuration_unavailable",
                    "messageEnKey": "research_scenario_configuration_unavailable_en",
                    "details": {},
                    "sha256": scenario_sha256(
                        {
                            "code": "configuration_context_unavailable",
                            "recipeHash": recipe.sha256,
                        }
                    ),
                }
            )
        else:
            issues.extend(
                self._scenario_rules_applicability_issues(
                    recipe,
                    context.simulator.rules,
                    supported_tactics=context.legacy_domain_v2.tactical_patterns,
                )
            )
            authority = dict(recipe.configuration)
            if (
                str(authority["bundleRevisionId"]) != context.bundle_revision_id
                or str(authority["semanticBundleHash"]) != context.bundle.sha256
                or str(authority["effectiveRuntimeConfigHash"])
                != context.effective_runtime_config_hash
            ):
                issues.append(
                    {
                        "issueId": "ISS-V4-AUTHORITY",
                        "class": "rules_inapplicable",
                        "severity": "error",
                        "code": "configuration_authority_mismatch",
                        "predicateIds": [],
                        "fieldPaths": ["configuration"],
                        "messageArKey": "research_scenario_configuration_mismatch",
                        "messageEnKey": "research_scenario_configuration_mismatch_en",
                        "details": {},
                        "sha256": scenario_sha256(
                            {
                                "code": "configuration_authority_mismatch",
                                "recipeHash": recipe.sha256,
                            }
                        ),
                    }
                )
            if recipe.scenario.origin.exact_trace_prefix is not None:
                try:
                    normalized_exact_state = (
                        ExactStateNormalizerV1(context.simulator.rules)
                        .normalize(recipe.scenario.origin.exact_trace_prefix)
                        .to_dict()
                    )
                except (AssertionError, KeyError, TypeError, ValueError) as exc:
                    issues.append(
                        {
                            "issueId": "ISS-V4-EXACT-STATE",
                            "class": "schema_invalid",
                            "severity": "error",
                            "code": "exact_trace_prefix_invalid",
                            "predicateIds": [],
                            "fieldPaths": ["scenario.origin.exactTracePrefix"],
                            "messageArKey": "research_scenario_exact_trace_invalid",
                            "messageEnKey": "research_scenario_exact_trace_invalid_en",
                            "details": {
                                "errorType": type(exc).__name__,
                                "message": str(exc),
                            },
                            "sha256": scenario_sha256(
                                {
                                    "code": "exact_trace_prefix_invalid",
                                    "message": str(exc),
                                    "recipeHash": recipe.sha256,
                                }
                            ),
                        }
                    )
        if (
            recipe.resources["generationLane"] == "ortools"
            and not self.deal_generator.available()
        ):
            issues.append(
                {
                    "issueId": "ISS-V4-ORTOOLS",
                    "class": "rules_inapplicable",
                    "severity": "error",
                    "code": "ortools_unavailable",
                    "predicateIds": [],
                    "fieldPaths": ["resources.generationLane"],
                    "messageArKey": "research_issue_ortools_unavailable",
                    "messageEnKey": "research_issue_ortools_unavailable_en",
                    "details": {},
                    "sha256": scenario_sha256(
                        {
                            "code": "ortools_unavailable",
                            "recipeHash": recipe.sha256,
                        }
                    ),
                }
            )
        valid = not any(item["severity"] == "error" for item in issues)
        result: dict[str, Any] = {
            "schemaVersion": 2,
            "valid": valid,
            "recipe": recipe.to_dict(),
            "scenarioPlanHash": recipe.capability_plan.sha256,
            "configurationAuthority": copy.deepcopy(dict(recipe.configuration)),
            "capabilityPlan": recipe.capability_plan.to_dict(),
            "issues": issues,
        }
        if normalized_exact_state is not None:
            result["normalizedExactState"] = normalized_exact_state
        return result

    def _validate_recipe_v4_ortools(
        self,
        recipe: ResearchRecipeV4,
        context: ResearchExecutionContextV1,
        backend: OrToolsScenarioResearchBackendV2,
    ) -> dict[str, Any]:
        preflight = backend.preflight(recipe)
        native_plan = (
            backend.native_capability_plan(recipe)
            if backend.authority.backend == "ortools_cp_sat_v3"
            else None
        )
        workflow_copy = self._ortools_localization_projection_v2(context)["groups"][
            "workflow"
        ]
        issues = [
            self._ortools_preflight_issue_v2(
                item, recipe_hash=recipe.sha256, workflow_copy=workflow_copy
            )
            for item in preflight["issues"]
        ]
        requested_workers = recipe.resources.get("shardWorkerCount")
        search_profile = dict(
            (backend.authority.search_budgets.get("profiles") or {}).get(
                recipe.proof.budget_profile
            ) or {}
        )
        if requested_workers is not None and requested_workers > int(search_profile.get("maximumShardWorkers", 0)):
            issues.append(ScenarioDiagnosticV1.build({
                "issueId": "ISS-V4-SHARD-WORKER-BUDGET",
                "class": "schema_invalid",
                "severity": "error",
                "code": "shard_worker_budget_exceeded",
                "predicateIds": [],
                "fieldPaths": ["resources.shardWorkerCount"],
                "messageArKey": "research_scenario_help_shard_worker_count",
                "messageEnKey": "research_scenario_help_shard_worker_count_en",
                "details": {
                    "requested": requested_workers,
                    "maximum": search_profile.get("maximumShardWorkers"),
                },
            }).to_dict())
        authority = dict(recipe.configuration)
        if (
            str(authority["bundleRevisionId"]) != context.bundle_revision_id
            or str(authority["semanticBundleHash"]) != context.bundle.sha256
            or str(authority["effectiveRuntimeConfigHash"])
            != context.effective_runtime_config_hash
        ):
            issues.append(
                ScenarioDiagnosticV1.build(
                    {
                        "issueId": "ISS-V4-ORTOOLS-AUTHORITY",
                        "class": "rules_inapplicable",
                        "severity": "error",
                        "code": "configuration_authority_mismatch",
                        "predicateIds": [],
                        "fieldPaths": ["configuration"],
                        "messageArKey": "research_scenario_configuration_mismatch",
                        "messageEnKey": "research_scenario_configuration_mismatch_en",
                        "details": {"backendIdentityHash": backend.identity.sha256},
                    }
                ).to_dict()
            )
        normalized_exact_state: dict[str, Any] | None = None
        prefix = recipe.scenario.origin.exact_trace_prefix
        if prefix is not None and prefix.adjudication_events:
            adjudication_authority = self._adjudication_research_authority_v1(context)
            for event in prefix.adjudication_events:
                try:
                    if adjudication_authority is None:
                        raise ValueError(
                            "selected bundle has no adjudication authority"
                        )
                    adjudication_authority.validate_authored_event(event)
                except (KeyError, TypeError, ValueError) as exc:
                    issues.append(
                        ScenarioDiagnosticV1.build(
                            {
                                "issueId": (
                                    f"ISS-V4-ADJUDICATION-{event.sha256[:16].upper()}"
                                ),
                                "class": "schema_invalid",
                                "severity": "error",
                                "code": "exact_adjudication_event_invalid",
                                "predicateIds": [],
                                "fieldPaths": [
                                    "scenario.origin.exactTracePrefix.adjudicationEvents"
                                ],
                                "messageArKey": (
                                    "research_scenario_exact_trace_invalid"
                                ),
                                "messageEnKey": (
                                    "research_scenario_exact_trace_invalid_en"
                                ),
                                "details": {
                                    "eventHash": event.sha256,
                                    "errorType": type(exc).__name__,
                                    "errorHash": scenario_sha256(
                                        {
                                            "type": type(exc).__name__,
                                            "message": str(exc),
                                        }
                                    ),
                                },
                            }
                        ).to_dict()
                    )
        if prefix is not None and not issues:
            validation_profiles = dict(
                backend.authority.search_budgets.get("profiles") or {}
            )
            validation_budget = dict(
                validation_profiles.get("interactive_validation_v1") or {}
            )
            maximum_seconds = float(validation_budget.get("maximumWallSeconds") or 1.0)
            try:
                normalized = OrToolsExactStateNormalizerV1(
                    context.bundle.rules,
                    native_cp_sat=(backend.identity.backend_id == "ortools_cp_sat_v3"),
                ).normalize(
                    prefix,
                    decision_point=recipe.scenario.decision_point,
                    maximum_seconds=maximum_seconds,
                    **({"recorded_escalation_opportunities": recipe.scenario.origin.recorded_escalation_opportunities,
                        "opportunity_order": backend.authority.quality_execution.escalation_opportunity_order}
                       if backend.identity.backend_id == "ortools_cp_sat_v3"
                       and backend.authority.quality_execution is not None
                       and recipe.scenario.decision_point["kind"] == "escalation"
                       and recipe.scenario.origin.recorded_escalation_opportunities else {}),
                )
                self._validate_exact_adjudication_events_v1(
                    prefix,
                    self._adjudication_research_authority_v1(context),
                    current_stage={
                        "bid": "auction",
                        "escalation": "escalation",
                        "project": "projects",
                        "play": "play",
                    }[str(recipe.scenario.decision_point["kind"])],
                    next_actor=normalized.next_actor,
                )
                normalized_exact_state = normalized.to_dict()
            except ExactNormalizationStopped as exc:
                source_stage = exc.partial_receipt["evidence"].get("sourceStage")
                issues.append(
                    ScenarioDiagnosticV1.build(
                        {
                            "issueId": "ISS-V4-ORTOOLS-EXACT-NORMALIZATION-INCOMPLETE",
                            "class": "normalization_incomplete",
                            "severity": "error",
                            "code": "exact_normalization_incomplete",
                            "predicateIds": [],
                            "fieldPaths": ["scenario.origin.exactTracePrefix"],
                            "messageArKey": "research_scenario_support_incomplete",
                            "messageEnKey": "research_scenario_support_incomplete_en",
                            "details": {
                                "backendIdentityHash": backend.identity.sha256,
                                "exhaustionReason": exc.partial_receipt["exhaustionReason"],
                                "stage": "exact_normalization",
                                "partialReceiptHash": exc.partial_receipt["sha256"],
                                **({"sourceStage": source_stage} if isinstance(source_stage, str) else {}),
                            },
                        }
                    ).to_dict()
                )
            except (
                AssertionError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                issue_code = str(getattr(exc, "code", "exact_trace_prefix_invalid"))
                issue_path = (
                    "scenario.decisionPoint"
                    if issue_code == "project_decision_not_strategic"
                    else "scenario.origin.exactTracePrefix"
                )
                issues.append(
                    ScenarioDiagnosticV1.build(
                        {
                            "issueId": "ISS-V4-ORTOOLS-EXACT-STATE",
                            "class": "schema_invalid",
                            "severity": "error",
                            "code": issue_code,
                            "predicateIds": [],
                            "fieldPaths": [issue_path],
                            "messageArKey": "research_scenario_exact_trace_invalid",
                            "messageEnKey": "research_scenario_exact_trace_invalid_en",
                            "details": {
                                "backendIdentityHash": backend.identity.sha256,
                                "errorType": type(exc).__name__,
                                "errorHash": scenario_sha256(
                                    {
                                        "type": type(exc).__name__,
                                        "message": str(exc),
                                    }
                                ),
                            },
                        }
                    ).to_dict()
                )
        result = {
            "schemaVersion": 2,
            "valid": not any(item["severity"] == "error" for item in issues),
            "recipe": recipe.to_dict(),
            "scenarioPlanHash": recipe.capability_plan.sha256,
            "configurationAuthority": copy.deepcopy(dict(recipe.configuration)),
            "capabilityPlan": recipe.capability_plan.to_dict(),
            "nativeCapabilityPlan": native_plan,
            "issues": issues,
        }
        if normalized_exact_state is not None:
            result["normalizedExactState"] = normalized_exact_state
        return result

    def estimate(
        self, payload: Mapping[str, Any], *, new_authoring: bool = False,
    ) -> dict[str, Any]:
        recipe, context = self._build_recipe_and_context(payload)
        if isinstance(recipe, ResearchRecipeV4):
            issue = self._new_recipe_authoring_issue(context) if new_authoring else None
            validated = (
                {"valid": False, "recipe": recipe.to_dict(), "issues": [issue]}
                if issue else self._validate_recipe_v4_cached(recipe, context)
            )
            return self._estimate_recipe_v4(recipe, validated, context)
        validated = self.validate_recipe(payload)
        resources = recipe.resources
        seed_count = int(resources["seedCount"])
        candidates = (
            seed_count * int(resources["candidatesPerSeed"])
            if resources["generationLane"] == "ortools"
            else seed_count
        )
        estimate = ResearchEstimateV1(
            seed_count,
            candidates,
            math.ceil(seed_count / int(resources["seedsPerShard"])),
            (
                candidates * float(resources["solverTimeoutSeconds"])
                if resources["generationLane"] == "ortools"
                else 0.0
            ),
            candidates * 40,
            tuple(
                item["messageAr"]
                for item in validated["issues"]
                if item["severity"] == "warning"
            ),
        )
        tactical_upper_bound = (
            candidates
            * (
                context.maximum_tactical_decisions
                if context is not None
                else self.maximum_tactical_decisions
            )
            if self._tactical_verification_required(recipe)
            else 0
        )
        return {
            **estimate.to_dict(),
            "tacticalVerificationDecisionUpperBound": tactical_upper_bound,
            "valid": validated["valid"],
            "issues": validated["issues"],
        }

    def _estimate_recipe_v4(
        self,
        recipe: ResearchRecipeV4,
        validated: Mapping[str, Any],
        context: ResearchExecutionContextV1 | None,
    ) -> dict[str, Any]:
        ortools_backend = self._cached_ortools_scenario_backend_v2(context)
        if ortools_backend is not None:
            return self._estimate_recipe_v4_ortools(
                recipe,
                validated,
                ortools_backend,
            )
        resources = recipe.resources
        seed_count = int(resources["seedCount"])
        candidate_count = (
            1 if recipe.scenario.origin.kind == "exact_trace_prefix" else seed_count
        )
        exact_state = validated.get("normalizedExactState")
        decision_kind = str(recipe.scenario.decision_point["kind"])
        selector = dict(recipe.scenario.decision_point.get("selector") or {})
        normalized_state = (
            dict(exact_state.get("state") or {})
            if isinstance(exact_state, Mapping)
            else {}
        )
        remaining_cards: int | None = None
        hand_counts: list[int] = []
        if isinstance(exact_state, Mapping):
            hands = normalized_state.get("hands") or {}
            if isinstance(hands, Mapping):
                hand_counts = [len(cards) for cards in hands.values()]
                remaining_cards = sum(hand_counts)
        elif recipe.scenario.origin.kind == "search" and decision_kind == "play":
            remaining_cards = int(selector.get("remainingCards", 8))
            quotient, remainder = divmod(remaining_cards, 4)
            hand_counts = [
                quotient + (1 if index < remainder else 0) for index in range(4)
            ]
        requested_mode = recipe.proof.mode
        strategy_class = getattr(recipe.proof, "strategy_class", None) or (
            "frozen_policy"
            if requested_mode == "policy_conditioned"
            else "pure_deterministic"
        )
        selected_budget_profile = str(recipe.proof.budget_profile)
        resource_budget_profile = str(resources["proofBudgetProfile"])
        budget_registry = getattr(self, "_scenario_proof_budget_registry", None)
        if not isinstance(budget_registry, ScenarioProofBudgetRegistryV1):
            budget_registry = ScenarioProofBudgetRegistryV1.load_default()
            self._scenario_proof_budget_registry = budget_registry
        try:
            budget = budget_registry.get(selected_budget_profile)
        except ValueError:
            budget = None

        requested_depth: int | None = None
        if requested_mode == "exact":
            requested_depth = remaining_cards
        elif requested_mode == "bounded":
            horizon = dict(recipe.proof.horizon or {})
            unit = str(horizon.get("unit") or "")
            value = int(horizon.get("value") or 0)
            requested_depth = (
                remaining_cards
                if unit == "round_end"
                else value * 4
                if unit == "tricks"
                else value
            )
        elif requested_mode == "policy_conditioned":
            requested_depth = (
                len(FULL_DECK) + (2 * len(Seat))
                if decision_kind == "bid"
                else (remaining_cards + 1)
                if decision_kind in {"escalation", "project"}
                and remaining_cards is not None
                else remaining_cards
            )

        estimated_hidden_worlds: int | None = None
        if recipe.proof.visibility == "actor_information_set" and decision_kind in {
            "bid",
            "escalation",
            "project",
        }:
            group_sizes: list[int] = []
            if decision_kind == "bid":
                # The actor sees five cards plus the public purchase card. The
                # other initial hands and every exact undealt position remain
                # distinct because later purchase ownership can change.
                group_sizes = [5, 5, 5, *([1] * 11)]
            elif normalized_state:
                actor = str(
                    exact_state.get("nextActor")
                    if isinstance(exact_state, Mapping)
                    else selector.get("actingSeat") or selector.get("owner") or ""
                )
                fixed: dict[int, str] = {}

                def bind(raw_card: Any, seat: str) -> None:
                    if isinstance(raw_card, Mapping):
                        card_id = raw_card.get("id")
                    else:
                        card_id = raw_card
                    if type(card_id) is int:
                        existing = fixed.get(card_id)
                        if existing not in {None, seat}:
                            raise ValueError(
                                "observer-visible card ownership is contradictory"
                            )
                        fixed[card_id] = seat

                initial_hands = normalized_state.get("initialHands")
                if isinstance(initial_hands, Mapping) and isinstance(
                    initial_hands.get(actor), list
                ):
                    for card in initial_hands[actor]:
                        bind(card, actor)
                recipient = normalized_state.get("purchaseRecipient")
                purchase_card = normalized_state.get("purchaseCard")
                if isinstance(recipient, str) and purchase_card is not None:
                    bind(purchase_card, recipient)
                for project in normalized_state.get("projects") or ():
                    if (
                        not isinstance(project, Mapping)
                        or project.get("revealed") is not True
                    ):
                        continue
                    owner = project.get("owner")
                    if not isinstance(owner, str):
                        continue
                    for card in project.get("cards") or ():
                        bind(card, owner)
                counts = {seat: 0 for seat in ("north", "west", "south", "east")}
                for owner in fixed.values():
                    counts[owner] += 1
                group_sizes = [8 - counts[seat] for seat in counts]
            else:
                # Search has no candidate world yet. Use the largest complete-
                # hand hidden partition and let the exact candidate preflight
                # refine it before proof.
                group_sizes = [0, 8, 8, 8]
            if group_sizes and min(group_sizes) >= 0:
                estimated_hidden_worlds = math.factorial(sum(group_sizes))
                for size in group_sizes:
                    estimated_hidden_worlds //= math.factorial(size)
        elif recipe.proof.visibility == "actor_information_set" and hand_counts:
            allocations: list[int] = []
            for actor_index in range(len(hand_counts)):
                sizes = [
                    count
                    for index, count in enumerate(hand_counts)
                    if index != actor_index
                ]
                unknown = sum(sizes)
                worlds = 1
                for size in sizes[:-1]:
                    worlds *= math.comb(unknown, size)
                    unknown -= size
                allocations.append(worlds)
            estimated_hidden_worlds = max(allocations, default=1)

        legal_root_count = (
            len(exact_state["legalActions"])
            if isinstance(exact_state, Mapping)
            else max(hand_counts, default=32)
        )
        base_action_domain_upper_bound = max(
            [1, legal_root_count, *hand_counts],
        )
        exact_contract = (
            str(dict(exact_state.get("state") or {}).get("contract") or "")
            if isinstance(exact_state, Mapping)
            else ""
        )
        required_contracts: set[str] = set()
        try:
            required_contracts = {
                str(predicate.arguments.get("contract") or "")
                for predicate in recipe.scenario.constraints.predicates()
                if predicate.hardness == "hard" and predicate.kind == "auction.contract"
            }
        except AttributeError:
            required_contracts = set()
        hokum_possible = (
            exact_contract == "hokum"
            if exact_contract
            else required_contracts != {"sun"}
        )
        ekka_action_variants = 3 if hokum_possible else 0
        action_domain_upper_bound = min(
            11,
            base_action_domain_upper_bound + ekka_action_variants,
        )
        ekka_sequence_exponent = (
            math.ceil(requested_depth / 4)
            if hokum_possible and requested_depth is not None
            else 0
        )
        ekka_sequence_expression = _power_expression(2, ekka_sequence_exponent)
        forecast_depth_within_budget = (
            requested_depth is not None
            and budget is not None
            and requested_depth
            <= max(budget.maximum_depth, budget.maximum_horizon_actions)
        )
        ekka_sequence_factor = (
            1 << ekka_sequence_exponent if forecast_depth_within_budget else None
        )
        target_information_sets: int | None = None
        opponent_information_sets: int | None = None
        target_policy_count: str | None = None
        opponent_policy_count: str | None = None
        policy_product: str | None = None
        tree_state_upper_bound: int | None = None
        policy_product_exceeds_budget: bool | None = None
        information_tree_exceeds_budget: bool | None = None
        if (
            requested_mode == "policy_conditioned"
            and requested_depth is not None
            and budget is not None
        ):
            policy_worlds = estimated_hidden_worlds or 1
            tree_state_upper_bound = (
                policy_worlds * max(1, legal_root_count) * (1 + requested_depth)
            )
            information_tree_exceeds_budget = (
                tree_state_upper_bound > budget.maximum_state_space
            )
        if (
            recipe.proof.visibility == "actor_information_set"
            and requested_mode in {"exact", "bounded"}
            and requested_depth is not None
            and estimated_hidden_worlds is not None
            and forecast_depth_within_budget
            and estimated_hidden_worlds <= budget.maximum_hidden_worlds
        ):
            branching = max(1, action_domain_upper_bound)
            continuation_depth = max(0, requested_depth - 1)
            information_sets_per_world = (
                _geometric_state_upper_bound(branching, continuation_depth - 1)
                if continuation_depth
                else 0
            )
            # Each side is conservatively allowed every information-set node;
            # runtime merging and seat parity can only reduce these counts.
            target_information_sets = (
                estimated_hidden_worlds * information_sets_per_world
            )
            opponent_information_sets = target_information_sets
            target_policy_count = _power_expression(branching, target_information_sets)
            opponent_policy_count = _power_expression(
                branching, opponent_information_sets
            )
            policy_exponent = target_information_sets + opponent_information_sets
            policy_product = _power_expression(branching, policy_exponent)
            tree_state_upper_bound = (
                action_domain_upper_bound
                * estimated_hidden_worlds
                * _geometric_state_upper_bound(branching, continuation_depth)
                * int(ekka_sequence_factor or 1)
            )
            information_tree_exceeds_budget = (
                tree_state_upper_bound > budget.maximum_state_space
            )
            policy_product_exceeds_budget = _power_exceeds_limit(
                branching,
                policy_exponent,
                multiplier=(
                    action_domain_upper_bound
                    * estimated_hidden_worlds
                    * max(1, requested_depth)
                ),
                additive=tree_state_upper_bound,
                limit=budget.maximum_state_space,
            )

        estimated_state_space: int | None = None
        if requested_depth is not None and hand_counts:
            if requested_mode == "exact":
                leaves = math.prod(math.factorial(count) for count in hand_counts)
            elif requested_mode == "bounded" and (
                budget is None
                or requested_depth
                <= max(budget.maximum_depth, budget.maximum_horizon_actions)
            ):
                leaves = max(hand_counts, default=1) ** requested_depth
            else:
                leaves = 0
            if leaves and ekka_sequence_factor is not None:
                leaves *= ekka_sequence_factor
            if leaves:
                estimated_state_space = (1 + requested_depth * leaves) * (
                    estimated_hidden_worlds or 1
                )

        refusal_reason: str | None = None
        if selected_budget_profile != resource_budget_profile:
            refusal_reason = "proof_budget_profile_mismatch"
        elif budget is None:
            refusal_reason = "unknown_proof_budget_profile"
        elif (
            estimated_hidden_worlds is not None
            and estimated_hidden_worlds > budget.maximum_hidden_worlds
        ):
            refusal_reason = "hidden_worlds_exceed_proof_budget"
        elif (
            requested_mode == "policy_conditioned"
            and requested_depth is not None
            and requested_depth > budget.maximum_horizon_actions
        ):
            refusal_reason = "policy_horizon_exceeds_proof_budget"
        elif (
            requested_mode == "policy_conditioned"
            and requested_depth is not None
            and requested_depth > budget.maximum_depth
        ):
            refusal_reason = "policy_depth_exceeds_proof_budget"
        elif information_tree_exceeds_budget is True:
            refusal_reason = "estimated_information_tree_exceeds_proof_budget"
        elif policy_product_exceeds_budget is True:
            refusal_reason = "estimated_pure_policy_space_exceeds_proof_budget"
        elif requested_mode == "exact":
            if remaining_cards is None:
                refusal_reason = "exact_state_required_for_exact_admission"
            elif remaining_cards > budget.maximum_depth:
                refusal_reason = "exact_depth_exceeds_proof_budget"
            elif (
                estimated_state_space is not None
                and estimated_state_space > budget.maximum_state_space
            ):
                refusal_reason = "estimated_state_space_exceeds_exact_profile"
        elif requested_mode == "bounded" and requested_depth is not None:
            if requested_depth > budget.maximum_horizon_actions:
                refusal_reason = "bounded_horizon_exceeds_proof_budget"
            elif requested_depth > budget.maximum_depth:
                refusal_reason = "bounded_depth_exceeds_proof_budget"
            elif (
                estimated_state_space is not None
                and estimated_state_space > budget.maximum_state_space
            ):
                refusal_reason = "estimated_state_space_exceeds_bounded_profile"
        proof_admissible = refusal_reason is None
        mode_relaxable_reasons = {
            "exact_state_required_for_exact_admission",
            "exact_depth_exceeds_proof_budget",
            "estimated_state_space_exceeds_exact_profile",
            "estimated_information_tree_exceeds_proof_budget",
            "estimated_pure_policy_space_exceeds_proof_budget",
        }
        recommended_modes = (
            ["bounded", "policy_conditioned"]
            if requested_mode == "exact" and refusal_reason in mode_relaxable_reasons
            else ["policy_conditioned"]
            if requested_mode == "bounded"
            and refusal_reason
            in {
                "bounded_horizon_exceeds_proof_budget",
                "bounded_depth_exceeds_proof_budget",
                "estimated_state_space_exceeds_bounded_profile",
                "estimated_information_tree_exceeds_proof_budget",
                "estimated_pure_policy_space_exceeds_proof_budget",
            }
            else []
        )
        by_capability: dict[str, list[str]] = {}
        for row in recipe.capability_plan.rows:
            by_capability.setdefault(str(row["constructionCapability"]), []).append(
                str(row["predicateId"])
            )
        for values in by_capability.values():
            values.sort()
        requested_result_count = int(recipe.output["resultCount"])
        requested_video_count = int(recipe.render_policy["videoCount"])
        output_admissible = requested_result_count <= candidate_count
        output_reason = (
            None
            if output_admissible
            else "requested_result_count_exceeds_candidate_budget"
        )
        output = {
            "kind": str(recipe.output["kind"]),
            "requestedResultCount": requested_result_count,
            "requestedVideoCount": requested_video_count,
            "maximumCandidateCount": candidate_count,
            "admissible": output_admissible,
            "reason": output_reason,
            "recommendedResultCount": min(requested_result_count, candidate_count),
            "requiresResourceExpansion": not output_admissible,
        }
        output = {key: value for key, value in output.items() if value is not None}
        proof = {
            "requestedMode": requested_mode,
            "visibility": recipe.proof.visibility,
            "strategyClass": strategy_class,
            "horizon": copy.deepcopy(recipe.proof.horizon),
            "admissible": proof_admissible,
            "reason": refusal_reason,
            "recommendedMode": recommended_modes[0] if recommended_modes else None,
            "recommendedModes": recommended_modes,
            "estimatedRootActionUpperBound": legal_root_count,
            "actionDomainUpperBound": action_domain_upper_bound,
            "estimatedEkkaActionVariantUpperBound": ekka_action_variants,
            "ekkaDeclarationSequenceUpperBound": ekka_sequence_expression,
            "targetInformationSetUpperBound": target_information_sets,
            "opponentInformationSetUpperBound": opponent_information_sets,
            "targetPolicyCountUpperBound": target_policy_count,
            "opponentPolicyCountUpperBound": opponent_policy_count,
            "policyProductUpperBound": policy_product,
            "estimatedTreeStateUpperBound": tree_state_upper_bound,
            "policyProductExceedsBudget": policy_product_exceeds_budget,
            "informationTreeExceedsBudget": information_tree_exceeds_budget,
            "estimatedStateSpaceClass": (
                "late_game_manageable"
                if remaining_cards is not None and remaining_cards <= 16
                else "unbounded_search_forecast"
            ),
            "hiddenWorldClass": (
                "complete_state"
                if recipe.proof.visibility == "complete_state"
                else "information_set_enumeration"
            ),
            "budgetProfile": recipe.proof.budget_profile,
            "resourceBudgetProfile": resource_budget_profile,
            "remainingCardCount": remaining_cards,
            "requestedDepth": requested_depth,
            "budgetMaximumDepth": (
                budget.maximum_depth if budget is not None else None
            ),
            "budgetMaximumHorizonActions": (
                budget.maximum_horizon_actions if budget is not None else None
            ),
            "estimatedStateSpace": estimated_state_space,
            "estimatedHiddenWorldUpperBound": estimated_hidden_worlds,
            "budgetMaximumHiddenWorlds": (
                budget.maximum_hidden_worlds if budget is not None else None
            ),
            "budgetStateSpaceLimit": (
                budget.maximum_state_space if budget is not None else None
            ),
            "budgetProfileHash": budget.sha256 if budget is not None else None,
            "proofBudgetRegistryHash": budget_registry.sha256,
        }
        result = {
            "schemaVersion": 2,
            "valid": bool(validated["valid"]),
            "launchAdmissible": (
                bool(validated["valid"]) and proof_admissible and output_admissible
            ),
            "configurationAuthority": copy.deepcopy(dict(recipe.configuration)),
            "scenarioPlanHash": recipe.capability_plan.sha256,
            "capabilityPlanHash": recipe.capability_plan.sha256,
            "predicatesByConstructionCapability": by_capability,
            "postconditionOnlyPredicates": by_capability.get("postcondition_only", []),
            "exactInputOnlyPredicates": by_capability.get("exact_input_only", []),
            "unsupportedPredicates": by_capability.get("unsupported", []),
            "seedCount": seed_count,
            "candidateCount": candidate_count,
            "shardCount": math.ceil(seed_count / int(resources["seedsPerShard"])),
            "maximumSolverSeconds": (
                candidate_count * float(resources["solverTimeoutSeconds"])
                if resources["generationLane"] == "ortools"
                else 0.0
            ),
            "analysisDecisionUpperBound": candidate_count * 40,
            "tacticalVerificationDecisionUpperBound": candidate_count
            * (context.maximum_tactical_decisions if context is not None else 0),
            "output": output,
            "proof": proof,
            "expectedCostCategory": (
                "offline_heavyweight"
                if requested_mode == "exact"
                and recipe.proof.visibility == "actor_information_set"
                else "bounded"
                if requested_mode == "bounded"
                else "late_game_exact"
                if requested_mode == "exact"
                else "policy_conditioned"
            ),
            "warnings": ["This estimate is a cost forecast, not a feasibility proof."],
            "issues": copy.deepcopy(list(validated["issues"])),
        }
        result["proof"] = {
            key: value for key, value in proof.items() if value is not None
        }
        return {**result, "estimateHash": scenario_sha256(result)}

    def _estimate_recipe_v4_ortools(
        self,
        recipe: ResearchRecipeV4,
        validated: Mapping[str, Any],
        backend: OrToolsScenarioResearchBackendV2,
    ) -> dict[str, Any]:
        neutral = backend.estimate(recipe)
        construction_plan = TailFirstStagePlannerV1(
            native_cp_sat=backend.authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3,
        ).plan(recipe.scenario)
        if (getattr(backend.authority.quality_execution, 'family_proposal_mode', None) is not None
                and neutral['requestedCombinationSupported']):
            from .ortools_tail_first_candidate import family_tail_first_plan_v1
            _, construction_plan = family_tail_first_plan_v1(backend._rules, backend.authority.quality_execution,
                scenario=recipe.scenario, request=recipe.proof, playing_quality=recipe.playing_quality)
        proof_budget = dict(neutral.get("proofBudget") or {})
        execution_parameters = self._ortools_execution_parameters_v2(
            recipe,
            backend,
            enforce_request_limits=False,
        )
        candidate_universe = int(execution_parameters["maximumCandidates"])
        candidate_limit = int(execution_parameters["candidateLimit"])
        variant_limit = int(execution_parameters["maximumVariants"])
        requested_results = int(recipe.output["resultCount"])
        candidate_admissible = candidate_universe <= candidate_limit
        output_admissible = (
            candidate_admissible
            and requested_results <= variant_limit
            and requested_results <= candidate_universe
        )
        output_reason = (
            "requested_seed_count_exceeds_candidate_budget"
            if not candidate_admissible
            else "requested_result_count_exceeds_variant_budget"
            if requested_results > variant_limit
            else "requested_result_count_exceeds_candidate_budget"
            if requested_results > candidate_universe
            else None
        )
        output = {
            "kind": str(recipe.output["kind"]),
            "requestedResultCount": requested_results,
            "requestedVideoCount": int(recipe.render_policy["videoCount"]),
            "maximumCandidateCount": candidate_limit,
            "maximumVariantCount": variant_limit,
            "admissible": output_admissible,
            "reason": (output_reason),
            "recommendedResultCount": min(
                requested_results, candidate_limit, variant_limit, candidate_universe
            ),
            "requiresResourceExpansion": not output_admissible,
        }
        strategy_class = getattr(recipe.proof, "strategy_class", None) or (
            "frozen_policy"
            if recipe.proof.mode == "policy_conditioned"
            else "pure_deterministic"
        )
        normalized = validated.get("normalizedExactState")
        normalized_state = (
            dict(normalized.get("state") or {})
            if isinstance(normalized, Mapping)
            else {}
        )
        hands = normalized_state.get("hands")
        selector = dict(recipe.scenario.decision_point.get("selector") or {})
        # An unspecified position may be any play in the trick. Position zero
        # gives the largest remaining tree and is a safe resource upper bound.
        position = int(selector.get("position", 0))
        hand_counts: list[int] = []
        remaining_cards = (
            sum(len(cards) for cards in hands.values())
            if isinstance(hands, Mapping)
            and all(isinstance(cards, list) for cards in hands.values())
            else None
        )
        if isinstance(hands, Mapping) and all(
            isinstance(cards, list) for cards in hands.values()
        ):
            hand_counts = [len(cards) for cards in hands.values()]
        if (
            remaining_cards is None
            and recipe.scenario.origin.kind == "search"
            and recipe.scenario.decision_point["kind"] == "play"
        ):
            per_seat_remaining = selector.get("remainingCards")
            if type(per_seat_remaining) is int:
                remaining_cards = per_seat_remaining * 4 - position
                hand_counts = [
                    per_seat_remaining,
                    *([per_seat_remaining - 1] * position),
                    *([per_seat_remaining] * (3 - position)),
                ]
        decision_kind = str(recipe.scenario.decision_point["kind"])
        if decision_kind in {"bid", "escalation"} or (
            decision_kind == "project" and remaining_cards is None
        ):
            # Exact/bounded non-play proofs graft the remaining play graph after
            # the root decision. Auction/escalation normalize before play;
            # search project roots have no constructed hands yet.
            remaining_cards = 32
            hand_counts = [8, 8, 8, 8]
        legal_actions = (
            list(normalized.get("legalActions") or [])
            if isinstance(normalized, Mapping)
            else []
        )
        estimated_hidden_worlds: int | None = None
        if recipe.proof.visibility == "actor_information_set" and hand_counts:
            acting_seat = str(
                (normalized.get("nextActor") if isinstance(normalized, Mapping) else None)
                or selector.get("actingSeat") or selector.get("owner") or ""
            )
            opponent_sizes = (
                [
                    len(cards)
                    for seat, cards in hands.items()
                    if seat != acting_seat and isinstance(cards, list)
                ]
                if isinstance(hands, Mapping)
                else hand_counts[1:]
            )
            unknown_count = sum(opponent_sizes)
            estimated_hidden_worlds = 1
            for size in opponent_sizes[:-1]:
                estimated_hidden_worlds *= math.comb(unknown_count, size)
                unknown_count -= size
        elif (
            recipe.proof.visibility == "actor_information_set"
            and recipe.scenario.origin.kind == "search"
            and type(selector.get("remainingCards")) is int
        ):
            # A search estimate has no constructed hands yet. Treat every
            # opponent as holding the declared remaining count; this is a safe
            # upper bound once current-trick public cards are removed.
            per_opponent = int(selector["remainingCards"])
            unknown_count = per_opponent * 3
            estimated_hidden_worlds = 1
            for _ in range(2):
                estimated_hidden_worlds *= math.comb(unknown_count, per_opponent)
                unknown_count -= per_opponent
        hidden_world_budget = int(proof_budget.get("maximumHiddenWorlds") or 0)
        hard_contracts = {
            str(predicate.arguments.get("contract"))
            for predicate in recipe.scenario.constraints.predicates()
            if predicate.hardness == "hard" and predicate.kind == "auction.contract"
        }
        known_contract = str(normalized_state.get("contract") or "") or (
            next(iter(hard_contracts)) if len(hard_contracts) == 1 else ""
        )
        actor_information_contract_admissible = (
            recipe.proof.visibility != "actor_information_set"
            or decision_kind == "bid"
            or known_contract in {"sun", "hokum", "ashkal"}
        )
        hidden_worlds_admissible = (
            recipe.proof.visibility != "actor_information_set"
            or (
                estimated_hidden_worlds is not None
                and estimated_hidden_worlds <= hidden_world_budget
            )
        )
        requested_depth = remaining_cards or 0
        if recipe.proof.mode == "bounded":
            horizon = dict(recipe.proof.horizon or {})
            horizon_value = int(horizon.get("value") or 0)
            horizon_unit = str(horizon.get("unit") or "")
            if horizon_unit == "tricks":
                requested_depth = min(
                    requested_depth,
                    max(0, horizon_value * 4 - position),
                )
            elif horizon_unit in {"plies", "decisions"}:
                requested_depth = min(requested_depth, horizon_value)
        late_play_admissible = (
            bool(neutral.get("launchAdmissible"))
            and decision_kind == "play"
            and recipe.scenario.origin.kind == "exact_trace_prefix"
            and normalized_state.get("phase") == "playing"
            and remaining_cards is not None
            and requested_depth <= int(proof_budget.get("maximumDepth") or 0)
            and bool(legal_actions)
            and actor_information_contract_admissible
            and hidden_worlds_admissible
        )
        nonplay_policy_admissible = (
            bool(neutral.get("launchAdmissible"))
            and decision_kind in {"bid", "escalation", "project"}
            and recipe.scenario.origin.kind in {"search", "exact_trace_prefix"}
            and recipe.proof.mode == "policy_conditioned"
            and recipe.proof.visibility == "complete_state"
            and (recipe.scenario.origin.kind == "search" or bool(legal_actions))
            and int(proof_budget.get("maximumNodes") or 0) >= 2
            and int(proof_budget.get("maximumDepth") or 0) >= 1
        )
        nonplay_complete_admissible = (
            bool(neutral.get("launchAdmissible"))
            and decision_kind in {"bid", "escalation", "project"}
            and recipe.scenario.origin.kind in {"search", "exact_trace_prefix"}
            and recipe.proof.mode in {"exact", "bounded"}
            and recipe.proof.visibility == "complete_state"
            and (recipe.scenario.origin.kind == "search" or bool(legal_actions))
            and recipe.proof.line_universe is None
            and recipe.proof.claim != "winning_continuation_line_count"
            and int(proof_budget.get("maximumNodes") or 0) >= 2
            and int(proof_budget.get("maximumDepth") or 0) >= 1
        )
        nonplay_actor_information_admissible = (
            bool(neutral.get("launchAdmissible"))
            and decision_kind in {"bid", "escalation", "project"}
            and recipe.scenario.origin.kind in {"search", "exact_trace_prefix"}
            and recipe.proof.mode == "exact"
            and recipe.proof.visibility == "actor_information_set"
            and recipe.proof.claim != "winning_continuation_line_count"
            and actor_information_contract_admissible
            and hidden_worlds_admissible
            and (recipe.scenario.origin.kind == "search" or bool(legal_actions))
            and int(proof_budget.get("maximumNodes") or 0) >= 2
            and int(proof_budget.get("maximumDepth") or 0) >= 1
        )
        search_play_admissible = bool(
            neutral.get("launchAdmissible")
            and decision_kind == "play"
            and recipe.scenario.origin.kind == "search"
            and actor_information_contract_admissible
            and hidden_worlds_admissible
        )
        execution_admissible = (
            late_play_admissible
            or nonplay_policy_admissible
            or nonplay_complete_admissible
            or nonplay_actor_information_admissible
            or search_play_admissible
        )
        proof_reason = neutral.get("refusalReason")
        if (
            neutral.get("launchAdmissible")
            and decision_kind == "play"
            and not (late_play_admissible or search_play_admissible)
        ):
            proof_reason = "exact_play_state_exceeds_selected_proof_tier"
            if not actor_information_contract_admissible:
                proof_reason = "actor_information_set_contract_unsupported"
            elif not hidden_worlds_admissible:
                proof_reason = "hidden_worlds_exceed_proof_budget"
        elif (
            neutral.get("launchAdmissible")
            and decision_kind in {"bid", "escalation", "project"}
            and recipe.proof.visibility == "actor_information_set"
            and not nonplay_actor_information_admissible
        ):
            proof_reason = (
                "hidden_worlds_exceed_proof_budget"
                if not hidden_worlds_admissible
                else "nonplay_actor_information_set_preflight_refused"
            )
        branch_sequence = _play_branch_sequence_upper_bound_v2(
            hand_counts,
            action_multiplier=1,
        )
        if decision_kind != "play":
            root_branching = len(legal_actions) or 32
            if recipe.proof.mode == "policy_conditioned":
                branch_sequence = (root_branching,)
                requested_depth = 1
            else:
                requested_depth += 1
                if recipe.proof.mode == "bounded":
                    requested_depth = min(
                        requested_depth,
                        int((recipe.proof.horizon or {}).get("value") or 0),
                    )
                branch_sequence = (root_branching, *branch_sequence)
        elif recipe.proof.mode == "policy_conditioned":
            branch_sequence = tuple(1 for _ in branch_sequence)
        if not branch_sequence and requested_depth:
            fallback_actions = len(legal_actions) if legal_actions else 32
            branch_sequence = tuple(fallback_actions for _ in range(requested_depth))
        branch_sequence = branch_sequence[:requested_depth]
        action_upper_bound = (
            len(legal_actions)
            if legal_actions
            else branch_sequence[0]
            if branch_sequence
            else 1
        )
        graph_worlds = int(
            (estimated_hidden_worlds or 1)
            if recipe.proof.visibility == "actor_information_set"
            else 1
        )
        displayed_worlds = (
            neutral["estimatedHiddenWorlds"]
            if neutral.get("refusalReason") == "playing_quality_first_bid_hidden_worlds_exceed_budget"
            else graph_worlds
        )
        per_world_nodes, per_world_edges = _tree_upper_bound_from_branches_v2(
            branch_sequence
        )
        estimated_graph_nodes = per_world_nodes * graph_worlds
        estimated_graph_edges = per_world_edges * graph_worlds

        estimated_count_limbs = 0
        if recipe.proof.claim == "winning_continuation_line_count":
            if recipe.proof.visibility == "complete_state":
                line_upper_bound = (
                    1
                    if recipe.proof.line_universe == "policy_conditioned"
                    else math.prod(branch_sequence[1:])
                )
                estimated_count_limbs = required_line_count_limbs_v1(line_upper_bound)

        execution_capabilities = build_scenario_execution_capabilities_v1(backend)
        matching_capabilities = [
            item
            for item in execution_capabilities["combinations"]
            if item["origin"] == recipe.scenario.origin.kind
            and item["decisionKind"] == decision_kind
            and item["visibility"] == recipe.proof.visibility
            and recipe.proof.claim in item["claims"]
            and (
                recipe.proof.claim != "winning_continuation_line_count"
                or recipe.proof.line_universe in item["lineUniverses"]
            )
        ]
        alternative_proof_modes = sorted(
            {
                str(item["mode"])
                for item in matching_capabilities
                if item["mode"] != recipe.proof.mode
            },
            key=("exact", "bounded", "policy_conditioned").index,
        )
        requested_combination_supported = any(
            item["mode"] == recipe.proof.mode for item in matching_capabilities
        )
        hard_predicate_count = sum(
            predicate.hardness == "hard" for predicate in recipe.scenario.predicates()
        )
        estimated_model_variables, estimated_model_constraints = (
            _ortools_model_size_upper_bound_v2(
                compiled_stages=neutral.get("compiledStages") or (),
                hard_predicate_count=hard_predicate_count,
                requested_depth=requested_depth,
                graph_nodes=estimated_graph_nodes,
                root_actions=action_upper_bound,
                count_limbs=(estimated_count_limbs or None),
                maximum_prefix_lift_attempts=int(
                    execution_parameters["maximumPrefixLiftAttempts"]
                ),
            )
        )
        thresholds = dict(backend.authority.refusal_thresholds)
        maximum_worlds = min(
            int(thresholds["maximumHiddenWorlds"]),
            int(proof_budget["maximumHiddenWorlds"]),
        )
        maximum_states = min(
            int(thresholds["maximumEstimatedStates"]),
            int(proof_budget["maximumStateSpace"]),
        )
        depth_admissible = requested_depth <= int(proof_budget["maximumDepth"])
        worlds_threshold_admissible = graph_worlds <= maximum_worlds
        states_admissible = estimated_graph_nodes <= maximum_states
        limbs_admissible = estimated_count_limbs <= int(
            thresholds["maximumLineCountLimbs"]
        )
        model_variables_admissible = estimated_model_variables <= int(
            thresholds["maximumModelVariables"]
        )
        model_constraints_admissible = estimated_model_constraints <= int(
            thresholds["maximumModelConstraints"]
        )
        execution_admissible = bool(
            execution_admissible
            and requested_combination_supported
            and depth_admissible
            and worlds_threshold_admissible
            and states_admissible
            and limbs_admissible
            and model_variables_admissible
            and model_constraints_admissible
        )
        refusal_reason = None
        if not neutral.get("launchAdmissible"):
            refusal_reason = neutral.get("refusalReason")
        elif output["reason"]:
            refusal_reason = output["reason"]
        elif not requested_combination_supported:
            refusal_reason = "requested_proof_combination_unsupported"
        elif not depth_admissible:
            refusal_reason = "requested_depth_exceeds_proof_budget"
        elif not worlds_threshold_admissible:
            refusal_reason = "hidden_worlds_exceed_refusal_threshold"
        elif not states_admissible:
            refusal_reason = "estimated_states_exceed_refusal_threshold"
        elif not limbs_admissible:
            refusal_reason = "line_count_limbs_exceed_refusal_threshold"
        elif not model_variables_admissible:
            refusal_reason = "model_variables_exceed_refusal_threshold"
        elif not model_constraints_admissible:
            refusal_reason = "model_constraints_exceed_refusal_threshold"
        elif not execution_admissible:
            refusal_reason = proof_reason
        quality_cost_estimable = recipe.playing_quality is None
        analysis_decision_upper_bound = len(legal_actions)
        quality_execution = backend.authority.quality_execution
        if (backend.authority.backend == "ortools_cp_sat_v3"
                and recipe.playing_quality is not None and quality_execution is not None):
            analysis_decision_upper_bound = candidate_universe * (
                2 * len(Seat) + len(Seat) + len(FULL_DECK)
                + quality_execution.maximum_escalation_decisions
            )
        first_result_cost_class = (
            "not_estimable"
            if execution_admissible and not quality_cost_estimable
            else "within_selected_tier"
            if execution_admissible
            else "exceeds_selected_tier"
        )
        requested_results_cost_class = (
            "not_estimable"
            if execution_admissible and output_admissible and not quality_cost_estimable
            else "within_selected_tier"
            if execution_admissible and output_admissible
            else "exceeds_selected_tier"
        )
        proof = {
            "requestedMode": recipe.proof.mode,
            "visibility": recipe.proof.visibility,
            "strategyClass": strategy_class,
            "horizon": copy.deepcopy(recipe.proof.horizon),
            "admissible": execution_admissible,
            "reason": None if execution_admissible else refusal_reason,
            "recommendedMode": None,
            "recommendedModes": [],
            "estimatedRootActionUpperBound": action_upper_bound,
            "actionDomainUpperBound": action_upper_bound,
            "estimatedEkkaActionVariantUpperBound": 0,
            "ekkaDeclarationSequenceUpperBound": "not_compiled",
            "targetInformationSetUpperBound": None,
            "opponentInformationSetUpperBound": None,
            "targetPolicyCountUpperBound": None,
            "opponentPolicyCountUpperBound": None,
            "policyProductUpperBound": None,
            "estimatedTreeStateUpperBound": None,
            "policyProductExceedsBudget": None,
            "informationTreeExceedsBudget": None,
            "estimatedStateSpaceClass": (
                "late_game_manageable" if execution_admissible else "refused"
            ),
            "hiddenWorldClass": (
                "complete_state"
                if recipe.proof.visibility == "complete_state"
                else "information_set_enumeration"
            ),
            "budgetProfile": recipe.proof.budget_profile,
            "resourceBudgetProfile": str(recipe.resources["proofBudgetProfile"]),
            "remainingCardCount": remaining_cards,
            "requestedDepth": requested_depth,
            "budgetMaximumDepth": proof_budget.get("maximumDepth"),
            "budgetMaximumHorizonActions": proof_budget.get("maximumHorizonActions"),
            "estimatedStateSpace": estimated_graph_nodes,
            "estimatedHiddenWorldUpperBound": estimated_hidden_worlds,
            "budgetMaximumHiddenWorlds": proof_budget.get("maximumHiddenWorlds"),
            "budgetStateSpaceLimit": proof_budget.get("maximumStateSpace"),
            "budgetProfileHash": (
                scenario_sha256(proof_budget) if proof_budget else None
            ),
            "proofBudgetRegistryHash": scenario_sha256(
                dict(backend.authority.proof_budgets)
            ),
        }
        by_capability: dict[str, list[str]] = {}
        native_plan = validated.get("nativeCapabilityPlan")
        if isinstance(native_plan, Mapping):
            for row in native_plan["rows"]:
                construction = (
                    "unsupported"
                    if row["nativeStatus"] != "native_exact"
                    else "cp_sat_exact"
                    if row["hardness"] == "hard"
                    else "guided_search"
                    if row["hardness"] == "prefer"
                    else "postcondition_only"
                )
                by_capability.setdefault(construction, []).append(row["predicateId"])
        else:
            for row in recipe.capability_plan.rows:
                by_capability.setdefault(str(row["constructionCapability"]), []).append(
                    str(row["predicateId"])
                )
        for predicate_ids in by_capability.values():
            predicate_ids.sort()
        requested_shard_count = math.ceil(
            int(recipe.resources["seedCount"]) / int(recipe.resources["seedsPerShard"])
        )
        actual_shard_count = (
            1
            if recipe.scenario.origin.kind == "exact_trace_prefix"
            else requested_shard_count
        )
        maximum_shard_workers = int(execution_parameters["maximumShardWorkers"])
        requested_shard_workers = int(recipe.resources.get("shardWorkerCount", maximum_shard_workers))
        effective_shard_workers = min(
            actual_shard_count, requested_shard_workers, maximum_shard_workers,
            self.maximum_local_shard_workers,
        )
        maximum_shard_wall_seconds = (
            float(execution_parameters["maximumCampaignSeconds"])
            if backend.authority.backend == "ortools_cp_sat_v3"
            else min(
                float(execution_parameters["maximumSolveSeconds"]),
                float(execution_parameters["maximumCampaignSeconds"]),
            )
        )
        desired_parallelism = min(
            requested_results,
            requested_shard_workers,
            int(recipe.resources["seedCount"]),
        )
        parallelism_limited = bool(
            recipe.scenario.origin.kind == "search"
            and requested_results > 1
            and min(actual_shard_count, requested_shard_workers, maximum_shard_workers)
            < desired_parallelism
        )
        sharding = {
            "configuredSeedsPerShard": int(recipe.resources["seedsPerShard"]),
            "requestedShardCount": requested_shard_count,
            "actualShardCount": actual_shard_count,
            "maximumShardWorkers": maximum_shard_workers,
            "requestedShardWorkers": requested_shard_workers,
            "effectiveShardWorkers": effective_shard_workers,
            "maximumShardWallSeconds": maximum_shard_wall_seconds,
            "parallelismLimitedByShardPlan": parallelism_limited,
            "recommendedSeedsPerShard": (
                1 if parallelism_limited else int(recipe.resources["seedsPerShard"])
            ),
        }
        estimate_warnings = [
            "This estimate is a cost forecast, not a feasibility proof.",
            "Model and graph sizes are conservative deterministic upper bounds.",
        ]
        if parallelism_limited:
            estimate_warnings.append("parallelism_limited_by_shard_plan")
        if not quality_cost_estimable:
            estimate_warnings.append("playing_quality_full_round_cost_not_estimated")
        result = {
            "schemaVersion": 2,
            "valid": bool(validated["valid"]),
            "launchAdmissible": bool(
                validated["valid"]
                and execution_admissible
                and output_admissible
                and requested_combination_supported
            ),
            "configurationAuthority": copy.deepcopy(dict(recipe.configuration)),
            "scenarioPlanHash": recipe.capability_plan.sha256,
            "capabilityPlanHash": recipe.capability_plan.sha256,
            "nativeCapabilityPlanHash": (
                native_plan["sha256"] if isinstance(native_plan, Mapping) else None
            ),
            "predicatesByConstructionCapability": by_capability,
            "postconditionOnlyPredicates": by_capability.get("postcondition_only", []),
            "exactInputOnlyPredicates": by_capability.get("exact_input_only", []),
            "unsupportedPredicates": by_capability.get("unsupported", []),
            "seedCount": int(recipe.resources["seedCount"]),
            "candidateCount": candidate_universe,
            "shardCount": actual_shard_count,
            "sharding": sharding,
            "maximumSolverSeconds": float(execution_parameters["maximumSolveSeconds"]),
            "maximumCampaignSeconds": float(
                execution_parameters["maximumCampaignSeconds"]
            ),
            "analysisDecisionUpperBound": analysis_decision_upper_bound,
            "tacticalVerificationDecisionUpperBound": 0,
            "backendIdentity": copy.deepcopy(neutral["backendIdentity"]),
            "compiledStages": copy.deepcopy(list(neutral["compiledStages"])),
            "constructionPlan": {
                "schemaVersion": 1,
                "planner": construction_plan.planner,
                "fallbackReason": construction_plan.fallback_reason,
                "prefixLength": construction_plan.prefix_length,
                "tailOffset": construction_plan.tail_offset,
                "tailLength": construction_plan.tail_length,
                "sha256": construction_plan.sha256,
            },
            "resourceTier": str(recipe.resources["proofBudgetProfile"]),
            "refusalReason": (
                None
                if execution_admissible and output_admissible
                else (str(refusal_reason) if refusal_reason is not None else None)
            ),
            "estimatedModelVariables": estimated_model_variables,
            "estimatedModelConstraints": estimated_model_constraints,
            "estimatedGraphNodes": estimated_graph_nodes,
            "estimatedGraphEdges": estimated_graph_edges,
            "estimatedHiddenWorlds": (
                str(displayed_worlds)
                if displayed_worlds > (1 << 53) - 1
                else displayed_worlds
            ),
            "estimatedCountLimbs": estimated_count_limbs,
            "firstResultCostClass": first_result_cost_class,
            "requestedResultsCostClass": requested_results_cost_class,
            "alternativeProofModes": alternative_proof_modes,
            "requestedCombinationSupported": requested_combination_supported,
            "exhaustiveSearchInfeasibilityEstablishable": bool(
                neutral.get("exhaustiveSearchInfeasibilityEstablishable")
                # Preserve frozen V2 estimates. V3 cannot infer exhaustive
                # construction coverage from an exact per-candidate proof.
                or (
                    backend.authority.backend != "ortools_cp_sat_v3"
                    and recipe.scenario.origin.kind == "search"
                    and recipe.proof.mode == "exact"
                    and recipe.proof.visibility == "complete_state"
                    and requested_combination_supported
                    and execution_admissible
                )
            ),
            "output": output,
            "proof": proof,
            "expectedCostCategory": (
                "bounded"
                if execution_admissible and recipe.proof.mode == "bounded"
                else "policy_conditioned"
                if execution_admissible and recipe.proof.mode == "policy_conditioned"
                else "late_game_exact"
                if execution_admissible
                else "not_compiled"
            ),
            "warnings": estimate_warnings,
            "issues": copy.deepcopy(list(validated["issues"])),
        }
        result["proof"] = {
            key: value for key, value in proof.items() if value is not None
        }
        return {**result, "estimateHash": scenario_sha256(result)}

    @staticmethod
    def _require_v4_launch_admission(
        recipe: ResearchRecipeV4,
        estimate: Mapping[str, Any],
    ) -> None:
        if estimate.get("launchAdmissible") is True:
            return
        proof = copy.deepcopy(dict(estimate.get("proof") or {}))
        output = copy.deepcopy(dict(estimate.get("output") or {}))
        reason = str(
            output.get("reason") or proof.get("reason") or "estimate_not_admissible"
        )
        core = {
            "schemaVersion": 1,
            "code": "research_launch_not_admissible",
            "recipeCode": recipe.code,
            "recipeHash": recipe.sha256,
            "configurationAuthority": copy.deepcopy(dict(recipe.configuration)),
            "scenarioPlanHash": recipe.capability_plan.sha256,
            "estimateHash": str(estimate.get("estimateHash") or ""),
            "reason": reason,
            "recommendedModes": list(proof.get("recommendedModes") or []),
            "output": output,
            "proof": proof,
        }
        raise ResearchLaunchAdmissionError({**core, "sha256": scenario_sha256(core)})

    def create_recipe(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        intent = copy.deepcopy(dict(payload))
        key = None
        intent_hash = None
        if idempotency_key is not None:
            key = validate_idempotency_key(idempotency_key)
            intent_hash = recipe_mutation_intent_hash("create", intent)
            receipt = self.store.get_recipe_mutation_receipt(key)
            if receipt is not None:
                return replay_recipe_mutation_receipt(
                    receipt,
                    mutation_kind="create",
                    request_intent_hash=intent_hash,
                )
        if type(intent.get("schemaVersion")) is not int or intent["schemaVersion"] != 4:
            raise ValueError(
                "new Research recipe creation requires schemaVersion 4; "
                "V3 remains available only for read, run, retire, and conversion preview"
            )
        recipe, creation_context = self._build_recipe_and_context(intent)
        if issue := self._new_recipe_authoring_issue(creation_context):
            raise ValueError(issue["details"]["reason"])
        validated = self._validate_recipe_v4_cached(recipe, creation_context)
        if not validated["valid"]:
            raise ValueError(json.dumps(validated["issues"], ensure_ascii=False))
        validated_recipe = validated["recipe"]
        if (
            type(validated_recipe.get("schemaVersion")) is not int
            or validated_recipe["schemaVersion"] != 4
        ):
            raise ValueError("Research recipe creation validation must produce V4")
        if key is not None and intent_hash is not None:
            return self.store.put_recipe_with_receipt(
                validated_recipe,
                idempotency_key=key,
                request_intent_hash=intent_hash,
            )
        created = self.store.put_recipe(validated_recipe)
        return {**created, "lifecycleStatus": self.store.recipe_status(created["code"])}

    def _project_recipe_read_native_v3(
        self, item: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = copy.deepcopy(dict(item))
        if (
            result.get("schemaVersion") == 4
            and dict(result.get("resources") or {}).get("generationPolicy")
            == "ortools_cp_sat_v3"
        ):
            recipe, context = self._build_recipe_and_context({
                key: value for key, value in result.items()
                if key not in {"lifecycleStatus", "nativeCapabilityPlan"}
            })
            backend = self._cached_ortools_scenario_backend_v2(context)
            if backend is None or backend.authority.backend != "ortools_cp_sat_v3":
                raise ValueError("stored V3 recipe bundle authority is unavailable")
            result["nativeCapabilityPlan"] = backend.native_capability_plan(recipe)
        return result

    def list_recipes(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return [
            {**item, "lifecycleStatus": self.store.recipe_status(item["code"])}
            for item in self.store.list_recipes(status=status, limit=limit)
        ]

    def list_recipe_page(
        self,
        *,
        status: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        anchor = (
            None if cursor is None else RecipeListCursorV1.decode(cursor, status=status)
        )
        page = self.store.list_recipe_page(
            status=status,
            limit=limit,
            after_created_at=(None if anchor is None else anchor.created_at),
            after_recipe_code=(None if anchor is None else anchor.recipe_code),
        )
        items = [
            {
                **copy.deepcopy(dict(item["recipe"])),
                "lifecycleStatus": str(item["lifecycleStatus"]),
            }
            for item in page["items"]
        ]
        next_anchor = page["nextAnchor"]
        next_cursor = None
        if next_anchor is not None:
            next_cursor = RecipeListCursorV1.from_anchor(
                next_anchor["createdAt"],
                next_anchor["recipeCode"],
                status=status,
            ).encode()
        return {"items": items, "nextCursor": next_cursor}

    def get_recipe(
        self, code: str, *, include_native_plan: bool = False
    ) -> dict[str, Any] | None:
        item = self.store.get_recipe(code)
        if item is None:
            return None
        result = {**item, "lifecycleStatus": self.store.recipe_status(code)}
        return (
            self._project_recipe_read_native_v3(result)
            if include_native_plan else result
        )

    def preview_recipe_v4_conversion(
        self,
        code: str,
        *,
        decision_point: Mapping[str, Any] | None = None,
        outcome: Mapping[str, Any] | None = None,
        proof: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        source = self.store.get_recipe(code)
        if source is None:
            return None
        if type(source.get("schemaVersion")) is not int or source["schemaVersion"] != 3:
            raise ValueError(
                "only stored Research V3 recipes support V4 conversion preview"
            )
        return preview_v3_to_v4_conversion(
            source,
            decision_point=decision_point,
            outcome=outcome,
            proof=proof,
        )

    def retire_recipe(
        self,
        code: str,
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if idempotency_key is None:
            return self.store.retire_recipe(code)
        key = validate_idempotency_key(idempotency_key)
        intent_hash = recipe_mutation_intent_hash("retire", {"recipeCode": str(code)})
        receipt = self.store.get_recipe_mutation_receipt(key)
        if receipt is not None:
            return replay_recipe_mutation_receipt(
                receipt,
                mutation_kind="retire",
                request_intent_hash=intent_hash,
            )
        return self.store.retire_recipe_with_receipt(
            code,
            idempotency_key=key,
            request_intent_hash=intent_hash,
        )

    def submit(
        self,
        recipe_code: str,
        *,
        idempotency_key: str,
        comparison_campaign_code: str | None = None,
    ) -> dict[str, Any]:
        return self._submit_recipe_run(
            recipe_code,
            idempotency_key=idempotency_key,
            comparison_campaign_code=comparison_campaign_code,
            schedule=True,
        )

    def _deal_generator_identity(self) -> dict[str, Any]:
        descriptor = getattr(self.deal_generator, "descriptor", None)
        if not isinstance(descriptor, PluginDescriptor):
            raise TypeError("Research deal generator lacks its plugin descriptor")
        descriptor_document = descriptor.to_dict()
        if (
            not descriptor.deterministic
            or descriptor_document != OrToolsCpSatDealGeneratorV1.descriptor.to_dict()
        ):
            raise ValueError(
                "Research requires the canonical deterministic OR deal generator"
            )
        implementation_type = type(self.deal_generator)
        implementation_module = inspect.getmodule(implementation_type)
        if implementation_module is None:
            raise ValueError(
                "Research deal-generator implementation module is unavailable"
            )
        try:
            implementation_source = inspect.getsource(implementation_module)
            implementation_class_source = inspect.getsource(implementation_type)
        except (OSError, TypeError) as exc:
            raise ValueError(
                "Research deal-generator implementation source is unavailable"
            ) from exc
        normalized_source = implementation_source.replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        normalized_class_source = implementation_class_source.replace(
            "\r\n", "\n"
        ).replace("\r", "\n")
        implementation = {
            "schemaVersion": 1,
            "kind": "research_deal_generator_implementation_v1",
            "module": str(implementation_type.__module__),
            "qualname": str(implementation_type.__qualname__),
            "sourceSha256": hashlib.sha256(
                normalized_source.encode("utf-8")
            ).hexdigest(),
            "classSourceSha256": hashlib.sha256(
                normalized_class_source.encode("utf-8")
            ).hexdigest(),
        }
        descriptor_hash = canonical_sha256(descriptor_document)
        core = {
            "schemaVersion": 1,
            "kind": "research_deal_generator_identity_v1",
            "descriptor": descriptor_document,
            "descriptorHash": descriptor_hash,
            "implementation": implementation,
            "implementationHash": canonical_sha256(implementation),
        }
        return {**core, "sha256": canonical_sha256(core)}

    def _scenario_executor_identity_v4(
        self,
        recipe: ResearchRecipeV4,
        context: ResearchExecutionContextV1 | None,
    ) -> dict[str, Any]:
        """Bind one V4 run to its frozen legacy or OR-Tools executor."""

        backend = self._cached_ortools_scenario_backend_v2(context)
        if backend is None:
            core = {
                "id": "research_scenario_search_v1",
                "version": "1.0.0",
                "constructionPlanHash": recipe.capability_plan.construction_hash,
                "runtimeVerificationPlanHash": recipe.capability_plan.runtime_verification_hash,
                "proofPlanHash": recipe.capability_plan.proof_plan_hash,
            }
            return {**core, "sha256": scenario_sha256(core)}
        core = {
            "schemaVersion": 1,
            "kind": "ortools_scenario_execution_identity_v1",
            "backendIdentity": backend.identity.to_dict(),
            "executorIdentity": backend.executor_identity,
            "loweringManifestHash": backend.identity.predicate_catalog_hash,
            "capabilityPlanHash": recipe.capability_plan.sha256,
            "nativeCapabilityPlanHash": (
                backend.native_capability_plan(recipe)["sha256"]
                if backend.authority.backend == "ortools_cp_sat_v3"
                else None
            ),
            "constructionPlanHash": recipe.capability_plan.construction_hash,
            "proofPlanHash": recipe.capability_plan.proof_plan_hash,
        }
        return {**core, "sha256": scenario_sha256(core)}

    def _submit_recipe_run(
        self,
        recipe_code: str,
        *,
        idempotency_key: str,
        comparison_campaign_code: str | None = None,
        schedule: bool,
    ) -> dict[str, Any]:
        recipe_value = self.store.get_recipe(recipe_code)
        if recipe_value is None:
            raise KeyError(recipe_code)
        validated = self.validate_recipe(recipe_value)
        if not validated["valid"]:
            raise ValueError(json.dumps(validated["issues"], ensure_ascii=False))
        recipe, context = self._build_recipe_and_context(validated["recipe"])
        if (
            recipe.status != "published"
            or self.store.recipe_status(recipe_code) != "published"
        ):
            raise ValueError("only a published research recipe can be launched")
        launch_admission: dict[str, Any] | None = None
        if isinstance(recipe, ResearchRecipeV4):
            estimate = self._estimate_recipe_v4(recipe, validated, context)
            self._require_v4_launch_admission(
                recipe,
                estimate,
            )
            admission_core = {
                "schemaVersion": 1,
                "recipeHash": recipe.sha256,
                "estimate": copy.deepcopy(estimate),
            }
            launch_admission = {
                **admission_core,
                "sha256": scenario_sha256(admission_core),
            }
        deal_generator_identity = (
            self._deal_generator_identity()
            if recipe.resources["generationLane"] == "ortools"
            and not isinstance(recipe, ResearchRecipeV4)
            else None
        )
        scenario_executor_identity = (
            self._scenario_executor_identity_v4(recipe, context)
            if isinstance(recipe, ResearchRecipeV4)
            else None
        )
        identity = {
            "schemaVersion": 1,
            "recipeHash": recipe.sha256,
            "idempotencyKey": str(idempotency_key),
            "dealGeneratorIdentityHash": (
                deal_generator_identity["sha256"]
                if deal_generator_identity is not None
                else None
            ),
            "scenarioExecutorIdentityHash": (
                scenario_executor_identity["sha256"]
                if scenario_executor_identity is not None
                else None
            ),
        }
        if launch_admission is not None:
            identity["launchAdmissionHash"] = launch_admission["sha256"]
        if comparison_campaign_code is not None:
            identity["comparisonCampaignCode"] = str(comparison_campaign_code)
        run_hash = canonical_sha256(identity)
        run_code = f"RUN-{run_hash[:12].upper()}"
        payload = {
            "schemaVersion": 1,
            "runCode": run_code,
            "runHash": run_hash,
            "recipeCode": recipe.code,
            "recipeHash": recipe.sha256,
            "configurationAuthority": (
                copy.deepcopy(dict(recipe.configuration))
                if isinstance(recipe, (ResearchRecipeV3, ResearchRecipeV4))
                else None
            ),
            "researchContextHash": (
                context.context_hash if context is not None else None
            ),
            "generationLane": (
                "exact_input"
                if isinstance(recipe, ResearchRecipeV4)
                and recipe.scenario.origin.kind == "exact_trace_prefix"
                else recipe.resources["generationLane"]
            ),
            "dealGeneratorIdentity": copy.deepcopy(deal_generator_identity),
            "dealGeneratorDescriptorHash": (
                deal_generator_identity["descriptorHash"]
                if deal_generator_identity is not None
                else None
            ),
            "comparisonCampaignCode": comparison_campaign_code,
            "scenarioExecutorIdentityHash": identity["scenarioExecutorIdentityHash"],
            "status": "queued",
            "progress": 0.0,
            "resultCount": 0,
            "quarantineCount": 0,
            "events": [{"type": "queued", "at": _now()}],
            "createdAt": _now(),
        }
        if launch_admission is not None:
            payload["launchAdmission"] = copy.deepcopy(launch_admission)
            payload["requestedResultCount"] = int(recipe.output["resultCount"])
            payload["requestedVideoCount"] = int(recipe.render_policy["videoCount"])
        if (
            scenario_executor_identity is not None
            and scenario_executor_identity.get("kind")
            == "ortools_scenario_execution_identity_v1"
        ):
            payload["scenarioExecutorIdentity"] = copy.deepcopy(
                scenario_executor_identity
            )
        created = self.store.create_run(payload)
        if not isinstance(recipe, ResearchRecipeV4):
            shards = partition_research_shards(
                run_code=run_code,
                run_hash=run_hash,
                recipe_hash=recipe.sha256,
                seed_start=int(recipe.resources["seedStart"]),
                seed_count=int(recipe.resources["seedCount"]),
                seeds_per_shard=int(recipe.resources["seedsPerShard"]),
                maximum_attempts=self.maximum_shard_attempts,
            )
            self.store.create_shards(tuple(item.to_dict() for item in shards))
        if schedule and created["status"] in {"queued", "running"}:
            self._schedule_run(run_code)
        return created

    def _schedule_run(self, run_code: str) -> None:
        with self._lock:
            if run_code in self._scheduled_runs:
                return
            self._scheduled_runs.add(run_code)
        try:
            self.executor.submit(self._execute, run_code)
        except Exception:
            with self._lock:
                self._scheduled_runs.discard(run_code)
            raise

    def _assert_run_execution_context(
        self,
        run: Mapping[str, Any],
        recipe: ResearchRecipe,
        context: ResearchExecutionContextV1 | None,
    ) -> None:
        if isinstance(recipe, ResearchRecipeV4):
            if context is None:
                raise ValueError("Research V4 run lacks its frozen execution context")
            stored_authority = run.get("configurationAuthority")
            if not isinstance(stored_authority, Mapping) or dict(
                stored_authority
            ) != dict(recipe.configuration):
                raise ValueError("Research V4 run configuration authority mismatch")
            admission = run.get("launchAdmission")
            if not isinstance(admission, Mapping) or set(admission) != {
                "schemaVersion",
                "recipeHash",
                "estimate",
                "sha256",
            }:
                raise ValueError(
                    "Research V4 run lacks a closed launch admission receipt"
                )
            admission_core = {
                key: copy.deepcopy(value)
                for key, value in admission.items()
                if key != "sha256"
            }
            if (
                admission.get("schemaVersion") != 1
                or admission.get("recipeHash") != recipe.sha256
                or admission.get("sha256") != scenario_sha256(admission_core)
            ):
                raise ValueError("Research V4 run launch admission receipt mismatch")
            estimate = admission.get("estimate")
            if not isinstance(estimate, Mapping):
                raise ValueError("Research V4 run launch estimate is unavailable")
            estimate_core = {
                key: copy.deepcopy(value)
                for key, value in estimate.items()
                if key != "estimateHash"
            }
            if (
                estimate.get("estimateHash") != scenario_sha256(estimate_core)
                or estimate.get("launchAdmissible") is not True
                or dict(estimate.get("configurationAuthority") or {})
                != dict(recipe.configuration)
                or str(estimate.get("scenarioPlanHash") or "")
                != recipe.capability_plan.sha256
                or dict(estimate.get("proof") or {}).get("admissible") is not True
            ):
                raise ValueError("Research V4 run launch estimate binding mismatch")
            if (
                run.get("requestedResultCount") != recipe.output["resultCount"]
                or run.get("requestedVideoCount") != recipe.render_policy["videoCount"]
            ):
                raise ValueError("Research V4 run output request binding mismatch")
            if str(run.get("researchContextHash") or "") != context.context_hash:
                raise ValueError("Research V4 run context identity mismatch")
            expected_lane = (
                "exact_input"
                if recipe.scenario.origin.kind == "exact_trace_prefix"
                else str(recipe.resources["generationLane"])
            )
            if str(run.get("generationLane") or "") != expected_lane:
                raise ValueError("Research V4 run generation lane mismatch")
            if (
                run.get("dealGeneratorIdentity") is not None
                or run.get("dealGeneratorDescriptorHash") is not None
            ):
                raise ValueError(
                    "Research V4 run cannot freeze the legacy deal generator"
                )
            expected_executor = self._scenario_executor_identity_v4(
                recipe,
                context,
            )
            if (
                str(run.get("scenarioExecutorIdentityHash") or "")
                != expected_executor["sha256"]
            ):
                raise ValueError("Research V4 scenario executor identity mismatch")
            stored_executor = run.get("scenarioExecutorIdentity")
            if (
                expected_executor.get("kind")
                == "ortools_scenario_execution_identity_v1"
            ):
                if (
                    not isinstance(stored_executor, Mapping)
                    or dict(stored_executor) != expected_executor
                ):
                    raise ValueError("Research V4 OR-Tools executor authority mismatch")
            elif stored_executor is not None:
                raise ValueError(
                    "legacy Research V4 run cannot carry an OR-Tools executor authority"
                )
            return
        if recipe.resources["generationLane"] == "ortools":
            expected_generator = self._deal_generator_identity()
            stored_generator = run.get("dealGeneratorIdentity")
            if (
                not isinstance(stored_generator, Mapping)
                or dict(stored_generator) != expected_generator
            ):
                raise ValueError("Research run deal-generator identity mismatch")
            if str(run.get("dealGeneratorDescriptorHash") or "") != str(
                expected_generator["descriptorHash"]
            ):
                raise ValueError("Research run deal-generator descriptor mismatch")
        elif (
            run.get("dealGeneratorIdentity") is not None
            or run.get("dealGeneratorDescriptorHash") is not None
        ):
            raise ValueError("Organic Research run cannot freeze an OR deal generator")
        stored_context_hash = run.get("researchContextHash")
        if isinstance(recipe, ResearchRecipeV3):
            if context is None:
                raise ValueError("Research V3 run lacks its frozen execution context")
            stored_authority = run.get("configurationAuthority")
            if not isinstance(stored_authority, Mapping) or dict(
                stored_authority
            ) != dict(recipe.configuration):
                raise ValueError("Research run configuration authority mismatch")
            if (
                not stored_context_hash
                or str(stored_context_hash) != context.context_hash
            ):
                raise ValueError("Research run context identity mismatch")
            if (
                str(recipe.configuration["effectiveRuntimeConfigHash"])
                != context.effective_runtime_config_hash
            ):
                raise ValueError("Research run effective configuration mismatch")
            return
        if (
            context is not None
            and stored_context_hash is not None
            and str(stored_context_hash) != context.context_hash
        ):
            raise ValueError("historical_record_requires_readjudication")

    def create_comparison_campaign(
        self,
        recipe_code: str,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        base_value = self.store.get_recipe(recipe_code)
        if base_value is None:
            raise KeyError(recipe_code)
        validated_base = self.validate_recipe(base_value)
        if not validated_base["valid"]:
            raise ValueError(json.dumps(validated_base["issues"], ensure_ascii=False))
        base, context = self._build_recipe_and_context(validated_base["recipe"])
        if (
            base.status != "published"
            or self.store.recipe_status(recipe_code) != "published"
        ):
            raise ValueError(
                "only a published research recipe can start a comparison campaign"
            )
        if isinstance(base, ResearchRecipeV4):
            raise ValueError(
                "Research V4 has one OR-Tools qualification authority; "
                "comparison campaigns are available only for frozen V1-V3 recipes"
            )
        campaign_identity = {
            "schemaVersion": 1,
            "baseRecipeHash": base.sha256,
            "idempotencyKey": str(idempotency_key),
            "generationLanes": ["ortools", "organic_runtime"],
        }
        campaign_hash = canonical_sha256(campaign_identity)
        campaign_code = f"CMP-{campaign_hash[:12].upper()}"
        existing = self.store.get_comparison_campaign(campaign_code)
        if existing is not None:
            return self._hydrate_comparison_campaign(existing)

        lane_titles = {
            "ortools": "OR-Tools",
            "organic_runtime": self._localized(
                "research_comparison_lane_organic", context
            ),
        }
        prepared_recipes: dict[str, ResearchRecipe] = {}
        prepared_contexts: dict[str, ResearchExecutionContextV1 | None] = {}
        organic_removed_resource_keys = (
            "candidatesPerSeed",
            "solverTimeoutSeconds",
            "maximumConflicts",
            "auctionConstructionMode",
            "strategicAuction",
            "playConstructionMode",
            "strategicPlay",
            "tacticalConstruction",
            "tacticalDealBalance",
        )
        for lane in ("ortools", "organic_runtime"):
            clone = base.to_dict()
            clone.pop("code", None)
            clone.pop("sha256", None)
            clone["titleAr"] = f"{base.title_ar[:130]} · {lane_titles[lane]}"
            clone["resources"] = {**dict(base.resources), "generationLane": lane}
            if lane == "organic_runtime":
                for key in organic_removed_resource_keys:
                    clone["resources"].pop(key, None)
            clone["status"] = "published"
            validated_lane = self.validate_recipe(clone)
            if not validated_lane["valid"]:
                raise ValueError(
                    json.dumps(validated_lane["issues"], ensure_ascii=False)
                )
            lane_recipe, lane_context = self._build_recipe_and_context(
                validated_lane["recipe"]
            )
            if lane_recipe.status != "published":
                raise ValueError("comparison lane recipe must be published")
            prepared_recipes[lane] = lane_recipe
            prepared_contexts[lane] = lane_context

        base_authority = self._comparison_authority(base, context)
        for lane in ("ortools", "organic_runtime"):
            lane_authority = self._comparison_authority(
                prepared_recipes[lane], prepared_contexts[lane]
            )
            if lane_authority != base_authority:
                raise ValueError(
                    f"comparison lane {lane} configuration authority mismatch"
                )

        lane_recipes: dict[str, str] = {}
        for lane in ("ortools", "organic_runtime"):
            lane_recipe = prepared_recipes[lane]
            self.store.put_recipe(lane_recipe.to_dict())
            lane_recipes[lane] = lane_recipe.code

        lane_runs: dict[str, str] = {}
        for lane in ("organic_runtime", "ortools"):
            lane_run = self._submit_recipe_run(
                lane_recipes[lane],
                idempotency_key=f"{idempotency_key}:{lane}",
                comparison_campaign_code=campaign_code,
                schedule=False,
            )
            lane_runs[lane] = str(lane_run["runCode"])
        campaign = {
            "schemaVersion": 1,
            "campaignCode": campaign_code,
            "campaignHash": campaign_hash,
            "baseRecipeCode": base.code,
            "baseRecipeHash": base.sha256,
            "generationLanes": ["ortools", "organic_runtime"],
            "recipeCodes": lane_recipes,
            "runCodes": lane_runs,
            "matchedInputs": {
                "recipeSchemaVersion": int(base.to_dict()["schemaVersion"]),
                "profiles": copy.deepcopy(dict(base.profiles)),
                "situation": copy.deepcopy(dict(base.situation)),
                "tactics": [copy.deepcopy(dict(item)) for item in base.tactics],
                "qualification": copy.deepcopy(dict(base.qualification)),
                "resources": {
                    key: copy.deepcopy(value)
                    for key, value in dict(base.resources).items()
                    if key != "generationLane"
                },
                "seedStart": int(base.resources["seedStart"]),
                "seedCount": int(base.resources["seedCount"]),
                "analysisMode": str(base.resources["analysisMode"]),
                "generationLaneDelta": {
                    "ortools": {
                        "generationLane": "ortools",
                        "removedResourceKeys": [],
                    },
                    "organic_runtime": {
                        "generationLane": "organic_runtime",
                        "removedResourceKeys": [
                            key
                            for key in organic_removed_resource_keys
                            if key in base.resources
                        ],
                    },
                },
                "comparisonProfileHash": (
                    context.comparison_profile_hash
                    if context is not None
                    else self.comparison_profile_hash
                ),
                "configurationAuthorityHash": (
                    str(base.configuration["sha256"])
                    if isinstance(base, ResearchRecipeV3)
                    else None
                ),
                "effectiveRuntimeConfigHash": (
                    str(base.configuration["effectiveRuntimeConfigHash"])
                    if isinstance(base, ResearchRecipeV3)
                    else None
                ),
                "configurationSnapshotArtifact": (
                    copy.deepcopy(dict(base.configuration["snapshotArtifact"]))
                    if isinstance(base, ResearchRecipeV3)
                    else None
                ),
                "researchContextHash": (
                    context.context_hash
                    if isinstance(base, ResearchRecipeV3) and context is not None
                    else None
                ),
                **(
                    {
                        "diversity": copy.deepcopy(dict(base.diversity)),
                        "renderPolicy": copy.deepcopy(dict(base.render_policy)),
                    }
                    if isinstance(base, ResearchRecipeV2)
                    else {}
                ),
            },
            "createdAt": _now(),
        }
        stored = self.store.put_comparison_campaign(campaign)
        for lane in ("organic_runtime", "ortools"):
            run = self.store.get_run(lane_runs[lane])
            if run is not None and run["status"] in {"queued", "running"}:
                self._schedule_run(lane_runs[lane])
        return self._hydrate_comparison_campaign(stored)

    @staticmethod
    def _comparison_authority(
        recipe: ResearchRecipe,
        context: ResearchExecutionContextV1 | None,
    ) -> dict[str, Any]:
        if isinstance(recipe, ResearchRecipeV3) and context is None:
            raise ValueError(
                "Research V3 comparison lacks its frozen execution context"
            )
        return {
            "recipeSchemaVersion": int(recipe.to_dict()["schemaVersion"]),
            "profiles": copy.deepcopy(dict(recipe.profiles)),
            "analysisMode": str(recipe.resources["analysisMode"]),
            "configuration": (
                copy.deepcopy(dict(recipe.configuration))
                if isinstance(recipe, ResearchRecipeV3)
                else None
            ),
            "executionContext": (
                {
                    "bundleRevisionId": context.bundle_revision_id,
                    "effectiveRuntimeConfigHash": (
                        context.effective_runtime_config_hash
                    ),
                    "contextHash": context.context_hash,
                }
                if context is not None
                else None
            ),
        }

    def get_comparison_campaign(self, code: str) -> dict[str, Any] | None:
        item = self.store.get_comparison_campaign(code)
        return None if item is None else self._hydrate_comparison_campaign(item)

    def list_comparison_campaigns(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return [
            self._hydrate_comparison_campaign(item)
            for item in self.store.list_comparison_campaigns(limit=limit)
        ]

    def comparison_campaign_events(self, code: str) -> list[dict[str, Any]]:
        campaign = self.store.get_comparison_campaign(code)
        if campaign is None:
            raise KeyError(code)
        events: list[dict[str, Any]] = []
        for lane, run_code in dict(campaign["runCodes"]).items():
            run = self.store.get_run(str(run_code))
            for event in (run or {}).get("events", []):
                events.append(
                    {
                        **copy.deepcopy(dict(event)),
                        "generationLane": str(lane),
                        "runCode": str(run_code),
                    }
                )
        return sorted(
            events,
            key=lambda item: (str(item.get("at") or ""), str(item["generationLane"])),
        )

    def _hydrate_comparison_campaign(
        self, campaign: Mapping[str, Any]
    ) -> dict[str, Any]:
        base_value = self.store.get_recipe(str(campaign["baseRecipeCode"]))
        try:
            if base_value is None:
                raise ValueError("comparison base recipe is unavailable")
            _base_recipe, context = self._build_recipe_and_context(base_value)
        except (KeyError, PermissionError, RuntimeError, ValueError) as exc:
            lane_runs: dict[str, dict[str, Any]] = {}
            statuses: list[str] = []
            progress_values: list[float] = []
            for lane, run_code in sorted(dict(campaign["runCodes"]).items()):
                run = self.store.get_run(str(run_code))
                if run is None:
                    continue
                statuses.append(str(run.get("status") or "failed"))
                progress_values.append(float(run.get("progress", 0.0)))
                lane_runs[str(lane)] = {
                    "runCode": str(run_code),
                    "status": str(run.get("status") or "failed"),
                    "progress": float(run.get("progress", 0.0)),
                    "evidenceStatus": "historical_record_requires_readjudication",
                    "error": copy.deepcopy(run.get("error")),
                }
            terminal = {"completed", "failed", "cancelled"}
            if any(status == "failed" for status in statuses):
                status = "failed"
            elif any(status == "running" for status in statuses):
                status = "running"
            elif any(status == "queued" for status in statuses):
                status = "queued"
            elif statuses and all(status == "completed" for status in statuses):
                status = "completed"
            elif statuses and all(status in terminal for status in statuses):
                status = "cancelled"
            else:
                status = "failed"
            reason = "historical_record_requires_readjudication"
            comparison = {
                "schemaVersion": 1,
                "evidenceStatus": reason,
                "eligibleForPromotion": False,
                "passed": False,
                "recommendation": "insufficient_evidence",
                "reasons": [reason],
                "sourceErrorType": type(exc).__name__,
            }
            comparison["sha256"] = canonical_sha256(comparison)
            return {
                **copy.deepcopy(dict(campaign)),
                "status": status,
                "progress": (
                    sum(progress_values) / len(progress_values)
                    if progress_values
                    else 0.0
                ),
                "runs": lane_runs,
                "comparison": comparison,
            }
        comparison_profile = (
            context.comparison_profile
            if context is not None
            else self.comparison_profile
        )
        comparison_profile_hash = (
            context.comparison_profile_hash
            if context is not None
            else self.comparison_profile_hash
        )
        runs: dict[str, dict[str, Any]] = {}
        statuses: list[str] = []
        progress_values: list[float] = []
        for lane, run_code in dict(campaign["runCodes"]).items():
            run = self.store.get_run(str(run_code))
            if run is None:
                continue
            statuses.append(str(run["status"]))
            progress_values.append(float(run.get("progress", 0.0)))
            lane_stats = self.statistics(run_code=str(run_code))
            lane_performance = copy.deepcopy(dict(lane_stats.get("performance") or {}))
            qualified_count = int(run.get("qualifiedCount", 0))
            distinct_selected_count = int(run.get("selectedCount", 0))
            attempted_seeds = int(run.get("attemptedSeeds", 0))
            generated_candidates = int(run.get("structuralCandidates", 0))
            elapsed_seconds = float(run.get("elapsedSeconds", 0.0))
            cpu_seconds = (
                float(run["cpuSeconds"]) if run.get("cpuSeconds") is not None else None
            )
            semantic_defect_count = sum(
                int(value)
                for value in dict(lane_stats.get("semanticDefects") or {}).values()
            )
            runs[str(lane)] = {
                "runCode": str(run_code),
                "status": str(run["status"]),
                "progress": float(run.get("progress", 0.0)),
                "attemptedSeeds": attempted_seeds,
                "generatedCandidates": generated_candidates,
                "qualifiedCount": qualified_count,
                "distinctSelectedFinalistCount": distinct_selected_count,
                "rejectedCount": int(run.get("rejectedCount", 0)),
                "quarantineCount": int(run.get("quarantineCount", 0)),
                "constructionPreScreenCandidates": int(
                    run.get("constructionPreScreenCandidates", 0)
                ),
                "constructionPreScreenRejected": int(
                    run.get("constructionPreScreenRejected", 0)
                ),
                "tacticalViabilityPreScreenCandidates": int(
                    run.get("tacticalViabilityPreScreenCandidates", 0)
                ),
                "tacticalViabilityPreScreenRejected": int(
                    run.get("tacticalViabilityPreScreenRejected", 0)
                ),
                "constructionTargetMissCount": int(
                    run.get("constructionTargetMissCount", 0)
                ),
                "constructionPreScreenSeconds": float(
                    run.get("constructionPreScreenSeconds", 0.0)
                ),
                "tacticalViabilityPreScreenSeconds": float(
                    run.get("tacticalViabilityPreScreenSeconds", 0.0)
                ),
                "analyzedCandidateCount": int(run.get("analyzedCandidateCount", 0)),
                "error": copy.deepcopy(run.get("error")),
                "elapsedWallSeconds": elapsed_seconds,
                "cpuSeconds": cpu_seconds,
                "qualifiedPerWallHour": float(
                    lane_stats.get("qualifiedPerWallHour", 0.0)
                ),
                "qualifiedPerCpuHour": lane_stats.get("qualifiedPerCpuHour"),
                "qualifiedPer100Candidates": (
                    qualified_count * 100.0 / generated_candidates
                    if generated_candidates > 0
                    else 0.0
                ),
                "distinctSelectedFinalistsPer100Candidates": (
                    distinct_selected_count * 100.0 / generated_candidates
                    if generated_candidates > 0
                    else 0.0
                ),
                "distinctSelectedFinalistsPer100Seeds": (
                    distinct_selected_count * 100.0 / attempted_seeds
                    if attempted_seeds > 0
                    else 0.0
                ),
                "distinctSelectedFinalistsPerWallHour": (
                    distinct_selected_count * 3600.0 / elapsed_seconds
                    if elapsed_seconds > 0
                    else 0.0
                ),
                "distinctSelectedFinalistsPerCpuHour": (
                    distinct_selected_count * 3600.0 / cpu_seconds
                    if cpu_seconds is not None and cpu_seconds > 0
                    else None
                ),
                "secondsPerQualified": (
                    elapsed_seconds / qualified_count if qualified_count > 0 else None
                ),
                "cpuSecondsPerQualified": (
                    cpu_seconds / qualified_count
                    if cpu_seconds is not None and qualified_count > 0
                    else None
                ),
                "cpuSecondsPerDistinctSelectedFinalist": (
                    cpu_seconds / distinct_selected_count
                    if cpu_seconds is not None and distinct_selected_count > 0
                    else None
                ),
                "semanticDefectCount": semantic_defect_count,
                "stageSeconds": copy.deepcopy(
                    dict(lane_stats.get("stageSeconds") or {})
                ),
                "performanceEvidence": lane_performance,
            }
        terminal = {"completed", "failed", "cancelled"}
        if any(status == "failed" for status in statuses):
            status = "failed"
        elif len(statuses) == 2 and all(value in terminal for value in statuses):
            status = (
                "completed"
                if all(value == "completed" for value in statuses)
                else "cancelled"
            )
        elif any(status == "running" for status in statuses):
            status = "running"
        else:
            status = "queued"
        minimum_qualified = int(comparison_profile["minimumQualifiedPerLane"])
        minimum_distinct = int(
            comparison_profile["minimumDistinctSelectedFinalistsPerLane"]
        )
        minimum_attempted = int(comparison_profile["minimumAttemptedSeedsPerLane"])
        required_cpu_boundary = str(comparison_profile["requiredPromotionCpuBoundary"])
        minimum_yield_multiplier = float(comparison_profile["minimumYieldMultiplier"])
        minimum_cost_reduction = float(comparison_profile["minimumCostReduction"])
        ortools = runs.get("ortools")
        organic = runs.get("organic_runtime")
        smoke_reasons: list[str] = []
        release_reasons: list[str] = []
        yield_multiplier: float | None = None
        cost_reduction: float | None = None
        yield_gate_passed = False
        cost_gate_passed = False
        measurement_comparable = False
        cpu_boundary_comparable = False
        if status != "completed" or ortools is None or organic is None:
            smoke_reasons.append("comparison_incomplete")
            release_reasons.append("comparison_incomplete")
        else:
            for lane, evidence in (("ortools", ortools), ("organic_runtime", organic)):
                if int(evidence["qualifiedCount"]) < minimum_qualified:
                    smoke_reasons.append(f"insufficient_qualified:{lane}")
                if int(evidence["attemptedSeeds"]) < minimum_attempted:
                    release_reasons.append(f"insufficient_release_attempts:{lane}")
                if int(evidence["distinctSelectedFinalistCount"]) < minimum_distinct:
                    release_reasons.append(f"insufficient_distinct_finalists:{lane}")
                if (
                    bool(comparison_profile["requireZeroSemanticDefects"])
                    and int(evidence["semanticDefectCount"]) > 0
                ):
                    smoke_reasons.append(f"semantic_defects:{lane}")
                    release_reasons.append(f"semantic_defects:{lane}")

            environments = []
            cpu_boundaries = []
            for lane, evidence in (("ortools", ortools), ("organic_runtime", organic)):
                performance_evidence = dict(evidence["performanceEvidence"])
                environment_evidence = dict(
                    performance_evidence.get("environment") or {}
                )
                declaration = environment_evidence.get("declaration")
                comparison_hashes = list(
                    environment_evidence.get("comparisonHashes") or ()
                )
                if (
                    environment_evidence.get("status") != "measured"
                    or not isinstance(declaration, Mapping)
                    or len(comparison_hashes) != 1
                ):
                    release_reasons.append(
                        f"measurement_environment_unavailable:{lane}"
                    )
                else:
                    environments.append(str(comparison_hashes[0]))
                cpu_evidence = dict(performance_evidence.get("cpu") or {})
                boundary = cpu_evidence.get("measurementBoundary")
                if cpu_evidence.get("availability") != "measured" or not boundary:
                    release_reasons.append(f"cpu_evidence_unavailable:{lane}")
                else:
                    cpu_boundaries.append(str(boundary))
            measurement_comparable = (
                len(environments) == 2 and len(set(environments)) == 1
            )
            if len(environments) == 2 and not measurement_comparable:
                release_reasons.append("measurement_environment_mismatch")
            cpu_boundary_comparable = (
                len(cpu_boundaries) == 2 and len(set(cpu_boundaries)) == 1
            )
            if len(cpu_boundaries) == 2 and not cpu_boundary_comparable:
                release_reasons.append("cpu_measurement_boundary_mismatch")
            if (
                not cpu_boundary_comparable
                or not cpu_boundaries
                or cpu_boundaries[0] != required_cpu_boundary
            ):
                release_reasons.append("cpu_boundary_not_promotion_eligible")

            organic_yield = organic.get("distinctSelectedFinalistsPerCpuHour")
            ortools_yield = ortools.get("distinctSelectedFinalistsPerCpuHour")
            if organic_yield is None or ortools_yield is None:
                release_reasons.append("distinct_cpu_yield_unavailable")
            elif float(organic_yield) > 0:
                yield_multiplier = float(ortools_yield) / float(organic_yield)
                yield_gate_passed = yield_multiplier >= minimum_yield_multiplier
            elif float(ortools_yield) > 0:
                yield_gate_passed = True
            organic_cost = organic.get("cpuSecondsPerDistinctSelectedFinalist")
            ortools_cost = ortools.get("cpuSecondsPerDistinctSelectedFinalist")
            if (
                organic_cost is not None
                and ortools_cost is not None
                and float(organic_cost) > 0
            ):
                cost_reduction = 1.0 - float(ortools_cost) / float(organic_cost)
                cost_gate_passed = cost_reduction >= minimum_cost_reduction
            if not (yield_gate_passed or cost_gate_passed):
                release_reasons.append("performance_threshold_not_met")
        smoke_passed = not smoke_reasons
        release_reasons = sorted(set(release_reasons))
        passed = not release_reasons
        comparison = {
            "schemaVersion": 2,
            "profileHash": comparison_profile_hash,
            "smokeEvidence": {
                "scope": "smoke_only_not_promotion_evidence",
                "minimumQualifiedPerLane": minimum_qualified,
                "passed": smoke_passed,
                "reasons": sorted(set(smoke_reasons)),
            },
            "minimumDistinctSelectedFinalistsPerLane": minimum_distinct,
            "minimumAttemptedSeedsPerLane": minimum_attempted,
            "minimumYieldMultiplier": minimum_yield_multiplier,
            "minimumCostReduction": minimum_cost_reduction,
            "requiredPromotionCpuBoundary": required_cpu_boundary,
            "measurementComparable": measurement_comparable,
            "cpuBoundaryComparable": cpu_boundary_comparable,
            "yieldMultiplier": yield_multiplier,
            "costReduction": cost_reduction,
            "yieldGatePassed": yield_gate_passed,
            "costGatePassed": cost_gate_passed,
            "passed": passed,
            "eligibleForPromotion": passed,
            "recommendation": (
                "promote_ortools"
                if passed
                else "insufficient_evidence"
                if any(
                    reason == "comparison_incomplete"
                    or reason.startswith("insufficient_")
                    or "unavailable" in reason
                    or reason.endswith("not_promotion_eligible")
                    for reason in release_reasons
                )
                else "retain_current"
            ),
            "reasons": release_reasons,
        }
        comparison["sha256"] = canonical_sha256(comparison)
        return {
            **copy.deepcopy(dict(campaign)),
            "status": status,
            "progress": sum(progress_values) / len(progress_values)
            if progress_values
            else 0.0,
            "runs": runs,
            "comparison": comparison,
        }

    def get_run(self, run_code: str) -> dict[str, Any] | None:
        return self.store.get_run(run_code)

    def get_operator_run(self, run_code: str) -> dict[str, Any] | None:
        value = self.store.get_run(run_code, include_coordinator_evidence=False)
        return None if value is None else _operator_research_run_v1(value)

    def list_runs(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return self.store.list_runs(status=status, limit=limit)

    def list_operator_runs(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return [
            _operator_research_run_v1(value)
            for value in self.store.list_runs(
                status=status, limit=limit, include_coordinator_evidence=False
            )
        ]

    def cancel(self, run_code: str) -> dict[str, Any]:
        existing = self.store.get_run(run_code)
        if existing is None:
            raise KeyError(run_code)
        if existing["status"] not in {"queued", "running"}:
            raise ValueError("research run cannot be cancelled from its current state")
        with self._lock:
            self._cancelled.add(run_code)
        self.store.cancel_shards(run_code)
        campaign_progress = copy.deepcopy(dict(existing.get("campaignProgress") or {}))
        return self.store.update_run(
            run_code,
            status="cancelled",
            progress=float(existing["progress"]),
            campaignProgress={
                **campaign_progress,
                "state": "cancelled",
                "stopReason": "cancelled",
            },
            events=[*existing.get("events", []), {"type": "cancelled", "at": _now()}],
        )

    def events(self, run_code: str) -> list[dict[str, Any]]:
        item = self.store.get_run(run_code)
        if item is None:
            raise KeyError(run_code)
        return list(item.get("events") or ())

    def _verify_native_result_documents_cached(
        self,
        value: Mapping[str, Any],
        *,
        recipe: Mapping[str, Any],
        documents: Mapping[str, Any],
        campaign_document: Mapping[str, Any] | None,
        configuration_snapshot: Mapping[str, Any],
    ) -> None:
        # All CAS bytes, hashes, owners and result sections are checked by the
        # caller on every read. Bind actual inputs, not their claimed identities.
        settings = dict(configuration_snapshot.get("components") or {}).get(
            "challenge", {}
        ).get("payload", {}).get("scenarioResearchV4", {})
        limits = settings.get("cacheLimits", {})
        maximum_entries = limits.get("maximumEntries", 0)
        maximum_bytes = limits.get("maximumBytes", 0)
        cache_enabled = (
            type(maximum_entries) is int and maximum_entries > 0
            and type(maximum_bytes) is int and maximum_bytes >= 64
        )
        inputs = {
            "recipe": recipe,
            "documents": documents,
            "campaign_document": campaign_document,
            "configuration_snapshot": configuration_snapshot,
        }
        if not cache_enabled:
            validate_ortools_result_artifact_documents_v3(value, **inputs)
            return
        cache_key = canonical_sha256({
            "validator": "native_result_documents_v3",
            "result": canonical_sha256(value),
            "recipe": canonical_sha256(recipe),
            "documents": {name: canonical_sha256(document)
                          for name, document in documents.items()},
            "campaign": canonical_sha256(campaign_document),
            "configuration": canonical_sha256(configuration_snapshot),
        })
        # Published entries certify immutable content, so an already-verified
        # read need not wait behind an unrelated cold proof. Eviction only
        # removes reuse; it cannot invalidate the proof for this exact key.
        if cache_key in self._result_verification_cache:
            return
        with self._result_verification_lock:
            while self._result_verification_cache and (
                len(self._result_verification_cache) > maximum_entries
                or len(self._result_verification_cache) * 64 > maximum_bytes
            ):
                del self._result_verification_cache[
                    next(iter(self._result_verification_cache))
                ]
            if cache_key in self._result_verification_cache:
                return
            validate_ortools_result_artifact_documents_v3(value, **inputs)
            while self._result_verification_cache and (
                len(self._result_verification_cache) >= maximum_entries
                or (len(self._result_verification_cache) + 1) * 64 > maximum_bytes
            ):
                del self._result_verification_cache[
                    next(iter(self._result_verification_cache))
                ]
            self._result_verification_cache[cache_key] = None

    def _validate_stored_scenario_result_v4(
        self,
        value: Mapping[str, Any],
        *,
        run: Mapping[str, Any] | None = None,
        recipe_value: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        identity = dict(dict(value.get("provenance") or {}).get("resultIdentity") or {})
        if identity.get("kind") in {
            ORTOOLS_RESULT_IDENTITY_KIND,
            ORTOOLS_RESULT_IDENTITY_V3_KIND,
        }:
            run = run or self.store.get_run(str(value.get("runCode") or ""))
            recipe_value = recipe_value or self.store.get_recipe(
                str(value.get("recipeCode") or "")
            )
            if not isinstance(run, Mapping) or not isinstance(recipe_value, Mapping):
                raise RuntimeError(
                    "stored OR-Tools Research run or recipe authority is unavailable"
                )
            recipe = ResearchRecipeV4.build(recipe_value)
            validated = validate_ortools_result_sections_v1(
                value,
                recipe=recipe.to_dict(),
            )
            authority = dict(validated["researchAuthority"])
            if identity.get("kind") == ORTOOLS_RESULT_IDENTITY_KIND:
                resolved_recipe, context = self._build_recipe_and_context(recipe_value)
                if not isinstance(resolved_recipe, ResearchRecipeV4) or context is None:
                    raise RuntimeError(
                        "stored OR-Tools Research recipe authority mismatch"
                    )
                self._assert_run_execution_context(run, resolved_recipe, context)
                backend = self._cached_ortools_scenario_backend_v2(context)
                if backend is None:
                    raise RuntimeError(
                        "stored OR-Tools Research backend is unavailable"
                    )
                expected_domain_hashes = {
                    "auction": AuctionDomainIRV1.from_rules(
                        context.bundle.rules.data
                    ).sha256,
                    "escalation": EscalationDomainIRV1.from_rules(
                        context.bundle.rules.data
                    ).sha256,
                    "play": context.bundle.rules.play_domain.sha256,
                    "projects": context.bundle.rules.project_domain.sha256,
                    "scoring": ScoringDomainIRV1.from_rules(
                        context.bundle.rules.data
                    ).sha256,
                    "structural": context.bundle.rules.structural_domain.sha256,
                }
                if (
                    authority.get("backendVersion") != backend.identity.backend_version
                    or authority.get("solverVersion") != backend.identity.solver_version
                    or authority.get("bundleRevisionId") != context.bundle_revision_id
                    or authority.get("bundleHash") != context.bundle.sha256
                    or authority.get("loweringManifestHash")
                    != backend.identity.predicate_catalog_hash
                    or authority.get("domainIrHashes") != expected_domain_hashes
                ):
                    raise RuntimeError("stored OR-Tools Research authority mismatch")
            loaded_semantic: dict[str, dict[str, Any]] = {}
            for name, media_type in ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES.items():
                document, owner_hash = _load_owned_result_json_artifact_v1(
                    self.artifact_store,
                    dict(dict(validated["artifacts"])[name]),
                    media_type=media_type,
                    owner_kind="ortools_research_semantic_artifact_v1",
                )
                if (
                    owner_hash != identity.get("executorSemanticHash")
                    or scenario_sha256(document)
                    != dict(validated["artifacts"])[name]["sha256"]
                ):
                    raise RuntimeError(
                        f"stored OR-Tools Research {name} artifact mismatch"
                    )
                loaded_semantic[name] = document
            if identity.get("kind") == ORTOOLS_RESULT_IDENTITY_V3_KIND:
                authority_artifacts = dict(validated.get("authorityArtifacts") or {})
                loaded_authority: dict[str, dict[str, Any]] = {}
                for name, media_type in ORTOOLS_AUTHORITY_ARTIFACT_MEDIA_TYPES.items():
                    if name == "interestAdmission" and name not in authority_artifacts:
                        continue
                    document, owner_hash = _load_owned_result_json_artifact_v1(
                        self.artifact_store,
                        dict(authority_artifacts.get(name) or {}),
                        media_type=media_type,
                        owner_kind="ortools_research_authority_artifact_v1",
                    )
                    if owner_hash != validated.get("resultHash"):
                        raise RuntimeError(
                            f"stored OR-Tools Research {name} authority artifact mismatch"
                        )
                    loaded_authority[name] = document
                snapshot = loaded_authority["configurationSnapshot"]
                manifest = loaded_authority["loweringManifest"]
                if (
                    snapshot.get("sha256") != identity.get("configurationSnapshotHash")
                    or snapshot.get("bundleHash") != authority.get("bundleHash")
                    or manifest.get("sha256") != authority.get("loweringManifestHash")
                    or loaded_authority["resultIdentity"] != identity
                    or loaded_authority["executionReceipt"]
                    != dict(validated.get("executionReceipt") or {})
                ):
                    raise RuntimeError(
                        "stored OR-Tools Research authority artifact substitution detected"
                    )
                version_parts = tuple(
                    int(item)
                    for item in str(authority.get("backendVersion") or "").split(".")
                    if item.isdigit()
                )
                campaign_document = None
                if (
                    authority.get("backendId") == "ortools_cp_sat_v3"
                    and len(version_parts) == 3
                    and version_parts >= (3, 1, 0)
                ):
                    campaign_reference = run.get("scenarioResearchEvidence")
                    if not isinstance(campaign_reference, Mapping):
                        raise RuntimeError(
                            "stored OR-Tools Research campaign evidence is unavailable"
                        )
                    campaign_document = _load_result_json_artifact_v1(
                        self.artifact_store,
                        {
                            key: copy.deepcopy(campaign_reference.get(key))
                            for key in _CAS_ARTIFACT_REFERENCE_KEYS
                        },
                        media_type=(
                            "application/vnd.balootx.ortools-research-campaign-v1+json"
                        ),
                    )
                self._verify_native_result_documents_cached(
                    validated,
                    recipe=recipe.to_dict(),
                    documents=loaded_semantic,
                    campaign_document=campaign_document,
                    configuration_snapshot=snapshot,
                )
                if "interestAdmission" in loaded_authority:
                    validate_native_interest_artifact_v1(
                        validated, loaded_semantic["completeTrace"], snapshot,
                        loaded_authority["interestAdmission"],
                    )
            return validate_scenario_result_authority_v1(
                validated,
                run=run,
                recipe=recipe.to_dict(),
            )
        try:
            validated = validate_scenario_result_v4(
                value,
                artifact_store=self.artifact_store,
            )
        except (FileNotFoundError, KeyError) as exc:
            raise RuntimeError("stored Research V4 artifact is unavailable") from exc
        run = run or self.store.get_run(str(validated.get("runCode") or ""))
        recipe_value = recipe_value or self.store.get_recipe(
            str(validated.get("recipeCode") or "")
        )
        if not isinstance(run, Mapping) or not isinstance(recipe_value, Mapping):
            raise RuntimeError(
                "stored Research V4 run or recipe authority is unavailable"
            )
        recipe = copy.deepcopy(dict(recipe_value))
        recipe_core = {key: item for key, item in recipe.items() if key != "sha256"}
        if recipe.get("schemaVersion") != 4 or recipe.get("sha256") != scenario_sha256(
            recipe_core
        ):
            raise RuntimeError("stored Research V4 recipe authority mismatch")
        return validate_scenario_result_authority_v1(
            validated,
            run=run,
            recipe=recipe,
        )

    def _validate_stored_result(self, value: Mapping[str, Any]) -> dict[str, Any]:
        run = self.store.get_run(str(value.get("runCode") or ""))
        if not isinstance(run, Mapping):
            raise RuntimeError("stored Research result run authority is unavailable")
        recipe_value = self.store.get_recipe(str(run.get("recipeCode") or ""))
        if not isinstance(recipe_value, Mapping):
            raise RuntimeError("stored Research result recipe authority is unavailable")
        recipe_is_v4 = recipe_value.get("schemaVersion") == 4
        result_is_v4 = value.get("schemaVersion") == 2
        if recipe_is_v4 != result_is_v4:
            raise RuntimeError("stored Research result schema authority mismatch")
        if result_is_v4:
            return self._validate_stored_scenario_result_v4(
                value,
                run=run,
                recipe_value=recipe_value,
            )
        return dict(value)

    def list_results(self, **filters: Any) -> dict[str, Any]:
        page = self.store.list_results(**filters)
        page["items"] = [self._validate_stored_result(item) for item in page["items"]]
        return page

    def list_sequential_search_receipts(
        self,
        run_code: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        if type(run_code) is not str or not run_code.strip():
            raise ValueError(
                "sequential search receipt run code must be a non-empty string"
            )
        if type(offset) is not int or offset < 0:
            raise ValueError("sequential search receipt offset must be non-negative")
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError(
                "sequential search receipt limit must be between 1 and 500"
            )
        if self.store.get_run(run_code) is None:
            raise KeyError(run_code)
        page = self.store.list_sequential_search_receipts(
            run_code=run_code,
            admitted_only=True,
            offset=offset,
            limit=limit,
        )
        return build_sequential_search_receipt_page(
            page,
            run_code=run_code,
            offset=offset,
            limit=limit,
        )

    def get_result(self, code: str) -> dict[str, Any] | None:
        value = self.store.get_result(code)
        if value is None:
            return None
        if value.get("resultCode") != str(code):
            raise RuntimeError("stored Research result owner identity mismatch")
        return self._validate_stored_result(value)

    def get_exact_state(
        self,
        code: str,
        *,
        observer_seat: str,
    ) -> dict[str, Any]:
        """Return the immutable V4 position through its acting player's view."""

        result = self.get_result(code)
        if result is None:
            raise KeyError(code)
        if result.get("schemaVersion") != 2:
            raise ValueError("Research exact-state view requires a V4 result")
        artifacts = result.get("artifacts")
        projection = result.get("scenarioProjection")
        if not isinstance(artifacts, Mapping) or not isinstance(projection, Mapping):
            raise ValueError("Research V4 exact-state authority is unavailable")
        identity = dict(
            dict(result.get("provenance") or {}).get("resultIdentity") or {}
        )
        if identity.get("kind") in {
            ORTOOLS_RESULT_IDENTITY_KIND,
            ORTOOLS_RESULT_IDENTITY_V3_KIND,
        }:
            require_runtime_compatibility_passed_v1(
                dict(result.get("runtimeCompatibility") or {})
            )
            reference = dict(artifacts.get("completeTrace") or {})
            trace_document, trace_owner_hash = _load_owned_result_json_artifact_v1(
                self.artifact_store,
                reference,
                media_type=ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES["completeTrace"],
                owner_kind="ortools_research_semantic_artifact_v1",
            )
            authority = dict(result.get("researchAuthority") or {})
            if trace_owner_hash != identity.get(
                "executorSemanticHash"
            ) or scenario_sha256(trace_document) != authority.get("traceHash"):
                raise ValueError("OR-Tools Research complete-trace authority mismatch")
            exact_state = trace_document.get("exactNormalization")
            if not isinstance(exact_state, Mapping):
                raise ValueError(
                    "OR-Tools Research exact-state document is unavailable"
                )
        else:
            reference = artifacts.get("exactState")
            if not isinstance(reference, Mapping):
                raise ValueError("Research V4 exact-state artifact is unavailable")
            try:
                exact_state = json.loads(
                    self.artifact_store.get(str(reference.get("uri") or "")).decode(
                        "utf-8"
                    )
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Research V4 exact-state artifact is invalid") from exc
            if not isinstance(exact_state, Mapping):
                raise ValueError("Research V4 exact-state artifact is invalid")
        return build_observer_safe_exact_state_v1(
            exact_state=exact_state,
            scenario_projection=projection,
            observer_seat=observer_seat,
            decision_reference={
                "resultCode": str(code),
                "decisionOrdinal": 1,
            },
        )

    def get_playing_quality_audit(self, code: str) -> dict[str, Any]:
        """Project the immutable all-decision receipt without exposing private state."""

        result = self.get_result(code)
        if result is None:
            raise KeyError(code)
        identity = dict(dict(result.get("provenance") or {}).get("resultIdentity") or {})
        if identity.get("kind") != ORTOOLS_RESULT_IDENTITY_V3_KIND:
            raise ValueError("Research playing-quality audit requires an OR-Tools V3 result")
        reference = dict(dict(result.get("artifacts") or {}).get("completeTrace") or {})
        trace, owner_hash = _load_owned_result_json_artifact_v1(
            self.artifact_store,
            reference,
            media_type=ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES["completeTrace"],
            owner_kind="ortools_research_semantic_artifact_v1",
        )
        authority = dict(result.get("researchAuthority") or {})
        if (
            owner_hash != identity.get("executorSemanticHash")
            or scenario_sha256(trace) != authority.get("traceHash")
        ):
            raise ValueError("OR-Tools Research complete-trace authority mismatch")
        evidence = trace.get("playingQualityEvidence")
        if not isinstance(evidence, Mapping) or not isinstance(evidence.get("receipt"), Mapping):
            raise ValueError("Research playing-quality evidence is unavailable")
        receipt = dict(evidence["receipt"])
        snapshot, snapshot_owner = _load_owned_result_json_artifact_v1(
            self.artifact_store,
            dict(dict(result.get("authorityArtifacts") or {}).get("configurationSnapshot") or {}),
            media_type=ORTOOLS_AUTHORITY_ARTIFACT_MEDIA_TYPES["configurationSnapshot"],
            owner_kind="ortools_research_authority_artifact_v1",
        )
        if (snapshot_owner != result.get("resultHash")
                or snapshot.get("sha256") != identity.get("configurationSnapshotHash")
                or snapshot.get("bundleHash") != authority.get("bundleHash")
                or scenario_sha256({key: value for key, value in snapshot.items() if key != "sha256"}) != snapshot.get("sha256")):
            raise ValueError("Research playing-quality frozen configuration mismatch")
        from baloot_platform.core.rules import RuleBook
        rules = RuleBook(snapshot["components"]["rules"]["payload"])
        if rules.sha256 != snapshot["components"]["rules"]["sha256"]:
            raise ValueError("Research playing-quality frozen rules mismatch")
        # get_result freshly validates quality against the frozen recipe and bundle.
        # The trace hash above binds this projection to that same validated evidence.
        if receipt.get("scope") != "all_decisions" or receipt.get("status") != "complete":
            raise ValueError("Research all-decision quality audit is incomplete")

        auction = dict(receipt.get("auction") or {})
        source_rows = [
            (row, auction.get("artifacts") or {})
            for row in auction.get("decisions", ())
            if isinstance(row, Mapping)
        ]
        post = receipt.get("postAuction")
        if isinstance(post, Mapping):
            escalation = post.get("escalation")
            if isinstance(escalation, Mapping):
                artifacts = dict(escalation.get("artifacts") or {})
                recorded_events = escalation.get("recordedEvents")
                for index, reference in enumerate(escalation.get("decisions", ())):
                    artifact = dict(artifacts[reference["artifactHash"]])
                    quality = dict(dict(artifact["proofModel"])["actionQuality"])
                    proof = dict(artifact["proof"])
                    recorded = escalation.get("operation") == "audit_recorded" or (
                        escalation.get("operation") == "audit_then_generate"
                        and isinstance(recorded_events, list)
                        and index < len(recorded_events)
                    )
                    source_rows.append(
                        ({
                            "decisionKind": "escalation",
                            "actor": reference["actor"],
                            "action": proof["selectedAction"],
                            "actualActionOptimal": quality["actualActionOptimal"],
                            "regret": quality["regret"],
                            "selectionReason": (
                                "recorded_action"
                                if recorded
                                else "native_actor_information_set_best_value"
                            ),
                            "proofArtifactHash": reference["artifactHash"],
                            "actionQualityHash": quality["sha256"],
                        }, artifacts)
                    )
            project_play = post.get("projectPlay")
            if isinstance(project_play, Mapping):
                source_rows.extend(
                    (row, project_play.get("artifacts") or {})
                    for row in project_play.get("decisions", ())
                    if isinstance(row, Mapping)
                )
        kinds = {kind: 0 for kind in ("bid", "escalation", "project", "play")}
        rows: list[dict[str, Any]] = []
        for ordinal, (row, artifacts) in enumerate(source_rows, 1):
            kind = str(row.get("decisionKind") or "")
            if kind not in kinds:
                raise ValueError("Research playing-quality decision kind is invalid")
            artifact = artifacts.get(row.get("proofArtifactHash"))
            proof = artifact.get("proof") if isinstance(artifact, Mapping) else None
            root_actions = proof.get("rootActions") if isinstance(proof, Mapping) else None
            if not isinstance(root_actions, list) or not root_actions:
                raise ValueError("Research playing-quality proof roots are unavailable")
            request = artifact.get("request")
            model = artifact.get("proofModel")
            graph = artifact.get("graph")
            if not isinstance(request, Mapping) or not isinstance(model, Mapping) or not isinstance(graph, Mapping):
                raise ValueError("Research playing-quality proof boundary is unavailable")
            mode, visibility = request.get("mode"), request.get("visibility")
            completion = proof.get("completionStatus")
            analysis_scope = model.get("analysisScope")
            action_quality = model.get("actionQuality")
            scope = ((action_quality.get("scope") if isinstance(action_quality, Mapping)
                      and action_quality.get("scope") == "observer_safe_policy_conditioned" else None)
                     or proof.get("completionScope")
                     or (analysis_scope.get("coverage") if isinstance(analysis_scope, Mapping) else None)
                     or graph.get("scope")
                     or (action_quality.get("scope") if isinstance(action_quality, Mapping) else None))
            if (mode not in {"exact", "bounded"}
                    or visibility not in {"complete_state", "actor_information_set"}
                    or completion != ("exact_complete" if mode == "exact" else "bounded_complete")
                    or not isinstance(scope, str) or not scope):
                raise ValueError("Research playing-quality proof boundary is inconsistent")
            kinds[kind] += 1
            core = {
                "ordinal": ordinal,
                "kind": kind,
                "actor": row.get("actor"),
                "action": row.get("action"),
                "legalActionCount": len(root_actions),
                "actualActionOptimal": row.get("actualActionOptimal"),
                "regret": row.get("regret"),
                "selectionReason": row.get("selectionReason"),
                "proofMode": mode,
                "proofVisibility": visibility,
                "proofCompletionStatus": completion,
                "proofScope": scope,
                "proofArtifactHash": row.get("proofArtifactHash"),
                "actionQualityHash": row.get("actionQualityHash"),
            }
            rows.append({**core, "sha256": scenario_sha256(core)})
        featured: dict[str, Any] | None = None
        if dict(receipt.get("qualityExecution") or {}).get("featuredPlayNativeProof") is True:
            prefix = ExactTracePrefixV1.build(trace["exactTracePrefix"])
            exact_state_id = dict(trace["exactNormalization"])["exactStateId"]
            play = dict(dict(receipt["postAuction"])["projectPlay"])
            matches = [row for row in play["decisions"]
                       if row.get("decisionKind", "play") == "play"
                       and row.get("playIndex") == len(prefix.play_actions)]
            if len(matches) != 1 or matches[0].get("exactStateId") != exact_state_id:
                raise ValueError("featured native play decision is ambiguous")
            selected = matches[0]
            artifact = dict(play["artifacts"])[selected["proofArtifactHash"]]
            proof_model = dict(artifact["proofModel"])
            if native_featured_play_proof_supported_v1(artifact):
                proof = dict(artifact["proof"])
                hidden = proof.get("hiddenWorlds")
                featured = {
                    "exactStateId": exact_state_id,
                    "playIndex": selected["playIndex"],
                    "actor": selected["actor"],
                    "action": selected["action"],
                    "proofArtifactHash": selected["proofArtifactHash"],
                    "proofModelKind": proof_model["kind"],
                    "completionStatus": proof["completionStatus"],
                    **({"hiddenWorlds": {
                        "method": (hidden["coverage"]["method"]
                                   if proof_model["kind"] == "actor_information_set_cegis_model_v1"
                                   else hidden["method"]),
                        "count": hidden["count"], "complete": hidden["complete"],
                    }}
                       if hidden is not None else {}),
                }
        core = {
            "schemaVersion": 1,
            "resultCode": code,
            "resultHash": result.get("resultHash"),
            "scope": "all_decisions",
            "status": "complete",
            "wholeRoundCertified": False,
            "operation": receipt.get("operation"),
            "qualityReceiptHash": receipt.get("sha256"),
            "completeTraceHash": authority.get("traceHash"),
            "decisionCount": len(rows),
            "optimalDecisionCount": sum(row["actualActionOptimal"] is True for row in rows),
            "zeroRegretDecisionCount": sum(row["regret"] == 0 for row in rows),
            "countsByKind": kinds,
            "decisions": rows,
            **({"featuredPlayProof": featured} if featured is not None else {}),
        }
        return {**core, "sha256": scenario_sha256(core)}

    def list_audit_traces(self, **filters: Any) -> dict[str, Any]:
        return self.audit_traces.list_records(**filters)

    def get_audit_trace(self, identifier: str) -> dict[str, Any] | None:
        record = self.audit_traces.get_record(identifier)
        return None if record is None else record.to_dict()

    def load_audit_trace(self, identifier: str) -> CompleteGameTraceV1:
        """Load a trusted trace for internal observer-safe projection only."""
        return self.audit_traces.load_trace(identifier)

    def get_decisions(self, code: str) -> list[dict[str, Any]]:
        result = self.get_result(code)
        if result is None:
            raise KeyError(code)
        if result.get("schemaVersion") == 2:
            return scenario_decision_audit_v1(result)
        artifacts_value = result.get("artifacts")
        if not isinstance(artifacts_value, Mapping):
            raise ValueError("Research result artifacts are unavailable")
        artifacts = dict(artifacts_value)

        trace_reference = dict(artifacts.get("completeTrace") or {})
        trace_document, trace_owner_hash = _load_owned_result_json_artifact_v1(
            self.artifact_store,
            trace_reference,
            media_type=COMPLETE_TRACE_MEDIA_TYPE,
            owner_kind="complete_trace_v1",
        )
        trace = CompleteGameTraceV1.from_dict(trace_document)
        if trace_owner_hash != trace.sha256:
            raise ValueError("Research complete-trace semantic owner identity mismatch")
        if str(result.get("traceHash") or "") != trace.sha256:
            raise ValueError("Research result complete-trace membership mismatch")

        recipe_value = self.store.get_recipe(str(result.get("recipeCode") or ""))
        run_value = self.store.get_run(str(result.get("runCode") or ""))
        if not isinstance(recipe_value, Mapping) or not isinstance(run_value, Mapping):
            raise ValueError("Research result recipe or run authority is unavailable")
        recipe, context = self._build_recipe_and_context(recipe_value)
        self._assert_run_execution_context(run_value, recipe, context)
        result_provenance = dict(result.get("provenance") or {})
        if (
            str(result.get("recipeCode") or "") != recipe.code
            or str(result_provenance.get("recipeHash") or "") != recipe.sha256
            or str(run_value.get("recipeCode") or "") != recipe.code
            or str(run_value.get("recipeHash") or "") != recipe.sha256
        ):
            raise ValueError("Research result recipe authority mismatch")
        if str(result.get("createdAt") or "") != str(run_value.get("createdAt") or ""):
            raise ValueError("Research result creation authority mismatch")

        retained_trace_reference = {
            key: copy.deepcopy(trace_reference[key])
            for key in _CAS_ARTIFACT_REFERENCE_KEYS
        }
        expected_audit_source = {
            "kind": "research_candidate",
            "runCode": str(result.get("runCode") or ""),
            "runHash": str(run_value.get("runHash") or ""),
            "recipeCode": recipe.code,
            "recipeHash": recipe.sha256,
            "shardId": str(result_provenance.get("researchShardId") or ""),
            "shardHash": str(result_provenance.get("researchShardHash") or ""),
            "shardIndex": result_provenance.get("researchShardIndex"),
            "seed": result_provenance.get("researchSeed"),
            "candidateHash": str(result_provenance.get("candidateHash") or ""),
            "generationLane": str(result_provenance.get("generationLane") or ""),
        }
        trace_occurrences: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self.audit_traces.list_records(
                run_code=expected_audit_source["runCode"],
                offset=offset,
                limit=500,
            )
            trace_occurrences.extend(
                copy.deepcopy(dict(item))
                for item in page["items"]
                if str(item.get("traceHash") or "") == trace.sha256
            )
            next_offset = page.get("nextOffset")
            if next_offset is None:
                break
            next_value = int(next_offset)
            if next_value <= offset:
                raise ValueError("Research audit occurrence pagination did not advance")
            offset = next_value
        if not trace_occurrences:
            raise ValueError("Research result retained audit occurrence is unavailable")
        matching_occurrences = [
            item
            for item in trace_occurrences
            if all(
                dict(item.get("source") or {}).get(key) == expected
                for key, expected in expected_audit_source.items()
            )
            and dict(item.get("completeTrace") or {}) == retained_trace_reference
            and item.get("status") == "finalized"
            and item.get("attemptStatus") == "completed"
            and item.get("disposition") == str(result.get("disposition") or "")
        ]
        if not matching_occurrences:
            raise ValueError("Research result retained audit occurrence mismatch")

        analyzers = context.analyzers if context is not None else self.analyzers
        analysis_mode = str(recipe.resources.get("analysisMode") or "")
        audit_analysis_mode = (
            "deep" if self._uses_finalist_primary_policy(recipe) else analysis_mode
        )
        canonical_analysis_documents: dict[str, dict[str, Any]] = {}

        def load_analysis(
            artifact_name: str,
            *,
            mode: str,
            label: str,
        ) -> TraceAnalysisResultV1:
            document, owner_hash = _load_owned_result_json_artifact_v1(
                self.artifact_store,
                dict(artifacts.get(artifact_name) or {}),
                media_type=_TRACE_ANALYSIS_MEDIA_TYPE,
                owner_kind="trace_analysis_v1",
            )
            expected_analyzer = analyzers.get(mode)
            if expected_analyzer is None:
                raise ValueError(f"Research result {label} authority is unavailable")
            descriptor = expected_analyzer.descriptor
            expected_profile_hash = str(expected_analyzer.profile.sha256)
            if (
                str(document.get("analyzerId") or "") != str(descriptor.id)
                or str(document.get("analyzerVersion") or "") != str(descriptor.version)
                or str(document.get("profileHash") or "") != expected_profile_hash
            ):
                raise ValueError(f"Research result {label} authority mismatch")
            expected_adaptive_profile_hash = None
            if (
                str(document.get("analyzerId") or "")
                == "adaptive_finalist_trace_analyzer_v1"
                and document.get("schemaVersion") == 3
            ):
                expected_adaptive_profile_hash = (
                    frozen_adaptive_finalist_profile_hash_v1(trace)
                )
                if expected_adaptive_profile_hash != expected_profile_hash:
                    raise ValueError(
                        f"Research adaptive {label} does not match frozen trace authority"
                    )
            parsed = TraceAnalysisResultV1.from_dict(
                document,
                ledger=trace.decision_ledger,
                expected_adaptive_profile_hash=expected_adaptive_profile_hash,
            )
            if owner_hash != parsed.sha256:
                raise ValueError(f"Research {label} semantic owner identity mismatch")
            if parsed.trace_hash != trace.sha256:
                raise ValueError(f"Research {label} belongs to another complete trace")
            return parsed

        def assert_canonical_analysis(
            analysis: TraceAnalysisResultV1,
            *,
            mode: str,
            label: str,
        ) -> None:
            canonical_document = canonical_analysis_documents.get(mode)
            if canonical_document is None:
                expected_analyzer = analyzers.get(mode)
                if expected_analyzer is None:
                    raise ValueError(
                        f"Research result {label} authority is unavailable"
                    )
                canonical_document = expected_analyzer.analyze_trace(trace).to_dict()
                canonical_analysis_documents[mode] = canonical_document
            if analysis.to_dict() != canonical_document:
                raise ValueError(
                    f"Research result {label} is not the canonical analyzer output"
                )

        analysis = load_analysis(
            "analysis",
            mode=analysis_mode,
            label="analysis",
        )
        audit_analysis = load_analysis(
            "auditAnalysis",
            mode=audit_analysis_mode,
            label="audit analysis",
        )
        if str(result_provenance.get("auditAnalysisHash") or "") != (
            audit_analysis.sha256
        ):
            raise ValueError("Research result audit-analysis membership mismatch")
        if self._uses_finalist_primary_policy(recipe):
            if context is None:
                raise ValueError(
                    "Research result finalist parity context is unavailable"
                )
            primary_document = _load_result_json_artifact_v1(
                self.artifact_store,
                dict(artifacts.get("primaryDecisionScoreAnalysis") or {}),
                media_type=_PRIMARY_DECISION_SCORE_ANALYSIS_MEDIA_TYPE,
            )
            if primary_document != audit_analysis.to_dict():
                raise ValueError(
                    "Research result primary decision-score analysis mismatch"
                )
            parity_document = _load_result_json_artifact_v1(
                self.artifact_store,
                dict(artifacts.get("decisionScoreParity") or {}),
                media_type=_DECISION_SCORE_PARITY_MEDIA_TYPE,
            )
            expected_parity = self._decision_score_parity_evidence(
                trace,
                audit_analysis,
                context=context,
            )
            if (
                parity_document != expected_parity
                or parity_document.get("matched") is not True
            ):
                raise ValueError(
                    "Research result decision-score parity evidence mismatch"
                )
            if (
                result_provenance.get("decisionScoreParityStatus") != "matched"
                or str(result_provenance.get("primaryDecisionScoreAnalysisHash") or "")
                != audit_analysis.sha256
                or str(result_provenance.get("decisionScoreParityHash") or "")
                != str(parity_document["sha256"])
            ):
                raise ValueError(
                    "Research result primary audit-analysis parity mismatch"
                )
        elif (
            audit_analysis.sha256 != analysis.sha256
            or "primaryDecisionScoreAnalysis" in artifacts
            or "decisionScoreParity" in artifacts
        ):
            raise ValueError(
                "Research non-finalist analysis authorities are inconsistent"
            )
        assert_canonical_analysis(
            analysis,
            mode=analysis_mode,
            label="analysis",
        )
        assert_canonical_analysis(
            audit_analysis,
            mode=audit_analysis_mode,
            label="audit analysis",
        )
        score_evidence_value = result.get("scoreEvidence")
        if not isinstance(score_evidence_value, Mapping):
            raise ValueError("Research result score evidence is unavailable")
        score_evidence = copy.deepcopy(dict(score_evidence_value))
        declared_score_hash = _sha256_digest_v1(
            score_evidence.pop("sha256", None),
            field="Research result score evidence hash",
        )
        if canonical_sha256(score_evidence) != declared_score_hash:
            raise ValueError("Research result score evidence checksum mismatch")
        selection_evidence = score_evidence.get("decisionSelectionQualification")
        if (
            not isinstance(selection_evidence, Mapping)
            or str(selection_evidence.get("analysisHash") or "") != analysis.sha256
        ):
            raise ValueError("Research result analysis membership mismatch")

        audit_reference = dict(artifacts.get("decisionAudit") or {})
        if set(audit_reference) != _RESULT_ARTIFACT_REFERENCE_KEYS:
            raise ValueError(
                "Research DecisionAuditV2 reference is not closed or owned"
            )
        if audit_reference.get("ownerKind") != "decision_audit_v2":
            raise ValueError("Research DecisionAuditV2 semantic owner kind mismatch")
        audit_owner_hash = _sha256_digest_v1(
            audit_reference.get("ownerHash"),
            field="Research DecisionAuditV2 semantic owner hash",
        )
        audit_document = load_decision_audit_v2(
            self.artifact_store,
            audit_reference,
            expected_trace_hash=trace.sha256,
            ledger=trace.decision_ledger,
            expected_engine_version=trace.engine_version,
            expected_bundle_hash=trace.bundle_hash,
            expected_rules_hash=trace.rules_hash,
        )
        if audit_owner_hash != audit_document["sha256"]:
            raise ValueError(
                "Research DecisionAuditV2 semantic owner identity mismatch"
            )
        validate_governed_decision_audit_v2(
            audit_document,
            trace=trace,
            rules=(
                context.simulator.rules if context is not None else self.simulator.rules
            ),
        )

        # The render adapter owns the immutable result identity contract.  Import
        # locally to avoid the package initialization cycle while reusing that SSOT.
        from .render_service import _validated_research_result_hash

        _validated_research_result_hash(
            result,
            result_code=code,
            trace_hash=trace.sha256,
            analysis_hash=analysis.sha256,
            audit_analysis_hash=audit_analysis.sha256,
            audit_hash=str(audit_document["sha256"]),
        )
        if result.get("decisionCount") != audit_document["decisionCount"]:
            raise ValueError("Research result decision-audit count mismatch")
        analysis_decision_ids = {item.decision_id for item in audit_analysis.decisions}
        decisions = [copy.deepcopy(dict(item)) for item in audit_document["decisions"]]
        for item in decisions:
            if str(item.get("decisionId") or "") in analysis_decision_ids:
                validate_selected_decision_audit_v2(item, analysis=audit_analysis)
        return decisions

    def get_decision(self, code: str, ordinal: int) -> dict[str, Any]:
        for item in self.get_decisions(code):
            if int(item["ordinal"]) == int(ordinal):
                return item
        raise KeyError(f"{code}:{ordinal}")

    def get_perspectives(self, code: str, observer_seat: str) -> list[dict[str, Any]]:
        """Load exactly one isolated observer timeline from retained analysis."""
        result = self.get_result(code)
        if result is None:
            raise KeyError(code)
        observer = Seat.parse(observer_seat).name.lower()
        if result.get("schemaVersion") == 2:
            acting_seats = list(
                dict(result.get("libraryProjection") or {}).get("actingSeats") or ()
            )
            if acting_seats != [observer]:
                raise ValueError(
                    "Research V4 perspectives are available only to the acting observer"
                )
            return []
        reference = dict(result["artifacts"]["analysis"])
        payload = json.loads(self.artifact_store.get(reference["uri"]).decode("utf-8"))
        laffa = dict(payload.get("laffaEvidence") or {})
        rows = [
            copy.deepcopy(dict(item))
            for item in laffa.get("perspectives") or ()
            if str(item.get("observerSeat") or "").lower() == observer
        ]
        if any(
            str(item.get("observerSeat") or "").lower() != observer for item in rows
        ):
            raise RuntimeError("observer perspective isolation failed")
        return sorted(
            rows,
            key=lambda item: (
                str(item.get("decisionId") or ""),
                str(item.get("actingSeat") or ""),
                str(item.get("truthMode") or ""),
            ),
        )

    def verify_play_compiler(self, code: str) -> dict[str, Any]:
        """Run the explicit fixed-trace CP-SAT parity proof on one result.

        This is intentionally operator-triggered. Ordinary research does not
        pay for a second full CP-SAT solve after every generated Laffa.
        """
        if self.play_compiler is None or self.project_compiler is None:
            raise RuntimeError(
                "the project and play compiler plugins must both be active"
            )
        result = self.get_result(code)
        if result is None:
            raise KeyError(code)
        recipe_value = self.store.get_recipe(str(result["recipeCode"]))
        if recipe_value is None:
            raise ValueError("research result recipe is unavailable")
        _recipe, context = self._build_recipe_and_context(recipe_value)
        simulator = context.simulator if context is not None else self.simulator
        compiled_domain = (
            context.compiled_domain if context is not None else self.compiled_domain
        )
        reference = dict(dict(result.get("artifacts") or {}).get("completeTrace") or {})
        if not reference.get("uri"):
            raise ValueError("research result lacks its complete trace artifact")
        trace_payload = json.loads(
            self.artifact_store.get(reference["uri"]).decode("utf-8")
        )
        trusted_state = trace_payload.get("trustedFinalState")
        if not isinstance(trusted_state, Mapping):
            raise ValueError("research trace lacks trusted final state evidence")
        project_plan = self.project_compiler.verify_trace(
            trusted_state, rules=simulator.rules
        )
        plan = self.play_compiler.verify_trace(
            trusted_state,
            rules=simulator.rules,
            derived_project_points=project_plan.project_points,
            derived_baloot_points=project_plan.baloot_points,
        )
        evidence = {
            "schemaVersion": 2,
            "resultCode": str(code),
            "resultHash": str(result["resultHash"]),
            "traceHash": str(trace_payload.get("sha256") or ""),
            "domainHash": (
                compiled_domain.sha256 if compiled_domain is not None else None
            ),
            "plugins": {
                "projects": self.project_compiler.descriptor.to_dict(),
                "play": self.play_compiler.descriptor.to_dict(),
            },
            "outcome": "matched",
            "projectPlan": project_plan.to_dict(),
            "playPlan": plan.to_dict(),
        }
        evidence["sha256"] = canonical_sha256(evidence)
        artifact = self.artifact_store.put(
            json.dumps(
                evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            media_type="application/vnd.balootx.cp-sat-round-parity-v2+json",
            name=f"{evidence['sha256']}.json",
        )
        return {
            "schemaVersion": 2,
            "resultCode": str(code),
            "outcome": "matched",
            "solverStatus": plan.solver_status,
            "projectSolverStatus": project_plan.solver_status,
            "projectPlanHash": project_plan.sha256,
            "projectPoints": list(project_plan.project_points),
            "balootPoints": list(project_plan.baloot_points),
            "projectRevealCount": len(project_plan.reveal_schedule),
            "playPlanHash": plan.sha256,
            "evidenceHash": evidence["sha256"],
            "evidence": artifact,
        }

    def list_disagreements(self, **filters: Any) -> list[dict[str, Any]]:
        rows = self.store.list_disagreements(**filters)
        classified: list[dict[str, Any]] = []
        for row in rows:
            effective, semantic = effective_disagreement_category(
                str(row.get("category") or ""), int(row.get("schemaVersion", 1))
            )
            classified.append(
                {
                    **row,
                    "effectiveCategory": effective,
                    "semanticDefect": semantic,
                }
            )
        return classified

    def statistics(self, *, run_code: str | None = None) -> dict[str, Any]:
        if run_code is not None and (type(run_code) is not str or not run_code.strip()):
            raise ValueError(
                "sequential search receipt run code must be a non-empty string"
            )
        if run_code is not None and self.store.get_run(run_code) is None:
            raise KeyError(run_code)
        aggregate = self.store.aggregate_stats(run_code=run_code)
        with self.store.iter_admitted_sequential_search_receipts(
            run_code=run_code,
            batch_size=1000,
        ) as receipt_rows:
            sequential_search = build_sequential_search_receipt_rollup(receipt_rows)
        performance_snapshot = self.store.performance_evidence_snapshot(
            run_code=run_code
        )
        runs = list(performance_snapshot["runs"])
        performance_shards = list(performance_snapshot["shards"])
        performance_run_codes = {
            str(item.get("runCode") or "") for item in performance_shards
        }
        canonical_runs: list[dict[str, Any]] = []
        for source_run in runs:
            if source_run is None:
                continue
            measured_run = dict(source_run)
            raw_receipt = measured_run.get("performanceReceipt")
            raw_partial_receipt = measured_run.get("partialPerformanceReceipt")
            if raw_receipt is not None and raw_partial_receipt is not None:
                raise ValueError(
                    "Research V4 run has completed and partial performance receipts"
                )
            if (
                isinstance(raw_receipt, Mapping)
                and raw_receipt.get("measurementScope")
                == "research_v4_completed_run_v1"
            ):
                receipt = validate_v4_performance_receipt(
                    raw_receipt,
                    run_code=str(measured_run["runCode"]),
                )
                if measured_run.get("status") != "completed":
                    raise ValueError(
                        "Research V4 completed-run performance receipt is non-terminal"
                    )
                if str(measured_run["runCode"]) in performance_run_codes:
                    raise ValueError(
                        "Research run has duplicate shard and completed-run performance evidence"
                    )
                performance_shards.append(
                    {
                        "runCode": str(measured_run["runCode"]),
                        "shardId": f"{measured_run['runCode']}:v4-completed-run",
                        "shardIndex": 0,
                        "status": "completed",
                        "attemptCount": 1,
                        "retryCount": 0,
                        "reclaimCount": 0,
                        "result": copy.deepcopy(receipt),
                    }
                )
                performance_run_codes.add(str(measured_run["runCode"]))
                measured_run.update(receipt)
                measured_run["elapsedSeconds"] = receipt["shardWallWorkSeconds"]
            elif raw_receipt is not None:
                if not isinstance(raw_receipt, Mapping):
                    raise ValueError("Research run performance receipt is invalid")
                try:
                    measured_run = _project_ortools_campaign_statistics_run(
                        measured_run, raw_receipt
                    )
                except ValueError as exc:
                    raise ValueError(f"{exc} [run={measured_run['runCode']}]") from exc
                if str(measured_run["runCode"]) in performance_run_codes:
                    raise ValueError(
                        "Research run has duplicate shard and campaign performance evidence"
                    )
                performance_shards.append(
                    {
                        "runCode": str(measured_run["runCode"]),
                        "shardId": f"{measured_run['runCode']}:ortools-campaign-run",
                        "shardIndex": 0,
                        "status": "completed",
                        "attemptCount": 1,
                        "retryCount": None,
                        "reclaimCount": None,
                        "result": dict(measured_run),
                    }
                )
                performance_run_codes.add(str(measured_run["runCode"]))
            elif raw_partial_receipt is not None:
                partial_receipt = validate_v4_partial_performance_receipt(
                    raw_partial_receipt,
                    run_code=str(measured_run["runCode"]),
                )
                expected_run_status = (
                    "cancelled"
                    if partial_receipt["terminalStatus"] == "cancelled"
                    else "failed"
                )
                if measured_run.get("status") != expected_run_status:
                    raise ValueError(
                        "Research V4 partial performance receipt is non-terminal"
                    )
                if str(measured_run["runCode"]) in performance_run_codes:
                    raise ValueError(
                        "Research run has duplicate shard and partial-run performance evidence"
                    )
                performance_shards.append(
                    {
                        "runCode": str(measured_run["runCode"]),
                        "shardId": f"{measured_run['runCode']}:v4-partial-run",
                        "shardIndex": 0,
                        "status": expected_run_status,
                        "attemptCount": 1,
                        "retryCount": 0,
                        "reclaimCount": 0,
                        "result": copy.deepcopy(partial_receipt),
                    }
                )
                performance_run_codes.add(str(measured_run["runCode"]))
            elif (
                measured_run.get("status") == "completed"
                and str(measured_run.get("runCode") or "") not in performance_run_codes
            ):
                performance_shards.append(
                    {
                        "runCode": str(measured_run["runCode"]),
                        "shardId": f"{measured_run['runCode']}:legacy-completed-run",
                        "shardIndex": 0,
                        "status": "completed",
                        "attemptCount": 1,
                        "retryCount": None,
                        "reclaimCount": None,
                        "result": dict(measured_run),
                    }
                )
                performance_run_codes.add(str(measured_run["runCode"]))
            canonical_runs.append(measured_run)
        runs = canonical_runs
        performance = build_research_performance_rollup(performance_shards)
        performance_by_lane = build_research_performance_by_lane(performance_shards)
        operational_keys = (
            "requested",
            "attemptedSeeds",
            "attemptedCandidates",
            "structuralCandidates",
            "auctionFeasible",
            "solverOptimal",
            "solverIncomplete",
            "runtimeParityPassed",
            "semanticReplayPassed",
            "proofCompleted",
            "playCompilerAttempted",
            "playCompilerOptimal",
            "playCompilerFeasible",
            "playCompilerIncomplete",
            "strategicSelectionDecisions",
            "strategicExactDecisions",
            "strategicProvisionalDecisions",
            "strategicFallbackDecisions",
            "qualifiedCount",
            "rejectedCount",
            "quarantineCount",
            "tacticalVerificationCandidates",
            "tacticalVerificationDecisions",
            "constructionPreScreenCandidates",
            "constructionPreScreenRejected",
            "tacticalViabilityPreScreenCandidates",
            "tacticalViabilityPreScreenRejected",
            "constructionTargetMissCount",
            "analyzedCandidateCount",
        )
        measured_runs = [
            item
            for item in runs
            if item is not None
            and "attemptedSeeds" in item
            and "elapsedSeconds" in item
        ]
        operational = {
            key: sum(int(item.get(key, 0)) for item in measured_runs)
            for key in operational_keys
        }
        stage_seconds = {
            key: sum(float(item.get(key) or 0.0) for item in measured_runs)
            for key in (
                "constructionSeconds",
                "simulationSeconds",
                "semanticParitySeconds",
                "analysisSeconds",
                "tacticalVerificationSeconds",
                "constructionPreScreenSeconds",
                "tacticalViabilityPreScreenSeconds",
                "candidatePipelineSeconds",
            )
        }
        elapsed = sum(float(item.get("elapsedSeconds", 0.0)) for item in measured_runs)
        cpu_measured_runs = [
            item for item in measured_runs if item.get("cpuSeconds") is not None
        ]
        cpu_seconds = sum(
            float(item.get("cpuSeconds", 0.0)) for item in cpu_measured_runs
        )
        cpu_boundaries = sorted(
            {
                str(item.get("cpuMeasurementBoundary"))
                for item in cpu_measured_runs
                if item.get("cpuMeasurementBoundary")
            }
        )
        cpu_complete = (
            bool(measured_runs)
            and len(cpu_measured_runs) == len(measured_runs)
            and len(cpu_boundaries) == 1
        )
        qualified = operational["qualifiedCount"]
        lane_totals: dict[str, dict[str, float | int]] = {}
        for item in measured_runs:
            lane = str(item.get("generationLane") or "ortools")
            current = lane_totals.setdefault(
                lane,
                {
                    "runCount": 0,
                    "attemptedSeeds": 0,
                    "attemptedCandidates": 0,
                    "generatedCandidates": 0,
                    "solverIncomplete": 0,
                    "semanticReplayPassed": 0,
                    "proofCompleted": 0,
                    "playCompilerAttempted": 0,
                    "playCompilerOptimal": 0,
                    "playCompilerFeasible": 0,
                    "playCompilerIncomplete": 0,
                    "strategicSelectionDecisions": 0,
                    "strategicExactDecisions": 0,
                    "strategicProvisionalDecisions": 0,
                    "strategicFallbackDecisions": 0,
                    "qualifiedCount": 0,
                    "rejectedCount": 0,
                    "quarantineCount": 0,
                    "tacticalVerificationCandidates": 0,
                    "tacticalVerificationDecisions": 0,
                    "tacticalVerificationSeconds": 0.0,
                    "constructionPreScreenCandidates": 0,
                    "constructionPreScreenRejected": 0,
                    "tacticalViabilityPreScreenCandidates": 0,
                    "tacticalViabilityPreScreenRejected": 0,
                    "constructionTargetMissCount": 0,
                    "analyzedCandidateCount": 0,
                    "constructionPreScreenSeconds": 0.0,
                    "tacticalViabilityPreScreenSeconds": 0.0,
                    "constructionSeconds": 0.0,
                    "simulationSeconds": 0.0,
                    "semanticParitySeconds": 0.0,
                    "analysisSeconds": 0.0,
                    "candidatePipelineSeconds": 0.0,
                    "elapsedWallSeconds": 0.0,
                    "cpuSeconds": 0.0,
                    "cpuMeasuredRunCount": 0,
                },
            )
            current["runCount"] = int(current["runCount"]) + 1
            current["attemptedSeeds"] = int(current["attemptedSeeds"]) + int(
                item.get("attemptedSeeds", 0)
            )
            current["attemptedCandidates"] = int(current["attemptedCandidates"]) + int(
                item.get("attemptedCandidates", 0)
            )
            current["generatedCandidates"] = int(current["generatedCandidates"]) + int(
                item.get("structuralCandidates", 0)
            )
            current["solverIncomplete"] = int(current["solverIncomplete"]) + int(
                item.get("solverIncomplete", 0)
            )
            current["semanticReplayPassed"] = int(
                current["semanticReplayPassed"]
            ) + int(item.get("semanticReplayPassed", 0))
            current["proofCompleted"] = int(current["proofCompleted"]) + int(
                item.get("proofCompleted", 0)
            )
            current["playCompilerAttempted"] = int(
                current["playCompilerAttempted"]
            ) + int(item.get("playCompilerAttempted", 0))
            current["playCompilerOptimal"] = int(current["playCompilerOptimal"]) + int(
                item.get("playCompilerOptimal", 0)
            )
            current["playCompilerFeasible"] = int(
                current["playCompilerFeasible"]
            ) + int(item.get("playCompilerFeasible", 0))
            current["playCompilerIncomplete"] = int(
                current["playCompilerIncomplete"]
            ) + int(item.get("playCompilerIncomplete", 0))
            current["strategicSelectionDecisions"] = int(
                current["strategicSelectionDecisions"]
            ) + int(item.get("strategicSelectionDecisions", 0))
            current["strategicExactDecisions"] = int(
                current["strategicExactDecisions"]
            ) + int(item.get("strategicExactDecisions", 0))
            current["strategicProvisionalDecisions"] = int(
                current["strategicProvisionalDecisions"]
            ) + int(item.get("strategicProvisionalDecisions", 0))
            current["strategicFallbackDecisions"] = int(
                current["strategicFallbackDecisions"]
            ) + int(item.get("strategicFallbackDecisions", 0))
            current["qualifiedCount"] = int(current["qualifiedCount"]) + int(
                item.get("qualifiedCount", 0)
            )
            current["rejectedCount"] = int(current["rejectedCount"]) + int(
                item.get("rejectedCount", 0)
            )
            current["quarantineCount"] = int(current["quarantineCount"]) + int(
                item.get("quarantineCount", 0)
            )
            current["tacticalVerificationCandidates"] = int(
                current["tacticalVerificationCandidates"]
            ) + int(item.get("tacticalVerificationCandidates", 0))
            current["tacticalVerificationDecisions"] = int(
                current["tacticalVerificationDecisions"]
            ) + int(item.get("tacticalVerificationDecisions", 0))
            current["tacticalVerificationSeconds"] = float(
                current["tacticalVerificationSeconds"]
            ) + float(item.get("tacticalVerificationSeconds") or 0.0)
            current["constructionPreScreenCandidates"] = int(
                current["constructionPreScreenCandidates"]
            ) + int(item.get("constructionPreScreenCandidates", 0))
            current["constructionPreScreenRejected"] = int(
                current["constructionPreScreenRejected"]
            ) + int(item.get("constructionPreScreenRejected", 0))
            current["tacticalViabilityPreScreenCandidates"] = int(
                current["tacticalViabilityPreScreenCandidates"]
            ) + int(item.get("tacticalViabilityPreScreenCandidates", 0))
            current["tacticalViabilityPreScreenRejected"] = int(
                current["tacticalViabilityPreScreenRejected"]
            ) + int(item.get("tacticalViabilityPreScreenRejected", 0))
            current["constructionTargetMissCount"] = int(
                current["constructionTargetMissCount"]
            ) + int(item.get("constructionTargetMissCount", 0))
            current["analyzedCandidateCount"] = int(
                current["analyzedCandidateCount"]
            ) + int(item.get("analyzedCandidateCount", 0))
            current["constructionPreScreenSeconds"] = float(
                current["constructionPreScreenSeconds"]
            ) + float(item.get("constructionPreScreenSeconds") or 0.0)
            current["tacticalViabilityPreScreenSeconds"] = float(
                current["tacticalViabilityPreScreenSeconds"]
            ) + float(item.get("tacticalViabilityPreScreenSeconds") or 0.0)
            current["constructionSeconds"] = float(
                current["constructionSeconds"]
            ) + float(item.get("constructionSeconds") or 0.0)
            current["simulationSeconds"] = float(current["simulationSeconds"]) + float(
                item.get("simulationSeconds") or 0.0
            )
            current["semanticParitySeconds"] = float(
                current["semanticParitySeconds"]
            ) + float(item.get("semanticParitySeconds") or 0.0)
            current["analysisSeconds"] = float(current["analysisSeconds"]) + float(
                item.get("analysisSeconds") or 0.0
            )
            current["candidatePipelineSeconds"] = float(
                current["candidatePipelineSeconds"]
            ) + float(item.get("candidatePipelineSeconds", 0.0) or 0.0)
            current["elapsedWallSeconds"] = float(
                current["elapsedWallSeconds"]
            ) + float(item.get("elapsedSeconds") or 0.0)
            if item.get("cpuSeconds") is not None:
                current["cpuSeconds"] = float(current["cpuSeconds"]) + float(
                    item["cpuSeconds"]
                )
                current["cpuMeasuredRunCount"] = int(current["cpuMeasuredRunCount"]) + 1
        by_generation_lane: dict[str, dict[str, float | int]] = {}
        for lane, current in sorted(lane_totals.items()):
            lane_elapsed = float(current["elapsedWallSeconds"])
            lane_attempts = int(current["attemptedSeeds"])
            lane_candidates = int(current["generatedCandidates"])
            lane_qualified = int(current["qualifiedCount"])
            lane_cpu = float(current["cpuSeconds"])
            lane_cpu_boundaries = {
                str(item.get("cpuMeasurementBoundary"))
                for item in measured_runs
                if str(item.get("generationLane") or "ortools") == lane
                and item.get("cpuSeconds") is not None
                and item.get("cpuMeasurementBoundary")
            }
            lane_cpu_complete = (
                int(current["runCount"]) > 0
                and int(current["cpuMeasuredRunCount"]) == int(current["runCount"])
                and len(lane_cpu_boundaries) == 1
            )
            by_generation_lane[lane] = {
                **current,
                "cpuSeconds": lane_cpu if lane_cpu_complete else None,
                "cpuMeasurementAvailability": (
                    "measured"
                    if lane_cpu_complete
                    else "mixed"
                    if int(current["cpuMeasuredRunCount"]) > 0
                    else "unavailable"
                ),
                "qualifiedPerWallHour": lane_qualified * 3600.0 / lane_elapsed
                if lane_elapsed > 0
                else 0.0,
                "qualifiedPerCpuHour": (
                    lane_qualified * 3600.0 / lane_cpu
                    if lane_cpu_complete and lane_cpu > 0
                    else None
                ),
                "qualifiedPer100Candidates": lane_qualified * 100.0 / lane_candidates
                if lane_candidates > 0
                else 0.0,
                "averageSecondsPerAttempt": lane_elapsed / lane_attempts
                if lane_attempts > 0
                else 0.0,
            }
        return {
            "schemaVersion": 1,
            "scope": "run" if run_code is not None else "all",
            "runCode": run_code,
            **aggregate,
            "funnel": operational,
            "measuredRunCount": len(measured_runs),
            "cpuMeasuredRunCount": len(cpu_measured_runs),
            "elapsedWallSeconds": elapsed,
            "cpuSeconds": cpu_seconds if cpu_complete else None,
            "cpuMeasurementBoundary": (
                cpu_boundaries[0]
                if cpu_complete
                else ("mixed" if cpu_boundaries else None)
            ),
            "cpuMeasurementNote": (
                "Sum of calling OS worker-thread CPU. Native work on those threads is "
                "included; work on any auxiliary threads is excluded."
                if cpu_complete
                else "Complete CPU values and one declared boundary are unavailable."
                if cpu_measured_runs
                else None
            ),
            "qualifiedPerWallHour": (qualified * 3600.0 / elapsed)
            if elapsed > 0
            else 0.0,
            "qualifiedPerCpuHour": (
                qualified * 3600.0 / cpu_seconds
                if cpu_complete and cpu_seconds > 0
                else None
            ),
            "qualifiedPer100Attempts": (
                qualified * 100.0 / operational["attemptedSeeds"]
                if operational["attemptedSeeds"] > 0
                else 0.0
            ),
            "stageSeconds": stage_seconds,
            "byGenerationLane": by_generation_lane,
            "performance": performance,
            "performanceByGenerationLane": performance_by_lane,
            "sequentialSearch": sequential_search,
        }

    def _cancel_check(self, run_code: str) -> bool:
        with self._lock:
            locally_cancelled = run_code in self._cancelled
        if locally_cancelled:
            return True
        status_lookup = getattr(self.store, "run_status", None)
        status = (
            status_lookup(run_code)
            if status_lookup is not None
            else (self.store.get_run(run_code) or {}).get("status")
        )
        return status is None or status == "cancelled"

    @staticmethod
    def _optimality_cut_event(
        *,
        candidate: Any,
        trace_hash: str,
        audit: Mapping[str, Any],
        maximum_regret: float,
    ) -> dict[str, Any]:
        violating = [
            str(item["decisionId"])
            for item in audit["decisions"]
            if not bool(item["forced"])
            and float(item["expectedRegret"]) > float(maximum_regret)
        ]
        if not violating:
            raise ValueError("an optimality cut requires at least one proven violation")
        return {
            "type": "optimality_cut",
            "cutType": "complete_deck_assignment_v1",
            "candidateHash": candidate.candidate_hash,
            "candidateOrdinal": int(candidate.ordinal),
            "precedingNoGoodCutCount": int(candidate.ordinal),
            "traceHash": str(trace_hash),
            "violatingDecisionIds": violating,
            "at": _now(),
        }

    def _compile_deal_spec(
        self,
        recipe: ResearchRecipe,
        *,
        seed: int,
        context: ResearchExecutionContextV1 | None = None,
    ) -> DealConstraintSpecV1:
        hard_projects = [
            item
            for item in recipe.situation["projects"]
            if item.get("mode", "hard") == "hard"
        ]
        project = None
        if hard_projects:
            item = hard_projects[0]
            project = {
                "seat": item["seat"],
                "contract": item["contract"],
                "kind": item["kind"],
                "value": self._project_engine_value(item, context=context),
                "highRank": item["highRank"],
            }
        explicit_perspectives = tuple(
            perspective
            for item in recipe.tactics
            for perspective in item["perspectives"]
        )
        objective_target = (
            recipe.situation.get("objectiveSeat")
            or (explicit_perspectives[0] if explicit_perspectives else None)
            or Seat(int(seed) % 4).name.lower()
        )
        assumed_buyer = recipe.situation.get("buyer")
        buyer_relation = str(
            recipe.situation.get("constructionBuyerRelation") or "automatic"
        )
        if assumed_buyer is None and (recipe.tactics or buyer_relation != "automatic"):
            # Tactical final-hand objectives need a purchase layout.  The
            # actor-safe auction remains authoritative; a different natural
            # buyer is recorded later as a target miss, not a discrepancy.
            target = Seat.parse(objective_target)
            partner_buyer_tactics = {
                "small_card_big_gain",
                "entry_unblocking",
                "partner_enablement",
            }
            use_partner = buyer_relation == "partner" or (
                buyer_relation == "automatic"
                and all(
                    item["kind"] in partner_buyer_tactics for item in recipe.tactics
                )
            )
            assumed_buyer = (
                target.partner.name.lower() if use_partner else target.name.lower()
            )
        tactical_deal_balance = None
        balance_raw = dict(recipe.resources.get("tacticalDealBalance") or {})
        if balance_raw:
            from .tactical_construction import TacticalDealBalanceProfileV1

            balance_profile = TacticalDealBalanceProfileV1.build(balance_raw)
            if balance_profile.enabled:
                tactical_deal_balance = {
                    "maximumTeamHighCardDifference": (
                        balance_profile.maximum_team_high_card_difference
                    ),
                    "minimumHighCardsPerSeat": (
                        balance_profile.minimum_high_cards_per_seat
                    ),
                    "maximumHighCardsPerSeat": (
                        balance_profile.maximum_high_cards_per_seat
                    ),
                    "minimumTargetLowCards": (balance_profile.minimum_target_low_cards),
                }
        requested_contracts = tuple(recipe.situation.get("contracts") or ())
        if recipe.tactics or tactical_deal_balance is not None:
            if len(requested_contracts) != 1:
                raise ValueError(
                    "contract-sensitive tactical construction requires exactly "
                    "one requested contract"
                )
            tactical_contract = requested_contracts[0]
            tactical_trump = recipe.situation.get("trumpSuit")
        else:
            tactical_contract = None
            tactical_trump = None
        payload = {
            "schemaVersion": 1,
            "enabled": True,
            "backend": "ortools_cp_sat_v1",
            "assumedBuyer": assumed_buyer,
            "purchaseSuits": recipe.situation.get("purchaseSuits") or [],
            "initialRequiredCards": {},
            "finalRequiredCards": recipe.situation.get("requiredCards") or {},
            "finalForbiddenCards": recipe.situation.get("forbiddenCards") or {},
            "initialSuitCountRanges": {},
            "finalSuitCountRanges": (
                recipe.situation.get("finalSuitCountRanges") or {}
            ),
            "projectPattern": project,
            "maximumSolutions": int(recipe.resources["candidatesPerSeed"]),
            "maximumConflicts": int(recipe.resources["maximumConflicts"]),
            "safetyTimeoutSeconds": float(recipe.resources["solverTimeoutSeconds"]),
            "solverWorkers": 1,
            "seedOffset": int(seed),
            "tags": [item["kind"] for item in recipe.tactics],
            "tacticalObjectiveWeights": {
                item["kind"]: max(1, int(round(float(item["weight"]) * 100)))
                for item in recipe.tactics
            },
            "objectiveTargetSeat": objective_target,
            "tacticalDealBalance": tactical_deal_balance,
            "tacticalContract": tactical_contract,
            "tacticalTrump": tactical_trump,
        }
        construction_identity = canonical_sha256(
            {
                "schemaVersion": 1,
                "kind": "research_deal_constraint_identity_v1",
                "payload": payload,
            }
        )
        payload.update(
            {
                "id": f"research_construction_{construction_identity[:16]}",
                "revision": construction_identity[:16],
            }
        )
        return DealConstraintSpecV1.build(payload)

    def _project_engine_value(
        self,
        requirement: Mapping[str, Any],
        *,
        context: ResearchExecutionContextV1 | None = None,
    ) -> int:
        simulator = context.simulator if context is not None else self.simulator
        kind = str(requirement["kind"])
        contract = Contract.parse(str(requirement["contract"])).effective.value
        if kind == "four_hundred":
            return int(
                simulator.rules.data["projects"]["four_of_kind"]["values_game_points"][
                    contract
                ]["ace"]
            )
        if kind == "hundred_four_kind" and int(requirement["value"]) in {100, 400}:
            return int(
                simulator.rules.data["projects"]["four_of_kind"]["values_game_points"][
                    contract
                ][str(requirement["highRank"])]
            )
        return int(requirement["value"])

    def _compile_auction(
        self,
        recipe: ResearchRecipe,
        deck_order: tuple[int, ...],
        *,
        required_actions: tuple[Mapping[str, Any], ...] | None = None,
        enforce_recipe: bool = True,
        context: ResearchExecutionContextV1 | None = None,
    ) -> AuctionPlanV1 | None:
        if self.auction_compiler is None:
            raise RuntimeError(
                "the active Research path requires the governed OR-Tools auction compiler"
            )
        simulator = context.simulator if context is not None else self.simulator
        dealer = Seat.parse(recipe.situation["dealer"])
        buyer = (
            Seat.parse(recipe.situation["buyer"])
            if enforce_recipe and recipe.situation.get("buyer") is not None
            else None
        )
        layout = DealLayoutV1.build(
            simulator.structural_domain, dealer=dealer, buyer=buyer
        )
        purchase_suit = Card(int(deck_order[layout.purchase_position])).suit
        plan = self.auction_compiler.compile(
            dealer=dealer,
            purchase_suit=purchase_suit,
            rules=simulator.rules,
            required_buyer=buyer,
            required_contracts=tuple(
                Contract.parse(item) for item in recipe.situation.get("contracts") or ()
            )
            if enforce_recipe
            else (),
            required_purchase_round=(
                recipe.situation.get("purchaseRound") if enforce_recipe else None
            ),
            required_trump=(
                Suit.parse(recipe.situation["trumpSuit"])
                if enforce_recipe and recipe.situation.get("trumpSuit") is not None
                else None
            ),
            required_actions=required_actions,
            maximum_conflicts=int(recipe.resources["maximumConflicts"]),
            timeout_seconds=float(recipe.resources["solverTimeoutSeconds"]),
        )
        return plan

    @staticmethod
    def _auction_actions_from_trace(trace: Any) -> tuple[Mapping[str, Any], ...]:
        history = tuple(trace.trusted_final_state.get("biddingHistory") or ())
        return tuple(
            {
                "ordinal": index,
                "roundIndex": int(item["round"]),
                "actor": str(item["seat"]),
                "action": str(item["action"]),
                "trumpSuit": item.get("trumpSuit"),
            }
            for index, item in enumerate(history, start=1)
        )

    def _compile_escalation(
        self,
        recipe: ResearchRecipe,
        auction: AuctionPlanV1,
        *,
        context: ResearchExecutionContextV1 | None = None,
    ) -> tuple[Mapping[str, Any], ...] | None:
        plan = self._compile_escalation_plan(recipe, auction, context=context)
        return None if plan is None else tuple(item.to_dict() for item in plan.actions)

    def _compile_escalation_plan(
        self,
        recipe: ResearchRecipe,
        auction: AuctionPlanV1,
        *,
        context: ResearchExecutionContextV1 | None = None,
    ) -> EscalationPlanV1 | None:
        if self.escalation_compiler is None:
            raise RuntimeError(
                "the active Research path requires the governed OR-Tools escalation compiler"
            )
        simulator = context.simulator if context is not None else self.simulator
        required = tuple(recipe.situation.get("escalations") or ())
        plan = self.escalation_compiler.compile(
            dealer=Seat.parse(recipe.situation["dealer"]),
            declarer=auction.declarer,
            contract=auction.contract,
            match_score=tuple(int(item) for item in recipe.situation["matchScore"]),
            rules=simulator.rules,
            required_level=(required[0] if required else "normal"),
            maximum_conflicts=int(recipe.resources["maximumConflicts"]),
            timeout_seconds=float(recipe.resources["solverTimeoutSeconds"]),
        )
        return plan

    def _compile_complete_play(
        self,
        recipe: ResearchRecipe,
        deck_order: tuple[int, ...],
        auction: AuctionPlanV1,
        escalation: EscalationPlanV1,
        *,
        required_cards: tuple[Card, ...] | None = None,
        required_ekka_leads: tuple[bool, ...] | None = None,
        objective_target_seat: Seat | None = None,
        context: ResearchExecutionContextV1 | None = None,
    ) -> tuple[ProjectPlanV1, PlayPlanV1, tuple[Mapping[str, Any], ...]]:
        if self.project_compiler is None or self.play_compiler is None:
            raise RuntimeError(
                "the active Research path requires project and complete-play compilers"
            )
        simulator = context.simulator if context is not None else self.simulator
        dealer = Seat.parse(recipe.situation["dealer"])
        layout = DealLayoutV1.build(
            simulator.structural_domain,
            dealer=dealer,
            buyer=auction.buyer,
        )
        hands = {
            seat: tuple(
                Card(int(deck_order[position]))
                for position in layout.positions_for(seat, final=True)
            )
            for seat in Seat
        }
        project_plan = self.project_compiler.compile(
            hands=hands,
            contract=auction.contract,
            dealer=dealer,
            trump=auction.trump,
            rules=simulator.rules,
            timeout_seconds=float(recipe.resources["solverTimeoutSeconds"]),
            maximum_conflicts=int(recipe.resources["maximumConflicts"]),
        )
        tactical_precursor = None
        tactical_profile_raw = dict(recipe.resources.get("tacticalConstruction") or {})
        if tactical_profile_raw and required_cards is None:
            from .tactical_construction import TacticalPlayConstructionProfileV1

            tactical_profile = TacticalPlayConstructionProfileV1.build(
                tactical_profile_raw
            )
            if tactical_profile.enabled:
                gated_tactics = [
                    str(item["kind"])
                    for item in recipe.tactics
                    if item["mode"] in {"hard", "qualification_gate"}
                ]
                supported = [
                    kind for kind in gated_tactics if kind == "small_card_big_gain"
                ]
                if tactical_profile.require_precursor_for_gated_tactics and set(
                    gated_tactics
                ) != set(supported):
                    raise ValueError(
                        "the active tactical play constructor cannot compile every gated tactic"
                    )
                if supported:
                    if objective_target_seat is None:
                        raise ValueError(
                            "tactical play construction requires an objective seat"
                        )
                    tactical_precursor = {
                        "kind": "small_card_big_gain",
                        "targetSeat": objective_target_seat.name.lower(),
                        "maximumPayoffTricks": (tactical_profile.maximum_payoff_tricks),
                        "minimumLaterRawPoints": (
                            tactical_profile.minimum_later_raw_points
                        ),
                    }
        play_plan = self.play_compiler.compile(
            hands=hands,
            contract=auction.contract,
            trump=auction.trump,
            dealer=dealer,
            buyer=auction.buyer,
            declarer=auction.declarer,
            rules=simulator.rules,
            escalation=escalation.level,
            counting_team=resolve_counting_team(
                declarer=auction.declarer,
                escalation=escalation.level,
                last_escalating_team=(
                    escalation.actions[-1].actor.team if escalation.actions else None
                ),
            ),
            project_points=project_plan.project_points,
            baloot_points=project_plan.baloot_points,
            locked=escalation.locked,
            required_cards=required_cards,
            required_ekka_leads=required_ekka_leads,
            tactical_precursor=tactical_precursor,
            timeout_seconds=float(recipe.resources["solverTimeoutSeconds"]),
            maximum_conflicts=int(recipe.resources["maximumConflicts"]),
        )
        second_trick_order: list[Seat] = []
        actor = play_plan.tricks[1].leader
        for _position in range(4):
            second_trick_order.append(actor)
            actor = simulator.structural_domain.offset(actor, 1)
        project_plan = self.project_compiler.with_reveal_schedule(
            project_plan,
            second_trick_order,
            domain=simulator.rules.project_domain,
            trump_suit=auction.trump,
        )
        actions: list[Mapping[str, Any]] = []
        for trick in play_plan.tricks:
            actor = trick.leader
            for position, card in enumerate(trick.cards):
                actions.append(
                    {
                        "actor": actor.name.lower(),
                        "card": card.to_dict(),
                        "ekka": bool(trick.ekka_lead and position == 0),
                    }
                )
                actor = simulator.structural_domain.offset(actor, 1)
        if len(actions) != 32:
            raise AssertionError("complete-play compiler did not emit 32 actions")
        return project_plan, play_plan, tuple(actions)

    @staticmethod
    def _played_line(simulation: Any) -> tuple[tuple[Card, ...], tuple[bool, ...]]:
        completed = list(
            simulation.final_trace.trusted_final_state.get("completedTricks") or ()
        )
        if len(completed) != 8:
            raise ValueError("a complete actor-safe line requires eight tricks")
        cards: list[Card] = []
        ekka_leads: list[bool] = []
        for trick in completed:
            plays = list(dict(trick).get("plays") or ())
            if len(plays) != 4:
                raise ValueError("a complete actor-safe trick requires four plays")
            for position, play in enumerate(plays):
                value = dict(play).get("ekka", False)
                if type(value) is not bool:
                    raise ValueError("actor-safe Ekka evidence must be boolean")
                if position > 0 and value:
                    raise ValueError("actor-safe Ekka must be a trick lead")
            ekka_leads.append(bool(dict(plays[0]).get("ekka", False)))
            cards.extend(Card.parse(dict(play)["card"]) for play in plays)
        if len(cards) != 32:
            raise AssertionError("a complete actor-safe line must contain 32 cards")
        return tuple(cards), tuple(ekka_leads)

    @staticmethod
    def _compiled_project_plan_matches(
        project_plan: ProjectPlanV1,
        state: Mapping[str, Any],
        completed: list[Mapping[str, Any]],
        *,
        project_domain: ProjectDomainIRV1,
    ) -> bool:
        """Verify CP-SAT projects against this game's runtime and reference views."""

        try:
            plan_payload = project_plan.to_dict()
            plan_hash = str(plan_payload.pop("sha256"))
            if canonical_sha256(plan_payload) != plan_hash:
                return False

            hands = {
                Seat.parse(seat): tuple(Card.parse(card) for card in cards)
                for seat, cards in dict(state.get("initialHands") or {}).items()
            }
            if (
                set(hands) != set(Seat)
                or any(len(cards) != 8 for cards in hands.values())
                or sorted(card.id for cards in hands.values() for card in cards)
                != list(range(32))
            ):
                return False
            contract = Contract.parse(state["contract"])
            dealer = Seat.parse(state["dealer"])
            trump = (
                Suit.parse(state["trumpSuit"])
                if state.get("trumpSuit") is not None
                else None
            )
            reference = ReferenceProjectInterpreterV1(project_domain)
            reference_projects: list[CompiledProjectV1] = []
            for seat in Seat:
                selected = reference.select(
                    reference.candidates(seat, hands[seat], contract)
                )
                reference_projects.extend(
                    CompiledProjectV1(
                        project.owner,
                        project.kind,
                        project.cards,
                        project.value,
                        project.high_rank,
                        project.category_priority,
                    )
                    for project in selected
                )
            reference_projects, reference_winner = project_domain.resolve_declarations(
                reference_projects,
                dealer=dealer,
                trump_suit=trump,
                second_trick_leader=(Seat.parse(completed[0]["winner"]) if completed else None),
            )
            declaration_seats = list(project_domain.declaration_seats(dealer))
            reference_projects.sort(
                key=lambda project: (
                    declaration_seats.index(project.owner),
                    -project.value,
                    -project.category_priority,
                    -project.high_rank,
                )
            )
            reference_project_points = project_domain.project_points(reference_projects)
            reference_baloot_teams = reference.complete_baloot_teams(
                hands,
                reference_projects,
                contract,
                trump,
            )
            reference_baloot_points = [0, 0]
            for team in reference_baloot_teams:
                reference_baloot_points[int(team)] += project_domain.baloot_value

            if len(completed) != 8:
                return False
            second_plays = list(dict(completed[1]).get("plays") or ())
            if len(second_plays) != 4:
                return False
            reference_schedule: list[tuple[str, int, int, tuple[str, ...]]] = []
            for position, play in enumerate(second_plays):
                owner = Seat.parse(dict(play)["seat"])
                reference_projects, revealed = project_domain.reveal_for_turn(
                    reference_projects,
                    owner,
                    trump_suit=trump,
                    second_trick_leader=Seat.parse(completed[0]["winner"]),
                )
                if revealed:
                    reference_schedule.append(
                        (
                            owner.name.lower(),
                            1,
                            position,
                            tuple(project.kind for project in revealed),
                        )
                    )

            def compiled_row(project: Any) -> tuple[Any, ...]:
                return (
                    project.owner.name.lower(),
                    project.team.name.lower(),
                    str(project.kind),
                    tuple(card.id for card in project.cards),
                    int(project.value),
                    int(project.high_rank),
                    int(project.category_priority),
                    bool(project.declared),
                    int(project.declaration_order),
                    bool(project.scoring),
                    bool(project.revealed),
                )

            if sorted(map(compiled_row, project_plan.projects)) != sorted(
                map(compiled_row, reference_projects)
            ):
                return False
            if (
                project_plan.winning_team != reference_winner
                or project_plan.project_points != reference_project_points
                or project_plan.baloot_teams != reference_baloot_teams
                or project_plan.baloot_points != tuple(reference_baloot_points)
                or tuple(
                    (
                        item.owner.name.lower(),
                        item.trick_index,
                        item.play_position,
                        item.project_kinds,
                    )
                    for item in project_plan.reveal_schedule
                )
                != tuple(reference_schedule)
            ):
                return False

            def runtime_row(project: Mapping[str, Any]) -> tuple[Any, ...]:
                owner = Seat.parse(project["owner"])
                declaration_order = int(project.get("declarationOrder", 999))
                declared = bool(project.get("declared", declaration_order != 999))
                raw_cards = project.get("cards")
                cards = (
                    None
                    if raw_cards is None
                    else tuple(Card.parse(card).id for card in raw_cards)
                )
                return (
                    owner.name.lower(),
                    str(project["team"]),
                    str(project["kind"]),
                    cards,
                    int(project["value"]),
                    int(project["highRank"]),
                    declared,
                    declaration_order,
                    bool(project["scoring"]),
                    bool(project["revealed"]),
                )

            def compiled_runtime_row(project: Any) -> tuple[Any, ...]:
                return (
                    project.owner.name.lower(),
                    project.team.name.lower(),
                    str(project.kind),
                    (
                        tuple(card.id for card in project.cards)
                        if project.revealed
                        else None
                    ),
                    int(project.value),
                    int(project.high_rank),
                    bool(project.declared),
                    int(project.declaration_order),
                    bool(project.scoring),
                    bool(project.revealed),
                )

            runtime_projects = [
                dict(project) for project in state.get("projects") or ()
            ]
            if sorted(map(runtime_row, runtime_projects)) != sorted(
                map(compiled_runtime_row, project_plan.projects)
            ):
                return False
            for runtime_project in runtime_projects:
                if "categoryPriority" not in runtime_project:
                    continue
                identity = runtime_row(runtime_project)
                matching = [
                    project
                    for project in project_plan.projects
                    if compiled_runtime_row(project) == identity
                ]
                if (
                    len(matching) != 1
                    or int(runtime_project["categoryPriority"])
                    != matching[0].category_priority
                ):
                    return False

            expected_winner = (
                project_plan.winning_team.name.lower()
                if project_plan.winning_team is not None
                else None
            )
            score = dict(state.get("roundScore") or {})
            if (
                state.get("projectWinningTeam") != expected_winner
                or list(score.get("projectPoints") or ())
                != list(project_plan.project_points)
                or list(score.get("balootPoints") or ())
                != list(project_plan.baloot_points)
                or set(state.get("balootTeams") or ())
                != {team.name.lower() for team in project_plan.baloot_teams}
            ):
                return False

            if "events" not in state:
                return True
            events = [dict(event) for event in state.get("events") or ()]
            holdings = [
                event
                for event in events
                if event.get("type") == "project_holdings_detected"
            ]
            if len(holdings) != 1 or int(holdings[0].get("count", -1)) != len(
                project_plan.projects
            ):
                return False
            declared_events = [
                event for event in events if event.get("type") == "project_declared"
            ]
            expected_declarations = [
                {
                    "owner": project.owner.name.lower(),
                    "kind": project.kind,
                    "value": project.value,
                    "highRank": project.high_rank,
                    "declarationOrder": project.declaration_order,
                }
                for project in sorted(
                    (project for project in project_plan.projects if project.declared),
                    key=lambda project: project.declaration_order,
                )
            ]
            actual_declarations = [
                {
                    key: event.get(key)
                    for key in (
                        "owner",
                        "kind",
                        "value",
                        "highRank",
                        "declarationOrder",
                    )
                }
                for event in declared_events
            ]
            if actual_declarations != expected_declarations:
                return False

            compared_events = [
                event for event in events if event.get("type") == "projects_compared"
            ]
            if project_plan.projects:
                expected_compared_declarations = [
                    {
                        "owner": project.owner.name.lower(),
                        "kind": project.kind,
                        "value": project.value,
                        "highRank": project.high_rank,
                        "declarationOrder": project.declaration_order,
                    }
                    for project in project_plan.projects
                    if project.declared
                ]
                if (
                    len(compared_events) != 1
                    or compared_events[0].get("winningTeam") != expected_winner
                    or compared_events[0].get("declarations")
                    != expected_compared_declarations
                ):
                    return False
            elif compared_events:
                return False

            reveal_events = [
                event for event in events if event.get("type") == "projects_revealed"
            ]
            actual_schedule: list[tuple[str, int, int, tuple[str, ...]]] = []
            second_order = [str(dict(play).get("seat")) for play in second_plays]
            for event in reveal_events:
                owner = str(event.get("owner"))
                if owner not in second_order:
                    return False
                event_projects = [
                    dict(project) for project in event.get("projects") or ()
                ]
                expected_projects = [
                    project
                    for project in project_plan.projects
                    if project.owner.name.lower() == owner and project.revealed
                ]
                if sorted(map(runtime_row, event_projects)) != sorted(
                    map(compiled_runtime_row, expected_projects)
                ):
                    return False
                actual_schedule.append(
                    (
                        owner,
                        1,
                        second_order.index(owner),
                        tuple(str(project["kind"]) for project in event_projects),
                    )
                )
            if tuple(actual_schedule) != tuple(reference_schedule):
                return False

            collected = [
                event for event in events if event.get("type") == "projects_collected"
            ]
            if project_plan.projects:
                if len(collected) != 1 or collected[0].get("owners") != sorted(
                    {item[0] for item in reference_schedule}
                ):
                    return False
            elif collected:
                return False

            baloot_events = [
                event
                for event in events
                if event.get("type")
                in {"baloot_declared", "baloot_included_in_project"}
            ]
            baloot_event_teams = [str(event.get("team")) for event in baloot_events]
            if len(baloot_event_teams) != len(set(baloot_event_teams)) or set(
                baloot_event_teams
            ) != {team.name.lower() for team in project_plan.baloot_teams}:
                return False
            return True
        except (IndexError, KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _compiled_play_mismatches(
        play_plan: PlayPlanV1,
        simulation: Any,
        *,
        project_plan: ProjectPlanV1 | None = None,
        project_domain: ProjectDomainIRV1 | None = None,
        scoring_domain: ScoringDomainIRV1 | None = None,
        structural_domain: Any | None = None,
        expected_deck_order: tuple[int, ...] | None = None,
        expected_deck_commitment: str | None = None,
        expected_candidate_hash: str | None = None,
        expected_dealer: Seat | str | None = None,
        expected_round_number: int | None = None,
        expected_next_dealer: Seat | str | None = None,
        starting_match_score: tuple[int, int] | None = None,
    ) -> tuple[str, ...]:
        """Compare one compiled game with its own authoritative replay.

        Cross-lane game identity and CP-SAT structural/decision objectives are
        intentionally outside this gate. Only the immutable deal, project,
        play, round, and match semantics of this exact frozen game may affect
        semantic quarantine.
        """
        trace = simulation.final_trace
        state = trace.trusted_final_state
        if not isinstance(state, Mapping):
            return ("compiler_parity_defect",)
        completed = [dict(trick) for trick in state.get("completedTricks") or ()]
        plan_payload = play_plan.to_dict()
        declared_plan_hash = str(plan_payload.pop("sha256", ""))
        if (
            len(play_plan.tricks) != 8
            or sum(len(trick.cards) for trick in play_plan.tricks) != 32
            or sorted(card.id for trick in play_plan.tricks for card in trick.cards)
            != list(range(32))
            or canonical_sha256(plan_payload) != declared_plan_hash
            or len(completed) != 8
        ):
            return ("compiler_parity_defect",)
        if (
            str(state.get("phase")) != GamePhase.SCORED.value
            or state.get("currentPlayer") is not None
            or state.get("currentTrick") is not None
        ):
            return ("compiler_parity_defect",)
        generation = dict(state.get("generation") or {})
        frozen_round_number = (
            int(expected_round_number)
            if expected_round_number is not None
            else (
                int(generation["roundNumber"])
                if generation.get("roundNumber") is not None
                else (
                    int(state["roundNumber"])
                    if state.get("roundNumber") is not None
                    else None
                )
            )
        )
        if frozen_round_number is not None and (
            int(state.get("roundNumber", -1)) != frozen_round_number
            or (
                generation.get("roundNumber") is not None
                and int(generation["roundNumber"]) != frozen_round_number
            )
        ):
            return ("compiler_parity_defect",)
        frozen_dealer = (
            expected_dealer if expected_dealer is not None else generation.get("dealer")
        )
        if frozen_dealer is None:
            return ("compiler_parity_defect",)
        normalized_dealer = Seat.parse(frozen_dealer).name.lower()
        if str(state.get("dealer")) != normalized_dealer or (
            generation.get("dealer") is not None
            and Seat.parse(generation["dealer"]).name.lower() != normalized_dealer
        ):
            return ("compiler_parity_defect",)
        initial_hands: dict[Seat, tuple[Card, ...]] | None = None
        if state.get("initialHands") is not None:
            try:
                initial_hands = {
                    Seat.parse(seat): tuple(Card.parse(card) for card in cards)
                    for seat, cards in dict(state["initialHands"]).items()
                }
            except (KeyError, TypeError, ValueError):
                return ("compiler_parity_defect",)
            if (
                set(initial_hands) != set(Seat)
                or any(len(cards) != 8 for cards in initial_hands.values())
                or sorted(card.id for cards in initial_hands.values() for card in cards)
                != list(range(32))
            ):
                return ("compiler_parity_defect",)

        if expected_deck_order is not None:
            try:
                deck_order = tuple(int(card) for card in expected_deck_order)
                if len(deck_order) != 32 or sorted(deck_order) != list(range(32)):
                    return ("compiler_parity_defect",)
                commitment = hashlib.sha256(bytes(deck_order)).hexdigest()
                if (
                    expected_deck_commitment is not None
                    and str(expected_deck_commitment) != commitment
                ):
                    return ("compiler_parity_defect",)
                shuffle = dict(state.get("shuffle") or {})
                deal_generator = dict(generation.get("dealGenerator") or {})
                if (
                    structural_domain is None
                    or tuple(int(card) for card in shuffle.get("deckOrder") or ())
                    != deck_order
                    or str(shuffle.get("deckCommitment") or "") != commitment
                    or str(shuffle.get("algorithm") or "") != "governed_preordered_v1"
                    or str(deal_generator.get("deckCommitment") or "") != commitment
                    or (
                        expected_candidate_hash is not None
                        and str(deal_generator.get("candidateHash") or "")
                        != str(expected_candidate_hash)
                    )
                ):
                    return ("compiler_parity_defect",)
                layout = DealLayoutV1.build(
                    structural_domain,
                    dealer=Seat.parse(normalized_dealer),
                    buyer=Seat.parse(state["buyer"]),
                )
                expected_hands = {
                    seat: tuple(
                        Card(deck_order[position])
                        for position in layout.positions_for(seat, final=True)
                    )
                    for seat in Seat
                }
                if initial_hands is None or any(
                    sorted(card.id for card in initial_hands[seat])
                    != sorted(card.id for card in expected_hands[seat])
                    for seat in Seat
                ):
                    return ("compiler_parity_defect",)
            except (KeyError, TypeError, ValueError):
                return ("compiler_parity_defect",)

        remaining = (
            {seat: {card.id for card in cards} for seat, cards in initial_hands.items()}
            if initial_hands is not None
            else None
        )
        baloot_play_rows: list[tuple[str, int]] = []
        for expected, actual in zip(play_plan.tricks, completed, strict=True):
            plays = list(actual.get("plays") or ())
            if len(plays) != 4:
                return ("compiler_parity_defect",)
            cards: list[int] = []
            actor = expected.leader
            for position, raw_play in enumerate(plays):
                if not isinstance(raw_play, Mapping):
                    return ("compiler_parity_defect",)
                play = dict(raw_play)
                try:
                    card = Card.parse(play["card"])
                except (KeyError, TypeError, ValueError):
                    return ("compiler_parity_defect",)
                strict_play_record = initial_hands is not None
                try:
                    seat = (
                        Seat.parse(play["seat"])
                        if play.get("seat") is not None
                        else actor
                    )
                except (TypeError, ValueError):
                    return ("compiler_parity_defect",)
                if (
                    (strict_play_record or "seat" in play)
                    and seat != actor
                    or (
                        (strict_play_record or "trickIndex" in play)
                        and int(play.get("trickIndex", -1)) != expected.index
                    )
                    or (
                        (strict_play_record or "position" in play)
                        and int(play.get("position", -1)) != position
                    )
                    or (
                        "ekka" in play
                        and (
                            type(play["ekka"]) is not bool
                            or bool(play["ekka"])
                            != bool(expected.ekka_lead and position == 0)
                        )
                    )
                    or ("baloot" in play and type(play["baloot"]) is not bool)
                ):
                    return ("compiler_parity_defect",)
                if bool(play.get("baloot", False)):
                    baloot_play_rows.append((seat.name.lower(), card.id))
                if remaining is not None:
                    legal_cards = play.get("legalCards")
                    if not isinstance(legal_cards, list):
                        return ("compiler_parity_defect",)
                    try:
                        legal_ids = [Card.parse(item).id for item in legal_cards]
                    except (TypeError, ValueError):
                        return ("compiler_parity_defect",)
                    if (
                        len(legal_ids) != len(set(legal_ids))
                        or card.id not in legal_ids
                        or not set(legal_ids).issubset(remaining[seat])
                        or card.id not in remaining[seat]
                    ):
                        return ("compiler_parity_defect",)
                    remaining[seat].remove(card.id)
                cards.append(card.id)
                actor = (
                    structural_domain.offset(actor, 1)
                    if structural_domain is not None
                    else actor.next()
                )
            if (
                int(actual.get("index", -1)) != expected.index
                or str(actual.get("leader")) != expected.leader.name.lower()
                or tuple(cards) != tuple(card.id for card in expected.cards)
                or str(actual.get("winner")) != expected.winner.name.lower()
                or int(actual.get("rawPoints", -1)) != expected.raw_points
            ):
                return ("compiler_parity_defect",)
        if remaining is not None and any(remaining.values()):
            return ("compiler_parity_defect",)
        if "hands" in state and any(state.get("hands", {}).values()):
            return ("compiler_parity_defect",)
        if project_plan is not None:
            if project_domain is None:
                return ("compiler_parity_defect",)
            if not ResearchFoundryServiceV1._compiled_project_plan_matches(
                project_plan,
                state,
                completed,
                project_domain=project_domain,
            ):
                return ("compiler_parity_defect",)
            scripted = dict(generation.get("scriptedPlayCompiler") or {})
            if (
                scripted.get("projectPlanHash") is not None
                and str(scripted["projectPlanHash"]) != project_plan.sha256
            ):
                return ("compiler_parity_defect",)
        score = dict(state.get("roundScore") or {})
        expected_score = play_plan.to_dict()["score"]
        for key in (
            "rawCardPoints",
            "cardGamePoints",
            "projectPoints",
            "balootPoints",
            "totalBeforeAward",
            "awardedPoints",
        ):
            if list(score.get(key) or ()) != list(expected_score[key]):
                return ("compiler_parity_defect",)
        scalar_fields = (
            "winner",
            "countingTeam",
            "multiplier",
            "kabout",
            "gahwa",
        )
        if any(score.get(key) != expected_score[key] for key in scalar_fields):
            return ("compiler_parity_defect",)

        if scoring_domain is None:
            rules_payload = dict(
                dict(
                    dict(state.get("configurationSnapshot") or {}).get("components")
                    or {}
                ).get("rules")
                or {}
            ).get("payload")
            if not isinstance(rules_payload, Mapping):
                return ("compiler_parity_defect",)
            try:
                scoring_domain = ScoringDomainIRV1.from_rules(rules_payload)
            except (KeyError, TypeError, ValueError):
                return ("compiler_parity_defect",)
        frozen_match_score = (
            starting_match_score
            if starting_match_score is not None
            else generation.get("matchScore")
        )
        if frozen_match_score is None:
            return ("compiler_parity_defect",)
        try:
            match_start = tuple(int(value) for value in frozen_match_score)
            if len(match_start) != 2 or any(value < 0 for value in match_start):
                return ("compiler_parity_defect",)
            if (
                generation.get("matchScore") is not None
                and tuple(int(value) for value in generation["matchScore"])
                != match_start
            ):
                return ("compiler_parity_defect",)
            scoring_probe = OrToolsCpSatPlayCompilerV1.scoring_probe(
                domain=scoring_domain,
                contract=Contract.parse(state["contract"]),
                declarer=Seat.parse(state["declarer"]),
                escalation=str(state.get("escalation") or "normal"),
                trick_raw_points=tuple(trick.raw_points for trick in play_plan.tricks),
                trick_winner_teams=tuple(
                    trick.winner.team for trick in play_plan.tricks
                ),
                project_points=play_plan.project_points,
                baloot_points=play_plan.baloot_points,
                counting_team=play_plan.counting_team,
                starting_match_score=match_start,
            )
        except (KeyError, TypeError, ValueError):
            return ("compiler_parity_defect",)
        probe_round_fields = (
            "rawCardPoints",
            "cardGamePoints",
            "projectPoints",
            "balootPoints",
            "totalBeforeAward",
            "winner",
            "awardedPoints",
            "countingTeam",
            "multiplier",
            "kabout",
            "gahwa",
        )
        if any(scoring_probe[key] != expected_score[key] for key in probe_round_fields):
            return ("compiler_parity_defect",)
        try:
            reference = ReferenceScoringInterpreterV1(scoring_domain)
            reference_round = reference.score(
                contract=Contract.parse(state["contract"]),
                declarer=Seat.parse(state["declarer"]),
                escalation=str(state.get("escalation") or "normal"),
                trick_raw_points=tuple(trick.raw_points for trick in play_plan.tricks),
                trick_winners=tuple(trick.winner for trick in play_plan.tricks),
                project_points=play_plan.project_points,
                baloot_points=play_plan.baloot_points,
                counting_team=play_plan.counting_team,
            )
            reference_round_score = reference_round.to_dict()
            reference_match = reference.apply_match_award(
                starting_score=match_start,
                awarded_points=reference_round.awarded_points,
                round_winner=reference_round.winner,
                gahwa=reference_round.gahwa,
            )
        except (KeyError, TypeError, ValueError):
            return ("compiler_parity_defect",)
        reference_match_winner = (
            reference_match.winner.name.lower()
            if reference_match.winner is not None
            else None
        )
        if (
            any(
                reference_round_score[key] != scoring_probe[key]
                for key in probe_round_fields
            )
            or list(reference_match.final_score) != scoring_probe["finalMatchScore"]
            or reference_match_winner != scoring_probe["matchWinner"]
        ):
            return ("compiler_parity_defect",)
        if (
            list(state.get("matchScore") or ()) != scoring_probe["finalMatchScore"]
            or state.get("matchWinner") != scoring_probe["matchWinner"]
            or score.get("matchWinner") != scoring_probe["matchWinner"]
        ):
            return ("compiler_parity_defect",)

        expected_explanation = [
            f"counting_team={score.get('countingTeam')}",
            f"card_game_points={list(score.get('cardGamePoints') or ())}",
            f"projects={list(score.get('projectPoints') or ())}",
            f"baloot={list(score.get('balootPoints') or ())}",
        ]
        if (
            score.get("explanation") is not None
            and score.get("explanation") != expected_explanation
        ):
            return ("compiler_parity_defect",)

        if "events" in state:
            raw_events = state.get("events")
            if not isinstance(raw_events, list) or not raw_events:
                return ("compiler_parity_defect",)
            events = [dict(event) for event in raw_events]
            if any(
                int(event.get("sequence", -1)) != sequence
                for sequence, event in enumerate(events)
            ):
                return ("compiler_parity_defect",)
            created = [
                event for event in events if event.get("type") == "round_created"
            ]
            scored = [event for event in events if event.get("type") == "round_scored"]
            if (
                len(created) != 1
                or created[0] is not events[0]
                or created[0].get("dealer") != normalized_dealer
                or created[0].get("matchScore") != list(match_start)
                or int(created[0].get("roundNumber", -1))
                != int(state.get("roundNumber", -1))
                or int(created[0].get("seed", -1)) != int(state.get("seed", -1))
                or len(scored) != 1
                or scored[0] is not events[-1]
                or scored[0].get("score") != score
                or scored[0].get("matchScore") != scoring_probe["finalMatchScore"]
                or scored[0].get("matchWinner") != scoring_probe["matchWinner"]
            ):
                return ("compiler_parity_defect",)
            baloot_declared = [
                event for event in events if event.get("type") == "baloot_declared"
            ]
            expected_baloot_declared = [
                {
                    "seat": seat,
                    "team": Seat.parse(seat).team.name.lower(),
                }
                for seat, _card in baloot_play_rows
            ]
            actual_baloot_declared = [
                {"seat": event.get("seat"), "team": event.get("team")}
                for event in baloot_declared
            ]
            if actual_baloot_declared != expected_baloot_declared:
                return ("compiler_parity_defect",)

            if expected_next_dealer is not None:
                normalized_next_dealer = Seat.parse(expected_next_dealer).name.lower()
                lifecycle_next_dealers = [
                    str(value)
                    for value in (
                        state.get("nextDealer"),
                        scored[0].get("nextDealer"),
                        dict(state.get("terminalResult") or {}).get("nextDealer"),
                    )
                    if value is not None
                ]
                if lifecycle_next_dealers and any(
                    value != normalized_next_dealer for value in lifecycle_next_dealers
                ):
                    return ("compiler_parity_defect",)

        if all(
            hasattr(trace, field)
            for field in (
                "state_hash",
                "game_id",
                "game_hash",
                "rules_hash",
                "bundle_hash",
                "engine_version",
                "decision_ledger",
            )
        ):
            if (
                str(trace.state_hash) != canonical_sha256(state)
                or str(trace.game_id) != str(state.get("gameId") or "")
                or str(trace.rules_hash) != str(state.get("rulesHash") or "")
                or str(trace.bundle_hash) != str(state.get("bundleHash") or "")
                or str(trace.engine_version) != str(state.get("engineVersion") or "")
                or str(trace.decision_ledger.game_id) != str(trace.game_hash)
            ):
                return ("compiler_parity_defect",)
            if isinstance(trace, CompleteGameTraceV1):
                try:
                    CompleteGameTraceV1.from_dict(trace.to_dict(trusted=True))
                except (KeyError, TypeError, ValueError):
                    return ("compiler_parity_defect",)

        terminal_result = state.get("terminalResult")
        if terminal_result is not None:
            if not isinstance(terminal_result, Mapping):
                return ("compiler_parity_defect",)
            result = dict(terminal_result)
            if (
                result.get("roundScore") != score
                or result.get("matchScore") != scoring_probe["finalMatchScore"]
                or result.get("matchWinner") != scoring_probe["matchWinner"]
                or (
                    result.get("stateSha256") is not None
                    and str(result["stateSha256"]) != canonical_sha256(state)
                )
            ):
                return ("compiler_parity_defect",)
        return ()

    def _project_matches(
        self,
        recipe: ResearchRecipe,
        trace: Any,
        *,
        context: ResearchExecutionContextV1 | None = None,
    ) -> bool:
        required = [
            item
            for item in recipe.situation["projects"]
            if item.get("mode", "hard") == "hard"
        ]
        if not required:
            return True
        final_state = trace.trusted_final_state
        projects = list(final_state.get("projects") or ())
        actual_contract = Contract.parse(str(final_state.get("contract"))).effective
        for requirement in required:
            required_contract = Contract.parse(str(requirement["contract"])).effective
            required_high_rank = int(Rank.parse(str(requirement["highRank"])))
            found = any(
                actual_contract == required_contract
                and str(item.get("owner") or item.get("seat"))
                == str(requirement["seat"])
                and str(item.get("kind")) == str(requirement["kind"])
                and int(item.get("value", -1))
                == self._project_engine_value(requirement, context=context)
                and int(item.get("highRank", -1)) == required_high_rank
                for item in projects
            )
            if not found:
                return False
        return True

    def _semantic_mismatches(
        self,
        recipe: ResearchRecipe,
        simulation: Any,
        *,
        context: ResearchExecutionContextV1 | None = None,
        include_policy_target_mismatches: bool = True,
    ) -> list[str]:
        domain = context.legacy_domain_v2 if context is not None else self.domain
        trace = simulation.final_trace
        mismatches: list[str] = []
        if not simulation.publishable_laffa:
            mismatches.append("hard_constraint_violation")
        if include_policy_target_mismatches and self._policy_target_mismatches(
            recipe, simulation
        ):
            mismatches.append("compiler_parity_defect")
        if not self._project_matches(recipe, trace, context=context):
            mismatches.append("hard_constraint_violation")
        if str(trace.trusted_final_state.get("rulesHash")) != domain.rules_hash:
            mismatches.append("runtime_transition_defect")
        return sorted(set(mismatches))

    @staticmethod
    def _policy_target_mismatches(
        recipe: ResearchRecipe,
        simulation: Any,
    ) -> tuple[str, ...]:
        """Compare requested auction outcomes without claiming semantic parity."""

        state = simulation.final_trace.trusted_final_state
        mismatches: list[str] = []
        buyer = recipe.situation.get("buyer")
        if buyer is not None and str(state.get("buyer")) != str(buyer):
            mismatches.append("buyer_target_miss")
        contracts = set(recipe.situation.get("contracts") or ())
        if contracts and str(state.get("contract")) not in contracts:
            mismatches.append("contract_target_miss")
        trump_suit = recipe.situation.get("trumpSuit")
        if trump_suit is not None and str(state.get("trumpSuit")) != str(trump_suit):
            mismatches.append("trump_target_miss")
        purchase_round = recipe.situation.get("purchaseRound")
        if purchase_round is not None:
            winning_records = [
                item
                for item in state.get("biddingHistory") or ()
                if item.get("buyer") is not None and item.get("contract") is not None
            ]
            if not winning_records or int(winning_records[-1]["round"]) != int(
                purchase_round
            ):
                mismatches.append("purchase_round_target_miss")
        required_escalations = tuple(recipe.situation.get("escalations") or ())
        expected_escalation = (
            required_escalations[0] if required_escalations else "normal"
        )
        if str(state.get("escalation") or "normal") != expected_escalation:
            mismatches.append("escalation_target_miss")
        return tuple(sorted(set(mismatches)))

    def _forced_reason(
        self,
        snapshot: Any,
        *,
        context: ResearchExecutionContextV1 | None = None,
    ) -> dict[str, Any]:
        record = snapshot.decision
        if record.decision_kind == "bid":
            return {
                "code": "only_legal_bid",
                "rulePaths": ["bidding.legal_actions"],
                "textAr": self._localized(
                    "research_audit_reason_only_legal_bid", context
                ),
            }
        state = snapshot.state
        trick = state.current_trick
        if trick is None or not trick.plays:
            return {
                "code": "only_legal_card",
                "rulePaths": ["play.legal_moves"],
                "textAr": self._localized(
                    "research_audit_reason_only_legal_card", context
                ),
            }
        selected = Card.parse(dict(record.chosen_action))
        led = trick.led_suit
        hand_cards = tuple(
            Card(card_id)
            for card_id in range(32)
            if state.hands[int(record.acting_seat)] & (1 << card_id)
        )
        if any(card.suit == led for card in hand_cards):
            if state.trump_suit == led:
                return {
                    "code": "must_overtrump",
                    "rulePaths": [
                        "play.must_follow_suit",
                        "play.must_overtrump_when_possible",
                    ],
                    "textAr": self._localized(
                        "research_audit_reason_must_overtrump", context
                    ),
                }
            return {
                "code": "must_follow_suit",
                "rulePaths": ["play.must_follow_suit"],
                "textAr": self._localized(
                    "research_audit_reason_must_follow_suit", context
                ),
            }
        if state.trump_suit is not None and selected.suit == state.trump_suit:
            return {
                "code": "must_trump",
                "rulePaths": [
                    "play.must_trump_when_void",
                    "play.partner_winning_exception",
                ],
                "textAr": self._localized("research_audit_reason_must_trump", context),
            }
        return {
            "code": "only_legal_card",
            "rulePaths": ["play.legal_moves"],
            "textAr": self._localized("research_audit_reason_only_legal_card", context),
        }

    @staticmethod
    def _tactical_verification_required(recipe: ResearchRecipe) -> bool:
        return _tactical_verification_required_by_policy(recipe)

    def _verify_tactics(
        self,
        trace: Any,
        analysis: TraceAnalysisResultV1,
        *,
        recipe: ResearchRecipe,
        cancel_check: Callable[[], bool] | None,
        context: ResearchExecutionContextV1 | None = None,
        featured_truth_blocked_ids: Iterable[str] = (),
    ) -> tuple[TacticalVerificationResultV1, ...]:
        tactical_verifier = (
            context.tactical_verifier if context is not None else self.tactical_verifier
        )
        maximum_tactical_decisions = (
            context.maximum_tactical_decisions
            if context is not None
            else self.maximum_tactical_decisions
        )
        simulator = context.simulator if context is not None else self.simulator
        if tactical_verifier is None or maximum_tactical_decisions == 0:
            return ()
        if not self._tactical_verification_required(recipe):
            return ()
        projection_store = TraceDecisionProjectionStoreV1(trace, simulator.rules)
        analyzed = {item.decision_id: item for item in analysis.decisions}
        critical = set(analysis.laffa_evidence.critical_decision_ids)
        competence_only, unresolved = _post_trace_featured_truth_status_ids_v1(analysis)
        featured_blocked = {
            *(str(item) for item in featured_truth_blocked_ids),
            *competence_only,
            *unresolved,
        }
        configured_perspectives = [
            {str(seat) for seat in item["perspectives"]} for item in recipe.tactics
        ]
        preferred_acting_seats = (
            None
            if any(not seats for seats in configured_perspectives)
            else set().union(*configured_perspectives)
        )
        eligible = [
            snapshot
            for snapshot in projection_store.replay.decisions
            if snapshot.decision.decision_kind == "play"
            and not snapshot.decision.forced
            and snapshot.decision.decision_id not in featured_blocked
            and len(snapshot.decision.legal_actions) >= 2
            and sum(mask.bit_count() for mask in snapshot.state.hands)
            <= tactical_verifier.maximum_remaining_cards
        ]
        eligible.sort(
            key=lambda snapshot: (
                -(
                    preferred_acting_seats is None
                    or snapshot.decision.acting_seat.name.lower()
                    in preferred_acting_seats
                ),
                -(snapshot.decision.decision_id in critical),
                -snapshot.decision.ordinal,
                snapshot.decision.decision_id,
            )
        )
        return tuple(
            tactical_verifier.verify_decision(
                snapshot,
                analyzed[snapshot.decision.decision_id],
                cancel_check=cancel_check,
            )
            for snapshot in eligible[:maximum_tactical_decisions]
        )

    @staticmethod
    def _configuration_provenance(
        configuration: Mapping[str, Any] | None,
        context: ResearchExecutionContextV1 | None,
    ) -> dict[str, Any] | None:
        if configuration is None:
            return None
        if context is None:
            raise ValueError(
                "frozen Research configuration lacks its execution context"
            )
        authority = copy.deepcopy(dict(configuration))
        return {
            "configurationAuthorityHash": str(authority["sha256"]),
            "bundleRevisionId": str(authority["bundleRevisionId"]),
            "semanticBundleHash": str(authority["semanticBundleHash"]),
            "effectiveRuntimeConfigHash": str(authority["effectiveRuntimeConfigHash"]),
            "snapshotArtifact": copy.deepcopy(dict(authority["snapshotArtifact"])),
            "componentRevisions": copy.deepcopy(dict(authority["componentRevisions"])),
            "executionIdentities": copy.deepcopy(dict(authority["identities"])),
            "researchContextHash": context.context_hash,
        }

    @staticmethod
    def _requires_finalist_primary_policy(recipe: ResearchRecipe) -> bool:
        return (
            isinstance(recipe, ResearchRecipeV3)
            and str(recipe.resources.get("generationLane")) == "ortools"
        )

    @classmethod
    def _uses_finalist_primary_policy(cls, recipe: ResearchRecipe) -> bool:
        if not cls._requires_finalist_primary_policy(recipe):
            return False
        return (
            str(recipe.resources.get("auctionConstructionMode"))
            == "actor_safe_agent_verified"
            and str(recipe.resources.get("playConstructionMode"))
            == "actor_safe_agent_verified"
        )

    @staticmethod
    def _decision_score_parity_evidence(
        trace: CompleteGameTraceV1,
        deep_analysis: TraceAnalysisResultV1,
        *,
        context: ResearchExecutionContextV1,
    ) -> dict[str, Any]:
        """Compare one finalist line with its own direct deep audit.

        This is deliberately a same-game, same-public-state gate. Cross-lane
        game identity, the CP-SAT structural objective, content-selection
        scores, and the official Baloot round/match score are separate
        authorities and are never compared here.
        """

        deep_analyzer = context.analyzers["deep"]
        deep_profile_hash = str(deep_analyzer.profile.sha256)
        deep_identity_hash = str(context.analysis_identity_hashes["deep"])
        backend_id = str(context.exact_search_backend.descriptor.id)
        play_objective = str(deep_analyzer.profile.objective)
        simulator = context.simulator
        agreement = getattr(simulator, "decision_agreement", None)
        mismatches: list[dict[str, Any]] = []

        def record_mismatch(
            reason: str,
            *,
            decision_id: str | None = None,
            ordinal: int | None = None,
            decision_kind: str | None = None,
        ) -> None:
            mismatches.append(
                {
                    "decisionId": decision_id,
                    "ordinal": ordinal,
                    "decisionKind": decision_kind,
                    "reason": str(reason),
                }
            )

        if getattr(getattr(simulator, "agent_profile", None), "id", None) != (
            "finalist_candidate"
        ):
            record_mismatch("finalist_agent_profile_authority_mismatch")
        if agreement is None or not bool(
            getattr(agreement, "exact_primary_policy", False)
        ):
            record_mismatch("finalist_exact_primary_policy_missing")
        raw_primary_analysis_budget = getattr(
            agreement,
            "primary_analysis_budget",
            None,
        )
        primary_analysis_budget: dict[str, Any] | None = None
        if isinstance(raw_primary_analysis_budget, Mapping):
            try:
                primary_analysis_budget = canonical_plain_profile(
                    raw_primary_analysis_budget
                )
            except TypeError:
                record_mismatch("finalist_primary_analysis_budget_invalid")
        else:
            record_mismatch("finalist_primary_analysis_budget_missing")
        if str(getattr(simulator, "analysis_identity_hash", "")) != (
            deep_identity_hash
        ):
            record_mismatch("finalist_deep_analysis_identity_mismatch")
        if str(getattr(simulator, "trace_analysis_profile_hash", "")) != (
            deep_profile_hash
        ):
            record_mismatch("finalist_deep_analysis_profile_mismatch")
        if deep_analysis.trace_hash != trace.sha256:
            record_mismatch("direct_deep_trace_hash_mismatch")
        if deep_analysis.profile_hash != deep_profile_hash:
            record_mismatch("direct_deep_profile_hash_mismatch")

        policy_by_ordinal: dict[int, Mapping[str, Any]] = {}
        raw_policy_rows = trace.trusted_final_state.get("decisionPolicyTraces")
        if not isinstance(raw_policy_rows, (list, tuple)):
            record_mismatch("decision_policy_trace_ledger_missing")
            raw_policy_rows = ()
        for raw_row in raw_policy_rows:
            if not isinstance(raw_row, Mapping):
                record_mismatch("decision_policy_trace_row_invalid")
                continue
            raw_ordinal = raw_row.get("ordinal")
            if isinstance(raw_ordinal, bool) or not isinstance(raw_ordinal, int):
                record_mismatch("decision_policy_trace_ordinal_invalid")
                continue
            if raw_ordinal in policy_by_ordinal:
                record_mismatch(
                    "decision_policy_trace_ordinal_duplicate",
                    ordinal=raw_ordinal,
                )
                continue
            raw_trace = raw_row.get("trace")
            if not isinstance(raw_trace, Mapping):
                record_mismatch(
                    "decision_policy_trace_payload_invalid",
                    ordinal=raw_ordinal,
                )
                continue
            policy_by_ordinal[raw_ordinal] = raw_trace

        analysis_by_id: dict[str, Any] = {}
        for item in deep_analysis.decisions:
            if item.decision_id in analysis_by_id:
                record_mismatch(
                    "direct_deep_decision_duplicate",
                    decision_id=item.decision_id,
                    ordinal=item.ordinal,
                    decision_kind=item.decision_kind,
                )
                continue
            analysis_by_id[item.decision_id] = item

        records = tuple(trace.decision_ledger.records)
        expected_ordinals = {record.ordinal for record in records}
        unexpected_policy_ordinals = sorted(set(policy_by_ordinal) - expected_ordinals)
        for ordinal in unexpected_policy_ordinals:
            record_mismatch("decision_policy_trace_unbound", ordinal=ordinal)
        expected_ids = {record.decision_id for record in records}
        for decision_id in sorted(set(analysis_by_id) - expected_ids):
            item = analysis_by_id[decision_id]
            record_mismatch(
                "direct_deep_decision_unbound",
                decision_id=decision_id,
                ordinal=item.ordinal,
                decision_kind=item.decision_kind,
            )

        decision_rows: list[dict[str, Any]] = []
        for record in records:
            initial_mismatch_count = len(mismatches)
            item = analysis_by_id.get(record.decision_id)
            policy = policy_by_ordinal.get(record.ordinal)
            audit_hash: str | None = None
            generation_hash: str | None = None
            audit_rows_hash: str | None = None
            generation_rows_hash: str | None = None

            def fail(reason: str) -> None:
                record_mismatch(
                    reason,
                    decision_id=record.decision_id,
                    ordinal=record.ordinal,
                    decision_kind=record.decision_kind,
                )

            if record.decision_kind not in {"bid", "play"}:
                fail("unsupported_finalist_choice_decision_kind")
            if item is None:
                fail("direct_deep_decision_missing")
            else:
                try:
                    item.validate_ledger_binding(
                        record,
                        trace_analysis_profile_hash=deep_profile_hash,
                        rules_hash=trace.rules_hash,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    fail(f"direct_deep_ledger_binding_invalid:{exc}")
            if policy is None:
                fail("generation_policy_trace_missing")
            elif canonical_json_bytes_v1(policy.get("selectedAction")) != (
                canonical_json_bytes_v1(record.chosen_action)
            ):
                fail("generation_selected_action_mismatch")

            if not record.forced and policy is not None:
                if policy.get("source") != "finalist_multi_cohort_agent":
                    fail("generation_finalist_policy_source_mismatch")
                if policy.get("selectionMode") != "primary_policy_only":
                    fail("generation_finalist_selection_mode_mismatch")
                if policy.get("primaryActionPreserved") is not True:
                    fail("generation_primary_action_preservation_missing")
                if policy.get("observerSafe") is not True:
                    fail("generation_finalist_observer_safety_missing")
                if policy.get("decisionKind") != record.decision_kind:
                    fail("generation_finalist_decision_kind_mismatch")
                if policy.get("traceAnalysisProfileHash") != deep_profile_hash:
                    fail("generation_finalist_analysis_profile_mismatch")
                if policy.get("analysisIdentityHash") != deep_identity_hash:
                    fail("generation_finalist_analysis_identity_mismatch")
                raw_policy_budget = policy.get("analysisBudget")
                policy_budget: dict[str, Any] | None = None
                if isinstance(raw_policy_budget, Mapping):
                    try:
                        policy_budget = canonical_plain_profile(raw_policy_budget)
                    except TypeError:
                        pass
                if (
                    primary_analysis_budget is None
                    or policy_budget is None
                    or canonical_json_bytes_v1(policy_budget)
                    != canonical_json_bytes_v1(primary_analysis_budget)
                ):
                    fail("generation_finalist_analysis_budget_mismatch")

            if record.forced:
                audit_proof = item.forced_legality_proof if item is not None else None
                generation_proof = (
                    policy.get("forcedLegalityProof") if policy is not None else None
                )
                if (
                    item is not None
                    and item.primary_decision_score_evidence is not None
                ):
                    fail("forced_direct_deep_score_evidence_present")
                if (
                    policy is not None
                    and policy.get("primaryDecisionScoreEvidence") is not None
                ):
                    fail("forced_generation_score_evidence_present")
                if (
                    policy is not None
                    and policy.get("primaryDecisionScoreEvidenceHash") is not None
                ):
                    fail("forced_generation_score_hash_present")
                if not isinstance(audit_proof, Mapping):
                    fail("forced_direct_deep_legality_proof_missing")
                else:
                    audit_hash = str(audit_proof.get("sha256") or "") or None
                if not isinstance(generation_proof, Mapping):
                    fail("forced_generation_legality_proof_missing")
                else:
                    generation_hash = str(generation_proof.get("sha256") or "") or None
                if (
                    isinstance(audit_proof, Mapping)
                    and isinstance(generation_proof, Mapping)
                    and canonical_json_bytes_v1(generation_proof)
                    != (canonical_json_bytes_v1(audit_proof))
                ):
                    fail("forced_legality_proof_bytes_mismatch")
            else:
                expected_seed = derive_actor_observer_cohort_seed_v1(
                    public_prefix_hash=record.public_prefix_hash,
                    decision_kind=record.decision_kind,
                    acting_seat=record.acting_seat,
                    cohort_authority_hash=deep_identity_hash,
                    cohort_index=0,
                    governed_salt=PRIMARY_DECISION_COHORT_SALT_V1,
                )
                audit_evidence = (
                    item.primary_decision_score_evidence if item is not None else None
                )
                generation_value = (
                    policy.get("primaryDecisionScoreEvidence")
                    if policy is not None
                    else None
                )
                validated_audit = None
                validated_generation = None
                expected_bindings = {
                    "decision_kind": record.decision_kind,
                    "actor": record.acting_seat,
                    "public_prefix_hash": record.public_prefix_hash,
                    "rules_hash": trace.rules_hash,
                    "analysis_identity_hash": deep_identity_hash,
                    "trace_analysis_profile_hash": deep_profile_hash,
                    "cohort_seed_policy": (PRIMARY_DECISION_COHORT_SEED_POLICY_V1),
                    "cohort_index": 0,
                    "cohort_seed": expected_seed,
                }
                if audit_evidence is None:
                    fail("direct_deep_primary_score_evidence_missing")
                else:
                    try:
                        validated_audit = validate_primary_decision_score_evidence_v1(
                            audit_evidence,
                            **expected_bindings,
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        fail(f"direct_deep_primary_score_evidence_invalid:{exc}")
                if not isinstance(generation_value, Mapping):
                    fail("generation_primary_score_evidence_missing")
                else:
                    try:
                        validated_generation = (
                            validate_primary_decision_score_evidence_v1(
                                generation_value,
                                **expected_bindings,
                            )
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        fail(f"generation_primary_score_evidence_invalid:{exc}")
                expected_objective = (
                    BID_DECISION_OBJECTIVE_V1
                    if record.decision_kind == "bid"
                    else play_objective
                )
                expected_tie_break = (
                    BID_DECISION_TIE_BREAK_V1
                    if record.decision_kind == "bid"
                    else PLAY_DECISION_TIE_BREAK_V1
                )
                for label, evidence in (
                    ("direct_deep", validated_audit),
                    ("generation", validated_generation),
                ):
                    if evidence is None:
                        continue
                    if evidence.objective != expected_objective:
                        fail(f"{label}_objective_mismatch")
                    if evidence.tie_break != expected_tie_break:
                        fail(f"{label}_tie_break_mismatch")
                    if evidence.backend != backend_id:
                        fail(f"{label}_backend_mismatch")
                    if canonical_json_bytes_v1(evidence.recommended_action) != (
                        canonical_json_bytes_v1(record.chosen_action)
                    ):
                        fail(f"{label}_recommended_action_mismatch")
                if validated_audit is not None:
                    audit_hash = validated_audit.sha256
                    audit_rows_hash = validated_audit.rows_hash
                if validated_generation is not None:
                    generation_hash = validated_generation.sha256
                    generation_rows_hash = validated_generation.rows_hash
                    if policy is not None:
                        trace_bindings = {
                            "primaryDecisionScoreEvidenceHash": (
                                validated_generation.sha256
                            ),
                            "decisionScoreRowsHash": validated_generation.rows_hash,
                            "legalActionSetHash": (
                                validated_generation.legal_action_set_hash
                            ),
                        }
                        drifted = sorted(
                            key
                            for key, expected in trace_bindings.items()
                            if policy.get(key) != expected
                        )
                        if drifted:
                            fail(
                                "generation_score_trace_binding_mismatch:"
                                + ",".join(drifted)
                            )
                        if policy.get("forcedLegalityProof") is not None:
                            fail("non_forced_generation_legality_proof_present")
                        primary_trace = policy.get("primaryTrace")
                        if primary_trace is None:
                            fail("generation_primary_trace_missing")
                        elif not isinstance(primary_trace, Mapping):
                            fail("generation_primary_trace_invalid")
                        elif (
                            canonical_json_bytes_v1(
                                primary_trace.get("primaryDecisionScoreEvidence")
                            )
                            != validated_generation.canonical_bytes()
                            or primary_trace.get("primaryDecisionScoreEvidenceHash")
                            != validated_generation.sha256
                            or canonical_json_bytes_v1(
                                primary_trace.get("selectedAction")
                            )
                            != canonical_json_bytes_v1(record.chosen_action)
                        ):
                            fail("generation_primary_trace_binding_mismatch")
                if (
                    validated_audit is not None
                    and validated_generation is not None
                    and validated_generation.canonical_bytes()
                    != validated_audit.canonical_bytes()
                ):
                    fail("primary_decision_score_evidence_bytes_mismatch")

            decision_rows.append(
                {
                    "decisionId": record.decision_id,
                    "ordinal": record.ordinal,
                    "decisionKind": record.decision_kind,
                    "forced": record.forced,
                    "generationEvidenceHash": generation_hash,
                    "auditEvidenceHash": audit_hash,
                    "generationRowsHash": generation_rows_hash,
                    "auditRowsHash": audit_rows_hash,
                    "matched": len(mismatches) == initial_mismatch_count,
                }
            )

        mismatches.sort(
            key=lambda item: (
                int(item["ordinal"] or 0),
                str(item["decisionId"] or ""),
                str(item["reason"]),
            )
        )
        core = {
            "schemaVersion": 1,
            "authority": "finalist_primary_decision_score_parity_v1",
            "traceHash": trace.sha256,
            "deepAnalysisHash": deep_analysis.sha256,
            "deepAnalysisProfileHash": deep_profile_hash,
            "deepAnalysisIdentityHash": deep_identity_hash,
            "analysisBackend": backend_id,
            "decisionCount": len(records),
            "nonForcedDecisionCount": sum(not record.forced for record in records),
            "forcedDecisionCount": sum(record.forced for record in records),
            "matchedDecisionCount": sum(row["matched"] for row in decision_rows),
            "matched": not mismatches,
            "mismatches": mismatches,
            "decisions": decision_rows,
            "crossLaneComparison": False,
            "structuralObjectiveCompared": False,
            "officialBalootScoreCompared": False,
            "researchContextHash": context.context_hash,
        }
        return {**core, "sha256": canonical_sha256(core)}

    def _decision_audit(
        self,
        trace: Any,
        analysis: TraceAnalysisResultV1,
        tactical: tuple[TacticalVerificationResultV1, ...] = (),
        *,
        context: ResearchExecutionContextV1 | None = None,
        configuration: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        simulator = context.simulator if context is not None else self.simulator
        projection_store = TraceDecisionProjectionStoreV1(trace, simulator.rules)
        snapshots = {
            item.decision.decision_id: item
            for item in projection_store.replay.decisions
        }
        policy_traces = {
            int(item["ordinal"]): copy.deepcopy(dict(item.get("trace") or {}))
            for item in trace.trusted_final_state.get("decisionPolicyTraces") or ()
        }
        analyzed = {item.decision_id: item for item in analysis.decisions}
        tactical_by_decision = {item.oracle.decision_id: item for item in tactical}
        audited_actions: list[tuple[int, dict[str, Any]]] = []
        event_sequence_by_hash = {
            event.event_hash: event.sequence for event in trace.events
        }
        for record in trace.decision_ledger.records:
            item = analyzed[record.decision_id]
            snapshot = snapshots[record.decision_id]
            if record.decision_kind == "bid":
                if snapshot.bidding is None:
                    raise ValueError(
                        "bid audit requires its authoritative auction state"
                    )
                actor_info = {
                    "schemaVersion": 1,
                    "kind": "bid_information_state",
                    "observer": record.acting_seat.name.lower(),
                    "hand": [
                        card.to_dict()
                        for card in snapshot.state.hand(record.acting_seat)
                    ],
                    "purchaseCard": (
                        snapshot.state.purchase_card.to_dict()
                        if snapshot.state.purchase_card is not None
                        else None
                    ),
                    "dealer": snapshot.state.dealer.name.lower(),
                    "matchScore": list(snapshot.state.match_score),
                    "bidding": bidding_machine_projection_v1(snapshot.bidding),
                }
            else:
                actor_info = InformationState.from_complete(
                    snapshot.state, record.acting_seat
                ).to_dict()
            actual_equals_best = dict(item.actual_action) == dict(item.best_action)
            forced_reason = (
                self._forced_reason(snapshot, context=context)
                if record.forced
                else None
            )
            information_set = InformationSetEvidenceV1.build(
                observer=record.acting_seat,
                information_state=actor_info,
                legal_actions=record.legal_actions,
                alternatives=item.alternatives,
                forced=record.forced,
                analysis_hash=item.sha256,
                method=item.method,
                hidden_deal_support=item.hidden_deals,
                effective_sample_size=item.effective_sample_size,
            )
            sequential_response = SequentialResponseEvidenceV1.build(
                actual_action=item.actual_action,
                recommended_action=item.best_action,
                legal_actions=record.legal_actions,
                alternatives=item.alternatives,
                information_set=information_set,
                expected_regret=item.expected_regret,
                forced=record.forced,
            )
            truth = _decision_truth_label_v1(
                sequential_response_status=sequential_response.status,
                confidence=item.confidence,
                debate_score=item.debate_score,
            )
            verification = tactical_by_decision.get(record.decision_id)
            tactical_evidence = list(
                self._observer_tactical_evidence(record, item, context=context)
            )
            if verification is not None:
                tactical_evidence.extend(
                    {
                        **evidence.to_dict(),
                        "truthMode": TruthMode.ACTUAL_DEAL_ORACLE.value,
                        "evidenceBasis": "paired_exact_actual_deal_continuations",
                    }
                    for evidence in verification.evidence
                )
            policy = policy_traces.get(record.ordinal)
            selection_explanation = (
                {
                    "available": True,
                    "source": "recorded_agent_policy",
                    "summaryAr": self._localized(
                        "research_audit_selection_recorded", context
                    ),
                    "evidence": policy,
                }
                if policy
                else {
                    "available": False,
                    "source": "not_recorded",
                    "summaryAr": self._localized(
                        "research_audit_selection_not_recorded", context
                    ),
                    "evidence": None,
                }
            )
            analytical_explanation = {
                "status": "correct" if actual_equals_best else "mistake",
                "textAr": (
                    (forced_reason or {}).get("textAr")
                    if record.forced
                    else (
                        self._localized("research_audit_analysis_matches", context)
                        if actual_equals_best
                        else self._localized(
                            "research_audit_analysis_regret",
                            context,
                            regret=f"{item.expected_regret:.2f}",
                        )
                    )
                ),
                "causalContinuation": list(
                    (item.alternatives[0].get("principalVariation") or ())
                )
                if item.alternatives
                else [],
                "tacticalMechanisms": sorted(
                    {
                        mechanism
                        for evidence in tactical_evidence
                        for mechanism in evidence.get("mechanisms") or ()
                    }
                ),
            }
            audited_actions.append(
                (
                    event_sequence_by_hash.get(record.resulting_event_hash, 10**9),
                    {
                        "schemaVersion": 1,
                        "decisionId": record.decision_id,
                        "decisionOrdinal": record.ordinal,
                        "actingSeat": record.acting_seat.name.lower(),
                        "actingTeam": record.acting_seat.team.name.lower(),
                        "decisionKind": record.decision_kind,
                        "phase": record.phase,
                        "publicStateHash": record.public_prefix_hash,
                        "informationStateHash": canonical_sha256(actor_info),
                        "informationSetEvidence": information_set.to_dict(),
                        "sequentialResponseEvidence": sequential_response.to_dict(),
                        "legalActions": [dict(value) for value in record.legal_actions],
                        "rulePaths": (forced_reason or {}).get(
                            "rulePaths", ["engine.legal_actions"]
                        ),
                        "actualAction": dict(item.actual_action),
                        "actionSource": "agent",
                        "forced": record.forced,
                        "recommendedAction": dict(item.best_action),
                        "actualMatchesRecommendation": actual_equals_best,
                        "alternatives": [dict(value) for value in item.alternatives],
                        "expectedRegret": item.expected_regret,
                        "probabilityDrop": item.probability_drop,
                        "catastrophicCrossing": item.catastrophic_crossing,
                        "hiddenDealSupport": item.hidden_deals,
                        "effectiveSampleSize": item.effective_sample_size,
                        "confidence": item.confidence,
                        "convergence": item.method,
                        "principalContinuation": list(
                            (item.alternatives[0].get("principalVariation") or ())
                        )
                        if item.alternatives
                        else [],
                        "supportedMetrics": list(item.supported_metrics),
                        "unsupportedMetrics": list(item.unsupported_metrics),
                        "primaryDecisionScoreEvidence": (
                            item.primary_decision_score_evidence.to_dict()
                            if item.primary_decision_score_evidence is not None
                            else None
                        ),
                        "forcedLegalityProof": (
                            copy.deepcopy(dict(item.forced_legality_proof))
                            if item.forced_legality_proof is not None
                            else None
                        ),
                        "selectionExplanation": selection_explanation,
                        "analyticalExplanation": analytical_explanation,
                        "forcedExplanation": forced_reason,
                        "tacticalEvidence": tactical_evidence,
                        "actualDealOracle": (
                            verification.to_dict() if verification is not None else None
                        ),
                        "truthLabel": truth,
                        "provenance": {
                            "engineVersion": trace.engine_version,
                            "bundleHash": trace.bundle_hash,
                            "rulesHash": trace.rules_hash,
                            "agentRelease": dict(trace.agent_profiles).get(
                                record.acting_seat.name.lower()
                            ),
                            "analyzerId": analysis.analyzer_id,
                            "analyzerVersion": analysis.analyzer_version,
                            "analysisProfileHash": analysis.profile_hash,
                            "analysisHash": item.sha256,
                        },
                    },
                )
            )

        governed_event_types = {
            "project_declared",
            "projects_revealed",
            "baloot_declared",
            "baloot_included_in_project",
            "escalation_announced",
        }
        play_snapshots: list[tuple[int, Any]] = []
        for snapshot in projection_store.replay.decisions:
            if snapshot.decision.decision_kind != "play":
                continue
            sequence = event_sequence_by_hash.get(
                snapshot.decision.resulting_event_hash, 10**9
            )
            play_snapshots.append((sequence, snapshot))

        governed_count = 0
        incomplete_governed_count = 0
        prior_escalations: list[tuple[Seat, str, str | None]] = []
        scripted_escalations = tuple(
            copy.deepcopy(dict(item))
            for item in (
                dict(trace.trusted_final_state.get("generation") or {}).get(
                    "scriptedEscalations"
                )
                or ()
            )
        )
        escalation_ordinal = 0
        for event in trace.events:
            if event.event_type not in governed_event_types:
                continue
            payload = copy.deepcopy(dict(event.payload))
            raw_seat = payload.get("seat") or payload.get("owner")
            if raw_seat is None:
                incomplete_governed_count += 1
                continue
            seat = Seat.parse(str(raw_seat))
            trigger = next(
                (
                    snapshot
                    for sequence, snapshot in play_snapshots
                    if sequence > event.sequence
                    and snapshot.decision.acting_seat == seat
                ),
                None,
            )
            actor_info = (
                InformationState.from_complete(trigger.state, seat).to_dict()
                if trigger is not None
                else {
                    "schemaVersion": 1,
                    "kind": "governed_action_boundary",
                    "observer": seat.name.lower(),
                    "previousEventHash": event.previous_event_hash,
                }
            )
            is_escalation = event.event_type == "escalation_announced"
            action = {"eventType": event.event_type, **payload}
            automatic_reason = {
                "project_declared": (
                    self._localized("research_audit_project_declared", context),
                    ["rules.projects.declaration_timing"],
                ),
                "projects_revealed": (
                    self._localized("research_audit_projects_revealed", context),
                    ["rules.projects.reveal_timing"],
                ),
                "baloot_declared": (
                    self._localized("research_audit_baloot_declared", context),
                    ["rules.projects.baloot.declare_on_second_card"],
                ),
                "baloot_included_in_project": (
                    self._localized("research_audit_baloot_included", context),
                    ["rules.projects.baloot.siri_or_fifty_inclusion"],
                ),
            }
            reason_text, rule_paths = automatic_reason.get(
                event.event_type,
                (
                    self._localized("research_audit_escalation_unresolved", context),
                    [
                        "rules.escalation.enabled",
                        "rules.escalation.hokum_chain",
                        "rules.escalation.sun_chain",
                        "rules.escalation.response_actor_policy",
                        "rules.escalation.sun_score_threshold",
                    ],
                ),
            )
            escalation_legal_actions: tuple[dict[str, Any], ...] = ()
            escalation_alternatives: tuple[dict[str, Any], ...] = ()
            escalation_actual: dict[str, Any] | None = None
            escalation_analysis_hash: str | None = None
            escalation_selection: Mapping[str, Any] | None = None
            if is_escalation:
                escalation_ordinal += 1
                if (
                    trigger is None
                    or trigger.state.declarer is None
                    or trigger.state.contract is None
                ):
                    raise ValueError(
                        "escalation audit requires the authoritative contracted play state"
                    )
                interpreter = ReferenceEscalationInterpreterV1.from_rules(
                    simulator.rules.data
                )
                prior_state = ReferenceEscalationStateV1(
                    interpreter.domain.default_level
                )
                for prior_actor, prior_action, prior_mode in prior_escalations:
                    prior_state, _ = interpreter.apply(
                        prior_state,
                        declarer=trigger.state.declarer,
                        contract=trigger.state.contract,
                        match_score=tuple(trigger.state.match_score),
                        actor=prior_actor,
                        action=prior_action,
                        mode=prior_mode,
                    )
                actor_info = {
                    "schemaVersion": 1,
                    "kind": "escalation_information_state",
                    "observer": seat.name.lower(),
                    "observerHand": [
                        card.to_dict() for card in trigger.state.hand(seat)
                    ],
                    "dealer": trigger.state.dealer.name.lower(),
                    "buyer": (
                        trigger.state.buyer.name.lower()
                        if trigger.state.buyer is not None
                        else None
                    ),
                    "declarer": trigger.state.declarer.name.lower(),
                    "purchaseCard": (
                        trigger.state.purchase_card.to_dict()
                        if trigger.state.purchase_card is not None
                        else None
                    ),
                    "contract": trigger.state.contract.value,
                    "trumpSuit": (
                        trigger.state.trump_suit.name.lower()
                        if trigger.state.trump_suit is not None
                        else None
                    ),
                    "matchScore": list(trigger.state.match_score),
                    "currentEscalation": prior_state.level,
                    "locked": prior_state.locked,
                    "doubleSeat": (
                        prior_state.double_seat.name.lower()
                        if prior_state.double_seat is not None
                        else None
                    ),
                    "priorEscalations": [
                        {
                            "actor": prior_actor.name.lower(),
                            "action": prior_action,
                            "mode": prior_mode,
                        }
                        for prior_actor, prior_action, prior_mode in prior_escalations
                    ],
                    "biddingHistory": [
                        item.to_dict() for item in trigger.state.bid_history
                    ],
                }
                candidates: list[dict[str, Any]] = [{"action": "decline", "mode": None}]
                for candidate_action in ("double", "triple", "four", "gahwa"):
                    candidate_modes: tuple[str | None, ...] = (
                        ("open", "locked")
                        if trigger.state.contract == Contract.HOKUM
                        and candidate_action in {"double", "four"}
                        else (None,)
                    )
                    for candidate_mode in candidate_modes:
                        try:
                            _next_state, resolved = interpreter.apply(
                                prior_state,
                                declarer=trigger.state.declarer,
                                contract=trigger.state.contract,
                                match_score=tuple(trigger.state.match_score),
                                actor=seat,
                                action=candidate_action,
                                mode=candidate_mode,
                            )
                        except ValueError:
                            continue
                        candidate = {
                            "action": resolved.action,
                            "mode": resolved.resolved_mode,
                        }
                        if candidate not in candidates:
                            candidates.append(candidate)
                escalation_legal_actions = tuple(candidates)
                escalation_actual = {
                    "action": str(payload.get("action") or ""),
                    "mode": str(payload.get("mode") or "open"),
                }
                if escalation_actual not in escalation_legal_actions:
                    raise AssertionError(
                        "recorded escalation is outside the actor-safe legal action set"
                    )
                escalation_alternatives = tuple(
                    {
                        "action": dict(candidate),
                        "legality": "exact_legal",
                        "searchCompletionStatus": "not_applicable_rule_legality",
                        "exactShare": 0.0,
                        "strategicEvaluationAvailable": False,
                    }
                    for candidate in escalation_legal_actions
                )
                escalation_analysis_hash = canonical_sha256(
                    {
                        "schemaVersion": 1,
                        "method": "authoritative_actor_safe_escalation_legality_v1",
                        "informationState": actor_info,
                        "legalActions": list(escalation_legal_actions),
                        "actualAction": escalation_actual,
                        "rulesHash": trace.rules_hash,
                    }
                )
                if escalation_ordinal <= len(scripted_escalations):
                    recorded = scripted_escalations[escalation_ordinal - 1]
                    recorded_action = {
                        "action": str(recorded.get("action") or ""),
                        "mode": str(
                            recorded.get("resolvedMode")
                            or recorded.get("requestedMode")
                            or "open"
                        ),
                    }
                    if (
                        str(recorded.get("actor") or "") == seat.name.lower()
                        and recorded_action == escalation_actual
                    ):
                        escalation_selection = recorded
                prior_escalations.append(
                    (
                        seat,
                        escalation_actual["action"],
                        (
                            escalation_actual["mode"]
                            if escalation_actual["action"] in {"double", "four"}
                            else None
                        ),
                    )
                )
            governed_count += 1
            identity = {
                "traceHash": trace.sha256,
                "eventHash": event.event_hash,
                "decisionKind": "escalation" if is_escalation else "project",
            }
            transition_information_set = InformationSetEvidenceV1.build(
                observer=seat,
                information_state=actor_info,
                legal_actions=(
                    escalation_legal_actions if is_escalation else (action,)
                ),
                alternatives=(
                    escalation_alternatives if is_escalation else ({"action": action},)
                ),
                forced=not is_escalation,
                analysis_hash=(
                    escalation_analysis_hash if is_escalation else event.event_hash
                ),
                method=(
                    "authoritative_actor_safe_escalation_legality_v1"
                    if is_escalation
                    else "authoritative_rule_transition_v1"
                ),
                hidden_deal_support=0,
                effective_sample_size=0.0,
            )
            transition_sequential_response = SequentialResponseEvidenceV1.build(
                actual_action=(escalation_actual if is_escalation else action),
                recommended_action=(
                    {"status": "unavailable"} if is_escalation else action
                ),
                legal_actions=(
                    escalation_legal_actions if is_escalation else (action,)
                ),
                alternatives=(
                    escalation_alternatives if is_escalation else ({"action": action},)
                ),
                information_set=transition_information_set,
                expected_regret=0.0,
                forced=not is_escalation,
            )
            audited_actions.append(
                (
                    event.sequence,
                    {
                        "schemaVersion": 1,
                        "decisionId": f"ACT-{canonical_sha256(identity)[:20].upper()}",
                        "decisionOrdinal": None,
                        "actingSeat": seat.name.lower(),
                        "actingTeam": seat.team.name.lower(),
                        "decisionKind": "escalation" if is_escalation else "project",
                        "phase": "final_dealt" if is_escalation else "playing",
                        "publicStateHash": event.previous_event_hash
                        or canonical_sha256([]),
                        "informationStateHash": canonical_sha256(actor_info),
                        "informationSetEvidence": transition_information_set.to_dict(),
                        "sequentialResponseEvidence": transition_sequential_response.to_dict(),
                        "legalActions": (
                            [dict(item) for item in escalation_legal_actions]
                            if is_escalation
                            else [action]
                        ),
                        "rulePaths": rule_paths,
                        "actualAction": escalation_actual if is_escalation else action,
                        "actionSource": (
                            (
                                "recorded_compiled_escalation_plan"
                                if escalation_selection is not None
                                else "recorded_escalation_source_unknown"
                            )
                            if is_escalation
                            else "engine_automatic_rule"
                        ),
                        "forced": not is_escalation,
                        "recommendedAction": (
                            {"status": "unavailable"} if is_escalation else action
                        ),
                        "recommendationAvailable": not is_escalation,
                        "actualMatchesRecommendation": not is_escalation,
                        "alternatives": (
                            [dict(item) for item in escalation_alternatives]
                            if is_escalation
                            else [{"action": action}]
                        ),
                        "expectedRegret": 0.0,
                        "probabilityDrop": 0.0,
                        "catastrophicCrossing": False,
                        "hiddenDealSupport": 0,
                        "effectiveSampleSize": 0.0,
                        "confidence": 0.0 if is_escalation else 1.0,
                        "convergence": (
                            "exact_legal_actions_strategy_unresolved"
                            if is_escalation
                            else "authoritative_rule_transition_v1"
                        ),
                        "principalContinuation": [],
                        "supportedMetrics": (
                            ["legal_action_set", "rule_transition"]
                            if is_escalation
                            else ["rule_transition"]
                        ),
                        "unsupportedMetrics": (
                            ["all_counterfactual_metrics"] if is_escalation else []
                        ),
                        "selectionExplanation": {
                            "available": (
                                escalation_selection is not None
                                if is_escalation
                                else True
                            ),
                            "source": (
                                (
                                    "recorded_compiled_escalation_plan"
                                    if escalation_selection is not None
                                    else "not_recorded"
                                )
                                if is_escalation
                                else "engine_automatic_rule"
                            ),
                            "summaryAr": (
                                self._localized(
                                    "research_audit_selection_recorded", context
                                )
                                if is_escalation and escalation_selection is not None
                                else reason_text
                            ),
                            "evidence": (
                                {
                                    "eventSequence": event.sequence,
                                    "eventHash": event.event_hash,
                                    "recordedAction": dict(escalation_selection),
                                }
                                if escalation_selection is not None
                                else {
                                    "eventSequence": event.sequence,
                                    "eventHash": event.event_hash,
                                }
                            ),
                        },
                        "analyticalExplanation": {
                            "status": "unresolved" if is_escalation else "forced_rule",
                            "textAr": reason_text,
                            "causalContinuation": [],
                            "tacticalMechanisms": [],
                        },
                        "forcedExplanation": (
                            None
                            if is_escalation
                            else {"textAr": reason_text, "rulePaths": rule_paths}
                        ),
                        "tacticalEvidence": [],
                        "actualDealOracle": None,
                        "truthLabel": "provisional"
                        if is_escalation
                        else "exact_optimal",
                        "eventSequence": event.sequence,
                        "eventHash": event.event_hash,
                        "provenance": {
                            "engineVersion": trace.engine_version,
                            "bundleHash": trace.bundle_hash,
                            "rulesHash": trace.rules_hash,
                            "transitionAuthority": "baloot_engine",
                            "legalityAuthority": (
                                "reference_escalation_interpreter_v1"
                                if is_escalation
                                else "baloot_engine"
                            ),
                            "analysisHash": escalation_analysis_hash,
                        },
                    },
                )
            )

        audited_actions.sort(key=lambda item: (item[0], item[1]["decisionId"]))
        decisions = []
        for audit_ordinal, (_, item) in enumerate(audited_actions, start=1):
            decisions.append({**item, "ordinal": audit_ordinal})
        configuration_provenance = self._configuration_provenance(
            configuration, context
        )
        if configuration_provenance is not None:
            for item in decisions:
                item["provenance"] = {
                    **dict(item.get("provenance") or {}),
                    **copy.deepcopy(configuration_provenance),
                }
        choice_ids = {record.decision_id for record in trace.decision_ledger.records}
        choice_proofs = [
            dict(item.get("informationSetEvidence") or {})
            for item in decisions
            if item["decisionId"] in choice_ids
        ]
        choice_response_proofs = [
            dict(item.get("sequentialResponseEvidence") or {})
            for item in decisions
            if item["decisionId"] in choice_ids
        ]
        payload = {
            "schemaVersion": 2,
            "traceHash": trace.sha256,
            "decisionCount": len(decisions),
            "choiceDecisionCount": len(trace.decision_ledger.records),
            "governedActionCount": governed_count,
            "incompleteGovernedActionCount": incomplete_governed_count,
            "complete": (
                len(audited_actions)
                == len(trace.decision_ledger.records) + governed_count
                and incomplete_governed_count == 0
            ),
            "informationSetComplete": all(
                proof.get("status") != "incomplete"
                and bool(proof.get("observerSafe"))
                and bool(proof.get("actionCoverageComplete"))
                and bool(proof.get("actorChoiceBoundOnce"))
                for proof in choice_proofs
            ),
            "sequentialResponseComplete": all(
                proof.get("status") != "incomplete"
                and bool(proof.get("allAlternativesEvaluated"))
                and bool(proof.get("pairedActorSafeCohort"))
                for proof in choice_response_proofs
            ),
            "configurationProvenance": configuration_provenance,
            "decisions": decisions,
        }
        payload["sha256"] = canonical_sha256(payload)
        return payload

    @staticmethod
    def _tactical_action_id(action: Mapping[str, Any]) -> str:
        if "action" in action:
            return f"{action['action']}:{action.get('trumpSuit') or ''}"
        return str(
            action.get("code") or action.get("id") or Card.parse(dict(action)).code
        )

    def _observer_tactical_evidence(
        self,
        record: Any,
        item: Any,
        *,
        context: ResearchExecutionContextV1 | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Derive tactics that are provable from paired actor-safe evidence.

        Exact late-game causal tactics stay in the separate actual-deal oracle.
        Debate and purchase/contract choices can be proven without hidden-card
        disclosure from the already paired observer-safe alternatives.
        """
        if record.forced or len(item.alternatives) < 2:
            return ()
        tactical_verifier = (
            context.tactical_verifier if context is not None else self.tactical_verifier
        )
        profile = (
            tactical_verifier.detector.profile
            if tactical_verifier is not None
            else TacticalDetectorProfileV1.defaults()
        )
        ranked = tuple(
            sorted(
                (dict(value) for value in item.alternatives),
                key=lambda value: (
                    int(value.get("rank", 10**9)),
                    self._tactical_action_id(dict(value["action"])),
                ),
            )
        )
        selected = ranked[0]
        alternative = ranked[1]
        selected_id = self._tactical_action_id(dict(selected["action"]))
        alternative_id = self._tactical_action_id(dict(alternative["action"]))
        evidence: list[TacticalEvidenceV1] = []

        margin_gap = float(selected.get("expectedMargin", 0.0)) - float(
            alternative.get("expectedMargin", 0.0)
        )
        disagreement = max(
            0.0,
            min(
                1.0,
                float(selected.get("bestShare", 0.0)),
                float(alternative.get("bestShare", 0.0)),
            ),
        )
        if (
            abs(margin_gap) <= profile.debate_margin_gap
            and disagreement >= profile.debate_disagreement
        ):
            criteria = (
                ("hiddenDealDisagreement", disagreement),
                ("marginGap", margin_gap),
                ("maximumMarginGap", profile.debate_margin_gap),
                ("minimumDisagreement", profile.debate_disagreement),
            )
            evidence.append(
                TacticalEvidenceV1(
                    TacticalKind.DEBATE,
                    selected_id,
                    (alternative_id,),
                    disagreement * 100.0,
                    0,
                    ("paired_hidden_deal_disagreement",),
                    criteria,
                    canonical_sha256(
                        {
                            "analysisHash": item.sha256,
                            "kind": TacticalKind.DEBATE.value,
                            "criteria": dict(criteria),
                        }
                    ),
                )
            )

        if record.decision_kind == "bid":
            contract_gain = float(
                selected.get("contractSuccessProbability", 0.0)
            ) - float(alternative.get("contractSuccessProbability", 0.0))
            if (
                margin_gap >= profile.project_consequence_delta
                or contract_gain >= profile.contract_probability_delta
            ):
                criteria = (
                    ("contractProbabilityGain", contract_gain),
                    ("expectedMarginGain", margin_gap),
                    (
                        "minimumContractProbabilityGain",
                        profile.contract_probability_delta,
                    ),
                    ("minimumExpectedMarginGain", profile.project_consequence_delta),
                )
                evidence.append(
                    TacticalEvidenceV1(
                        TacticalKind.PROJECT_BIDDING,
                        selected_id,
                        (alternative_id,),
                        min(100.0, max(margin_gap * 5.0, contract_gain * 100.0)),
                        0,
                        ("purchase_or_contract_choice",),
                        criteria,
                        canonical_sha256(
                            {
                                "analysisHash": item.sha256,
                                "kind": TacticalKind.PROJECT_BIDDING.value,
                                "criteria": dict(criteria),
                            }
                        ),
                    )
                )

        return tuple(
            {
                **row.to_dict(),
                "truthMode": TruthMode.ACTOR_OBSERVER.value,
                "evidenceBasis": "paired_observer_safe_alternatives",
            }
            for row in evidence
        )

    def _qualification(
        self,
        recipe: ResearchRecipe,
        audit: Mapping[str, Any],
        analysis: TraceAnalysisResultV1,
        *,
        context: ResearchExecutionContextV1 | None = None,
    ) -> dict[str, Any]:
        decisions = list(audit["decisions"])
        non_forced = [item for item in decisions if not item["forced"]]
        minimum_truth = recipe.qualification["minimumTruth"]
        minimum_confidence = float(recipe.qualification["minimumConfidence"])
        maximum_regret = float(recipe.qualification["maximumRegret"])
        failures: list[str] = []
        compiled_domain = (
            context.compiled_domain if context is not None else self.compiled_domain
        )
        compiler_coverage = (
            compiled_domain.compiled.coverage if compiled_domain is not None else None
        )
        if (
            compiler_coverage is None
            or not compiler_coverage.executable_binding_complete
        ):
            failures.append("domain_executable_binding_coverage_incomplete")
        if (
            compiler_coverage is None
            or not compiler_coverage.behavioral_or_invariant_complete
        ):
            failures.append("domain_behavioral_or_invariant_coverage_unproven")
        if recipe.qualification["requireAllDecisionAudit"] and not audit["complete"]:
            failures.append("incomplete_decision_audit")
        if recipe.qualification["requireObserverSafety"] and not bool(
            audit.get("informationSetComplete")
        ):
            failures.append("observer_information_proof_incomplete")
        if recipe.qualification["requireAllDecisionAudit"] and not bool(
            audit.get("sequentialResponseComplete")
        ):
            failures.append("sequential_response_proof_incomplete")
        failures.extend(decision_consistency_qualification_failures(audit))
        if any(
            truth_rank(item["truthLabel"]) < truth_rank(minimum_truth)
            for item in non_forced
        ):
            failures.append("truth_below_required_tier")
        if any(float(item["confidence"]) < minimum_confidence for item in non_forced):
            failures.append("confidence_below_gate")
        if any(float(item["expectedRegret"]) > maximum_regret for item in non_forced):
            failures.append("surrounding_agent_regret")
        qualities = [
            max(
                0.0,
                100.0
                - float(item["expectedRegret"]) * 5.0
                - float(item["probabilityDrop"]) * 100.0,
            )
            for item in non_forced
        ]
        minimum_surrounding_quality = min(qualities, default=100.0)
        if minimum_surrounding_quality < float(
            recipe.qualification["minimumSurroundingQuality"]
        ):
            failures.append("surrounding_quality_below_gate")
        laffa = analysis.laffa_evidence
        competence_only, unresolved = _post_trace_featured_truth_status_ids_v1(analysis)
        featured_truth_blocked_ids = competence_only | unresolved
        analysis_critical_ids = {str(item) for item in laffa.critical_decision_ids}
        featured_truth_eligible_critical_ids = (
            analysis_critical_ids - featured_truth_blocked_ids
        )
        if not featured_truth_eligible_critical_ids:
            failures.append("no_verified_critical_point")
        tactical_rows = [
            {
                **dict(evidence),
                "actingSeat": decision["actingSeat"],
                "decisionId": decision["decisionId"],
            }
            for decision in decisions
            if str(decision["decisionId"]) not in featured_truth_blocked_ids
            for evidence in decision.get("tacticalEvidence") or ()
        ]
        tactic_scores: dict[str, float] = {}
        for row in tactical_rows:
            kind = str(row["kind"])
            tactic_scores[kind] = max(
                tactic_scores.get(kind, 0.0), float(row.get("magnitude", 0.0))
            )
        tactic_evaluations = _tactical_requirement_evaluations(
            recipe.tactics,
            tactical_rows,
        )
        for evaluation in tactic_evaluations:
            if evaluation["mode"] not in {"hard", "qualification_gate"}:
                continue
            if not evaluation["eligible"]:
                failures.append(f"tactical_gate_not_proven:{evaluation['kind']}")
        qualified = not failures
        confidence = min(
            (float(item["confidence"]) for item in non_forced), default=1.0
        )
        truth = min(
            (str(item["truthLabel"]) for item in non_forced),
            key=truth_rank,
            default="exact_optimal",
        )
        return {
            "qualified": qualified,
            "failures": failures,
            "truthLabel": truth if qualified else ("rejected" if failures else truth),
            "confidence": confidence,
            "minimumSurroundingQuality": minimum_surrounding_quality,
            "analysisCriticalDecisionIds": sorted(analysis_critical_ids),
            "featuredTruthEligibleCriticalDecisionIds": sorted(
                featured_truth_eligible_critical_ids
            ),
            "featuredTruthBlockedDecisionIds": sorted(featured_truth_blocked_ids),
            "tacticScores": tactic_scores,
            "tacticEvaluations": tactic_evaluations,
            "tacticalEvidenceCount": len(tactical_rows),
        }

    @staticmethod
    def _ortools_terminal_status_v2(result: Any) -> str:
        status = str(getattr(result, "status", ""))
        if status == "qualified":
            return "satisfied"
        if status == "infeasible":
            return "structurally_infeasible"
        if status == "not_found":
            return "not_found_within_budget"
        if status in {"feasible", "semantic_postcondition_miss"}:
            return "semantic_postcondition_miss"
        if status == "proof_failed":
            return "proof_failed"
        if status == "incomplete":
            return (
                "proof_incomplete"
                if getattr(result, "candidate", None) is not None
                or (getattr(result, "execution_receipt", {}) or {}).get("candidateAttemptReason") == "counterstrategy_budget"
                else "not_found_within_budget"
            )
        raise ValueError("OR-Tools Research returned an unknown terminal status")

    @staticmethod
    def _ortools_campaign_stop_reason_v2(
        campaign: Any,
        coordinator_receipt: Mapping[str, Any],
    ) -> str:
        if campaign.status == "completed":
            return "qualified_target_satisfied"
        if campaign.status == "structurally_infeasible":
            return "exact_structural_infeasibility"
        coordinator_status = str(coordinator_receipt.get("status") or "")
        if coordinator_status == "budget_exhausted":
            return "global_campaign_timeout"
        if coordinator_status == "cancelled":
            return "cancelled"
        if any(
            execution.status == "incomplete"
            and execution.proof is not None
            and execution.proof.get("completionStatus") == "proof_incomplete"
            for execution in campaign.executions
        ):
            return "proof_incomplete"
        if any(
            str(attempt.get("status") or "") == "failed"
            for attempt in coordinator_receipt.get("attempts", ())
            if isinstance(attempt, Mapping)
        ):
            return "worker_failed"
        if campaign.status == "incomplete":
            return "resource_incomplete"
        if coordinator_status == "not_found_within_budget":
            return "candidate_range_exhausted"
        return str(campaign.status)

    @staticmethod
    def _ortools_execution_parameters_v2(
        recipe: ResearchRecipeV4,
        backend: OrToolsScenarioResearchBackendV2,
        *,
        enforce_request_limits: bool = True,
    ) -> dict[str, Any]:
        profile_id = recipe.proof.budget_profile
        proof_profiles = dict(backend.authority.proof_budgets.get("profiles") or {})
        search_profiles = dict(backend.authority.search_budgets.get("profiles") or {})
        proof_budget = dict(proof_profiles.get(profile_id) or {})
        search_budget = dict(search_profiles.get(profile_id) or {})
        if not proof_budget or not search_budget:
            raise ValueError("OR-Tools Research budget profile is unavailable")
        if (
            proof_budget.get("maximumWorkers") != 1
            or search_budget.get("maximumWorkers") != 1
        ):
            raise ValueError("OR-Tools Research execution must use one solver worker")
        maximum_solve_seconds = min(
            float(proof_budget["maximumWallSeconds"]),
            float(recipe.resources["solverTimeoutSeconds"]),
        )
        maximum_campaign_seconds = float(search_budget["maximumWallSeconds"])
        maximum_candidates = (
            1
            if recipe.scenario.origin.kind == "exact_trace_prefix"
            else int(recipe.resources["seedCount"])
        )
        candidate_limit = int(search_budget["maximumCandidates"])
        variant_limit = int(search_budget["maximumVariants"])
        if enforce_request_limits and maximum_candidates > candidate_limit:
            raise ValueError("requested seed count exceeds frozen candidate budget")
        if enforce_request_limits and int(recipe.output["resultCount"]) > variant_limit:
            raise ValueError("requested result count exceeds frozen variant budget")
        maximum_shard_workers = int(search_budget["maximumShardWorkers"])
        requested_shard_workers = int(recipe.resources.get("shardWorkerCount", maximum_shard_workers))
        if enforce_request_limits and requested_shard_workers > maximum_shard_workers:
            raise ValueError("requested shard workers exceed frozen search budget")
        maximum_line_count_nodes = int(proof_budget["maximumLineCountNodes"])
        maximum_proof_nodes = int(proof_budget["maximumNodes"])
        if recipe.proof.line_universe is not None:
            maximum_proof_nodes = min(
                maximum_proof_nodes,
                maximum_line_count_nodes,
            )
        maximum_line_count_limbs = int(
            backend.authority.refusal_thresholds["maximumLineCountLimbs"]
        )
        diversity = dict(backend.authority.diversity_defaults)
        if diversity.get("symmetryPolicy") != "rules_preserving_v1":
            raise ValueError("OR-Tools Research symmetry policy is unsupported")
        if diversity.get("allowSymmetryOnlyVariants") is not False:
            raise ValueError("OR-Tools Research symmetry-only variants are forbidden")
        minimum_distance = diversity.get("minimumDecisionStateHammingDistance")
        if type(minimum_distance) is not int or minimum_distance < 1:
            raise ValueError(
                "OR-Tools Research minimum decision-state distance must be positive"
            )
        minimum_principal_variation_distance = diversity.get(
            "minimumPrincipalVariationActionDistance"
        )
        if (
            type(minimum_principal_variation_distance) is not int
            or minimum_principal_variation_distance < 1
        ):
            raise ValueError(
                "OR-Tools Research minimum principal-variation distance must be positive"
            )
        return {
            "maximumProofNodes": maximum_proof_nodes,
            "maximumProofDepth": int(proof_budget["maximumDepth"]),
            "maximumHiddenWorlds": int(proof_budget["maximumHiddenWorlds"]),
            **({"policySearch": proof_budget["policySearch"]} if "policySearch" in proof_budget else {}),
            "maximumLineCountNodes": maximum_line_count_nodes,
            "maximumLineCountLimbs": maximum_line_count_limbs,
            "maximumCandidates": maximum_candidates,
            "candidateLimit": candidate_limit,
            "maximumConflicts": int(search_budget["maximumConflicts"]),
            "maximumVariants": variant_limit,
            "maximumPrefixLiftAttempts": int(
                search_budget["maximumPrefixLiftAttempts"]
            ),
            "maximumShardWorkers": maximum_shard_workers,
            "requestedShardWorkers": requested_shard_workers,
            "cancellationPollMilliseconds": int(
                search_budget["cancellationCheckIntervalMilliseconds"]
            ),
            "symmetryPolicyId": diversity["symmetryPolicy"],
            "allowSymmetryOnlyVariants": diversity["allowSymmetryOnlyVariants"],
            "minimumDecisionStateDistance": minimum_distance,
            **(
                {
                    "minimumPrincipalVariationActionDistance": (
                        minimum_principal_variation_distance
                    )
                }
                if backend.authority.backend == "ortools_cp_sat_v3"
                else {}
            ),
            "maximumSeconds": maximum_solve_seconds,
            "maximumSolveSeconds": maximum_solve_seconds,
            "maximumCampaignSeconds": maximum_campaign_seconds,
            "seed": int(recipe.resources["seedStart"]),
            "proofBudgetHash": scenario_sha256(proof_budget),
            "searchBudgetHash": scenario_sha256(search_budget),
        }

    def _execute(self, run_code: str) -> None:
        """Coordinate a bounded set of workers over durable seed shards."""
        try:
            run = self.store.get_run(run_code)
            if run is None or run["status"] not in {"queued", "running"}:
                return
            recipe, context = self._build_recipe_and_context(
                self.store.get_recipe(run["recipeCode"]) or {}
            )
            self._assert_run_execution_context(run, recipe, context)
            if isinstance(recipe, ResearchRecipeV4):
                ortools_backend = self._cached_ortools_scenario_backend_v2(context)
                if ortools_backend is not None:
                    self._execute_v4_ortools_run(
                        run,
                        recipe,
                        context,
                        ortools_backend,
                    )
                    return
                if recipe.scenario.origin.kind == "exact_trace_prefix":
                    self._execute_v4_exact_run(run, recipe, context)
                else:
                    self._execute_v4_search_run(run, recipe, context)
                return
            shards = self.store.list_shards(run_code)
            if not shards:
                specs = partition_research_shards(
                    run_code=run_code,
                    run_hash=str(run["runHash"]),
                    recipe_hash=recipe.sha256,
                    seed_start=int(recipe.resources["seedStart"]),
                    seed_count=int(recipe.resources["seedCount"]),
                    seeds_per_shard=int(recipe.resources["seedsPerShard"]),
                    maximum_attempts=self.maximum_shard_attempts,
                )
                shards = self.store.create_shards(
                    tuple(item.to_dict() for item in specs)
                )
            if run["status"] == "queued":
                started_at = str(run.get("startedAt") or _now())
                self.store.update_run(
                    run_code,
                    status="running",
                    progress=float(run.get("progress", 0.0)),
                    startedAt=started_at,
                    events=[
                        *run.get("events", []),
                        {"type": "started", "shardCount": len(shards), "at": _now()},
                    ][-200:],
                )
            worker_count = min(self.maximum_local_shard_workers, len(shards))
            for worker_index in range(1, worker_count):
                self.executor.submit(
                    self._shard_worker_loop,
                    run_code,
                    worker_index,
                    context,
                )
            self._shard_worker_loop(run_code, 0, context)
        except Exception as exc:
            current = self.store.get_run(run_code)
            if current is not None and current.get("status") != "cancelled":
                self.store.update_run(
                    run_code,
                    status="failed",
                    progress=float(current.get("progress", 0.0)),
                    error={"type": type(exc).__name__, "message": str(exc)},
                    events=[
                        *current.get("events", []),
                        {"type": "failed", "message": str(exc), "at": _now()},
                    ][-200:],
                )
        finally:
            current = self.store.get_run(run_code)
            if current is None or current.get("status") in {
                "completed",
                "failed",
                "cancelled",
            }:
                with self._lock:
                    self._scheduled_runs.discard(run_code)

    @staticmethod
    def _runtime_proposal_queue(rules, execution, parameters, *, campaign_started,
                                cancel_check, progress_callback, worker_cpu_callback=None):
        rows = ResearchFoundryServiceV1._runtime_proposal_stream(
            rules, execution, parameters, campaign_started=campaign_started,
            cancel_check=cancel_check, progress_callback=progress_callback,
            worker_cpu_callback=worker_cpu_callback)
        try:
            return dict(rows)
        finally:
            rows.close()

    @staticmethod
    def _runtime_proposal_stream(rules, execution, parameters, *, campaign_started,
                                 cancel_check, progress_callback, worker_cpu_callback=None):
        """Yield ordered witness data; callers must close an abandoned stream."""
        from .construction_pre_screen import (screen_runtime_proposals_v1,
            refine_runtime_proposal_v1, parallel_runtime_proposal_seeds)

        deadline = campaign_started + parameters["maximumCampaignSeconds"]
        screen_deadline = min(deadline, time.perf_counter() + parameters["maximumSeconds"])
        emitted = 0
        seed_start = parameters["seed"]
        attempt_seconds = execution.proposal_maximum_seconds or parameters["maximumSeconds"]

        def witness(seed, refined):
            # Estimates never cross the qualification boundary; only witness data does.
            data = {key: refined[key] for key in (
                "prefix", "prefixHash", "rulesHash", "policyHash", "playIndex", "actor", "action")}
            data["sourceHash"] = refined["sha256"]
            data["sourceSeed"] = seed
            data["sha256"] = canonical_sha256(data)
            return data

        if parameters.get("requestedShardWorkers", 1) > 1:
            rows = parallel_runtime_proposal_seeds(rules, execution, parameters,
                deadline=screen_deadline, cancel_check=cancel_check,
                worker_cpu_callback=worker_cpu_callback)
            try:
                for seed, proposal_count, refined_rows in rows:
                    progress_callback({"currentStage": "runtime_proposal", "screenedSeed": seed,
                                       "proposalCount": proposal_count, "proposalOnly": True})
                    for refined in refined_rows:
                        if cancel_check() or time.perf_counter() >= screen_deadline:
                            return
                        ordinal = seed_start + emitted
                        emitted += 1
                        yield ordinal, witness(seed, refined)
                        if emitted == parameters["maximumCandidates"]:
                            return
            finally:
                rows.close()
            return
        for seed in range(seed_start, seed_start + execution.proposal_seed_count):
            remaining = screen_deadline - time.perf_counter()
            if remaining <= 0 or cancel_check():
                break
            screened = screen_runtime_proposals_v1(rules, seed=seed,
                rollout_policy_id=execution.early_play_policy_id,
                selection_policy=execution.video_interestingness,
                maximum_nodes=parameters["maximumProofNodes"], maximum_seconds=min(remaining, attempt_seconds),
                cancel_check=cancel_check,
                proposal_play_indices=getattr(execution, "proposal_play_indices", None),
                accepted_lesson_library=getattr(execution, "accepted_lesson_library", None))
            progress_callback({"currentStage": "runtime_proposal", "screenedSeed": seed,
                               "proposalCount": len(screened["proposals"]), "proposalOnly": True})
            for proposal in screened["proposals"]:
                remaining = screen_deadline - time.perf_counter()
                if remaining <= 0 or cancel_check():
                    break
                refined = refine_runtime_proposal_v1(rules, proposal,
                    selection_policy=execution.video_interestingness,
                    maximum_nodes=parameters["maximumProofNodes"], maximum_seconds=min(remaining, attempt_seconds),
                    maximum_worlds=parameters["maximumHiddenWorlds"], cancel_check=cancel_check)
                if refined["status"] != "priority_candidate":
                    continue
                ordinal = seed_start + emitted
                emitted += 1
                yield ordinal, witness(seed, refined)
                if emitted == parameters["maximumCandidates"]:
                    return

    def _execute_v4_ortools_run(
        self,
        run: Mapping[str, Any],
        recipe: ResearchRecipeV4,
        context: ResearchExecutionContextV1 | None,
        backend: OrToolsScenarioResearchBackendV2,
    ) -> None:
        """Run one bundle-bound CP-SAT Research request without legacy qualification."""

        if context is None:
            raise ValueError("OR-Tools Research execution context is unavailable")
        if recipe.playing_quality is not None:
            backend = backend.for_run()
        run_code = str(run["runCode"])
        started_at = str(run.get("startedAt") or _now())
        run_events = list(run.get("events") or ())
        event_sequence = max(
            [
                len(run_events),
                int(run.get("eventSequence") or 0),
                *[
                    int(event["sequence"])
                    for event in run_events
                    if type(event.get("sequence")) is int
                ],
            ]
        )
        if run["status"] == "queued":
            event_sequence += 1
            run = self.store.update_run(
                run_code,
                status="running",
                progress=0.0,
                startedAt=started_at,
                eventSequence=event_sequence,
                events=[
                    *run_events,
                    {
                        "type": "started",
                        "executionKind": backend.identity.backend_id,
                        "sequence": event_sequence,
                        "at": _now(),
                    },
                ][-200:],
            )
        if self._cancel_check(run_code):
            return
        parameters = self._ortools_execution_parameters_v2(recipe, backend)
        family_target = getattr(backend.authority.quality_execution, "family_target", None)
        if recipe.generation is not None:
            backend._validate_generation_request(recipe.generation, recipe.scenario, recipe.playing_quality)
        if family_target is not None:
            from .family_catalog import validate_family_target_scenario_v1
            validate_family_target_scenario_v1(family_target, recipe.scenario, rules=context.bundle.rules,
                execution=backend.authority.quality_execution, playing_quality=recipe.playing_quality)
        stored_shards = self.store.list_shards(run_code)
        if stored_shards:
            raise RuntimeError(
                "OR-Tools V4 uses immutable campaign receipts, not durable leased shards"
            )
        exact_input = recipe.scenario.origin.kind == "exact_trace_prefix"
        shards = partition_research_shards(
            run_code=run_code,
            run_hash=str(run["runHash"]),
            recipe_hash=recipe.sha256,
            seed_start=int(recipe.resources["seedStart"]),
            seed_count=(1 if exact_input else int(recipe.resources["seedCount"])),
            seeds_per_shard=(
                1 if exact_input else int(recipe.resources["seedsPerShard"])
            ),
            maximum_attempts=(1 if exact_input else self.maximum_shard_attempts),
        )
        campaign_started = time.perf_counter()
        progress_lock = threading.Lock()
        last_progress_at = 0.0
        terminal_attempts: dict[str, dict[str, Any]] = {}
        proof_retry_attempts: set[str] = set()
        candidate_receipts: dict[str, dict[str, Any]] = {}
        latest_stage_counters: dict[str, int] | None = None
        latest_quality_solver: dict[str, Any] | None = None
        attempted_by_worker: dict[tuple[str, int], int] = {}
        constructed_work: set[tuple[str, int, int]] = set()
        latest_work: dict[str, Any] = {}

        def publish_progress(evidence: Mapping[str, Any]) -> None:
            nonlocal event_sequence, last_progress_at, latest_stage_counters, latest_quality_solver
            value = copy.deepcopy(dict(evidence))
            if "incompleteCompilation" in value:
                value["incompleteCompilation"] = _operator_incomplete_compilation_v1(
                    value["incompleteCompilation"]
                )
            now = time.perf_counter()
            stage = str(value.get("currentStage") or "running")
            candidate_receipt = _operator_candidate_progress_receipt_v1(value)
            force = candidate_receipt is not None or stage in {
                "candidate_terminal",
                "variant_selection_terminal",
                "proof_retry_scheduled",
                "attempt_retry_scheduled",
                "transport_retry_scheduled",
            }
            with progress_lock:
                count = value.get("completedCandidates")
                if type(count) is int and count >= 0 and type(value.get("attempt", 0)) is int:
                    worker = (str(value.get("shardId", "campaign")), value.get("attempt", 0))
                    attempted_by_worker[worker] = max(attempted_by_worker.get(worker, 0), count)
                    if count and type(value.get("candidateHash")) is str:
                        constructed_work.add((*worker, count))
                latest_work.update(currentStage=stage)
                if type(value.get("candidateHash")) is str:
                    latest_work["candidateHash"] = value["candidateHash"]
                counters = _operator_stage_counters_v1(value.get("stageCounters"))
                if counters is not None:
                    latest_stage_counters = counters
                quality_solver = _operator_quality_solver_progress_v1(value.get("qualitySolver"))
                if quality_solver is not None:
                    latest_quality_solver = quality_solver
                elif stage in {"playingQuality", "candidate_terminal"}:
                    latest_quality_solver = None
                if not force and now - last_progress_at < 0.4:
                    return
                last_progress_at = now
                attempt_key = str(
                    candidate_receipt["receiptId"]
                    if candidate_receipt is not None
                    else (
                        f"{value.get('shardId', 'campaign')}:"
                        f"{value.get('attempt', 0)}:"
                        f"{value.get('executionOrdinal', 0)}"
                    )
                )
                if candidate_receipt is not None:
                    candidate_receipts[candidate_receipt["receiptId"]] = (
                        candidate_receipt
                    )
                if stage == "candidate_terminal":
                    terminal_attempts[attempt_key] = value
                elif stage in {"proof_retry_scheduled", "attempt_retry_scheduled"}:
                    proof_retry_attempts.add(attempt_key)
                elapsed = max(0.0, now - campaign_started)
                campaign_limit = float(parameters["maximumCampaignSeconds"])
                statuses = [
                    str(item.get("candidateStatus") or "")
                    for item in terminal_attempts.values()
                ]
                latest_terminal = next(reversed(terminal_attempts.values()), {})
                candidate_attempts = sum(
                    int(item.get("completedCandidates") or 0)
                    for item in terminal_attempts.values()
                )
                constructed_hashes = {
                    str(item["candidateHash"])
                    for item in terminal_attempts.values()
                    if item.get("candidateHash")
                }
                accounting = {
                    "schemaVersion": 1,
                    "state": "running",
                    "requestedCandidates": int(parameters["maximumCandidates"]),
                    "requestedQualifiedResults": int(recipe.output["resultCount"]),
                    "candidateAttempts": max(candidate_attempts, sum(attempted_by_worker.values())),
                    "constructedCandidates": max(len(constructed_hashes), len(constructed_work)),
                    "structurallyInfeasible": statuses.count("infeasible"),
                    "semanticPostconditionMiss": statuses.count(
                        "semantic_postcondition_miss"
                    ),
                    "proofFalse": statuses.count("proof_failed"),
                    "proofIncomplete": statuses.count("incomplete"),
                    "deepProofRetries": len(proof_retry_attempts),
                    "provisionalQualified": sum(
                        int(item.get("qualifiedCandidates") or 0)
                        for item in terminal_attempts.values()
                    ),
                    "provisionalResultHashes": sorted(
                        {
                            str(item["candidateHash"])
                            for item in terminal_attempts.values()
                            if item.get("candidateStatus") == "qualified"
                            and item.get("candidateHash")
                        }
                    ),
                    "provisionalResults": sorted(
                        (
                            copy.deepcopy(dict(item["candidateSummary"]))
                            for item in terminal_attempts.values()
                            if item.get("candidateStatus") == "qualified"
                            and isinstance(item.get("candidateSummary"), Mapping)
                        ),
                        key=lambda item: str(item.get("candidateHash") or ""),
                    ),
                    "qualified": 0,
                    "latestCandidateHash": latest_terminal.get("candidateHash") or latest_work.get("candidateHash"),
                    "latestCandidateStatus": latest_terminal.get("candidateStatus"),
                    "latestIncompleteCompilation": copy.deepcopy(
                        latest_terminal.get("incompleteCompilation")
                    ),
                    "currentSolverCounters": copy.deepcopy(latest_stage_counters),
                    "qualitySolver": copy.deepcopy(latest_quality_solver),
                    "currentStage": stage,
                    "elapsedSeconds": elapsed,
                    "remainingCampaignSeconds": max(0.0, campaign_limit - elapsed),
                    "maximumSolveSeconds": float(parameters["maximumSolveSeconds"]),
                    "maximumCampaignSeconds": campaign_limit,
                    "stopReason": None,
                    **_operator_candidate_progress_snapshot_v1(
                        candidate_receipts.values()
                    ),
                }
                current_progress_run = self.store.get_run(run_code)
                if current_progress_run is None or current_progress_run.get(
                    "status"
                ) not in {"queued", "running"}:
                    return
                event_sequence += 1
                event = {
                    "type": "progress",
                    **value,
                    "accounting": copy.deepcopy(accounting),
                    "sequence": event_sequence,
                    "at": _now(),
                }
                if candidate_receipt is not None:
                    event["candidateReceipt"] = copy.deepcopy(candidate_receipt)
                # The single receipt is carried by the event; the bounded history
                # lives on campaignProgress to avoid quadratic event payload growth.
                event["accounting"].pop("candidateReceipts", None)
                self.store.update_run(
                    run_code,
                    status="running",
                    progress=min(0.99, elapsed / campaign_limit),
                    campaignProgress=accounting,
                    eventSequence=event_sequence,
                    events=[*current_progress_run.get("events", []), event][-200:],
                )

        proposal_options = {}
        producer_options = {}
        proposals = {}
        construction_sources = {}
        streaming_proposals = False
        available_workers = min(
            parameters["requestedShardWorkers"], self.maximum_local_shard_workers,
        )
        campaign_workers = min(available_workers, len(shards))
        proposal_wall_seconds = proposal_cpu_seconds = 0.0
        native_campaign_seconds = parameters["maximumCampaignSeconds"]
        quality_execution = backend.authority.quality_execution
        if (quality_execution is not None and quality_execution.candidate_source is not None
                and not exact_input and recipe.scenario.decision_point["kind"] == "play"):
            if recipe.playing_quality is None:
                raise ValueError("runtime proposals require a search play recipe with playing quality")
            from .ortools_research_executor import validate_content_interest_rules_v1
            validate_content_interest_rules_v1(context.bundle.rules, recipe.playing_quality, quality_execution)
            proposal_started = time.perf_counter()
            proposal_cpu_started = time.thread_time()
            child_cpu_seconds = 0.0

            def report_proposal_cpu(seconds):
                nonlocal child_cpu_seconds
                child_cpu_seconds += seconds

            streaming_proposals = (quality_execution.candidate_source == "runtime_proposal_stream_v1"
                                   and available_workers > 1)
            screen_parameters = dict(parameters)
            screen_parameters["requestedShardWorkers"] = available_workers
            if streaming_proposals:
                screen_workers = max(1, available_workers // 3)
                screen_parameters["requestedShardWorkers"] = screen_workers
                campaign_workers = available_workers - screen_workers
            proposal_producer = self._runtime_proposal_stream if streaming_proposals else self._runtime_proposal_queue
            if family_target is not None:
                producer_options['scenario'] = recipe.scenario
                producer_options['playing_quality'] = recipe.playing_quality
                if getattr(quality_execution, 'family_proposal_mode', None) is not None:
                    from .ortools_tail_first_candidate import tail_first_proposal_stream_v1
                    proposal_producer = tail_first_proposal_stream_v1
                    producer_options.update(request=recipe.proof,
                        proof={'maximumNodes': parameters['maximumProofNodes'],
                               'maximumDepth': parameters['maximumProofDepth']},
                        search={'maximumPrefixLiftAttempts': parameters['maximumPrefixLiftAttempts']})
                else:
                    from .family_proposal_stream import family_proposal_stream_v1
                    proposal_producer = family_proposal_stream_v1
                    producer_options['auction_provider'] = backend.qualified_auction_provider(
                        context.bundle.rules, recipe.playing_quality)
                streaming_proposals = True
                # Native construction is serial; retain the existing bounded proof pool.
                campaign_workers = max(1, available_workers - 1)
            elif recipe.generation is not None:
                from .family_proposal_stream import family_query_proposal_stream_v1
                proposal_producer = family_query_proposal_stream_v1
                producer_options.update(generation=recipe.generation, scenario=recipe.scenario,
                    playing_quality=recipe.playing_quality,
                    auction_provider=backend.qualified_auction_provider(context.bundle.rules, recipe.playing_quality))
                streaming_proposals = True
                campaign_workers = max(1, available_workers - 1)
            def publish_proposal_progress(value):
                value = dict(value)
                source = value.pop('constructionSource', None)
                if source is not None:
                    # Full construction lines are private evidence, never live public progress.
                    digest = scenario_sha256(source)
                    construction_sources[digest] = self.artifact_store.put(scenario_json_bytes(source),
                        media_type='application/vnd.balootx.research-construction-source-v1+json',
                        name=f'{digest}.json')
                publish_progress(value)

            rows = proposal_producer(
                context.bundle.rules, quality_execution, screen_parameters,
                campaign_started=campaign_started,
                cancel_check=lambda: self._cancel_check(run_code),
                progress_callback=publish_proposal_progress,
                worker_cpu_callback=report_proposal_cpu,
                **producer_options,
            )
            if streaming_proposals:
                def recorded_proposals():
                    try:
                        for ordinal, witness in rows:
                            proposals[ordinal] = witness
                            yield ordinal, witness
                    finally:
                        rows.close()
                proposal_options["runtime_proposal_stream"] = recorded_proposals()
            else:
                proposals = rows
                proposal_wall_seconds = max(0.0, time.perf_counter() - proposal_started)
                proposal_cpu_seconds = max(0.0, time.thread_time() - proposal_cpu_started) + child_cpu_seconds
                proposal_options["runtime_proposals"] = proposals
            # Proposal work and native proof share the original campaign deadline.
            native_campaign_seconds = max(0.0,
                parameters["maximumCampaignSeconds"] - (time.perf_counter() - campaign_started))
        campaign = backend.execute_campaign(
            context.bundle.rules,
            recipe.scenario,
            recipe.proof,
            playing_quality=recipe.playing_quality,
            **({"generation": recipe.generation} if recipe.generation is not None else {}),
            result_count=int(recipe.output["resultCount"]),
            symmetry_policy_id=parameters["symmetryPolicyId"],
            allow_symmetry_only_variants=parameters["allowSymmetryOnlyVariants"],
            minimum_decision_state_distance=parameters["minimumDecisionStateDistance"],
            minimum_principal_variation_action_distance=parameters.get(
                "minimumPrincipalVariationActionDistance"
            ),
            maximum_proof_nodes=parameters["maximumProofNodes"],
            maximum_proof_depth=parameters["maximumProofDepth"],
            maximum_hidden_worlds=parameters["maximumHiddenWorlds"],
            policy_search=parameters.get("policySearch"),
            maximum_line_count_limbs=parameters["maximumLineCountLimbs"],
            maximum_conflicts=parameters["maximumConflicts"],
            maximum_prefix_lift_attempts=parameters["maximumPrefixLiftAttempts"],
            seed=parameters["seed"],
            maximum_candidates=parameters["maximumCandidates"],
            maximum_seconds=parameters["maximumSeconds"],
            maximum_campaign_seconds=native_campaign_seconds,
            shards=shards,
            shard_worker_count=campaign_workers,
            cancellation_poll_milliseconds=parameters["cancellationPollMilliseconds"],
            cancel_check=lambda: self._cancel_check(run_code),
            progress_callback=publish_progress,
            adjudication_authority=(
                context.bundle.component("adjudication")
                if "adjudication" in context.bundle.components
                else None
            ),
            **proposal_options,
        )
        if proposal_options:
            if streaming_proposals:
                # Parent screening CPU is already included by the overlapping campaign.
                proposal_cpu_seconds = child_cpu_seconds
            campaign = replace(campaign,
                wall_time_seconds=campaign.wall_time_seconds + proposal_wall_seconds,
                cpu_seconds=campaign.cpu_seconds + proposal_cpu_seconds)
        execution_references = [
            self.artifact_store.put(
                scenario_json_bytes(execution.to_dict()),
                media_type="application/vnd.balootx.ortools-research-execution-v1+json",
                name=f"{execution.sha256}.json",
            )
            for execution in campaign.executions
        ]
        proof_boundaries = _ortools_line_count_boundaries_v1(
            campaign,
            recipe,
            parameters,
        )
        campaign_document = {
            **campaign.to_dict(),
            **({"runtimeProposalQueue": list(proposals.values())}
               if proposal_options else {}),
            **({'proposalConstructionEvidence': construction_sources} if construction_sources else {}),
            "executionEvidence": [dict(item) for item in execution_references],
            "proofBoundaries": copy.deepcopy(proof_boundaries),
        }
        evidence_reference = self.artifact_store.put(
            scenario_json_bytes(campaign_document),
            media_type="application/vnd.balootx.ortools-research-campaign-v1+json",
            name=f"{campaign.sha256}.json",
        )
        current = self.store.get_run(run_code)
        if current is None:
            raise KeyError(run_code)
        if current.get("status") == "cancelled":
            campaign_progress = copy.deepcopy(
                dict(current.get("campaignProgress") or {})
            )
            self.store.update_run(
                run_code,
                campaignProgress={
                    **campaign_progress,
                    **_operator_candidate_progress_snapshot_v1(candidate_receipts.values()),
                    "proofIncomplete": max(int(campaign_progress.get("proofIncomplete", 0)),
                        sum(item.get("candidateStatus") == "incomplete"
                            for item in terminal_attempts.values())),
                    "state": "cancelled",
                    "stopReason": "cancelled",
                },
                scenarioResearchEvidence={
                    **copy.deepcopy(dict(evidence_reference)),
                    "proofBoundaries": copy.deepcopy(proof_boundaries),
                },
            )
            return
        semantic_status = {
            "completed": "satisfied",
            "structurally_infeasible": "structurally_infeasible",
            "not_found_within_budget": "not_found_within_budget",
            "semantic_postcondition_miss": "semantic_postcondition_miss",
            "proof_failed": "proof_failed",
            "incomplete": "proof_incomplete",
        }[campaign.status]
        if campaign.status == "incomplete" and campaign.executions and all(
            execution.status in {"incomplete", "not_found"}
            and self._ortools_terminal_status_v2(execution) == "not_found_within_budget"
            for execution in campaign.executions
        ):
            semantic_status = "not_found_within_budget"
        result_codes: list[str] = []
        audit_result_codes: list[str] = []
        results: list[dict[str, Any]] = []
        configuration_snapshot = context.bundle.configuration_snapshot()
        lowering_manifest = backend.lowering_manifest.catalog()

        def persist_execution(
            execution: Any, *, terminal_status: str
        ) -> dict[str, Any]:
            execution_payload: Mapping[str, Any] | Any = execution
            try:
                documents = execution.semantic_artifact_documents()
            except RuntimeError:
                candidate_document = (
                    execution.candidate.to_dict()
                    if getattr(execution, "candidate", None) is not None
                    else None
                )
                stage_receipts = dict(
                    (candidate_document or {}).get("stageReceipts") or {}
                )
                documents = {
                    "constructionModel": {
                        "schemaVersion": 1,
                        "status": (
                            "candidate_constructed"
                            if candidate_document is not None
                            else "not_constructed"
                        ),
                        "preflightHash": execution.preflight.sha256,
                        "compiledStages": list(execution.preflight.compiled_stages),
                        "stageModelHashes": {
                            stage: dict(receipt).get("modelHash")
                            for stage, receipt in sorted(stage_receipts.items())
                        },
                        "namedCore": list(
                            getattr(execution.predicate_evaluation, "named_core", ())
                            if execution.predicate_evaluation is not None
                            else ()
                        ),
                    },
                    "proofModel": {
                        "schemaVersion": 1,
                        "status": "not_compiled",
                        "reason": execution.status,
                    },
                    "proofGraph": {
                        "schemaVersion": 1,
                        "status": "graph_not_compiled",
                        "reason": execution.status,
                    },
                    "trace": (
                        copy.deepcopy(dict(candidate_document["trace"]))
                        if candidate_document is not None
                        else {
                            "schemaVersion": 1,
                            "status": "trace_not_available",
                            "reason": execution.status,
                            "playable": False,
                        }
                    ),
                    "proofOutcome": {
                        "schemaVersion": 1,
                        "status": "proof_not_run",
                        "reason": execution.status,
                    },
                }
                authority = ResearchAuthorityV1.build(
                    backend_id=backend.identity.backend_id,
                    backend_version=backend.identity.backend_version,
                    solver_version=backend.identity.solver_version,
                    bundle_revision_id=context.bundle_revision_id,
                    bundle_hash=context.bundle.sha256,
                    domain_ir_hashes={
                        "auction": AuctionDomainIRV1.from_rules(
                            context.bundle.rules.data
                        ).sha256,
                        "escalation": EscalationDomainIRV1.from_rules(
                            context.bundle.rules.data
                        ).sha256,
                        "play": context.bundle.rules.play_domain.sha256,
                        "projects": context.bundle.rules.project_domain.sha256,
                        "scoring": ScoringDomainIRV1.from_rules(
                            context.bundle.rules.data
                        ).sha256,
                        "structural": context.bundle.rules.structural_domain.sha256,
                    },
                    lowering_manifest_hash=execution.preflight.lowering_manifest_hash,
                    construction_model_hash=scenario_sha256(
                        documents["constructionModel"]
                    ),
                    proof_model_hash=scenario_sha256(documents["proofModel"]),
                    graph_hash=scenario_sha256(documents["proofGraph"]),
                    trace_hash=scenario_sha256(documents["trace"]),
                    proof_outcome_hash=scenario_sha256(documents["proofOutcome"]),
                ).to_dict()
                execution_payload = {
                    **execution.to_dict(),
                    "researchAuthority": authority,
                }
            artifact_documents = {
                "constructionModel": documents["constructionModel"],
                "proofModel": documents["proofModel"],
                "proofGraph": documents["proofGraph"],
                "completeTrace": documents["trace"],
                "proofOutcome": documents["proofOutcome"],
            }
            artifact_references = {
                name: self.artifact_store.put(
                    scenario_json_bytes(document),
                    media_type=ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES[name],
                    name=f"{scenario_sha256(document)}.json",
                )
                for name, document in artifact_documents.items()
            }
            complete_trace = dict(artifact_documents["completeTrace"])
            policy_continuation = complete_trace.get("policyContinuation")
            all_passed = (
                isinstance(policy_continuation, Mapping)
                and policy_continuation.get("allPassed") is True
            )
            exact_state = complete_trace.get("completeExactNormalization")
            match_score_before = None
            project_actions = None
            if not isinstance(exact_state, Mapping):
                exact_state = complete_trace.get("exactNormalization")
            else:
                complete_prefix = complete_trace.get("completeExactTracePrefix")
                if isinstance(complete_prefix, Mapping):
                    match_score_before = complete_prefix.get("matchScore")
                    project_actions = complete_prefix.get("projectActions")

            def runtime_check(
                semantic_result_hash: str,
                trace_hash: str,
            ) -> Any:
                if not isinstance(exact_state, Mapping):
                    raise ValueError(
                        "OR-Tools Research trace lacks exact normalization"
                    )
                if all_passed:
                    return quarantine_runtime_compatibility_failure_v1(
                        semantic_result_hash=semantic_result_hash,
                        bundle_hash=context.bundle.sha256,
                        trace_hash=trace_hash,
                        failure_code="all_pass_research_continuation_nonplayable",
                        failure=RuntimeError(
                            "all-pass Research continuation has no playable round"
                        ),
                        engine_version=__version__,
                    )
                return check_runtime_compatibility_v1(
                    exact_state=exact_state,
                    semantic_result_hash=semantic_result_hash,
                    bundle_hash=context.bundle.sha256,
                    trace_hash=trace_hash,
                    rules=context.bundle.rules,
                    engine_version=__version__,
                    match_score_before=match_score_before,
                    project_actions=project_actions,
                )

            verified_artifacts = execution._quality_verified_artifacts
            builder = (_build_ortools_research_result_v3
                if verified_artifacts else build_ortools_research_result_v3)
            result = builder(
                run=current,
                recipe=recipe.to_dict(),
                executor_result=execution_payload,
                campaign_semantic_hash=campaign.semantic_hash,
                terminal_status=terminal_status,
                artifact_documents=artifact_documents,
                artifact_references=artifact_references,
                configuration_snapshot=configuration_snapshot,
                lowering_manifest=lowering_manifest,
                authority_artifact_writer=lambda payload,
                media_type,
                name: self.artifact_store.put(
                    payload, media_type=media_type, name=name
                ),
                runtime_compatibility_check=runtime_check,
                created_at=str(current["createdAt"]),
                **({"verified_artifacts": verified_artifacts} if verified_artifacts else {}),
            )
            self.store.put_result(result)
            return result

        for execution in campaign.accepted_results:
            result = persist_execution(execution, terminal_status="satisfied")
            results.append(result)
            result_codes.append(str(result["resultCode"]))
        if campaign.status != "completed" and campaign.executions:
            audit_execution = campaign.executions[-1]
            if campaign.status in {"incomplete", "proof_failed", "semantic_postcondition_miss"}:
                audit_execution = next(
                    (item for item in reversed(campaign.executions) if item.status == campaign.status),
                    audit_execution,
                )
            terminal_result = persist_execution(
                audit_execution, terminal_status=semantic_status
            )
            audit_result_codes.append(str(terminal_result["resultCode"]))
        retained_count = len(result_codes)
        diversity_selection = _select_v4_results(recipe, results)
        selected_count = len(diversity_selection.selected_result_codes)
        completed_candidates = int(campaign.to_dict()["completedCandidates"])
        # A cancelled worker may not return an execution. Preserve observed work
        # separately from immutable completed-execution/qualification receipts.
        attempted_candidates = max(completed_candidates, sum(attempted_by_worker.values()))
        constructed_candidates = len(
            {
                execution.candidate.sha256
                for execution in campaign.executions
                if execution.candidate is not None
            }
        )
        constructed_candidates = max(constructed_candidates, len(constructed_work))
        rejected_count = sum(
            execution.candidate is not None and execution.status != "qualified"
            for execution in campaign.executions
        )
        duplicate_qualified_count = (
            sum(execution.status == "qualified" for execution in campaign.executions)
            - retained_count
            + diversity_selection.duplicate_count
        )
        quarantine_count = sum(
            dict(result.get("runtimeCompatibility") or {}).get("status") != "passed"
            for result in results
        ) + sum(
            execution.status == "compiler_disagreement"
            for execution in campaign.executions
        )
        campaign_receipt = campaign.to_dict()
        coordinator_receipt = dict(campaign_receipt.get("coordinatorReceipt") or {})
        coordinator_attempts = list(coordinator_receipt.get("attempts") or [])
        deep_proof_retries = sum(
            int(item.get("attempt", 0)) > 1
            and any(
                prior.get("shardId") == item.get("shardId")
                and int(prior.get("attempt", 0)) == int(item.get("attempt", 0)) - 1
                and dict(prior.get("partialEvidence") or {}).get("proofRetryEligible")
                is True
                for prior in coordinator_attempts
            )
            for item in coordinator_attempts
        )
        campaign_limit = float(parameters["maximumCampaignSeconds"])
        campaign_accounting = {
            "schemaVersion": 1,
            "state": "completed",
            "requestedCandidates": int(parameters["maximumCandidates"]),
            "requestedQualifiedResults": int(recipe.output["resultCount"]),
            "candidateAttempts": attempted_candidates,
            "constructedCandidates": constructed_candidates,
            "structurallyInfeasible": sum(
                execution.status == "infeasible" for execution in campaign.executions
            ),
            "semanticPostconditionMiss": sum(
                execution.status == "semantic_postcondition_miss"
                for execution in campaign.executions
            ),
            "proofFalse": sum(
                execution.status == "proof_failed" for execution in campaign.executions
            ),
            "proofIncomplete": max(
                sum(execution.status == "incomplete" for execution in campaign.executions),
                sum(item.get("candidateStatus") == "incomplete"
                    for item in terminal_attempts.values()),
            ),
            "deepProofRetries": deep_proof_retries,
            "provisionalQualified": sum(
                execution.status == "qualified" for execution in campaign.executions
            ),
            "provisionalResultHashes": sorted(
                {
                    execution.candidate.sha256
                    for execution in campaign.executions
                    if execution.status == "qualified"
                    and execution.candidate is not None
                }
            ),
            "provisionalResults": sorted(
                (
                    summary
                    for execution in campaign.accepted_results
                    if (
                        summary := operator_candidate_progress_summary_v1(
                            execution,
                            decision_kind=str(recipe.scenario.decision_point["kind"]),
                        )
                    )
                    is not None
                ),
                key=lambda item: str(item.get("candidateHash") or ""),
            ),
            "qualified": retained_count,
            "latestCandidateHash": (
                campaign.executions[-1].candidate.sha256
                if campaign.executions and campaign.executions[-1].candidate is not None
                else latest_work.get("candidateHash")
            ),
            "latestCandidateStatus": (
                campaign.executions[-1].status if campaign.executions else None
            ),
            "latestIncompleteCompilation": next(
                (
                    _operator_incomplete_compilation_v1(
                        execution.execution_receipt["incompleteCompilation"]
                    )
                    for execution in reversed(campaign.executions)
                    if "incompleteCompilation" in execution.execution_receipt
                ),
                None,
            ),
            "currentStage": latest_work.get("currentStage"),
            "currentSolverCounters": copy.deepcopy(latest_stage_counters),
            "qualitySolver": copy.deepcopy(latest_quality_solver),
            "elapsedSeconds": float(campaign.wall_time_seconds),
            "remainingCampaignSeconds": max(
                0.0, campaign_limit - float(campaign.wall_time_seconds)
            ),
            "maximumSolveSeconds": float(parameters["maximumSolveSeconds"]),
            "maximumCampaignSeconds": campaign_limit,
            "stopReason": self._ortools_campaign_stop_reason_v2(
                campaign,
                coordinator_receipt,
            ),
            **_operator_candidate_progress_snapshot_v1(candidate_receipts.values()),
        }
        output_receipt = _v4_output_receipt(
            recipe,
            retained_count=retained_count,
            qualified_count=retained_count,
            selected_count=selected_count,
        )
        completed_at = _now()
        event_sequence += 1
        self.store.update_run(
            run_code,
            status="completed",
            progress=1.0,
            resultCount=retained_count,
            qualifiedCount=retained_count,
            selectedCount=selected_count,
            duplicateQualifiedCount=duplicate_qualified_count,
            rejectedCount=rejected_count,
            quarantineCount=quarantine_count,
            resultCodes=result_codes,
            auditResultCodes=audit_result_codes,
            auditResultCount=len(audit_result_codes),
            attemptedCandidates=attempted_candidates,
            scenarioResearchStatus=semantic_status,
            scenarioResearchEvidence={
                **copy.deepcopy(dict(evidence_reference)),
                "proofBoundaries": copy.deepcopy(proof_boundaries),
            },
            scenarioYieldCounts={
                "attempted": attempted_candidates,
                "structurallyFeasible": constructed_candidates,
                "legalReplay": 0,
                "semanticSatisfied": sum(
                    execution.predicate_evaluation is not None
                    and execution.predicate_evaluation.satisfied
                    for execution in campaign.executions
                ),
                "proofCompleted": sum(
                    execution.proof is not None
                    and execution.proof.get("completionStatus")
                    in {
                        "exact_complete",
                        "bounded_complete",
                        "policy_complete",
                        "proof_failed",
                    }
                    for execution in campaign.executions
                ),
                "qualified": retained_count,
            },
            performanceReceipt=copy.deepcopy(campaign_receipt),
            campaignProgress=copy.deepcopy(campaign_accounting),
            campaignAccounting=copy.deepcopy(campaign_accounting),
            diversitySelection=diversity_selection.to_dict(),
            outputReceipt=output_receipt,
            completedAt=completed_at,
            eventSequence=event_sequence,
            events=[
                *current.get("events", []),
                {
                    "type": "completed",
                    "sequence": event_sequence,
                    "scenarioResearchStatus": semantic_status,
                    "resultCount": retained_count,
                    "campaignAccounting": copy.deepcopy(campaign_accounting),
                    "quarantineCount": quarantine_count,
                    "compilerDisagreement": any(
                        execution.status == "compiler_disagreement"
                        for execution in campaign.executions
                    ),
                    "at": completed_at,
                },
            ][-200:],
        )

    def _put_v4_scenario_result(
        self,
        *,
        run: Mapping[str, Any],
        recipe: ResearchRecipeV4,
        normalized: Any,
        proof: Any,
        satisfaction: Any,
        construction_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        constraint_evidence = build_constraint_evidence_v1(
            recipe,
            normalized,
            satisfaction,
            construction_receipt=construction_receipt,
        )
        documents = {
            "exactState": normalized.to_dict(),
            "scenarioSatisfaction": satisfaction.to_dict(),
            "constraintEvidence": constraint_evidence,
            "proofOutcome": proof.to_dict(),
        }
        artifacts = {
            key: self.artifact_store.put(
                scenario_json_bytes(document),
                media_type=SCENARIO_ARTIFACT_MEDIA_TYPES[key],
                name=f"{document['sha256']}.json",
            )
            for key, document in documents.items()
        }
        result = build_scenario_result_v4(
            run=run,
            recipe=recipe,
            normalized=normalized,
            satisfaction=satisfaction,
            proof=proof,
            constraint_evidence=constraint_evidence,
            artifacts=artifacts,
            created_at=str(run["createdAt"]),
            generation_lane=(
                "exact_input"
                if recipe.scenario.origin.kind == "exact_trace_prefix"
                else str(recipe.resources["generationLane"])
            ),
        )
        result = validate_scenario_result_v4(
            result,
            artifact_store=self.artifact_store,
        )
        result = validate_scenario_result_authority_v1(
            result,
            run=run,
            recipe=recipe.to_dict(),
        )
        self.store.put_result(result)
        return result

    def _execute_v4_exact_run(
        self,
        run: Mapping[str, Any],
        recipe: ResearchRecipeV4,
        context: ResearchExecutionContextV1 | None,
    ) -> None:
        """Execute one immutable exact-prefix V4 scenario without legacy projections."""

        if context is None:
            raise ValueError("Research V4 exact execution context is unavailable")
        prefix = recipe.scenario.origin.exact_trace_prefix
        if prefix is None:
            raise ValueError(
                "Research V4 exact execution requires an exact trace prefix"
            )
        started_clock = time.perf_counter()
        started_cpu = time.thread_time()
        run_code = str(run["runCode"])
        started_at = str(run.get("startedAt") or _now())
        attempted_candidates = 0
        normalized = None
        proof = None
        satisfaction = None
        simulation_seconds: float | None = None
        analysis_seconds: float | None = None
        semantic_parity_seconds: float | None = None
        candidate_pipeline_seconds: float | None = None

        def persist_partial_receipt(reason: str) -> None:
            current_run = self.store.get_run(run_code)
            if current_run is None:
                raise KeyError(run_code)
            if current_run.get("status") != "cancelled":
                return
            proof_payload = dict(getattr(proof, "payload", {}) or {})
            proof_runtime = dict(proof_payload.get("resources") or {})
            hidden_worlds = dict(proof_payload.get("hiddenWorlds") or {})
            resource_values = {
                "nodes": proof_runtime.get("nodes"),
                "cacheEntries": proof_runtime.get("cacheEntries"),
                "cacheBytes": proof_runtime.get("cacheBytes"),
                "hiddenWorlds": hidden_worlds.get("count"),
            }
            validated_resources = {
                key: int(value) if type(value) is int and value >= 0 else None
                for key, value in resource_values.items()
            }
            proof_resource_availability = (
                "not_started"
                if proof is None
                else "complete"
                if all(value is not None for value in validated_resources.values())
                else "partial"
            )
            cache_scopes: dict[str, dict[str, int]] = {}
            execution_context_factory = getattr(self, "execution_context_factory", None)
            if execution_context_factory is not None:
                for snapshot in execution_context_factory.cache_metrics_snapshots():
                    cache_scopes[str(snapshot["scope"])] = {
                        key: int(snapshot[key])
                        for key in (
                            "hits",
                            "misses",
                            "evictions",
                            "entries",
                            "capacity",
                        )
                    }
            peak_process_rss = process_lifetime_peak_rss_bytes()
            funnel = {
                "attempted": attempted_candidates,
                "structurallyFeasible": 0,
                "legalReplay": int(normalized is not None),
                "semanticSatisfied": int(
                    satisfaction is not None and satisfaction.status == "satisfied"
                ),
                "proofCompleted": int(
                    proof is not None
                    and proof_payload.get("completionStatus") != "proof_incomplete"
                ),
                "qualified": 0,
            }
            partial_receipt = _v4_partial_performance_receipt(
                {
                    "runCode": run_code,
                    "executionKind": "exact_trace_prefix",
                    "generationLane": "exact_input",
                    "terminalStatus": "cancelled",
                    "terminalReason": reason,
                    "evidenceCompleteness": {
                        "counters": "complete_through_terminal_observation_v1",
                        "timings": "complete_through_terminal_observation_v1",
                        "results": "retained_results_only_v1",
                        "proofResources": proof_resource_availability,
                    },
                    "attemptedSeeds": 0,
                    "attemptedCandidates": attempted_candidates,
                    "structuralCandidates": 0,
                    "auctionFeasible": 0,
                    "solverOptimal": 0,
                    "solverIncomplete": 0,
                    "proofCompleted": funnel["proofCompleted"],
                    "resultCount": 0,
                    "qualifiedCount": 0,
                    "rejectedCount": 0,
                    "quarantineCount": 0,
                    "resultCodes": [],
                    "funnel": funnel,
                    "constructionSeconds": None,
                    "simulationSeconds": simulation_seconds,
                    "semanticParitySeconds": semantic_parity_seconds,
                    "analysisSeconds": analysis_seconds,
                    "tacticalVerificationSeconds": None,
                    "candidatePipelineSeconds": candidate_pipeline_seconds,
                    "stageTimingBoundary": {
                        "construction": "not_applicable_exact_input_v1",
                        "simulation": "authoritative_exact_prefix_normalization_and_replay_v1",
                        "semanticParity": "runtime_scenario_verifier_v1",
                        "analysis": "unified_proof_service_call_v1",
                        "tacticalVerification": "not_applicable_no_dedicated_stage_v1",
                        "candidatePipeline": "normalize_prove_verify_until_cancellation_v1",
                    },
                    "firstFeasibleCandidateSeconds": None,
                    "firstOptimalCandidateSeconds": None,
                    "firstQualifiedResultSeconds": None,
                    "shardWallWorkSeconds": time.perf_counter() - started_clock,
                    "cpuSeconds": time.thread_time() - started_cpu,
                    "cpuMeasurementBoundary": CALLING_THREAD_CPU_BOUNDARY,
                    "peakProcessRssBytes": peak_process_rss,
                    "peakRssMeasurementBoundary": (
                        PROCESS_PEAK_RSS_BOUNDARY
                        if peak_process_rss is not None
                        else None
                    ),
                    "environment": build_research_environment_declaration(
                        generation_lane="exact_input",
                        configured_shard_workers=int(
                            getattr(self, "maximum_local_shard_workers", 1)
                        ),
                        effective_shard_workers=1,
                        cp_sat_workers_per_solve=None,
                        research_store=self.store,
                        artifact_store=self.artifact_store,
                    ),
                    "cacheScopes": cache_scopes,
                    "proofResources": {
                        **validated_resources,
                        "aggregation": (
                            "not_started_v1"
                            if proof is None
                            else "single_proof_invocation_v1"
                        ),
                        "availability": proof_resource_availability,
                    },
                    "cancellationObserved": True,
                }
            )
            self.store.update_run(
                run_code,
                partialPerformanceReceipt=copy.deepcopy(partial_receipt),
                events=[
                    *current_run.get("events", []),
                    {
                        "type": "partial_performance_receipt",
                        "terminalReason": reason,
                        "performanceReceiptHash": partial_receipt["sha256"],
                        "at": _now(),
                    },
                ][-200:],
            )

        if run["status"] == "queued":
            run = self.store.update_run(
                run_code,
                status="running",
                progress=0.0,
                startedAt=started_at,
                events=[
                    *run.get("events", []),
                    {
                        "type": "started",
                        "executionKind": "exact_trace_prefix",
                        "at": _now(),
                    },
                ][-200:],
            )
        if self._cancel_check(run_code):
            persist_partial_receipt("cancelled_before_candidate")
            return
        attempted_candidates = 1
        candidate_started = time.perf_counter()
        normalization_started = time.perf_counter()
        normalized = ExactStateNormalizerV1(context.bundle.rules).normalize(
            prefix,
            decision_point=recipe.scenario.decision_point,
        )
        simulation_seconds = time.perf_counter() - normalization_started
        analysis_started = time.perf_counter()
        proof = UnifiedScenarioProofServiceV1.from_default_authority().prove(
            normalized,
            recipe.proof,
            bundle_hash=str(context.bundle.sha256),
            cancel_check=lambda: self._cancel_check(run_code),
        )
        analysis_seconds = time.perf_counter() - analysis_started
        semantic_verification_started = time.perf_counter()
        satisfaction = ScenarioRuntimeVerifierV1().verify(
            recipe.scenario,
            normalized,
            proof_outcome=proof,
        )
        semantic_parity_seconds = time.perf_counter() - semantic_verification_started
        candidate_pipeline_seconds = time.perf_counter() - candidate_started
        current = self.store.get_run(run_code)
        if current is None:
            raise KeyError(run_code)
        if current.get("status") == "cancelled":
            persist_partial_receipt("cancelled_after_proof")
            return
        result = ResearchFoundryServiceV1._put_v4_scenario_result(
            self,
            run=current,
            recipe=recipe,
            normalized=normalized,
            satisfaction=satisfaction,
            proof=proof,
        )
        proof_payload = dict(proof.payload)
        proof_runtime = dict(proof_payload.get("resources") or {})
        hidden_worlds = dict(proof_payload.get("hiddenWorlds") or {})
        proof_completed = int(
            proof_payload.get("completionStatus") != "proof_incomplete"
        )
        qualified_count = int(result["disposition"] == "qualified")
        funnel = {
            "attempted": 1,
            "structurallyFeasible": 0,
            "legalReplay": 1,
            "semanticSatisfied": int(satisfaction.status == "satisfied"),
            "proofCompleted": proof_completed,
            "qualified": qualified_count,
        }
        cache_scopes: dict[str, dict[str, int]] = {}
        execution_context_factory = getattr(self, "execution_context_factory", None)
        if execution_context_factory is not None:
            for snapshot in execution_context_factory.cache_metrics_snapshots():
                scope = str(snapshot["scope"])
                cache_scopes[scope] = {
                    key: int(snapshot[key])
                    for key in (
                        "hits",
                        "misses",
                        "evictions",
                        "entries",
                        "capacity",
                    )
                }
        peak_process_rss = process_lifetime_peak_rss_bytes()
        environment = build_research_environment_declaration(
            generation_lane="exact_input",
            configured_shard_workers=int(
                getattr(self, "maximum_local_shard_workers", 1)
            ),
            effective_shard_workers=1,
            cp_sat_workers_per_solve=None,
            research_store=self.store,
            artifact_store=self.artifact_store,
        )
        performance_core = {
            "schemaVersion": 1,
            "measurementScope": "research_v4_completed_run_v1",
            "runCode": run_code,
            "executionKind": "exact_trace_prefix",
            "generationLane": "exact_input",
            "attemptedSeeds": 0,
            "attemptedCandidates": 1,
            "structuralCandidates": 0,
            "auctionFeasible": 0,
            "solverOptimal": 0,
            "solverIncomplete": 0,
            "runtimeParityPassed": 1,
            "semanticReplayPassed": 1,
            "proofCompleted": proof_completed,
            "resultCount": 1,
            "qualifiedCount": qualified_count,
            "rejectedCount": int(result["disposition"] == "rejected"),
            "quarantineCount": 0,
            "resultCodes": [str(result["resultCode"])],
            "distinctQualifiedResultHashes": (
                [str(result["resultHash"])] if qualified_count else []
            ),
            "qualifiedScenarioHashes": (
                [qualified_scenario_identity_hash_v1(result)] if qualified_count else []
            ),
            "funnel": funnel,
            "constructionSeconds": None,
            "simulationSeconds": simulation_seconds,
            "semanticParitySeconds": semantic_parity_seconds,
            "analysisSeconds": analysis_seconds,
            "tacticalVerificationSeconds": None,
            "candidatePipelineSeconds": candidate_pipeline_seconds,
            "stageTimingBoundary": {
                "construction": "not_applicable_exact_input_v1",
                "simulation": "authoritative_exact_prefix_normalization_and_replay_v1",
                "semanticParity": "runtime_scenario_verifier_v1",
                "analysis": "unified_proof_service_call_v1",
                "tacticalVerification": "not_applicable_no_dedicated_stage_v1",
                "candidatePipeline": "normalize_prove_verify_v1",
            },
            "firstFeasibleCandidateSeconds": None,
            "firstOptimalCandidateSeconds": None,
            "firstQualifiedResultSeconds": (
                time.perf_counter() - started_clock if qualified_count else None
            ),
            "shardWallWorkSeconds": time.perf_counter() - started_clock,
            "cpuSeconds": time.thread_time() - started_cpu,
            "cpuMeasurementBoundary": CALLING_THREAD_CPU_BOUNDARY,
            "peakProcessRssBytes": peak_process_rss,
            "peakRssMeasurementBoundary": (
                PROCESS_PEAK_RSS_BOUNDARY if peak_process_rss is not None else None
            ),
            "environment": environment,
            "cacheScopes": cache_scopes,
            "proofResources": {
                "nodes": (
                    int(proof_runtime["nodes"])
                    if type(proof_runtime.get("nodes")) is int
                    and proof_runtime["nodes"] >= 0
                    else 0
                ),
                "cacheEntries": (
                    int(proof_runtime["cacheEntries"])
                    if type(proof_runtime.get("cacheEntries")) is int
                    and proof_runtime["cacheEntries"] >= 0
                    else 0
                ),
                "cacheBytes": (
                    int(proof_runtime["cacheBytes"])
                    if type(proof_runtime.get("cacheBytes")) is int
                    and proof_runtime["cacheBytes"] >= 0
                    else 0
                ),
                "hiddenWorlds": (
                    int(hidden_worlds["count"])
                    if type(hidden_worlds.get("count")) is int
                    and hidden_worlds["count"] >= 0
                    else 0
                ),
                "aggregation": "single_proof_invocation_v1",
            },
            "cancellationObserved": proof_runtime.get("cancelled") is True,
            "terminalStatus": str(result["disposition"]),
        }
        performance_receipt = {
            **performance_core,
            "sha256": scenario_sha256(performance_core),
        }
        diversity_selection = _select_v4_results(recipe, [result])
        selected_count = len(diversity_selection.selected_result_codes)
        output_receipt = _v4_output_receipt(
            recipe,
            retained_count=1,
            qualified_count=qualified_count,
            selected_count=selected_count,
        )
        completed_at = _now()
        self.store.update_run(
            run_code,
            status="completed",
            progress=1.0,
            resultCount=1,
            resultCodes=[str(result["resultCode"])],
            qualifiedCount=qualified_count,
            selectedCount=selected_count,
            duplicateQualifiedCount=diversity_selection.duplicate_count,
            rejectedCount=int(result["disposition"] == "rejected"),
            quarantineCount=0,
            performanceReceipt=copy.deepcopy(performance_receipt),
            diversitySelection=diversity_selection.to_dict(),
            outputReceipt=copy.deepcopy(output_receipt),
            completedAt=completed_at,
            events=[
                *current.get("events", []),
                {
                    "type": "completed",
                    "executionKind": "exact_trace_prefix",
                    "resultCode": result["resultCode"],
                    "scenarioStatus": satisfaction.status,
                    "proofCompletion": proof.payload["completionStatus"],
                    "performanceReceiptHash": performance_receipt["sha256"],
                    "outputReceiptHash": output_receipt["sha256"],
                    "at": completed_at,
                },
            ][-200:],
        )

    def _execute_v4_search_run(
        self,
        run: Mapping[str, Any],
        recipe: ResearchRecipeV4,
        context: ResearchExecutionContextV1 | None,
    ) -> None:
        """Run one bounded deterministic V4 search and retain honest misses."""

        if context is None:
            raise ValueError("Research V4 search execution context is unavailable")
        started_clock = time.perf_counter()
        started_cpu = time.thread_time()
        run_code = str(run["runCode"])
        started_at = str(run.get("startedAt") or _now())
        if run["status"] == "queued":
            run = self.store.update_run(
                run_code,
                status="running",
                progress=0.0,
                startedAt=started_at,
                events=[
                    *run.get("events", []),
                    {
                        "type": "started",
                        "executionKind": "scenario_search",
                        "at": _now(),
                    },
                ][-200:],
            )
        executor = ScenarioSearchExecutorV1()
        seed_start = int(recipe.resources["seedStart"])
        seed_count = int(recipe.resources["seedCount"])
        candidates_per_seed = int(recipe.resources["candidatesPerSeed"])
        total = seed_count * candidates_per_seed
        target_result_count = int(recipe.output["resultCount"])
        attempted_seeds = 0
        attempted = 0
        retained: list[dict[str, Any]] = []
        status_counts: dict[str, int] = {}
        construction_receipts: list[dict[str, Any]] = []
        no_good_receipts: list[dict[str, Any]] = []
        no_good_decks: list[tuple[int, ...]] = []
        no_good_deck_hashes: set[str] = set()
        yield_counts = {
            "attempted": 0,
            "structurallyFeasible": 0,
            "legalReplay": 0,
            "semanticSatisfied": 0,
            "proofCompleted": 0,
            "qualified": 0,
        }
        structurally_infeasible = False
        auction_feasible = 0
        solver_optimal = 0
        solver_incomplete = 0
        construction_seconds = 0.0
        analysis_seconds = 0.0
        candidate_pipeline_seconds = 0.0
        proof_nodes = 0
        proof_cache_entries = 0
        proof_cache_bytes = 0
        proof_hidden_worlds = 0
        proof_invocation_count = 0
        proof_resource_fields_complete = {
            "nodes": True,
            "cacheEntries": True,
            "cacheBytes": True,
            "hiddenWorlds": True,
        }
        cancellation_observed = False
        first_feasible_candidate_seconds: float | None = None
        first_optimal_candidate_seconds: float | None = None
        first_qualified_result_seconds: float | None = None
        selected_count = 0

        def persist_partial_receipt(reason: str) -> None:
            current_run = self.store.get_run(run_code)
            if current_run is None:
                raise KeyError(run_code)
            if current_run.get("status") != "cancelled":
                return
            proof_resource_availability = (
                "not_started"
                if proof_invocation_count == 0
                else "complete"
                if all(proof_resource_fields_complete.values())
                else "partial"
            )
            proof_resource_totals = {
                "nodes": proof_nodes,
                "cacheEntries": proof_cache_entries,
                "cacheBytes": proof_cache_bytes,
                "hiddenWorlds": proof_hidden_worlds,
            }
            proof_resources = {
                key: (
                    None
                    if proof_invocation_count == 0
                    or not proof_resource_fields_complete[key]
                    else value
                )
                for key, value in proof_resource_totals.items()
            }
            cache_scopes: dict[str, dict[str, int]] = {}
            execution_context_factory = getattr(self, "execution_context_factory", None)
            if execution_context_factory is not None:
                for snapshot in execution_context_factory.cache_metrics_snapshots():
                    cache_scopes[str(snapshot["scope"])] = {
                        key: int(snapshot[key])
                        for key in (
                            "hits",
                            "misses",
                            "evictions",
                            "entries",
                            "capacity",
                        )
                    }
            peak_process_rss = process_lifetime_peak_rss_bytes()
            qualified_count = sum(
                item["disposition"] == "qualified" for item in retained
            )
            rejected_count = sum(item["disposition"] == "rejected" for item in retained)
            generation_lane = str(recipe.resources["generationLane"])
            partial_receipt = _v4_partial_performance_receipt(
                {
                    "runCode": run_code,
                    "executionKind": "scenario_search",
                    "generationLane": generation_lane,
                    "terminalStatus": "cancelled",
                    "terminalReason": reason,
                    "evidenceCompleteness": {
                        "counters": "complete_through_terminal_observation_v1",
                        "timings": "complete_through_terminal_observation_v1",
                        "results": "retained_results_only_v1",
                        "proofResources": proof_resource_availability,
                    },
                    "attemptedSeeds": attempted_seeds,
                    "attemptedCandidates": attempted,
                    "structuralCandidates": attempted,
                    "auctionFeasible": auction_feasible,
                    "solverOptimal": solver_optimal,
                    "solverIncomplete": solver_incomplete,
                    "proofCompleted": yield_counts["proofCompleted"],
                    "resultCount": len(retained),
                    "qualifiedCount": qualified_count,
                    "rejectedCount": rejected_count,
                    "quarantineCount": 0,
                    "resultCodes": [str(item["resultCode"]) for item in retained],
                    "funnel": copy.deepcopy(yield_counts),
                    "constructionSeconds": construction_seconds,
                    "simulationSeconds": None,
                    "semanticParitySeconds": None,
                    "analysisSeconds": analysis_seconds,
                    "tacticalVerificationSeconds": None,
                    "candidatePipelineSeconds": candidate_pipeline_seconds,
                    "stageTimingBoundary": {
                        "construction": "sum_structural_receipt_elapsed_seconds_v1",
                        "simulation": "unavailable_combined_candidate_pipeline_v1",
                        "semanticParity": "unavailable_combined_candidate_pipeline_v1",
                        "analysis": "sum_proof_resource_elapsed_wall_seconds_v1",
                        "tacticalVerification": "not_applicable_no_dedicated_stage_v1",
                        "candidatePipeline": "sum_scenario_search_executor_calls_until_cancellation_v1",
                    },
                    "firstFeasibleCandidateSeconds": first_feasible_candidate_seconds,
                    "firstOptimalCandidateSeconds": first_optimal_candidate_seconds,
                    "firstQualifiedResultSeconds": first_qualified_result_seconds,
                    "shardWallWorkSeconds": time.perf_counter() - started_clock,
                    "cpuSeconds": time.thread_time() - started_cpu,
                    "cpuMeasurementBoundary": CALLING_THREAD_CPU_BOUNDARY,
                    "peakProcessRssBytes": peak_process_rss,
                    "peakRssMeasurementBoundary": (
                        PROCESS_PEAK_RSS_BOUNDARY
                        if peak_process_rss is not None
                        else None
                    ),
                    "environment": build_research_environment_declaration(
                        generation_lane=generation_lane,
                        configured_shard_workers=int(
                            getattr(self, "maximum_local_shard_workers", 1)
                        ),
                        effective_shard_workers=1,
                        cp_sat_workers_per_solve=1,
                        research_store=self.store,
                        artifact_store=self.artifact_store,
                    ),
                    "cacheScopes": cache_scopes,
                    "proofResources": {
                        **proof_resources,
                        "aggregation": (
                            "not_started_v1"
                            if proof_invocation_count == 0
                            else "sum_per_proof_invocation_v1"
                        ),
                        "availability": proof_resource_availability,
                    },
                    "cancellationObserved": True,
                }
            )
            self.store.update_run(
                run_code,
                partialPerformanceReceipt=copy.deepcopy(partial_receipt),
                events=[
                    *current_run.get("events", []),
                    {
                        "type": "partial_performance_receipt",
                        "terminalReason": reason,
                        "performanceReceiptHash": partial_receipt["sha256"],
                        "at": _now(),
                    },
                ][-200:],
            )

        for seed_offset in range(seed_count):
            if self._cancel_check(run_code):
                persist_partial_receipt(
                    "cancelled_before_candidate"
                    if attempted == 0
                    else "cancelled_between_candidates"
                )
                return
            base_seed = seed_start + seed_offset
            for candidate_ordinal in range(candidates_per_seed):
                if self._cancel_check(run_code):
                    persist_partial_receipt(
                        "cancelled_before_candidate"
                        if attempted == 0
                        else "cancelled_between_candidates"
                    )
                    return
                if candidate_ordinal == 0:
                    attempted_seeds += 1
                candidate_seed = base_seed * 1_000 + candidate_ordinal
                candidate_started = time.perf_counter()
                candidate = executor.build_candidate(
                    recipe,
                    rules=context.bundle.rules,
                    bundle_hash=str(context.bundle.sha256),
                    seed=candidate_seed,
                    cancel_check=lambda: self._cancel_check(run_code),
                    excluded_deck_orders=(
                        tuple(no_good_decks)
                        if recipe.resources["generationPolicy"]
                        == "scenario_guided_actor_safe"
                        else ()
                    ),
                )
                candidate_pipeline_seconds += time.perf_counter() - candidate_started
                attempted += 1
                yield_counts["attempted"] = attempted
                status_counts[candidate.status] = (
                    status_counts.get(candidate.status, 0) + 1
                )
                construction_receipts.append(dict(candidate.construction_receipt))
                candidate_auction_feasible = candidate.construction_receipt.get(
                    "auctionFeasible"
                )
                if type(candidate_auction_feasible) is not bool:
                    raise ValueError(
                        "Research V4 candidate auction-feasibility evidence is unavailable"
                    )
                auction_feasible += int(candidate_auction_feasible)
                structural_payload = dict(candidate.construction_receipt["structural"])
                structural_elapsed = structural_payload.get("elapsedSeconds")
                if (
                    isinstance(structural_elapsed, (int, float))
                    and not isinstance(structural_elapsed, bool)
                    and float(structural_elapsed) >= 0
                ):
                    construction_seconds += float(structural_elapsed)
                solver_status = str(
                    structural_payload.get("solverStatus") or ""
                ).lower()
                if solver_status == "optimal":
                    solver_optimal += 1
                    if first_optimal_candidate_seconds is None:
                        first_optimal_candidate_seconds = (
                            time.perf_counter() - started_clock
                        )
                elif solver_status not in {"", "infeasible"}:
                    solver_incomplete += 1
                if structural_payload["status"] == "satisfied":
                    yield_counts["structurallyFeasible"] += 1
                    if first_feasible_candidate_seconds is None:
                        first_feasible_candidate_seconds = (
                            time.perf_counter() - started_clock
                        )
                if candidate.normalized is not None:
                    yield_counts["legalReplay"] += 1
                if (
                    candidate.satisfaction is not None
                    and candidate.satisfaction.status == "satisfied"
                ):
                    yield_counts["semanticSatisfied"] += 1
                if (
                    candidate.proof is not None
                    and candidate.proof.payload["completionStatus"]
                    != "proof_incomplete"
                ):
                    yield_counts["proofCompleted"] += 1
                if candidate.proof is not None:
                    proof_invocation_count += 1
                    proof_payload = dict(candidate.proof.payload)
                    proof_runtime = dict(proof_payload.get("resources") or {})
                    hidden_worlds = dict(proof_payload.get("hiddenWorlds") or {})
                    proof_elapsed = proof_runtime.get("elapsedWallSeconds")
                    if (
                        isinstance(proof_elapsed, (int, float))
                        and not isinstance(proof_elapsed, bool)
                        and float(proof_elapsed) >= 0
                    ):
                        analysis_seconds += float(proof_elapsed)
                    for key, current_total in (
                        ("nodes", proof_nodes),
                        ("cacheEntries", proof_cache_entries),
                        ("cacheBytes", proof_cache_bytes),
                    ):
                        value = proof_runtime.get(key)
                        if type(value) is int and value >= 0:
                            if key == "nodes":
                                proof_nodes = current_total + value
                            elif key == "cacheEntries":
                                proof_cache_entries = current_total + value
                            else:
                                proof_cache_bytes = current_total + value
                        else:
                            proof_resource_fields_complete[key] = False
                    hidden_count = hidden_worlds.get("count")
                    if type(hidden_count) is int and hidden_count >= 0:
                        proof_hidden_worlds += hidden_count
                    else:
                        proof_resource_fields_complete["hiddenWorlds"] = False
                    cancellation_observed = (
                        cancellation_observed or proof_runtime.get("cancelled") is True
                    )
                deck_order = structural_payload.get("deckOrder")
                applies_to_structural = (
                    recipe.resources["generationPolicy"] == "scenario_guided_actor_safe"
                )
                if deck_order is not None and (
                    candidate.status != "satisfied" or applies_to_structural
                ):
                    deck_hash = scenario_sha256(deck_order)
                    no_good_core = {
                        "schemaVersion": 1,
                        "candidateHash": candidate.sha256,
                        "candidateStatus": candidate.status,
                        "scope": (
                            "exact_deck"
                            if applies_to_structural
                            else "candidate_identity"
                        ),
                        "deckHash": deck_hash,
                        "appliedToFutureStructuralSolves": applies_to_structural,
                    }
                    no_good_receipts.append(
                        {**no_good_core, "sha256": scenario_sha256(no_good_core)}
                    )
                    if applies_to_structural and deck_hash not in no_good_deck_hashes:
                        no_good_decks.append(tuple(int(card) for card in deck_order))
                        no_good_deck_hashes.add(deck_hash)
                if candidate.status == "structurally_infeasible":
                    structurally_infeasible = True
                    break
                if (
                    candidate.normalized is not None
                    and candidate.proof is not None
                    and candidate.satisfaction is not None
                ):
                    current = self.store.get_run(run_code)
                    if current is None or current.get("status") == "cancelled":
                        if current is not None:
                            persist_partial_receipt(
                                "cancelled_before_result_persistence"
                            )
                        return
                    retained.append(
                        ResearchFoundryServiceV1._put_v4_scenario_result(
                            self,
                            run=current,
                            recipe=recipe,
                            normalized=candidate.normalized,
                            proof=candidate.proof,
                            satisfaction=candidate.satisfaction,
                            construction_receipt=candidate.construction_receipt,
                        )
                    )
                    if retained[-1]["disposition"] == "qualified":
                        yield_counts["qualified"] += 1
                        if first_qualified_result_seconds is None:
                            first_qualified_result_seconds = (
                                time.perf_counter() - started_clock
                            )
                    diversity_selection = _select_v4_results(recipe, retained)
                    selected_count = len(diversity_selection.selected_result_codes)
                    if selected_count >= target_result_count:
                        break
                self.store.update_run(
                    run_code,
                    status="running",
                    progress=min(attempted / total, 0.999999),
                    attemptedCandidates=attempted,
                    resultCount=len(retained),
                    qualifiedCount=sum(
                        item["disposition"] == "qualified" for item in retained
                    ),
                    selectedCount=selected_count,
                    rejectionStatusCounts=copy.deepcopy(status_counts),
                    scenarioYieldCounts=copy.deepcopy(yield_counts),
                    noGoodReceiptHashes=[
                        str(item["sha256"]) for item in no_good_receipts
                    ],
                )
            if structurally_infeasible or selected_count >= target_result_count:
                break
        diversity_selection = _select_v4_results(recipe, retained)
        selected_count = len(diversity_selection.selected_result_codes)
        terminal_status = (
            "satisfied"
            if selected_count >= target_result_count
            else "structurally_infeasible"
            if structurally_infeasible
            else "not_found_within_budget"
        )
        completed_at = _now()
        current = self.store.get_run(run_code)
        if current is None:
            return
        if current.get("status") == "cancelled":
            persist_partial_receipt("cancelled_after_candidate")
            return
        qualified_count = sum(item["disposition"] == "qualified" for item in retained)
        rejected_count = sum(item["disposition"] == "rejected" for item in retained)
        generation_lane = str(recipe.resources["generationLane"])
        cache_scopes: dict[str, dict[str, int]] = {}
        execution_context_factory = getattr(self, "execution_context_factory", None)
        if execution_context_factory is not None:
            for snapshot in execution_context_factory.cache_metrics_snapshots():
                scope = str(snapshot["scope"])
                cache_scopes[scope] = {
                    key: int(snapshot[key])
                    for key in (
                        "hits",
                        "misses",
                        "evictions",
                        "entries",
                        "capacity",
                    )
                }
        peak_process_rss = process_lifetime_peak_rss_bytes()
        environment = build_research_environment_declaration(
            generation_lane=generation_lane,
            configured_shard_workers=int(
                getattr(self, "maximum_local_shard_workers", 1)
            ),
            effective_shard_workers=1,
            cp_sat_workers_per_solve=1,
            research_store=self.store,
            artifact_store=self.artifact_store,
        )
        performance_core = {
            "schemaVersion": 1,
            "measurementScope": "research_v4_completed_run_v1",
            "runCode": run_code,
            "executionKind": "scenario_search",
            "generationLane": generation_lane,
            "attemptedSeeds": attempted_seeds,
            "attemptedCandidates": attempted,
            "structuralCandidates": attempted,
            "auctionFeasible": auction_feasible,
            "solverOptimal": solver_optimal,
            "solverIncomplete": solver_incomplete,
            "runtimeParityPassed": yield_counts["legalReplay"],
            "semanticReplayPassed": yield_counts["legalReplay"],
            "proofCompleted": yield_counts["proofCompleted"],
            "resultCount": len(retained),
            "qualifiedCount": qualified_count,
            "rejectedCount": rejected_count,
            "quarantineCount": 0,
            "resultCodes": [str(item["resultCode"]) for item in retained],
            "distinctQualifiedResultHashes": sorted(
                {
                    str(item["resultHash"])
                    for item in retained
                    if item["disposition"] == "qualified"
                }
            ),
            "qualifiedScenarioHashes": sorted(
                [
                    qualified_scenario_identity_hash_v1(item)
                    for item in retained
                    if item["disposition"] == "qualified"
                ]
            ),
            "funnel": copy.deepcopy(yield_counts),
            "constructionSeconds": construction_seconds,
            "simulationSeconds": None,
            "semanticParitySeconds": None,
            "analysisSeconds": analysis_seconds,
            "tacticalVerificationSeconds": None,
            "candidatePipelineSeconds": candidate_pipeline_seconds,
            "stageTimingBoundary": {
                "construction": "sum_structural_receipt_elapsed_seconds_v1",
                "simulation": "unavailable_combined_candidate_pipeline_v1",
                "semanticParity": "unavailable_combined_candidate_pipeline_v1",
                "analysis": "sum_proof_resource_elapsed_wall_seconds_v1",
                "tacticalVerification": "not_applicable_no_dedicated_stage_v1",
                "candidatePipeline": "sum_scenario_search_executor_calls_v1",
            },
            "firstFeasibleCandidateSeconds": first_feasible_candidate_seconds,
            "firstOptimalCandidateSeconds": first_optimal_candidate_seconds,
            "firstQualifiedResultSeconds": first_qualified_result_seconds,
            "shardWallWorkSeconds": time.perf_counter() - started_clock,
            "cpuSeconds": time.thread_time() - started_cpu,
            "cpuMeasurementBoundary": CALLING_THREAD_CPU_BOUNDARY,
            "peakProcessRssBytes": peak_process_rss,
            "peakRssMeasurementBoundary": (
                PROCESS_PEAK_RSS_BOUNDARY if peak_process_rss is not None else None
            ),
            "environment": environment,
            "cacheScopes": cache_scopes,
            "proofResources": {
                "nodes": proof_nodes,
                "cacheEntries": proof_cache_entries,
                "cacheBytes": proof_cache_bytes,
                "hiddenWorlds": proof_hidden_worlds,
                "aggregation": (
                    "sum_per_proof_invocation_v1"
                    if proof_invocation_count
                    else "no_proof_invocation_v1"
                ),
            },
            "cancellationObserved": cancellation_observed,
            "terminalStatus": terminal_status,
        }
        performance_receipt = {
            **performance_core,
            "sha256": scenario_sha256(performance_core),
        }
        output_receipt = _v4_output_receipt(
            recipe,
            retained_count=len(retained),
            qualified_count=qualified_count,
            selected_count=selected_count,
        )
        search_evidence_core = {
            "schemaVersion": 1,
            "runCode": run_code,
            "runHash": str(run["runHash"]),
            "recipeHash": recipe.sha256,
            "scenarioHash": recipe.scenario.sha256,
            "terminalStatus": terminal_status,
            "statusCounts": copy.deepcopy(status_counts),
            "yieldCounts": copy.deepcopy(yield_counts),
            "constructionReceipts": copy.deepcopy(construction_receipts),
            "noGoodReceipts": copy.deepcopy(no_good_receipts),
            "performanceReceiptHash": performance_receipt["sha256"],
            "diversitySelection": diversity_selection.to_dict(),
            "outputReceipt": copy.deepcopy(output_receipt),
        }
        search_evidence = {
            **search_evidence_core,
            "sha256": scenario_sha256(search_evidence_core),
        }
        search_evidence_ref = self.artifact_store.put(
            scenario_json_bytes(search_evidence),
            media_type=("application/vnd.balootx.scenario-search-evidence-v1+json"),
            name=f"{search_evidence['sha256']}.json",
        )
        self.store.update_run(
            run_code,
            status="completed",
            progress=1.0,
            resultCount=len(retained),
            resultCodes=[str(item["resultCode"]) for item in retained],
            qualifiedCount=qualified_count,
            selectedCount=selected_count,
            duplicateQualifiedCount=diversity_selection.duplicate_count,
            rejectedCount=rejected_count,
            quarantineCount=0,
            attemptedCandidates=attempted,
            scenarioSearchStatus=terminal_status,
            rejectionStatusCounts=copy.deepcopy(status_counts),
            constructionReceiptHashes=[
                str(item["sha256"]) for item in construction_receipts
            ],
            noGoodReceiptHashes=[str(item["sha256"]) for item in no_good_receipts],
            scenarioYieldCounts=copy.deepcopy(yield_counts),
            scenarioSearchEvidence=copy.deepcopy(dict(search_evidence_ref)),
            performanceReceipt=copy.deepcopy(performance_receipt),
            diversitySelection=diversity_selection.to_dict(),
            outputReceipt=copy.deepcopy(output_receipt),
            completedAt=completed_at,
            events=[
                *current.get("events", []),
                {
                    "type": "completed",
                    "executionKind": "scenario_search",
                    "scenarioSearchStatus": terminal_status,
                    "attemptedCandidates": attempted,
                    "resultCount": len(retained),
                    "scenarioYieldCounts": copy.deepcopy(yield_counts),
                    "noGoodReceiptHashes": [
                        str(item["sha256"]) for item in no_good_receipts
                    ],
                    "performanceReceiptHash": performance_receipt["sha256"],
                    "outputReceiptHash": output_receipt["sha256"],
                    "at": completed_at,
                },
            ][-200:],
        )

    def _shard_worker_loop(
        self,
        run_code: str,
        worker_index: int,
        context: ResearchExecutionContextV1 | None = None,
    ) -> None:
        worker_id = (
            f"{self._service_worker_id}:{int(worker_index)}:{threading.get_ident()}"
        )
        idle_polls = 0
        while not self._cancel_check(run_code):
            shard = self.store.claim_shard(
                run_code,
                worker_id=worker_id,
                lease_seconds=self.shard_lease_seconds,
            )
            if shard is None:
                state = self._reduce_sharded_run(run_code)
                if state in {"completed", "failed", "cancelled"}:
                    with self._lock:
                        self._scheduled_runs.discard(run_code)
                    return
                idle_polls += 1
                # Another worker still owns a live lease. Keep one bounded
                # recovery poller alive so an abandoned lease can be reclaimed.
                if idle_polls > int(self.shard_lease_seconds / 0.02) + 500:
                    return
                time.sleep(0.02)
                continue
            idle_polls = 0
            self._execute_claimed_shard(run_code, shard, worker_id, context)
            state = self._reduce_sharded_run(run_code)
            if state in {"completed", "failed", "cancelled"}:
                with self._lock:
                    self._scheduled_runs.discard(run_code)
                return

    def _reduce_sharded_run(self, run_code: str) -> str:
        """Reduce shard evidence in immutable shard-index order."""
        run = self.store.get_run(run_code)
        if run is None:
            return "failed"
        if run.get("status") == "cancelled":
            return "cancelled"
        shards = self.store.list_shards(run_code)
        if not shards:
            return str(run.get("status", "failed"))
        statuses = [str(item["status"]) for item in shards]
        completed = [item for item in shards if item["status"] == "completed"]
        failed = [item for item in shards if item["status"] == "failed"]
        metrics = [dict(item.get("result") or {}) for item in completed]
        additive = (
            "attemptedSeeds",
            "structuralCandidates",
            "auctionFeasible",
            "solverOptimal",
            "runtimeParityPassed",
            "semanticReplayPassed",
            "solverIncomplete",
            "playCompilerAttempted",
            "playCompilerOptimal",
            "playCompilerFeasible",
            "playCompilerIncomplete",
            "strategicSelectionDecisions",
            "strategicExactDecisions",
            "strategicProvisionalDecisions",
            "strategicFallbackDecisions",
            "resultCount",
            "qualifiedCount",
            "rejectedCount",
            "quarantineCount",
            "tacticalVerificationCandidates",
            "tacticalVerificationDecisions",
            "optimalityCutCount",
            "constructionPreScreenCandidates",
            "constructionPreScreenRejected",
            "tacticalViabilityPreScreenCandidates",
            "tacticalViabilityPreScreenRejected",
            "constructionTargetMissCount",
            "analyzedCandidateCount",
        )
        durations = (
            "tacticalVerificationSeconds",
            "constructionSeconds",
            "simulationSeconds",
            "semanticParitySeconds",
            "analysisSeconds",
            "shardWallWorkSeconds",
            "cpuSeconds",
            "constructionPreScreenSeconds",
            "tacticalViabilityPreScreenSeconds",
        )
        aggregate: dict[str, Any] = {
            key: sum(int(item.get(key, 0)) for item in metrics) for key in additive
        }
        aggregate.update(
            {
                key: sum(float(item.get(key, 0.0)) for item in metrics)
                for key in durations
            }
        )
        aggregate["requested"] = int(
            build_research_recipe(
                self.store.get_recipe(run["recipeCode"]) or {}
            ).output["resultCount"]
        )
        recipe = build_research_recipe(self.store.get_recipe(run["recipeCode"]) or {})
        aggregate["generationLane"] = str(recipe.resources["generationLane"])
        aggregate["cpuMeasurementBoundary"] = CALLING_THREAD_CPU_BOUNDARY
        aggregate["shardCount"] = len(shards)
        aggregate["completedShardCount"] = len(completed)
        aggregate["failedShardCount"] = statuses.count("failed")
        aggregate["cancelledShardCount"] = statuses.count("cancelled")
        processed_seeds = sum(int(item["seedCount"]) for item in completed)
        total_seeds = sum(int(item["seedCount"]) for item in shards)
        progress = processed_seeds / max(1, total_seeds)
        if all(status == "completed" for status in statuses):
            status = "completed"
            progress = 1.0
        elif any(status == "failed" for status in statuses) and all(
            status in {"completed", "failed", "cancelled"} for status in statuses
        ):
            status = "failed"
        else:
            status = "running"
        failed_errors = [
            {
                **copy.deepcopy(dict(item.get("error") or {})),
                "shardId": str(item["shardId"]),
                "shardIndex": int(item["shardIndex"]),
                "attemptCount": int(item.get("attemptCount", 0)),
            }
            for item in sorted(failed, key=lambda value: int(value["shardIndex"]))
        ]
        aggregate["error"] = (
            failed_errors[0] if status == "failed" and failed_errors else None
        )
        aggregate["shardErrors"] = failed_errors
        started_text = str(run.get("startedAt") or run.get("createdAt") or _now())
        try:
            started_at = datetime.fromisoformat(started_text)
            elapsed_wall = max(
                0.0, (datetime.now(timezone.utc) - started_at).total_seconds()
            )
        except ValueError:
            elapsed_wall = 0.0
        aggregate["elapsedWallSeconds"] = elapsed_wall
        # Compatibility name retained, now explicitly the run wall boundary.
        aggregate["elapsedSeconds"] = elapsed_wall
        if status == "completed" and isinstance(recipe, ResearchRecipeV2):
            run_results: list[dict[str, Any]] = []
            offset = 0
            while True:
                page = self.store.list_results(
                    run_code=run_code,
                    sort="result_code_asc",
                    offset=offset,
                    limit=500,
                )
                items = [dict(item) for item in page["items"]]
                run_results.extend(items)
                if not bool(page["hasMore"]):
                    break
                offset = int(page["nextOffset"])
            diversity_profile_payload = dict(recipe.diversity)
            diversity_profile_payload.pop("sha256", None)
            diversity_selection = select_research_results(
                run_results,
                profile=QualityDiversityProfileV1.build(diversity_profile_payload),
                requested=int(recipe.output["resultCount"]),
            )
            aggregate["diversitySelection"] = diversity_selection.to_dict()
            aggregate["selectedCount"] = len(diversity_selection.selected_result_codes)
            aggregate["duplicateQualifiedCount"] = diversity_selection.duplicate_count
        shard_events = []
        for shard, result in zip(completed, metrics):
            shard_events.append(
                {
                    "type": "shard_completed",
                    "shardId": shard["shardId"],
                    "shardIndex": int(shard["shardIndex"]),
                    "attemptedSeeds": int(result.get("attemptedSeeds", 0)),
                }
            )
        for error in failed_errors:
            shard_events.append(
                {
                    "type": "shard_failed",
                    "errorType": str(error.get("type") or ""),
                    "category": str(error.get("category") or "system_failure"),
                    "message": str(error.get("message") or ""),
                    "shardId": str(error["shardId"]),
                    "shardIndex": int(error["shardIndex"]),
                    "attemptCount": int(error["attemptCount"]),
                    "at": _now(),
                }
            )
        shard_events.sort(
            key=lambda item: (int(item.get("shardIndex", -1)), str(item["type"]))
        )
        terminal_event = (
            [
                {
                    "type": status,
                    "resultCount": aggregate["resultCount"],
                    "message": (
                        str(aggregate["error"].get("message") or "")
                        if status == "failed" and aggregate.get("error")
                        else ""
                    ),
                    "at": _now(),
                }
            ]
            if status in {"completed", "failed"}
            else []
        )
        base_events = [
            item
            for item in run.get("events", [])
            if item.get("type") in {"queued", "started", "cancelled"}
        ]
        self.store.update_run(
            run_code,
            status=status,
            progress=progress,
            **aggregate,
            events=[*base_events, *shard_events, *terminal_event][-200:],
        )
        return status

    def _execute_claimed_shard(
        self,
        run_code: str,
        shard: Mapping[str, Any],
        worker_id: str,
        context: ResearchExecutionContextV1 | None = None,
    ) -> None:
        run = self.store.get_run(run_code)
        if run is None or run["status"] not in {"running", "queued"}:
            return
        recipe_value = self.store.get_recipe(run["recipeCode"]) or {}
        if context is None:
            recipe, context = self._build_recipe_and_context(recipe_value)
        else:
            recipe = build_research_recipe(recipe_value)
        self._assert_run_execution_context(run, recipe, context)
        events: list[dict[str, Any]] = []
        result_count = 0
        result_codes: list[str] = []
        qualified_count = 0
        quarantine_count = 0
        requested = int(recipe.output["resultCount"])
        started_clock = time.perf_counter()
        started_cpu = time.thread_time()
        first_feasible_candidate_seconds: float | None = None
        first_optimal_candidate_seconds: float | None = None
        first_qualified_result_seconds: float | None = None
        cp_sat_worker_counts: set[int] = set()
        attempted_seeds = 0
        structural_candidates = 0
        auction_feasible = 0
        solver_optimal = 0
        # Canonical sequential generation emits only verified optimal candidates.
        # A non-optimal generator status fails the shard before a search receipt,
        # so successful shards cannot contain a generator-incomplete candidate.
        solver_incomplete = 0
        runtime_parity_passed = 0
        semantic_replay_passed = 0
        play_compiler_attempted = 0
        play_compiler_optimal = 0
        play_compiler_feasible = 0
        play_compiler_incomplete = 0
        strategic_selection_decisions = 0
        strategic_exact_decisions = 0
        strategic_provisional_decisions = 0
        strategic_fallback_decisions = 0
        rejected_count = 0
        tactical_verification_candidates = 0
        tactical_verification_decisions = 0
        tactical_verification_seconds = 0.0
        construction_seconds = 0.0
        simulation_seconds = 0.0
        semantic_parity_seconds = 0.0
        analysis_seconds = 0.0
        construction_pre_screen_seconds = 0.0
        construction_pre_screen_candidates = 0
        construction_pre_screen_rejected = 0
        tactical_viability_pre_screen_seconds = 0.0
        tactical_viability_pre_screen_candidates = 0
        tactical_viability_pre_screen_rejected = 0
        construction_target_miss_count = 0
        analyzed_candidate_count = 0
        optimality_cut_count = 0
        resources = recipe.resources
        generation_lane = str(resources["generationLane"])
        simulator = context.simulator if context is not None else self.simulator
        analyzers = context.analyzers if context is not None else self.analyzers
        domain = context.legacy_domain_v2 if context is not None else self.domain
        compiled_domain = (
            context.compiled_domain if context is not None else self.compiled_domain
        )
        configuration_provenance = self._configuration_provenance(
            recipe.configuration if isinstance(recipe, ResearchRecipeV3) else None,
            context,
        )
        standalone_configuration_evidence = (
            {"configurationProvenance": copy.deepcopy(configuration_provenance)}
            if configuration_provenance is not None
            else {}
        )
        score_profile = (
            context.score_profile if context is not None else self.score_profile
        )
        decision_score_profile = (
            context.decision_score_profile
            if context is not None
            else self.decision_score_profile
        )
        decision_score_profile_hash = (
            context.decision_score_profile_hash
            if context is not None
            else self.decision_score_profile_hash
        )
        tactical_verifier = (
            context.tactical_verifier if context is not None else self.tactical_verifier
        )
        analyzer = analyzers[resources["analysisMode"]]
        pre_screen = None
        if resources.get("constructionPreScreen"):
            pre_screen_profile = ConstructionPreScreenProfileV1.build(
                resources["constructionPreScreen"]
            )
            if pre_screen_profile.enabled:
                pre_screen = ConstructionPreScreenV1(pre_screen_profile)
        tactical_viability_pre_screen = None
        if resources.get("tacticalViabilityPreScreen"):
            if tactical_verifier is None:
                raise ValueError(
                    "tactical viability pre-screen requires the exact tactical verifier"
                )
            tactical_pre_screen_profile = TacticalViabilityPreScreenProfileV1.build(
                resources["tacticalViabilityPreScreen"]
            )
            if tactical_pre_screen_profile.enabled:
                tactical_viability_pre_screen = TacticalViabilityPreScreenV1(
                    tactical_pre_screen_profile,
                    tactical_verifier,
                )
        seeds = range(int(shard["seedStart"]), int(shard["seedEndExclusive"]))
        lease = _ShardLeaseHeartbeat(
            self.store,
            shard_id=str(shard["shardId"]),
            worker_id=worker_id,
            fencing_token=int(shard["fencingToken"]),
            lease_seconds=self.shard_lease_seconds,
        )
        lease.start()

        staged_candidate_effects: (
            list[tuple[str, tuple[Any, ...], dict[str, Any]]] | None
        ) = None
        retention_evaluations: dict[str, SequentialCandidateEvaluationV1] = {}

        def should_cancel() -> bool:
            return lease.lost or self._cancel_check(run_code)

        def build_candidate_evaluation(
            candidate: Any,
            outcome: SequentialCandidateOutcomeKindV1,
            failure_codes: Iterable[str] = (),
            *,
            evidence: Mapping[str, Any] | None = None,
        ) -> SequentialCandidateEvaluationV1:
            failures = tuple(sorted({str(item) for item in failure_codes}))
            evidence_core = {
                "schemaVersion": 1,
                "kind": "research_authoritative_candidate_evaluation_evidence_v1",
                "candidateHash": str(candidate.candidate_hash),
                "outcome": outcome.value,
                "failureCodes": list(failures),
                "evidence": copy.deepcopy(dict(evidence or {})),
            }
            return SequentialCandidateEvaluationV1.build(
                outcome=outcome,
                candidate_hash=str(candidate.candidate_hash),
                evaluation_evidence_hashes=(canonical_sha256(evidence_core),),
                failure_codes=failures,
            )

        def put_candidate_disagreement(
            disagreement: Mapping[str, Any],
            **kwargs: Any,
        ) -> None:
            if staged_candidate_effects is not None:
                staged_candidate_effects.append(
                    (
                        "put_disagreement",
                        (copy.deepcopy(dict(disagreement)),),
                        dict(kwargs),
                    )
                )
                return
            self.store.put_disagreement(disagreement, **kwargs)

        def put_candidate_result(result: Mapping[str, Any], **kwargs: Any) -> None:
            if staged_candidate_effects is not None:
                staged_candidate_effects.append(
                    ("put_result", (copy.deepcopy(dict(result)),), dict(kwargs))
                )
                return
            self.store.put_result(result, **kwargs)

        def commit_candidate_effects(
            effects: Iterable[tuple[str, tuple[Any, ...], dict[str, Any]]],
        ) -> None:
            for kind, args, kwargs in effects:
                if kind == "put_disagreement":
                    self.store.put_disagreement(*args, **kwargs)
                elif kind == "put_result":
                    self.store.put_result(*args, **kwargs)
                elif kind == "finalize_audit_trace":
                    self.audit_traces.finalize(*args, **kwargs)
                else:  # pragma: no cover - internal staging invariant
                    raise AssertionError(f"unknown staged candidate effect: {kind}")

        def candidate_metric_snapshot() -> dict[str, Any]:
            return {
                "analysisSeconds": analysis_seconds,
                "analyzedCandidateCount": analyzed_candidate_count,
                "auctionFeasible": auction_feasible,
                "constructionPreScreenCandidates": construction_pre_screen_candidates,
                "constructionPreScreenRejected": construction_pre_screen_rejected,
                "constructionPreScreenSeconds": construction_pre_screen_seconds,
                "constructionSeconds": construction_seconds,
                "constructionTargetMissCount": construction_target_miss_count,
                "eventCount": len(events),
                "firstFeasibleCandidateSeconds": first_feasible_candidate_seconds,
                "firstOptimalCandidateSeconds": first_optimal_candidate_seconds,
                "firstQualifiedResultSeconds": first_qualified_result_seconds,
                "optimalityCutCount": optimality_cut_count,
                "playCompilerAttempted": play_compiler_attempted,
                "playCompilerFeasible": play_compiler_feasible,
                "playCompilerIncomplete": play_compiler_incomplete,
                "playCompilerOptimal": play_compiler_optimal,
                "qualifiedCount": qualified_count,
                "quarantineCount": quarantine_count,
                "rejectedCount": rejected_count,
                "resultCodeCount": len(result_codes),
                "resultCount": result_count,
                "runtimeParityPassed": runtime_parity_passed,
                "semanticParitySeconds": semantic_parity_seconds,
                "semanticReplayPassed": semantic_replay_passed,
                "simulationSeconds": simulation_seconds,
                "solverOptimal": solver_optimal,
                "strategicExactDecisions": strategic_exact_decisions,
                "strategicFallbackDecisions": strategic_fallback_decisions,
                "strategicProvisionalDecisions": strategic_provisional_decisions,
                "strategicSelectionDecisions": strategic_selection_decisions,
                "structuralCandidates": structural_candidates,
                "tacticalVerificationCandidates": tactical_verification_candidates,
                "tacticalVerificationDecisions": tactical_verification_decisions,
                "tacticalVerificationSeconds": tactical_verification_seconds,
                "tacticalViabilityPreScreenCandidates": tactical_viability_pre_screen_candidates,
                "tacticalViabilityPreScreenRejected": tactical_viability_pre_screen_rejected,
                "tacticalViabilityPreScreenSeconds": tactical_viability_pre_screen_seconds,
            }

        def restore_candidate_metrics(snapshot: Mapping[str, Any]) -> None:
            nonlocal analysis_seconds, analyzed_candidate_count, auction_feasible
            nonlocal construction_pre_screen_candidates
            nonlocal construction_pre_screen_rejected
            nonlocal construction_pre_screen_seconds, construction_seconds
            nonlocal construction_target_miss_count
            nonlocal first_feasible_candidate_seconds
            nonlocal first_optimal_candidate_seconds
            nonlocal first_qualified_result_seconds, optimality_cut_count
            nonlocal play_compiler_attempted, play_compiler_feasible
            nonlocal play_compiler_incomplete, play_compiler_optimal
            nonlocal qualified_count, quarantine_count, rejected_count
            nonlocal result_count, runtime_parity_passed
            nonlocal semantic_parity_seconds, semantic_replay_passed
            nonlocal simulation_seconds, solver_optimal
            nonlocal strategic_exact_decisions, strategic_fallback_decisions
            nonlocal strategic_provisional_decisions
            nonlocal strategic_selection_decisions, structural_candidates
            nonlocal tactical_verification_candidates
            nonlocal tactical_verification_decisions
            nonlocal tactical_verification_seconds
            nonlocal tactical_viability_pre_screen_candidates
            nonlocal tactical_viability_pre_screen_rejected
            nonlocal tactical_viability_pre_screen_seconds

            analysis_seconds = float(snapshot["analysisSeconds"])
            analyzed_candidate_count = int(snapshot["analyzedCandidateCount"])
            auction_feasible = int(snapshot["auctionFeasible"])
            construction_pre_screen_candidates = int(
                snapshot["constructionPreScreenCandidates"]
            )
            construction_pre_screen_rejected = int(
                snapshot["constructionPreScreenRejected"]
            )
            construction_pre_screen_seconds = float(
                snapshot["constructionPreScreenSeconds"]
            )
            construction_seconds = float(snapshot["constructionSeconds"])
            construction_target_miss_count = int(
                snapshot["constructionTargetMissCount"]
            )
            first_feasible_candidate_seconds = snapshot["firstFeasibleCandidateSeconds"]
            first_optimal_candidate_seconds = snapshot["firstOptimalCandidateSeconds"]
            first_qualified_result_seconds = snapshot["firstQualifiedResultSeconds"]
            optimality_cut_count = int(snapshot["optimalityCutCount"])
            play_compiler_attempted = int(snapshot["playCompilerAttempted"])
            play_compiler_feasible = int(snapshot["playCompilerFeasible"])
            play_compiler_incomplete = int(snapshot["playCompilerIncomplete"])
            play_compiler_optimal = int(snapshot["playCompilerOptimal"])
            qualified_count = int(snapshot["qualifiedCount"])
            quarantine_count = int(snapshot["quarantineCount"])
            rejected_count = int(snapshot["rejectedCount"])
            result_count = int(snapshot["resultCount"])
            runtime_parity_passed = int(snapshot["runtimeParityPassed"])
            semantic_parity_seconds = float(snapshot["semanticParitySeconds"])
            semantic_replay_passed = int(snapshot["semanticReplayPassed"])
            simulation_seconds = float(snapshot["simulationSeconds"])
            solver_optimal = int(snapshot["solverOptimal"])
            strategic_exact_decisions = int(snapshot["strategicExactDecisions"])
            strategic_fallback_decisions = int(snapshot["strategicFallbackDecisions"])
            strategic_provisional_decisions = int(
                snapshot["strategicProvisionalDecisions"]
            )
            strategic_selection_decisions = int(snapshot["strategicSelectionDecisions"])
            structural_candidates = int(snapshot["structuralCandidates"])
            tactical_verification_candidates = int(
                snapshot["tacticalVerificationCandidates"]
            )
            tactical_verification_decisions = int(
                snapshot["tacticalVerificationDecisions"]
            )
            tactical_verification_seconds = float(
                snapshot["tacticalVerificationSeconds"]
            )
            tactical_viability_pre_screen_candidates = int(
                snapshot["tacticalViabilityPreScreenCandidates"]
            )
            tactical_viability_pre_screen_rejected = int(
                snapshot["tacticalViabilityPreScreenRejected"]
            )
            tactical_viability_pre_screen_seconds = float(
                snapshot["tacticalViabilityPreScreenSeconds"]
            )
            del events[int(snapshot["eventCount"]) :]
            del result_codes[int(snapshot["resultCodeCount"]) :]

        def disagreement_reference(
            disagreement: SemanticDisagreementV2,
        ) -> dict[str, str]:
            return {
                "disagreementId": disagreement.disagreement_id,
                "sha256": disagreement.sha256,
                "category": disagreement.category,
                "status": disagreement.status,
            }

        def finalize_audit_trace(
            record: Any,
            *,
            disposition: str,
            failures: list[str] | tuple[str, ...],
            disagreements: list[SemanticDisagreementV2]
            | tuple[SemanticDisagreementV2, ...] = (),
        ) -> None:
            kwargs = {
                "disposition": disposition,
                "failure_categories": list(failures),
                "disagreement_references": [
                    disagreement_reference(item) for item in disagreements
                ],
            }
            if staged_candidate_effects is not None:
                staged_candidate_effects.append(
                    ("finalize_audit_trace", (record.record_hash,), kwargs)
                )
                return
            self.audit_traces.finalize(record.record_hash, **kwargs)

        def retain_completed_trace(
            simulation: Any,
            candidate: Any,
            seed: int,
        ) -> Any | None:
            nonlocal quarantine_count, semantic_replay_passed
            trusted_final_state = getattr(
                simulation.final_trace,
                "trusted_final_state",
                None,
            )
            if isinstance(trusted_final_state, Mapping) and str(
                trusted_final_state.get("phase")
            ) not in {GamePhase.COMPLETE.value, GamePhase.SCORED.value}:
                retention_evaluations[str(candidate.candidate_hash)] = (
                    build_candidate_evaluation(
                        candidate,
                        SequentialCandidateOutcomeKindV1.INCOMPLETE,
                        ("authoritative_replay_incomplete",),
                        evidence={
                            "stage": "authoritative_complete_trace_replay",
                            "traceHash": simulation.final_trace.sha256,
                            "phase": trusted_final_state.get("phase"),
                        },
                    )
                )
                return None
            source = {
                "kind": "research_candidate",
                "runCode": run_code,
                "runHash": str(run["runHash"]),
                "recipeCode": recipe.code,
                "recipeHash": recipe.sha256,
                "shardId": str(shard["shardId"]),
                "shardHash": str(shard["sha256"]),
                "shardIndex": int(shard["shardIndex"]),
                "seed": int(seed),
                "candidateHash": str(candidate.candidate_hash),
                "candidateOrdinal": int(candidate.ordinal),
                "generationLane": generation_lane,
                "attemptCount": int(shard["attemptCount"]),
                "fencingToken": int(shard["fencingToken"]),
                "workerId": worker_id,
            }
            try:
                record = self.audit_traces.retain(
                    simulation.final_trace,
                    rules=simulator.rules,
                    source=source,
                    research_configuration=(
                        recipe.configuration
                        if isinstance(recipe, ResearchRecipeV3)
                        else None
                    ),
                )
            except (AssertionError, ValueError) as exc:
                message = str(exc)
                lower_message = message.lower()
                source_conflict = "source occurrence" in lower_message
                observer_failure = any(
                    marker in lower_message
                    for marker in ("observer", "projection", "hidden", "private")
                )
                category = (
                    "observer_information_defect"
                    if observer_failure
                    else "runtime_transition_defect"
                )
                stage = (
                    "candidate_output_determinism_admission"
                    if source_conflict
                    else (
                        "observer_safety_projection"
                        if observer_failure
                        else "authoritative_complete_trace_replay"
                    )
                )
                disagreement = SemanticDisagreementV2.build(
                    run_code=run_code,
                    candidate_hash=candidate.candidate_hash,
                    category=category,
                    solver_evidence={
                        **candidate.provenance_dict(),
                        **standalone_configuration_evidence,
                        "stage": "audit_trace_admission",
                    },
                    runtime_evidence={
                        **standalone_configuration_evidence,
                        "stage": stage,
                        "traceHash": simulation.final_trace.sha256,
                        "publishableLaffa": bool(simulation.publishable_laffa),
                        "errorType": type(exc).__name__,
                        "message": message,
                    },
                )
                put_candidate_disagreement(
                    disagreement.to_dict(),
                    shard_id=str(shard["shardId"]),
                    attempt_count=int(shard["attemptCount"]),
                    fencing_token=int(shard["fencingToken"]),
                    worker_id=worker_id,
                )
                quarantine_count += 1
                retention_evaluations[str(candidate.candidate_hash)] = (
                    build_candidate_evaluation(
                        candidate,
                        SequentialCandidateOutcomeKindV1.FATAL_QUARANTINE,
                        (category,),
                        evidence={
                            "stage": stage,
                            "traceHash": simulation.final_trace.sha256,
                            "disagreementHash": disagreement.sha256,
                        },
                    )
                )
                events.append(
                    {
                        "type": "semantic_replay_quarantine",
                        "seed": int(seed),
                        "candidateHash": candidate.candidate_hash,
                        "stage": stage,
                        "category": category,
                        "disagreementId": disagreement.disagreement_id,
                        "message": message,
                        "at": _now(),
                    }
                )
                return None
            semantic_replay_passed += 1
            return record

        try:
            for seed_index, seed in enumerate(seeds):
                if should_cancel():
                    return
                self.store.heartbeat_shard(
                    str(shard["shardId"]),
                    worker_id=worker_id,
                    fencing_token=int(shard["fencingToken"]),
                    lease_seconds=self.shard_lease_seconds,
                )
                attempted_seeds += 1
                if generation_lane == "organic_runtime":
                    simulation_started = time.perf_counter()
                    organic_simulation = simulator.simulate(
                        seed,
                        dealer=Seat.parse(recipe.situation["dealer"]),
                        match_score=tuple(
                            int(item) for item in recipe.situation["matchScore"]
                        ),
                        cancel_check=should_cancel,
                    )
                    simulation_seconds += time.perf_counter() - simulation_started
                    organic_model_hash = canonical_sha256(
                        {
                            "schemaVersion": 1,
                            "kind": "organic_runtime_baseline_v1",
                            "rulesHash": simulator.rules.sha256,
                            "agentRelease": simulator.agent_release,
                        }
                    )
                    organic_candidate = _OrganicResearchCandidateV1(
                        0,
                        canonical_sha256(
                            {
                                "schemaVersion": 1,
                                "kind": "organic_research_candidate_v1",
                                "recipeHash": recipe.sha256,
                                "lane": generation_lane,
                                "seed": seed,
                                "ordinal": 0,
                            }
                        ),
                        organic_model_hash,
                        organic_simulation.final_trace.engine_version,
                    )
                    candidate_rows = ((organic_candidate, organic_simulation),)
                else:
                    spec = self._compile_deal_spec(recipe, seed=seed, context=context)
                    cp_sat_worker_counts.add(int(spec.solver_workers))
                    candidate_rows = ()

                def evaluate_candidate(
                    candidate: Any,
                    _evaluation_index: int,
                    *,
                    prepared_simulation: Any | None = None,
                ) -> SequentialCandidateEvaluationV1:
                    nonlocal analysis_seconds, analyzed_candidate_count
                    nonlocal auction_feasible, construction_pre_screen_candidates
                    nonlocal construction_pre_screen_rejected
                    nonlocal construction_pre_screen_seconds, construction_seconds
                    nonlocal first_feasible_candidate_seconds
                    nonlocal first_optimal_candidate_seconds
                    nonlocal first_qualified_result_seconds, play_compiler_attempted
                    nonlocal play_compiler_feasible, play_compiler_incomplete
                    nonlocal play_compiler_optimal, qualified_count, quarantine_count
                    nonlocal rejected_count, result_count, runtime_parity_passed
                    nonlocal semantic_parity_seconds, simulation_seconds
                    nonlocal solver_optimal
                    nonlocal strategic_exact_decisions, strategic_fallback_decisions
                    nonlocal strategic_provisional_decisions
                    nonlocal strategic_selection_decisions, structural_candidates
                    nonlocal tactical_verification_candidates
                    nonlocal tactical_verification_decisions
                    nonlocal tactical_verification_seconds
                    nonlocal tactical_viability_pre_screen_candidates
                    nonlocal tactical_viability_pre_screen_rejected
                    nonlocal tactical_viability_pre_screen_seconds
                    nonlocal construction_target_miss_count
                    if should_cancel():
                        return build_candidate_evaluation(
                            candidate,
                            SequentialCandidateOutcomeKindV1.CANCELLED,
                            ("candidate_evaluation_cancelled",),
                            evidence={"stage": "candidate_evaluation_admission"},
                        )
                    self.store.heartbeat_shard(
                        str(shard["shardId"]),
                        worker_id=worker_id,
                        fencing_token=int(shard["fencingToken"]),
                        lease_seconds=self.shard_lease_seconds,
                    )
                    structural_candidates += 1
                    candidate_status = str(candidate.solver_status).lower()
                    if (
                        first_feasible_candidate_seconds is None
                        and candidate_status in {"feasible", "optimal"}
                    ):
                        first_feasible_candidate_seconds = (
                            time.perf_counter() - started_clock
                        )
                    if (
                        first_optimal_candidate_seconds is None
                        and candidate_status == "optimal"
                    ):
                        first_optimal_candidate_seconds = (
                            time.perf_counter() - started_clock
                        )
                    solver_optimal += int(candidate_status == "optimal")
                    auction_plan: AuctionPlanV1 | None = None
                    escalation_plan: EscalationPlanV1 | None = None
                    compiled_project_plan: ProjectPlanV1 | None = None
                    compiled_play_plan: PlayPlanV1 | None = None
                    if generation_lane == "organic_runtime":
                        if prepared_simulation is None:
                            raise AssertionError(
                                "organic lane lost its runtime simulation"
                            )
                        simulation = prepared_simulation
                        audit_trace_record = retain_completed_trace(
                            simulation,
                            candidate,
                            seed,
                        )
                        if audit_trace_record is None:
                            return retention_evaluations.get(
                                str(candidate.candidate_hash)
                            ) or build_candidate_evaluation(
                                candidate,
                                SequentialCandidateOutcomeKindV1.INCOMPLETE,
                                ("authoritative_replay_incomplete",),
                                evidence={
                                    "stage": "authoritative_complete_trace_replay",
                                    "traceHash": simulation.final_trace.sha256,
                                },
                            )
                        if not simulation.publishable_laffa:
                            events.append(
                                {
                                    "type": "recipe_infeasible",
                                    "seed": seed,
                                    "candidateHash": candidate.candidate_hash,
                                    "reason": "organic_runtime_all_pass",
                                    "at": _now(),
                                }
                            )
                            if audit_trace_record is not None:
                                finalize_audit_trace(
                                    audit_trace_record,
                                    disposition="rejected",
                                    failures=[
                                        "stage_organic_simulation",
                                        "organic_all_pass_complete_trace",
                                    ],
                                )
                            return build_candidate_evaluation(
                                candidate,
                                SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
                                ("organic_all_pass_complete_trace",),
                                evidence={
                                    "stage": "organic_simulation",
                                    "traceHash": simulation.final_trace.sha256,
                                },
                            )
                        auction_feasible += 1
                    else:
                        play_construction_mode = str(
                            resources.get("playConstructionMode")
                            or "cp_sat_feasibility_line"
                        )
                        auction_construction_mode = str(
                            resources.get("auctionConstructionMode")
                            or "cp_sat_feasibility_line"
                        )
                        scripted_plays: tuple[Mapping[str, Any], ...] | None = None
                        scripted_play_provenance: Mapping[str, Any] | None = None
                        play_selector: ActorSafeCpSatPlaySelectorV1 | None = None
                        if play_construction_mode == "cp_sat_actor_safe_late_game_v1":
                            play_selector = ActorSafeCpSatPlaySelectorV1(
                                ActorSafeCpSatPlayProfileV1.build(
                                    dict(resources.get("strategicPlay") or {})
                                )
                            )

                        scripted_bids: tuple[Mapping[str, Any], ...] | None = None
                        scripted_escalations: tuple[Mapping[str, Any], ...] | None = (
                            None
                        )
                        bid_selector: ActorSafeCpSatAuctionSelectorV1 | None = None
                        if auction_construction_mode == "cp_sat_actor_safe_decision_v1":
                            bid_selector = ActorSafeCpSatAuctionSelectorV1(
                                ActorSafeCpSatAuctionProfileV1.build(
                                    dict(resources.get("strategicAuction") or {})
                                ),
                                exact_search_backend=simulator.exact_search_backend,
                                trace_analysis_profile=(
                                    context.analyzers["deep"].profile
                                ),
                                analysis_identity_hash=(
                                    context.analysis_identity_hashes["deep"]
                                ),
                            )
                        if auction_construction_mode == "cp_sat_feasibility_line":
                            construction_started = time.perf_counter()
                            auction_plan = self._compile_auction(
                                recipe, candidate.deck_order, context=context
                            )
                            construction_seconds += (
                                time.perf_counter() - construction_started
                            )
                            if auction_plan is None:
                                solver_evidence = {
                                    **candidate.provenance_dict(),
                                    "stage": "auction_construction",
                                }
                                events.append(
                                    {
                                        "type": "recipe_infeasible",
                                        "seed": seed,
                                        "candidateHash": candidate.candidate_hash,
                                        "reason": "auction_infeasible",
                                        "solverEvidence": solver_evidence,
                                        "at": _now(),
                                    }
                                )
                                return build_candidate_evaluation(
                                    candidate,
                                    SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
                                    ("auction_infeasible",),
                                    evidence=solver_evidence,
                                )
                            construction_started = time.perf_counter()
                            escalation_plan = self._compile_escalation_plan(
                                recipe, auction_plan, context=context
                            )
                            construction_seconds += (
                                time.perf_counter() - construction_started
                            )
                            if escalation_plan is None:
                                events.append(
                                    {
                                        "type": "recipe_infeasible",
                                        "seed": seed,
                                        "candidateHash": candidate.candidate_hash,
                                        "reason": "escalation_infeasible",
                                        "at": _now(),
                                    }
                                )
                                return build_candidate_evaluation(
                                    candidate,
                                    SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
                                    ("escalation_infeasible",),
                                    evidence={"stage": "escalation_construction"},
                                )
                            scripted_bids = tuple(
                                item.to_dict() for item in auction_plan.actions
                            )
                            scripted_escalations = tuple(
                                item.to_dict() for item in escalation_plan.actions
                            )

                        if play_construction_mode == "cp_sat_feasibility_line":
                            if auction_plan is None or escalation_plan is None:
                                raise AssertionError(
                                    "feasibility play requires a compiled auction"
                                )
                            construction_started = time.perf_counter()
                            play_compiler_attempted += 1
                            try:
                                (
                                    compiled_project_plan,
                                    compiled_play_plan,
                                    scripted_plays,
                                ) = self._compile_complete_play(
                                    recipe,
                                    candidate.deck_order,
                                    auction_plan,
                                    escalation_plan,
                                    objective_target_seat=spec.objective_target_seat,
                                    context=context,
                                )
                            except ValueError as exc:
                                construction_seconds += (
                                    time.perf_counter() - construction_started
                                )
                                events.append(
                                    {
                                        "type": "recipe_infeasible",
                                        "seed": seed,
                                        "candidateHash": candidate.candidate_hash,
                                        "reason": "complete_play_infeasible",
                                        "message": str(exc),
                                        "at": _now(),
                                    }
                                )
                                return build_candidate_evaluation(
                                    candidate,
                                    SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
                                    ("complete_play_infeasible",),
                                    evidence={
                                        "stage": "complete_play",
                                        "message": str(exc),
                                    },
                                )
                            except RuntimeError as exc:
                                play_compiler_incomplete += 1
                                construction_seconds += (
                                    time.perf_counter() - construction_started
                                )
                                events.append(
                                    {
                                        "type": "solver_incomplete",
                                        "seed": seed,
                                        "candidateHash": candidate.candidate_hash,
                                        "stage": "complete_play",
                                        "message": str(exc),
                                        "at": _now(),
                                    }
                                )
                                return build_candidate_evaluation(
                                    candidate,
                                    SequentialCandidateOutcomeKindV1.INCOMPLETE,
                                    ("complete_play_solver_incomplete",),
                                    evidence={
                                        "stage": "complete_play",
                                        "message": str(exc),
                                    },
                                )
                            construction_seconds += (
                                time.perf_counter() - construction_started
                            )
                            scripted_play_provenance = {
                                "schemaVersion": 1,
                                "pluginId": self.play_compiler.descriptor.id,
                                "modelHash": compiled_play_plan.model_hash,
                                "planHash": compiled_play_plan.sha256,
                                "projectPlanHash": compiled_project_plan.sha256,
                                "solverStatus": compiled_play_plan.solver_status,
                                "selectionAuthority": "cp_sat_feasibility",
                                "tacticalPrecursor": (
                                    compiled_play_plan.tactical_precursor
                                ),
                            }
                        simulation_started = time.perf_counter()
                        simulation = simulator.simulate_preordered_deck(
                            seed,
                            candidate.deck_order,
                            deal_generator_provenance={
                                **candidate.provenance_dict(),
                                "auctionConstructionMode": auction_construction_mode,
                                "playConstructionMode": play_construction_mode,
                                "playPolicyRelease": getattr(
                                    simulator,
                                    "agent_release",
                                    str(recipe.profiles["agentRelease"]),
                                ),
                            },
                            dealer=Seat.parse(recipe.situation["dealer"]),
                            match_score=tuple(
                                int(item) for item in recipe.situation["matchScore"]
                            ),
                            scripted_bids=scripted_bids,
                            scripted_escalations=scripted_escalations,
                            scripted_plays=scripted_plays,
                            scripted_play_provenance=scripted_play_provenance,
                            bid_selector=bid_selector,
                            play_selector=play_selector,
                            cancel_check=should_cancel,
                        )
                        simulation_seconds += time.perf_counter() - simulation_started
                        audit_trace_record = retain_completed_trace(
                            simulation,
                            candidate,
                            seed,
                        )
                        if audit_trace_record is None:
                            return retention_evaluations.get(
                                str(candidate.candidate_hash)
                            ) or build_candidate_evaluation(
                                candidate,
                                SequentialCandidateOutcomeKindV1.INCOMPLETE,
                                ("authoritative_replay_incomplete",),
                                evidence={
                                    "stage": "authoritative_complete_trace_replay",
                                    "traceHash": simulation.final_trace.sha256,
                                },
                            )
                        if not simulation.publishable_laffa:
                            events.append(
                                {
                                    "type": "recipe_infeasible",
                                    "seed": seed,
                                    "candidateHash": candidate.candidate_hash,
                                    "reason": "constructed_auction_all_pass",
                                    "at": _now(),
                                }
                            )
                            if audit_trace_record is not None:
                                finalize_audit_trace(
                                    audit_trace_record,
                                    disposition="rejected",
                                    failures=[
                                        "stage_candidate_simulation",
                                        "all_pass_unscored_complete_trace",
                                    ],
                                )
                            return build_candidate_evaluation(
                                candidate,
                                SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
                                ("all_pass_unscored_complete_trace",),
                                evidence={
                                    "stage": "candidate_simulation",
                                    "traceHash": simulation.final_trace.sha256,
                                },
                            )
                        if auction_construction_mode in {
                            "actor_safe_agent_verified",
                            "cp_sat_actor_safe_decision_v1",
                        }:
                            construction_started = time.perf_counter()
                            auction_plan = self._compile_auction(
                                recipe,
                                candidate.deck_order,
                                required_actions=self._auction_actions_from_trace(
                                    simulation.final_trace
                                ),
                                enforce_recipe=False,
                                context=context,
                            )
                            construction_seconds += (
                                time.perf_counter() - construction_started
                            )
                            if auction_plan is None:
                                events.append(
                                    {
                                        "type": "compiler_parity_defect",
                                        "seed": seed,
                                        "candidateHash": candidate.candidate_hash,
                                        "stage": "actor_safe_auction_verification",
                                        "at": _now(),
                                    }
                                )
                                disagreement = SemanticDisagreementV2.build(
                                    run_code=run_code,
                                    candidate_hash=candidate.candidate_hash,
                                    category="compiler_parity_defect",
                                    solver_evidence={
                                        **candidate.provenance_dict(),
                                        **standalone_configuration_evidence,
                                        "stage": "actor_safe_auction_verification",
                                    },
                                    runtime_evidence={
                                        **standalone_configuration_evidence,
                                        "traceHash": simulation.final_trace.sha256,
                                        "auctionActions": self._auction_actions_from_trace(
                                            simulation.final_trace
                                        ),
                                    },
                                )
                                put_candidate_disagreement(
                                    disagreement.to_dict(),
                                    shard_id=str(shard["shardId"]),
                                    attempt_count=int(shard["attemptCount"]),
                                    fencing_token=int(shard["fencingToken"]),
                                    worker_id=worker_id,
                                )
                                quarantine_count += 1
                                finalize_audit_trace(
                                    audit_trace_record,
                                    disposition="quarantined",
                                    failures=[
                                        "stage_actor_safe_auction_verification",
                                        "compiler_parity_defect",
                                    ],
                                    disagreements=[disagreement],
                                )
                                return build_candidate_evaluation(
                                    candidate,
                                    SequentialCandidateOutcomeKindV1.FATAL_QUARANTINE,
                                    ("compiler_parity_defect",),
                                    evidence={
                                        "stage": "actor_safe_auction_verification",
                                        "traceHash": simulation.final_trace.sha256,
                                        "disagreementHash": disagreement.sha256,
                                    },
                                )
                            escalation_plan = self._compile_escalation_plan(
                                recipe, auction_plan, context=context
                            )
                            if escalation_plan is None or escalation_plan.actions:
                                events.append(
                                    {
                                        "type": "compiler_parity_defect",
                                        "seed": seed,
                                        "candidateHash": candidate.candidate_hash,
                                        "stage": "actor_safe_escalation_verification",
                                        "at": _now(),
                                    }
                                )
                                disagreement = SemanticDisagreementV2.build(
                                    run_code=run_code,
                                    candidate_hash=candidate.candidate_hash,
                                    category="compiler_parity_defect",
                                    solver_evidence={
                                        **candidate.provenance_dict(),
                                        **standalone_configuration_evidence,
                                        "stage": "actor_safe_escalation_verification",
                                    },
                                    runtime_evidence={
                                        **standalone_configuration_evidence,
                                        "traceHash": simulation.final_trace.sha256,
                                        "escalation": simulation.final_trace.trusted_final_state.get(
                                            "escalation"
                                        ),
                                    },
                                )
                                put_candidate_disagreement(
                                    disagreement.to_dict(),
                                    shard_id=str(shard["shardId"]),
                                    attempt_count=int(shard["attemptCount"]),
                                    fencing_token=int(shard["fencingToken"]),
                                    worker_id=worker_id,
                                )
                                quarantine_count += 1
                                finalize_audit_trace(
                                    audit_trace_record,
                                    disposition="quarantined",
                                    failures=[
                                        "stage_actor_safe_escalation_verification",
                                        "compiler_parity_defect",
                                    ],
                                    disagreements=[disagreement],
                                )
                                return build_candidate_evaluation(
                                    candidate,
                                    SequentialCandidateOutcomeKindV1.FATAL_QUARANTINE,
                                    ("compiler_parity_defect",),
                                    evidence={
                                        "stage": "actor_safe_escalation_verification",
                                        "traceHash": simulation.final_trace.sha256,
                                        "disagreementHash": disagreement.sha256,
                                    },
                                )
                        auction_feasible += 1
                        if play_construction_mode in {
                            "actor_safe_agent_verified",
                            "cp_sat_actor_safe_late_game_v1",
                        }:
                            if auction_plan is None or escalation_plan is None:
                                raise AssertionError(
                                    "actor-safe play verification lost its auction proof"
                                )
                            required_cards, required_ekka_leads = self._played_line(
                                simulation
                            )
                            construction_started = time.perf_counter()
                            play_compiler_attempted += 1
                            try:
                                (
                                    compiled_project_plan,
                                    compiled_play_plan,
                                    _verified_plays,
                                ) = self._compile_complete_play(
                                    recipe,
                                    candidate.deck_order,
                                    auction_plan,
                                    escalation_plan,
                                    required_cards=required_cards,
                                    required_ekka_leads=required_ekka_leads,
                                    context=context,
                                )
                            except (ValueError, RuntimeError) as exc:
                                if isinstance(exc, RuntimeError):
                                    play_compiler_incomplete += 1
                                construction_seconds += (
                                    time.perf_counter() - construction_started
                                )
                                event_type = (
                                    "compiler_parity_defect"
                                    if isinstance(exc, ValueError)
                                    else "solver_incomplete"
                                )
                                events.append(
                                    {
                                        "type": event_type,
                                        "seed": seed,
                                        "candidateHash": candidate.candidate_hash,
                                        "stage": "complete_play_verification",
                                        "message": str(exc),
                                        "at": _now(),
                                    }
                                )
                                if isinstance(exc, ValueError):
                                    disagreement = SemanticDisagreementV2.build(
                                        run_code=run_code,
                                        candidate_hash=candidate.candidate_hash,
                                        category="compiler_parity_defect",
                                        solver_evidence={
                                            **candidate.provenance_dict(),
                                            **standalone_configuration_evidence,
                                            "stage": "complete_play_verification",
                                            "message": str(exc),
                                        },
                                        runtime_evidence={
                                            **standalone_configuration_evidence,
                                            "traceHash": simulation.final_trace.sha256,
                                            "publishableLaffa": simulation.publishable_laffa,
                                            "playConstructionMode": play_construction_mode,
                                        },
                                    )
                                    put_candidate_disagreement(
                                        disagreement.to_dict(),
                                        shard_id=str(shard["shardId"]),
                                        attempt_count=int(shard["attemptCount"]),
                                        fencing_token=int(shard["fencingToken"]),
                                        worker_id=worker_id,
                                    )
                                    quarantine_count += 1
                                    finalize_audit_trace(
                                        audit_trace_record,
                                        disposition="quarantined",
                                        failures=[
                                            "stage_complete_play_verification",
                                            "compiler_parity_defect",
                                        ],
                                        disagreements=[disagreement],
                                    )
                                else:
                                    finalize_audit_trace(
                                        audit_trace_record,
                                        disposition="rejected",
                                        failures=[
                                            "stage_complete_play_verification",
                                            "solver_incomplete",
                                        ],
                                    )
                                if isinstance(exc, ValueError):
                                    return build_candidate_evaluation(
                                        candidate,
                                        SequentialCandidateOutcomeKindV1.FATAL_QUARANTINE,
                                        ("compiler_parity_defect",),
                                        evidence={
                                            "stage": "complete_play_verification",
                                            "traceHash": simulation.final_trace.sha256,
                                            "disagreementHash": disagreement.sha256,
                                        },
                                    )
                                return build_candidate_evaluation(
                                    candidate,
                                    SequentialCandidateOutcomeKindV1.INCOMPLETE,
                                    ("complete_play_solver_incomplete",),
                                    evidence={
                                        "stage": "complete_play_verification",
                                        "traceHash": simulation.final_trace.sha256,
                                        "message": str(exc),
                                    },
                                )
                            construction_seconds += (
                                time.perf_counter() - construction_started
                            )
                        if compiled_play_plan is None:
                            raise AssertionError(
                                "Research lost the complete-play proof"
                            )
                        play_compiler_optimal += int(
                            compiled_play_plan.solver_status == "optimal"
                        )
                        play_compiler_feasible += int(
                            compiled_play_plan.solver_status == "feasible"
                        )
                    strategic_summary = _strategic_selection_summary(
                        simulation.final_trace.trusted_final_state
                    )
                    strategic_selection_decisions += int(
                        strategic_summary["decisionCount"]
                    )
                    strategic_exact_decisions += int(
                        strategic_summary["exactDecisionCount"]
                    )
                    strategic_provisional_decisions += int(
                        strategic_summary["provisionalDecisionCount"]
                    )
                    strategic_fallback_decisions += int(
                        strategic_summary["fallbackDecisionCount"]
                    )
                    parity_started = time.perf_counter()
                    actor_safe_auction = generation_lane == "ortools" and str(
                        resources.get("auctionConstructionMode") or ""
                    ) in {
                        "actor_safe_agent_verified",
                        "cp_sat_actor_safe_decision_v1",
                    }
                    target_mismatches = list(
                        self._policy_target_mismatches(recipe, simulation)
                        if actor_safe_auction
                        else ()
                    )
                    if (
                        actor_safe_auction
                        and spec.assumed_buyer is not None
                        and recipe.situation.get("buyer") is None
                        and recipe.tactics
                        and str(simulation.final_trace.trusted_final_state.get("buyer"))
                        != spec.assumed_buyer.name.lower()
                    ):
                        target_mismatches.append("buyer_target_miss")
                    target_mismatches = sorted(set(target_mismatches))
                    mismatches = self._semantic_mismatches(
                        recipe,
                        simulation,
                        context=context,
                        include_policy_target_mismatches=not actor_safe_auction,
                    )
                    if compiled_play_plan is not None:
                        mismatches = sorted(
                            set(
                                [
                                    *mismatches,
                                    *self._compiled_play_mismatches(
                                        compiled_play_plan,
                                        simulation,
                                        project_plan=compiled_project_plan,
                                        project_domain=simulator.rules.project_domain,
                                        scoring_domain=ScoringDomainIRV1.from_rules(
                                            simulator.rules.data
                                        ),
                                        structural_domain=(
                                            simulator.rules.structural_domain
                                        ),
                                        expected_deck_order=tuple(
                                            int(item) for item in candidate.deck_order
                                        ),
                                        expected_deck_commitment=(
                                            hashlib.sha256(
                                                bytes(candidate.deck_order)
                                            ).hexdigest()
                                        ),
                                        expected_candidate_hash=(
                                            candidate.candidate_hash
                                        ),
                                        expected_dealer=recipe.situation["dealer"],
                                        expected_round_number=1,
                                        expected_next_dealer=(
                                            simulator.rules.structural_domain.next_dealer(
                                                Seat.parse(recipe.situation["dealer"])
                                            )
                                        ),
                                        starting_match_score=tuple(
                                            int(item)
                                            for item in recipe.situation["matchScore"]
                                        ),
                                    ),
                                ]
                            )
                        )
                    semantic_parity_seconds += time.perf_counter() - parity_started
                    if mismatches:
                        if generation_lane == "organic_runtime":
                            events.append(
                                {
                                    "type": "organic_recipe_miss",
                                    "seed": seed,
                                    "candidateHash": candidate.candidate_hash,
                                    "reasons": list(mismatches),
                                    "at": _now(),
                                }
                            )
                            finalize_audit_trace(
                                audit_trace_record,
                                disposition="rejected",
                                failures=[
                                    "stage_recipe_semantic_screen",
                                    "organic_recipe_miss",
                                    *[
                                        f"recipe_mismatch:{category}"
                                        for category in mismatches
                                    ],
                                ],
                            )
                            return build_candidate_evaluation(
                                candidate,
                                SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
                                (
                                    "organic_recipe_miss",
                                    *(
                                        f"recipe_mismatch:{category}"
                                        for category in mismatches
                                    ),
                                ),
                                evidence={
                                    "stage": "recipe_semantic_screen",
                                    "traceHash": simulation.final_trace.sha256,
                                    "mismatches": list(mismatches),
                                },
                            )
                        semantic_disagreements: list[SemanticDisagreementV2] = []
                        for category in mismatches:
                            disagreement = SemanticDisagreementV2.build(
                                run_code=run_code,
                                candidate_hash=candidate.candidate_hash,
                                category=category,
                                solver_evidence={
                                    **candidate.provenance_dict(),
                                    **standalone_configuration_evidence,
                                },
                                runtime_evidence={
                                    **standalone_configuration_evidence,
                                    "traceHash": simulation.final_trace.sha256,
                                    "publishableLaffa": simulation.publishable_laffa,
                                    "buyer": simulation.final_trace.trusted_final_state.get(
                                        "buyer"
                                    ),
                                    "contract": simulation.final_trace.trusted_final_state.get(
                                        "contract"
                                    ),
                                    "projects": simulation.final_trace.trusted_final_state.get(
                                        "projects"
                                    ),
                                },
                            )
                            put_candidate_disagreement(
                                disagreement.to_dict(),
                                shard_id=str(shard["shardId"]),
                                attempt_count=int(shard["attemptCount"]),
                                fencing_token=int(shard["fencingToken"]),
                                worker_id=worker_id,
                            )
                            semantic_disagreements.append(disagreement)
                            quarantine_count += 1
                        finalize_audit_trace(
                            audit_trace_record,
                            disposition="quarantined",
                            failures=[
                                "stage_semantic_parity_verification",
                                *mismatches,
                            ],
                            disagreements=semantic_disagreements,
                        )
                        return build_candidate_evaluation(
                            candidate,
                            SequentialCandidateOutcomeKindV1.FATAL_QUARANTINE,
                            tuple(mismatches),
                            evidence={
                                "stage": "semantic_parity_verification",
                                "traceHash": simulation.final_trace.sha256,
                                "disagreementHashes": [
                                    item.sha256 for item in semantic_disagreements
                                ],
                            },
                        )
                    if target_mismatches:
                        state = simulation.final_trace.trusted_final_state
                        winning_records = [
                            item
                            for item in state.get("biddingHistory") or ()
                            if item.get("buyer") is not None
                            and item.get("contract") is not None
                        ]
                        construction_target_miss_count += 1
                        events.append(
                            {
                                "type": "construction_target_miss",
                                "seed": seed,
                                "candidateHash": candidate.candidate_hash,
                                "stage": "actor_safe_policy_target_verification",
                                "reasons": target_mismatches,
                                "expected": {
                                    "buyer": recipe.situation.get("buyer"),
                                    "assumedBuyer": (
                                        spec.assumed_buyer.name.lower()
                                        if spec.assumed_buyer is not None
                                        else None
                                    ),
                                    "contracts": list(
                                        recipe.situation.get("contracts") or ()
                                    ),
                                    "purchaseRound": recipe.situation.get(
                                        "purchaseRound"
                                    ),
                                    "trumpSuit": recipe.situation.get("trumpSuit"),
                                    "escalation": (
                                        tuple(
                                            recipe.situation.get("escalations") or ()
                                        )[0]
                                        if recipe.situation.get("escalations")
                                        else "normal"
                                    ),
                                },
                                "actual": {
                                    "buyer": state.get("buyer"),
                                    "contract": state.get("contract"),
                                    "purchaseRound": (
                                        winning_records[-1].get("round")
                                        if winning_records
                                        else None
                                    ),
                                    "trumpSuit": state.get("trumpSuit"),
                                    "escalation": state.get("escalation") or "normal",
                                },
                                "at": _now(),
                            }
                        )
                        finalize_audit_trace(
                            audit_trace_record,
                            disposition="rejected",
                            failures=[
                                "stage_actor_safe_policy_target_verification",
                                "construction_target_miss",
                                *target_mismatches,
                            ],
                        )
                        return build_candidate_evaluation(
                            candidate,
                            SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
                            ("construction_target_miss", *target_mismatches),
                            evidence={
                                "stage": "actor_safe_policy_target_verification",
                                "traceHash": simulation.final_trace.sha256,
                                "targetMismatches": list(target_mismatches),
                            },
                        )
                    runtime_parity_passed += int(generation_lane == "ortools")
                    if self._requires_finalist_primary_policy(
                        recipe
                    ) and not self._uses_finalist_primary_policy(recipe):
                        events.append(
                            {
                                "type": "finalist_primary_policy_required",
                                "seed": seed,
                                "candidateHash": candidate.candidate_hash,
                                "auctionConstructionMode": resources.get(
                                    "auctionConstructionMode"
                                ),
                                "playConstructionMode": resources.get(
                                    "playConstructionMode"
                                ),
                                "at": _now(),
                            }
                        )
                        finalize_audit_trace(
                            audit_trace_record,
                            disposition="rejected",
                            failures=[
                                "stage_finalist_primary_policy_admission",
                                "finalist_primary_policy_required",
                            ],
                        )
                        return build_candidate_evaluation(
                            candidate,
                            SequentialCandidateOutcomeKindV1.INCOMPLETE,
                            ("finalist_primary_policy_unavailable",),
                            evidence={
                                "stage": "finalist_primary_policy_admission",
                                "traceHash": simulation.final_trace.sha256,
                            },
                        )
                    if tactical_viability_pre_screen is not None:
                        tactical_pre_screen_started = time.perf_counter()
                        tactical_pre_screen_evidence = (
                            tactical_viability_pre_screen.evaluate(
                                simulation.final_trace,
                                required_tactics=tuple(
                                    sorted(
                                        {
                                            str(item["kind"])
                                            for item in recipe.tactics
                                            if item["mode"]
                                            in {"hard", "qualification_gate"}
                                        }
                                    )
                                ),
                                cancel_check=should_cancel,
                            )
                        )
                        tactical_viability_pre_screen_seconds += (
                            time.perf_counter() - tactical_pre_screen_started
                        )
                        tactical_viability_pre_screen_candidates += 1
                        tactical_pre_screen_bytes = json.dumps(
                            tactical_pre_screen_evidence.to_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        tactical_pre_screen_ref = self.artifact_store.put(
                            tactical_pre_screen_bytes,
                            media_type=(
                                "application/vnd.balootx."
                                "tactical-viability-pre-screen-v1+json"
                            ),
                            name=f"{tactical_pre_screen_evidence.sha256}.json",
                        )
                        events.append(
                            {
                                "type": "tactical_viability_pre_screen",
                                "seed": seed,
                                "candidateHash": candidate.candidate_hash,
                                "rejected": tactical_pre_screen_evidence.rejected,
                                "requiredTactics": list(
                                    tactical_pre_screen_evidence.required_tactics
                                ),
                                "foundTactics": list(
                                    tactical_pre_screen_evidence.found_tactics
                                ),
                                "status": tactical_pre_screen_evidence.probe.status,
                                "evidence": tactical_pre_screen_ref,
                                "at": _now(),
                            }
                        )
                        if tactical_pre_screen_evidence.rejected:
                            tactical_viability_pre_screen_rejected += 1
                            finalize_audit_trace(
                                audit_trace_record,
                                disposition="rejected",
                                failures=[
                                    "stage_tactical_viability_pre_screen",
                                    "tactical_viability_pre_screen_rejected",
                                    *[
                                        f"required_tactic_absent:{kind}"
                                        for kind in tactical_pre_screen_evidence.required_tactics
                                    ],
                                ],
                            )
                            return build_candidate_evaluation(
                                candidate,
                                SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
                                (
                                    "tactical_viability_pre_screen_rejected",
                                    *(
                                        f"required_tactic_absent:{kind}"
                                        for kind in tactical_pre_screen_evidence.required_tactics
                                    ),
                                ),
                                evidence={
                                    "stage": "tactical_viability_pre_screen",
                                    "traceHash": simulation.final_trace.sha256,
                                    "preScreenHash": tactical_pre_screen_evidence.sha256,
                                },
                            )
                    if pre_screen is not None and generation_lane == "ortools":
                        pre_screen_started = time.perf_counter()
                        pre_screen_evidence = pre_screen.evaluate(
                            simulation.final_trace,
                            analyzers["screening"],
                            cancel_check=should_cancel,
                        )
                        construction_pre_screen_seconds += (
                            time.perf_counter() - pre_screen_started
                        )
                        construction_pre_screen_candidates += 1
                        evidence_bytes = json.dumps(
                            pre_screen_evidence.to_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        evidence_ref = self.artifact_store.put(
                            evidence_bytes,
                            media_type=(
                                "application/vnd.balootx."
                                "construction-pre-screen-v1+json"
                            ),
                            name=f"{pre_screen_evidence.sha256}.json",
                        )
                        events.append(
                            {
                                "type": "construction_pre_screen",
                                "seed": seed,
                                "candidateHash": candidate.candidate_hash,
                                "rejected": pre_screen_evidence.rejected,
                                "reasons": list(pre_screen_evidence.rejection_reasons),
                                "evidence": evidence_ref,
                                "at": _now(),
                            }
                        )
                        if pre_screen_evidence.rejected:
                            construction_pre_screen_rejected += 1
                            finalize_audit_trace(
                                audit_trace_record,
                                disposition="rejected",
                                failures=[
                                    "stage_construction_pre_screen",
                                    "construction_pre_screen_rejected",
                                    *list(pre_screen_evidence.rejection_reasons),
                                ],
                            )
                            return build_candidate_evaluation(
                                candidate,
                                SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION,
                                (
                                    "construction_pre_screen_rejected",
                                    *pre_screen_evidence.rejection_reasons,
                                ),
                                evidence={
                                    "stage": "construction_pre_screen",
                                    "traceHash": simulation.final_trace.sha256,
                                    "preScreenHash": pre_screen_evidence.sha256,
                                },
                            )
                    analysis_started = time.perf_counter()
                    analyzed_candidate_count += 1
                    try:
                        analysis = analyzer.analyze_trace(
                            simulation.final_trace,
                            cancel_check=should_cancel,
                        )
                        direct_deep_analysis = analysis
                        decision_score_parity: dict[str, Any] | None = None
                        decision_score_parity_ref: Mapping[str, Any] | None = None
                        direct_deep_analysis_ref: Mapping[str, Any] | None = None
                        if self._uses_finalist_primary_policy(recipe):
                            if context is None:
                                raise ValueError(
                                    "V3 finalist admission requires its frozen "
                                    "execution context"
                                )
                            if analyzer is not analyzers["deep"]:
                                direct_deep_analysis = analyzers["deep"].analyze_trace(
                                    simulation.final_trace,
                                    cancel_check=should_cancel,
                                )
                    except SearchTimeBudgetExceeded as exc:
                        analysis_seconds += time.perf_counter() - analysis_started
                        events.append(
                            {
                                "type": "candidate_analysis_incomplete",
                                "seed": seed,
                                "candidateHash": candidate.candidate_hash,
                                "reason": "wall_clock_budget_exhausted",
                                "at": _now(),
                            }
                        )
                        finalize_audit_trace(
                            audit_trace_record,
                            disposition="rejected",
                            failures=[
                                "stage_candidate_analysis",
                                "candidate_analysis_incomplete",
                                "analysis_wall_clock_budget_exhausted",
                            ],
                        )
                        return build_candidate_evaluation(
                            candidate,
                            SequentialCandidateOutcomeKindV1.INCOMPLETE,
                            ("analysis_wall_clock_budget_exhausted",),
                            evidence={
                                "stage": "candidate_analysis",
                                "traceHash": simulation.final_trace.sha256,
                                "completionStatus": "proof_incomplete",
                                "reason": str(exc),
                            },
                        )
                    if self._uses_finalist_primary_policy(recipe):
                        decision_score_parity = self._decision_score_parity_evidence(
                            simulation.final_trace,
                            direct_deep_analysis,
                            context=context,
                        )
                        deep_analysis_bytes = json.dumps(
                            direct_deep_analysis.to_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        direct_deep_analysis_ref = self.artifact_store.put(
                            deep_analysis_bytes,
                            media_type=_PRIMARY_DECISION_SCORE_ANALYSIS_MEDIA_TYPE,
                            name=f"{direct_deep_analysis.sha256}.json",
                        )
                        parity_bytes = json.dumps(
                            decision_score_parity,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        decision_score_parity_ref = self.artifact_store.put(
                            parity_bytes,
                            media_type=_DECISION_SCORE_PARITY_MEDIA_TYPE,
                            name=f"{decision_score_parity['sha256']}.json",
                        )
                        if not bool(decision_score_parity["matched"]):
                            mismatch_reasons = sorted(
                                {
                                    str(item["reason"])
                                    for item in decision_score_parity["mismatches"]
                                }
                            )
                            disagreement = SemanticDisagreementV2.build(
                                run_code=run_code,
                                candidate_hash=candidate.candidate_hash,
                                category="compiler_parity_defect",
                                solver_evidence={
                                    **candidate.provenance_dict(),
                                    **standalone_configuration_evidence,
                                    "stage": "decision_score_parity",
                                    "traceHash": simulation.final_trace.sha256,
                                    "completeTrace": copy.deepcopy(
                                        dict(audit_trace_record.complete_trace)
                                    ),
                                    "parityEvidence": decision_score_parity_ref,
                                    "mismatchReasons": mismatch_reasons,
                                },
                                runtime_evidence={
                                    **standalone_configuration_evidence,
                                    "stage": "decision_score_parity",
                                    "traceHash": simulation.final_trace.sha256,
                                    "deepAnalysisHash": (direct_deep_analysis.sha256),
                                    "deepAnalysis": direct_deep_analysis_ref,
                                    "parityEvidence": decision_score_parity_ref,
                                    "mismatchReasons": mismatch_reasons,
                                },
                            )
                            put_candidate_disagreement(
                                disagreement.to_dict(),
                                shard_id=str(shard["shardId"]),
                                attempt_count=int(shard["attemptCount"]),
                                fencing_token=int(shard["fencingToken"]),
                                worker_id=worker_id,
                            )
                            quarantine_count += 1
                            events.append(
                                {
                                    "type": "compiler_parity_defect",
                                    "stage": "decision_score_parity",
                                    "seed": seed,
                                    "candidateHash": candidate.candidate_hash,
                                    "mismatchCount": len(
                                        decision_score_parity["mismatches"]
                                    ),
                                    "mismatchReasons": mismatch_reasons,
                                    "parityEvidence": decision_score_parity_ref,
                                    "at": _now(),
                                }
                            )
                            finalize_audit_trace(
                                audit_trace_record,
                                disposition="quarantined",
                                failures=[
                                    "stage_decision_score_parity",
                                    "compiler_parity_defect",
                                    *[
                                        f"decision_score_parity:{reason}"
                                        for reason in mismatch_reasons
                                    ],
                                ],
                                disagreements=[disagreement],
                            )
                            analysis_seconds += time.perf_counter() - analysis_started
                            return build_candidate_evaluation(
                                candidate,
                                SequentialCandidateOutcomeKindV1.FATAL_QUARANTINE,
                                (
                                    "decision_score_parity_defect",
                                    *(
                                        f"decision_score_parity:{item}"
                                        for item in mismatch_reasons
                                    ),
                                ),
                                evidence={
                                    "stage": "decision_score_parity",
                                    "traceHash": simulation.final_trace.sha256,
                                    "analysisHash": direct_deep_analysis.sha256,
                                    "parityHash": decision_score_parity["sha256"],
                                    "disagreementHash": disagreement.sha256,
                                },
                            )
                    analysis_seconds += time.perf_counter() - analysis_started
                    tactical_started = time.perf_counter()
                    adjudication_evidence = dict(
                        getattr(analysis, "adjudication_evidence", None) or {}
                    )
                    post_trace_unresolved_ids = {
                        *(str(item) for item in analysis.unresolved_decision_ids),
                        *(
                            str(item)
                            for item in adjudication_evidence.get(
                                "unresolvedDecisionIds", ()
                            )
                        ),
                    }
                    post_trace_competent_ids = {
                        str(item)
                        for item in adjudication_evidence.get(
                            "competentDecisionIds", ()
                        )
                    }
                    tactical = self._verify_tactics(
                        simulation.final_trace,
                        analysis,
                        recipe=recipe,
                        cancel_check=should_cancel,
                        context=context,
                        featured_truth_blocked_ids=(
                            post_trace_competent_ids | post_trace_unresolved_ids
                        ),
                    )
                    if self._tactical_verification_required(recipe):
                        tactical_verification_candidates += 1
                        tactical_verification_decisions += len(tactical)
                        tactical_verification_seconds += (
                            time.perf_counter() - tactical_started
                        )
                    audit = self._decision_audit(
                        simulation.final_trace,
                        direct_deep_analysis,
                        tactical,
                        context=context,
                        configuration=(
                            recipe.configuration
                            if isinstance(recipe, ResearchRecipeV3)
                            else None
                        ),
                    )
                    qualification = self._qualification(
                        recipe, audit, analysis, context=context
                    )
                    if (
                        compiled_play_plan is not None
                        and compiled_play_plan.solver_status != "optimal"
                    ):
                        qualification = {
                            **qualification,
                            "qualified": False,
                            "truthLabel": "rejected",
                            "failures": [
                                *qualification["failures"],
                                "play_construction_solver_incomplete",
                            ],
                        }
                    governed_score = score_profile.score(analysis.laffa_evidence)
                    decision_selection = evaluate_decision_selection_v1(
                        decision_score_profile,
                        perspectives=analysis.laffa_evidence.perspectives,
                        analyses=analysis.decisions,
                        analysis_hash=analysis.sha256,
                        decision_kinds={
                            str(item["decisionId"]): str(item["decisionKind"])
                            for item in audit["decisions"]
                        },
                        tactical_verifications=tactical,
                        competent_decision_ids=post_trace_competent_ids,
                        unresolved_decision_ids=post_trace_unresolved_ids,
                        authority_evidence_hash=adjudication_evidence.get("sha256"),
                        authority_evidence=adjudication_evidence,
                    )
                    decision_score_rows = list(decision_selection["decisions"])
                    qualification = apply_decision_selection_qualification_v1(
                        qualification,
                        decision_selection,
                    )
                    if not governed_score.passed:
                        score_failures = [
                            f"score_profile_gate:{reason}"
                            for reason in governed_score.rejection_reasons
                        ]
                        qualification = {
                            **qualification,
                            "qualified": False,
                            "truthLabel": "rejected",
                            "failures": [*qualification["failures"], *score_failures],
                        }
                    incomplete_failures = sorted(
                        {
                            *(
                                ("play_construction_solver_incomplete",)
                                if compiled_play_plan is not None
                                and compiled_play_plan.solver_status != "optimal"
                                else ()
                            ),
                            *(
                                ("bounded_decision_analysis_unresolved",)
                                if post_trace_unresolved_ids
                                else ()
                            ),
                        }
                    )
                    if incomplete_failures:
                        finalize_audit_trace(
                            audit_trace_record,
                            disposition="rejected",
                            failures=[
                                "stage_bounded_candidate_evaluation",
                                *incomplete_failures,
                            ],
                        )
                        return build_candidate_evaluation(
                            candidate,
                            SequentialCandidateOutcomeKindV1.INCOMPLETE,
                            incomplete_failures,
                            evidence={
                                "stage": "bounded_candidate_evaluation",
                                "traceHash": simulation.final_trace.sha256,
                                "analysisHash": analysis.sha256,
                                "auditHash": audit["sha256"],
                                "qualificationHash": canonical_sha256(qualification),
                                "unresolvedDecisionIds": sorted(
                                    post_trace_unresolved_ids
                                ),
                            },
                        )
                    trace_bytes = _canonical_research_json_bytes(
                        simulation.final_trace.to_dict(trusted=True)
                    )
                    analysis_bytes = _canonical_research_json_bytes(analysis.to_dict())
                    audit_analysis_bytes = _canonical_research_json_bytes(
                        direct_deep_analysis.to_dict()
                    )
                    audit_bytes = _canonical_research_json_bytes(audit)
                    tactical_payload = {
                        "schemaVersion": 1,
                        "traceHash": simulation.final_trace.sha256,
                        "verifications": [item.to_dict() for item in tactical],
                        **standalone_configuration_evidence,
                    }
                    tactical_payload["sha256"] = canonical_sha256(tactical_payload)
                    tactical_bytes = json.dumps(
                        tactical_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    artifacts = {
                        "completeTrace": _bind_result_artifact_owner_v1(
                            audit_trace_record.complete_trace,
                            payload=trace_bytes,
                            media_type=COMPLETE_TRACE_MEDIA_TYPE,
                            owner_kind="complete_trace_v1",
                            owner_hash=simulation.final_trace.sha256,
                        ),
                        "analysis": _bind_result_artifact_owner_v1(
                            self.artifact_store.put(
                                analysis_bytes,
                                media_type=_TRACE_ANALYSIS_MEDIA_TYPE,
                                name=f"{analysis.sha256}.json",
                            ),
                            payload=analysis_bytes,
                            media_type=_TRACE_ANALYSIS_MEDIA_TYPE,
                            owner_kind="trace_analysis_v1",
                            owner_hash=analysis.sha256,
                        ),
                        "auditAnalysis": _bind_result_artifact_owner_v1(
                            self.artifact_store.put(
                                audit_analysis_bytes,
                                media_type=_TRACE_ANALYSIS_MEDIA_TYPE,
                                name=f"{direct_deep_analysis.sha256}.json",
                            ),
                            payload=audit_analysis_bytes,
                            media_type=_TRACE_ANALYSIS_MEDIA_TYPE,
                            owner_kind="trace_analysis_v1",
                            owner_hash=direct_deep_analysis.sha256,
                        ),
                        "decisionAudit": _bind_result_artifact_owner_v1(
                            self.artifact_store.put(
                                audit_bytes,
                                media_type=DECISION_AUDIT_V2_MEDIA_TYPE,
                                name=f"{audit['sha256']}.json",
                            ),
                            payload=audit_bytes,
                            media_type=DECISION_AUDIT_V2_MEDIA_TYPE,
                            owner_kind="decision_audit_v2",
                            owner_hash=str(audit["sha256"]),
                        ),
                        "tacticalEvidence": self.artifact_store.put(
                            tactical_bytes,
                            media_type="application/vnd.baloot.tactical-evidence+json",
                            name=f"{tactical_payload['sha256']}.json",
                        ),
                    }
                    if direct_deep_analysis_ref is not None:
                        artifacts["primaryDecisionScoreAnalysis"] = copy.deepcopy(
                            dict(direct_deep_analysis_ref)
                        )
                    if decision_score_parity_ref is not None:
                        artifacts["decisionScoreParity"] = copy.deepcopy(
                            dict(decision_score_parity_ref)
                        )
                    laffa = analysis.laffa_evidence
                    base_score = float(governed_score.score)
                    preference_scores = [
                        copy.deepcopy(dict(item))
                        for item in qualification["tacticEvaluations"]
                        if item["mode"] == "preference"
                    ]
                    score = _preference_selection_score(
                        base_score,
                        preference_scores,
                    )
                    score_evidence = {
                        "schemaVersion": 1,
                        "family": "research_content_selection",
                        "constructionObjective": {
                            "kind": (
                                "cp_sat_structural_guidance"
                                if generation_lane == "ortools"
                                else "organic_runtime_baseline"
                            ),
                            "score": candidate.structural_objective_score,
                            "isQualificationEvidence": False,
                            "proofBoundary": "guides construction only; replay and decision analysis govern acceptance",
                        },
                        "laffaThrillScore": governed_score.to_dict(),
                        "baseLaffaScore": governed_score.to_dict(),
                        "decisionChallengeScores": decision_score_rows,
                        "decisionSelectionQualification": {
                            key: copy.deepcopy(value)
                            for key, value in decision_selection.items()
                            if key != "decisions"
                        },
                        "peakDecisionChallengeScore": (
                            float(decision_score_rows[0]["score"]["score"])
                            if decision_score_rows
                            else None
                        ),
                        "preferenceScores": preference_scores,
                        "selectionScore": score,
                        "profileHash": recipe.profiles["scoreProfileHash"],
                        "decisionProfileHash": decision_score_profile_hash,
                        **(
                            copy.deepcopy(dict(audit["configurationProvenance"]))
                            if audit.get("configurationProvenance") is not None
                            else {}
                        ),
                    }
                    score_evidence["sha256"] = canonical_sha256(score_evidence)
                    identity = {
                        "schemaVersion": 1,
                        "runCode": run_code,
                        "shardId": shard["shardId"],
                        "shardIndex": int(shard["shardIndex"]),
                        "recipeHash": recipe.sha256,
                        "traceHash": simulation.final_trace.sha256,
                        "analysisHash": analysis.sha256,
                        "auditAnalysisHash": direct_deep_analysis.sha256,
                        "auditHash": audit["sha256"],
                        **(
                            {
                                "primaryDecisionScoreAnalysisHash": (
                                    direct_deep_analysis.sha256
                                ),
                                "decisionScoreParityHash": (
                                    decision_score_parity["sha256"]
                                ),
                            }
                            if decision_score_parity is not None
                            else {}
                        ),
                        "auctionPlanHash": (
                            auction_plan.sha256 if auction_plan is not None else None
                        ),
                    }
                    result_hash = canonical_sha256(identity)
                    analysis_critical_decision_ids = {
                        str(item) for item in laffa.critical_decision_ids
                    }
                    critical_decision_ids = set(
                        qualification["featuredTruthEligibleCriticalDecisionIds"]
                    )
                    acting_seats = sorted(
                        {
                            item.acting_seat.name.lower()
                            for item in laffa.perspectives
                            if item.decision_id in critical_decision_ids
                        }
                    )
                    observer_seats = sorted(
                        {
                            item.observer_seat.name.lower()
                            for item in laffa.perspectives
                            if item.decision_id in critical_decision_ids
                        }
                    )
                    trusted_final_state = dict(
                        simulation.final_trace.trusted_final_state
                    )
                    project_kinds = sorted(
                        {
                            str(item.get("kind") or "").strip()
                            for item in trusted_final_state.get("projects") or ()
                            if isinstance(item, Mapping)
                            and str(item.get("kind") or "").strip()
                        }
                    )
                    result = {
                        "schemaVersion": 1,
                        "resultCode": f"LAF-{result_hash[:10].upper()}",
                        "resultHash": result_hash,
                        "runCode": run_code,
                        "recipeCode": recipe.code,
                        "traceHash": simulation.final_trace.sha256,
                        "gameHash": simulation.final_trace.game_hash,
                        "score": score,
                        "scoreEvidence": score_evidence,
                        "truthLabel": qualification["truthLabel"],
                        "confidence": qualification["confidence"],
                        "disposition": "qualified"
                        if qualification["qualified"]
                        else "rejected",
                        "qualification": qualification,
                        "classification": laffa.classification,
                        "analysisCriticalDecisionIds": sorted(
                            analysis_critical_decision_ids
                        ),
                        "criticalDecisionIds": sorted(critical_decision_ids),
                        "featuredTruthBlockedDecisionIds": list(
                            qualification["featuredTruthBlockedDecisionIds"]
                        ),
                        "interestingSeats": acting_seats,
                        "tacticalTags": [item["kind"] for item in recipe.tactics],
                        "verifiedTacticalTags": sorted(qualification["tacticScores"]),
                        "decisionCount": audit["decisionCount"],
                        "strategicSelection": strategic_summary,
                        "libraryProjection": {
                            "schemaVersion": 1,
                            "actingSeats": acting_seats,
                            "observerSeats": observer_seats,
                            "contract": str(trusted_final_state.get("contract") or "")
                            .strip()
                            .lower()
                            or None,
                            "purchaseRound": trusted_final_state.get("purchaseRound"),
                            "trumpSuit": str(trusted_final_state.get("trumpSuit") or "")
                            .strip()
                            .lower()
                            or None,
                            "escalation": str(
                                trusted_final_state.get("escalation") or "normal"
                            )
                            .strip()
                            .lower(),
                            "projectKinds": project_kinds,
                            "generationLane": generation_lane,
                        },
                        "artifacts": artifacts,
                        "provenance": {
                            "generationLane": generation_lane,
                            "finalistPrimaryPolicy": bool(
                                decision_score_parity is not None
                            ),
                            "decisionScoreParityStatus": (
                                "matched"
                                if decision_score_parity is not None
                                else "not_applicable"
                            ),
                            "decisionScoreParityHash": (
                                decision_score_parity["sha256"]
                                if decision_score_parity is not None
                                else None
                            ),
                            "primaryDecisionScoreAnalysisHash": (
                                direct_deep_analysis.sha256
                                if decision_score_parity is not None
                                else None
                            ),
                            "auditAnalysisHash": direct_deep_analysis.sha256,
                            "generatorPlugin": candidate.provenance_dict()["pluginId"],
                            **(
                                copy.deepcopy(dict(audit["configurationProvenance"]))
                                if audit.get("configurationProvenance") is not None
                                else {}
                            ),
                            "domainHash": (
                                compiled_domain.sha256
                                if context is not None and compiled_domain is not None
                                else domain.sha256
                            ),
                            "legacyDomainV2Hash": (
                                domain.sha256 if context is not None else None
                            ),
                            "compiledDomainHash": (
                                compiled_domain.sha256
                                if compiled_domain is not None
                                else None
                            ),
                            "compilerExecutableBindingComplete": (
                                compiled_domain.compiled.coverage.executable_binding_complete
                                if compiled_domain is not None
                                else False
                            ),
                            "compilerBehavioralOrInvariantComplete": (
                                compiled_domain.compiled.coverage.behavioral_or_invariant_complete
                                if compiled_domain is not None
                                else False
                            ),
                            "recipeHash": recipe.sha256,
                            "researchShardId": shard["shardId"],
                            "researchShardHash": shard["sha256"],
                            "researchShardIndex": int(shard["shardIndex"]),
                            "researchSeed": int(seed),
                            "candidateHash": candidate.candidate_hash,
                            "modelHash": candidate.model_hash,
                            "solverStatus": candidate.solver_status,
                            "solverVersion": candidate.backend_version,
                            "structuralObjectiveScore": candidate.structural_objective_score,
                            "auctionConstructionMode": (
                                str(resources.get("auctionConstructionMode"))
                                if resources.get("auctionConstructionMode") is not None
                                else (
                                    "cp_sat_feasibility_line"
                                    if generation_lane == "ortools"
                                    else "not_applicable"
                                )
                            ),
                            "strategicAuctionProfileHash": (
                                str(
                                    (resources.get("strategicAuction") or {}).get(
                                        "sha256"
                                    )
                                )
                                if resources.get("strategicAuction") is not None
                                else None
                            ),
                            "auctionCompilerPlanHash": (
                                auction_plan.sha256
                                if auction_plan is not None
                                else None
                            ),
                            "auctionCompilerModelHash": (
                                auction_plan.model_hash
                                if auction_plan is not None
                                else None
                            ),
                            "auctionCompilerSolverStatus": (
                                auction_plan.solver_status
                                if auction_plan is not None
                                else None
                            ),
                            "projectCompilerPlanHash": (
                                compiled_project_plan.sha256
                                if compiled_project_plan is not None
                                else None
                            ),
                            "playCompilerPlanHash": (
                                compiled_play_plan.sha256
                                if compiled_play_plan is not None
                                else None
                            ),
                            "playCompilerModelHash": (
                                compiled_play_plan.model_hash
                                if compiled_play_plan is not None
                                else None
                            ),
                            "playCompilerSolverStatus": (
                                compiled_play_plan.solver_status
                                if compiled_play_plan is not None
                                else None
                            ),
                            "playConstructionMode": (
                                str(resources.get("playConstructionMode"))
                                if resources.get("playConstructionMode") is not None
                                else (
                                    "cp_sat_feasibility_line"
                                    if generation_lane == "ortools"
                                    else "not_applicable"
                                )
                            ),
                            "strategicPlayProfileHash": strategic_summary[
                                "profileHash"
                            ],
                            "strategicSelectionDecisionCount": strategic_summary[
                                "decisionCount"
                            ],
                            "strategicSelectionTruthLevels": strategic_summary[
                                "truthLevels"
                            ],
                            "strategicSelectionProofScope": strategic_summary[
                                "proofScope"
                            ],
                            "globalStrategicOptimalityClaimed": False,
                            **_tactical_verification_provenance(
                                recipe,
                                len(tactical),
                            ),
                            "engineVersion": simulation.final_trace.engine_version,
                            "bundleHash": simulation.final_trace.bundle_hash,
                            "rulesHash": simulation.final_trace.rules_hash,
                        },
                        "createdAt": str(run["createdAt"]),
                    }
                    if (
                        bool(qualification["qualified"])
                        and first_qualified_result_seconds is None
                    ):
                        first_qualified_result_seconds = (
                            time.perf_counter() - started_clock
                        )
                    put_candidate_result(
                        result,
                        attempt_count=int(shard["attemptCount"]),
                        fencing_token=int(shard["fencingToken"]),
                        worker_id=worker_id,
                    )
                    audit_disposition = (
                        "qualified" if qualification["qualified"] else "rejected"
                    )
                    finalize_audit_trace(
                        audit_trace_record,
                        disposition=audit_disposition,
                        failures=(
                            []
                            if qualification["qualified"]
                            else [
                                "stage_content_qualification",
                                *[str(item) for item in qualification["failures"]],
                            ]
                        ),
                    )
                    result_count += 1
                    result_codes.append(str(result["resultCode"]))
                    qualified_count += int(bool(qualification["qualified"]))
                    rejected_count += int(not bool(qualification["qualified"]))
                    publication_core = copy.deepcopy(result)
                    publication_core.pop("createdAt", None)
                    outcome = (
                        SequentialCandidateOutcomeKindV1.QUALIFIED
                        if qualification["qualified"]
                        else SequentialCandidateOutcomeKindV1.ORDINARY_REJECTION
                    )
                    return build_candidate_evaluation(
                        candidate,
                        outcome,
                        () if qualification["qualified"] else qualification["failures"],
                        evidence={
                            "stage": "authoritative_candidate_publication",
                            "traceHash": simulation.final_trace.sha256,
                            "analysisHash": analysis.sha256,
                            "directDeepAnalysisHash": direct_deep_analysis.sha256,
                            "auditHash": audit["sha256"],
                            "qualificationHash": canonical_sha256(qualification),
                            "resultPublicationCoreHash": canonical_sha256(
                                publication_core
                            ),
                        },
                    )

                if generation_lane == "organic_runtime":
                    for organic_candidate, organic_simulation in candidate_rows:
                        organic_evaluation = evaluate_candidate(
                            organic_candidate,
                            1,
                            prepared_simulation=organic_simulation,
                        )
                        if (
                            organic_evaluation.outcome
                            == SequentialCandidateOutcomeKindV1.CANCELLED
                        ):
                            return
                    continue

                evaluation_drafts: list[dict[str, Any]] = []
                primary_candidate_hashes: set[str] = set()

                def staged_sequential_evaluator(
                    candidate: Any,
                    evaluation_index: int,
                ) -> SequentialCandidateEvaluationV1:
                    nonlocal staged_candidate_effects
                    candidate_hash = str(candidate.candidate_hash)
                    repeated = candidate_hash in primary_candidate_hashes
                    before = candidate_metric_snapshot()
                    staged_candidate_effects = []
                    try:
                        evaluation = evaluate_candidate(candidate, evaluation_index)
                        effects = list(staged_candidate_effects)
                    except Exception:
                        if repeated:
                            restore_candidate_metrics(before)
                        raise
                    finally:
                        staged_candidate_effects = None
                    evaluation_drafts.append(
                        {
                            "candidate": candidate,
                            "evaluation": evaluation,
                            "effects": effects,
                            "metricSnapshot": before,
                            "repeat": repeated,
                        }
                    )
                    if repeated:
                        restore_candidate_metrics(before)
                    else:
                        primary_candidate_hashes.add(candidate_hash)
                    return evaluation

                search_result = run_sequential_candidate_search_v1(
                    self.deal_generator,
                    spec,
                    root_seed=int(seed),
                    dealer=Seat.parse(recipe.situation["dealer"]),
                    rules=simulator.rules,
                    evaluate_candidate=staged_sequential_evaluator,
                )
                search_receipt = search_result.receipt.verify().to_dict()
                self.store.put_sequential_search_receipt(
                    search_receipt,
                    run_code=run_code,
                    shard_id=str(shard["shardId"]),
                    attempt_count=int(shard["attemptCount"]),
                    fencing_token=int(shard["fencingToken"]),
                    worker_id=worker_id,
                    seed=int(seed),
                )

                accepted_candidate_hash = (
                    str(search_result.qualified_candidate.candidate_hash)
                    if search_result.qualified_candidate is not None
                    else None
                )
                primary_qualified_draft = next(
                    (
                        draft
                        for draft in evaluation_drafts
                        if not bool(draft["repeat"])
                        and draft["evaluation"].outcome
                        == SequentialCandidateOutcomeKindV1.QUALIFIED
                    ),
                    None,
                )
                repeat_draft = next(
                    (
                        draft
                        for draft in reversed(evaluation_drafts)
                        if bool(draft["repeat"])
                    ),
                    None,
                )

                committed_repeat_effects: list[
                    tuple[str, tuple[Any, ...], dict[str, Any]]
                ] = []
                for draft in evaluation_drafts:
                    evaluation = draft["evaluation"]
                    if not bool(draft["repeat"]):
                        if (
                            evaluation.outcome
                            == SequentialCandidateOutcomeKindV1.QUALIFIED
                            and str(draft["candidate"].candidate_hash)
                            != accepted_candidate_hash
                        ):
                            continue
                        commit_candidate_effects(draft["effects"])
                        continue
                    if accepted_candidate_hash is not None:
                        continue
                    if evaluation.outcome in {
                        SequentialCandidateOutcomeKindV1.FATAL_QUARANTINE,
                        SequentialCandidateOutcomeKindV1.INCOMPLETE,
                        SequentialCandidateOutcomeKindV1.CANCELLED,
                    }:
                        repeat_effects = [
                            effect
                            for effect in draft["effects"]
                            if effect[0] != "put_result"
                        ]
                        commit_candidate_effects(repeat_effects)
                        committed_repeat_effects.extend(repeat_effects)
                        if (
                            evaluation.outcome
                            == SequentialCandidateOutcomeKindV1.FATAL_QUARANTINE
                        ):
                            quarantine_count += sum(
                                1
                                for effect in repeat_effects
                                if effect[0] == "put_disagreement"
                            )

                if (
                    primary_qualified_draft is not None
                    and accepted_candidate_hash is None
                ):
                    primary_snapshot = primary_qualified_draft["metricSnapshot"]
                    result_count = int(primary_snapshot["resultCount"])
                    qualified_count = int(primary_snapshot["qualifiedCount"])
                    rejected_count = int(primary_snapshot["rejectedCount"])
                    first_qualified_result_seconds = primary_snapshot[
                        "firstQualifiedResultSeconds"
                    ]
                    del result_codes[int(primary_snapshot["resultCodeCount"]) :]

                    if search_receipt["stopReason"] == "fatal_quarantine":
                        disagreement_references = [
                            {
                                "disagreementId": str(effect[1][0]["disagreementId"]),
                                "sha256": str(effect[1][0]["sha256"]),
                                "category": str(effect[1][0]["category"]),
                                "status": str(effect[1][0]["status"]),
                            }
                            for effect in committed_repeat_effects
                            if effect[0] == "put_disagreement"
                        ]
                        finalized_record_hashes = {
                            str(effect[1][0])
                            for effect in committed_repeat_effects
                            if effect[0] == "finalize_audit_trace"
                        }
                        repeat_evaluation = (
                            repeat_draft["evaluation"]
                            if repeat_draft is not None
                            else None
                        )
                        if not disagreement_references:
                            determinism_disagreement = SemanticDisagreementV2.build(
                                run_code=run_code,
                                candidate_hash=str(
                                    primary_qualified_draft["candidate"].candidate_hash
                                ),
                                category="runtime_transition_defect",
                                solver_evidence={
                                    **primary_qualified_draft[
                                        "candidate"
                                    ].provenance_dict(),
                                    **standalone_configuration_evidence,
                                    "stage": "qualified_repeat_determinism",
                                    "primaryEvaluation": primary_qualified_draft[
                                        "evaluation"
                                    ].to_dict(),
                                    "searchReceiptHash": search_result.receipt.sha256,
                                },
                                runtime_evidence={
                                    **standalone_configuration_evidence,
                                    "stage": "qualified_repeat_determinism",
                                    "repeatEvaluation": (
                                        repeat_evaluation.to_dict()
                                        if repeat_evaluation is not None
                                        else None
                                    ),
                                    "searchReceiptHash": search_result.receipt.sha256,
                                },
                            )
                            self.store.put_disagreement(
                                determinism_disagreement.to_dict(),
                                shard_id=str(shard["shardId"]),
                                attempt_count=int(shard["attemptCount"]),
                                fencing_token=int(shard["fencingToken"]),
                                worker_id=worker_id,
                            )
                            quarantine_count += 1
                            disagreement_references.append(
                                disagreement_reference(determinism_disagreement)
                            )
                        record_hashes = {
                            str(effect[1][0])
                            for effect in primary_qualified_draft["effects"]
                            if effect[0] == "finalize_audit_trace"
                        }
                        for record_hash in sorted(
                            record_hashes - finalized_record_hashes
                        ):
                            self.audit_traces.finalize(
                                record_hash,
                                disposition="quarantined",
                                failure_categories=[
                                    "stage_qualified_repeat_determinism",
                                    *sorted(
                                        {
                                            str(item["category"])
                                            for item in disagreement_references
                                        }
                                    ),
                                ],
                                disagreement_references=disagreement_references,
                            )
                        events.append(
                            {
                                "type": "qualified_repeat_quarantine",
                                "seed": int(seed),
                                "candidateHash": str(
                                    primary_qualified_draft["candidate"].candidate_hash
                                ),
                                "disagreementIds": [
                                    str(item["disagreementId"])
                                    for item in disagreement_references
                                ],
                                "searchReceiptHash": search_result.receipt.sha256,
                                "at": _now(),
                            }
                        )

                events.append(
                    {
                        "type": "sequential_candidate_search",
                        "seed": int(seed),
                        "receiptHash": search_result.receipt.sha256,
                        "stopReason": search_receipt["stopReason"],
                        "emittedCandidateCount": search_receipt[
                            "emittedCandidateCount"
                        ],
                        "evaluatedCandidateCount": search_receipt[
                            "evaluatedCandidateCount"
                        ],
                        "rejectedCandidateCount": search_receipt[
                            "rejectedCandidateCount"
                        ],
                        "at": _now(),
                    }
                )
                if search_receipt["stopReason"] == "cancelled":
                    return
            cache_scopes: dict[str, dict[str, int]] = {}
            if self.execution_context_factory is not None:
                for (
                    snapshot
                ) in self.execution_context_factory.cache_metrics_snapshots():
                    scope = str(snapshot["scope"])
                    cache_scopes[scope] = {
                        key: int(snapshot[key])
                        for key in (
                            "hits",
                            "misses",
                            "evictions",
                            "entries",
                            "capacity",
                        )
                    }
            peak_process_rss = process_lifetime_peak_rss_bytes()
            environment = build_research_environment_declaration(
                generation_lane=generation_lane,
                configured_shard_workers=self.maximum_local_shard_workers,
                effective_shard_workers=min(
                    self.maximum_local_shard_workers,
                    len(self.store.list_shards(run_code)),
                ),
                cp_sat_workers_per_solve=(
                    next(iter(cp_sat_worker_counts))
                    if len(cp_sat_worker_counts) == 1
                    else None
                ),
                research_store=self.store,
                artifact_store=self.artifact_store,
            )
            self.store.complete_shard(
                str(shard["shardId"]),
                worker_id=worker_id,
                fencing_token=int(shard["fencingToken"]),
                result={
                    "schemaVersion": 1,
                    "shardId": shard["shardId"],
                    "shardHash": shard["sha256"],
                    "requested": requested,
                    "attemptedSeeds": attempted_seeds,
                    "structuralCandidates": structural_candidates,
                    "auctionFeasible": auction_feasible,
                    "solverOptimal": solver_optimal,
                    "solverIncomplete": solver_incomplete,
                    "runtimeParityPassed": runtime_parity_passed,
                    "semanticReplayPassed": semantic_replay_passed,
                    "playCompilerAttempted": play_compiler_attempted,
                    "playCompilerOptimal": play_compiler_optimal,
                    "playCompilerFeasible": play_compiler_feasible,
                    "playCompilerIncomplete": play_compiler_incomplete,
                    "strategicSelectionDecisions": strategic_selection_decisions,
                    "strategicExactDecisions": strategic_exact_decisions,
                    "strategicProvisionalDecisions": strategic_provisional_decisions,
                    "strategicFallbackDecisions": strategic_fallback_decisions,
                    "generationLane": generation_lane,
                    "resultCount": result_count,
                    "qualifiedCount": qualified_count,
                    "rejectedCount": rejected_count,
                    "quarantineCount": quarantine_count,
                    "tacticalVerificationCandidates": tactical_verification_candidates,
                    "tacticalVerificationDecisions": tactical_verification_decisions,
                    "tacticalVerificationSeconds": tactical_verification_seconds,
                    "constructionSeconds": construction_seconds,
                    "simulationSeconds": simulation_seconds,
                    "semanticParitySeconds": semantic_parity_seconds,
                    "analysisSeconds": analysis_seconds,
                    "constructionPreScreenCandidates": (
                        construction_pre_screen_candidates
                    ),
                    "constructionPreScreenRejected": (construction_pre_screen_rejected),
                    "tacticalViabilityPreScreenCandidates": (
                        tactical_viability_pre_screen_candidates
                    ),
                    "tacticalViabilityPreScreenRejected": (
                        tactical_viability_pre_screen_rejected
                    ),
                    "tacticalViabilityPreScreenSeconds": (
                        tactical_viability_pre_screen_seconds
                    ),
                    "constructionTargetMissCount": (construction_target_miss_count),
                    "constructionPreScreenSeconds": (construction_pre_screen_seconds),
                    "analyzedCandidateCount": analyzed_candidate_count,
                    "firstFeasibleCandidateSeconds": first_feasible_candidate_seconds,
                    "firstOptimalCandidateSeconds": first_optimal_candidate_seconds,
                    "firstQualifiedResultSeconds": first_qualified_result_seconds,
                    "shardWallWorkSeconds": time.perf_counter() - started_clock,
                    "cpuSeconds": time.thread_time() - started_cpu,
                    "cpuMeasurementBoundary": CALLING_THREAD_CPU_BOUNDARY,
                    "peakProcessRssBytes": peak_process_rss,
                    "peakRssMeasurementBoundary": (
                        PROCESS_PEAK_RSS_BOUNDARY
                        if peak_process_rss is not None
                        else None
                    ),
                    "environment": environment,
                    "cacheScopes": cache_scopes,
                    "optimalityCutCount": optimality_cut_count,
                    "resultCodes": sorted(set(result_codes)),
                    "events": events,
                },
            )
        except InterruptedError:
            if not self._cancel_check(run_code):
                self.store.fail_shard(
                    str(shard["shardId"]),
                    worker_id=worker_id,
                    fencing_token=int(shard["fencingToken"]),
                    error={"type": "InterruptedError", "message": "shard interrupted"},
                    retryable=True,
                )
        except Exception as exc:
            error, retryable = _shard_failure_evidence(exc)
            try:
                self.store.fail_shard(
                    str(shard["shardId"]),
                    worker_id=worker_id,
                    fencing_token=int(shard["fencingToken"]),
                    error=error,
                    retryable=retryable,
                )
            except RuntimeError:
                # A newer fencing token owns the shard; stale work may not
                # overwrite its state.
                return
        finally:
            lease.stop()
